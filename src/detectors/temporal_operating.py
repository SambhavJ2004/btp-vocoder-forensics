"""Source-disjoint folds, exact weighted thresholds and common-mask ranking."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from detectors.temporal_context import REJECT_ALL_THRESHOLD
from detectors.temporal_features import guarded_labels, sample_labels, smooth_probabilities
from detectors.temporal_metrics import evaluate_localization


def grouped_folds(group_ids, seed, n_folds=3):
    groups = sorted(set(group_ids))
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**31
        or not isinstance(n_folds, int)
        or isinstance(n_folds, bool)
        or not 2 <= n_folds <= len(groups)
        or any(not isinstance(g, str) or not g for g in groups)
    ):
        raise ValueError("Require seeded source groups and at least two populated folds.")
    ordered = sorted(
        groups, key=lambda g: (hashlib.sha256(f"{seed}:temporal3b:{g}".encode()).hexdigest(), g)
    )
    return {group: i % n_folds for i, group in enumerate(ordered)}


def sample_cells(record, config, score_guard):
    row = record["row"]
    n = row["num_samples"]
    starts = record["starts"]
    centers = np.minimum(starts + config.window_samples / 2, n)
    edges = np.r_[0, np.floor((centers[:-1] + centers[1:]) / 2).astype(np.int64), n]
    if len(edges) != len(starts) + 1 or (np.diff(edges) < 0).any():
        raise ValueError("Invalid sample projection cells.")
    truth = record.get("score_labels")
    if truth is None:
        truth = guarded_labels(sample_labels(row["interval_labels"], n), score_guard)
    counts = []
    for label in (0, 1):
        prefix = np.r_[0, np.cumsum(truth == label)]
        counts.append(np.diff(prefix[edges]))
    return counts


def weighted_control_cutoff(scores, weights, target):
    """Minimum representable >= cutoff obeying a tied, sample-weighted tail cap.

    Entire tied groups cross together. A cutoff one float above 1 is an explicit
    all-negative point, including sigmoid/smoothing values saturated at 1.
    """
    scores, weights = np.asarray(scores), np.asarray(weights)
    if (
        scores.ndim != 1
        or weights.shape != scores.shape
        or not len(scores)
        or not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
        or not np.isfinite(target)
        or not 0 <= target <= 1
    ):
        raise ValueError("Invalid scores, sample weights or FPR target.")
    keep = weights > 0
    scores, weights = scores[keep], weights[keep]
    if target == 1:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ss, ww = scores[order], weights[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    cumulative = np.cumsum(ww)[ends]
    exceeded = np.flatnonzero(cumulative > target * weights.sum())
    cutoff = float(np.nextafter(ss[ends[exceeded[0]]], np.inf))
    return min(cutoff, REJECT_ALL_THRESHOLD)


def control_operating_points(records, margins, config, score_guard, targets=(0.01, 0.05, 0.10)):
    controls = defaultdict(lambda: ([], []))
    for record, margin in zip(records, margins, strict=True):
        kind = record["row"]["example_class"]
        if kind not in ("null_control", "real_to_real_control"):
            continue
        from scipy.special import expit

        p = smooth_probabilities(expit(margin), 3)
        real, synthetic = sample_cells(record, config, score_guard)
        if synthetic.any():
            raise ValueError("Control has synthetic-labeled samples.")
        controls[kind][0].append(p)
        controls[kind][1].append(real)
    if set(controls) != {"null_control", "real_to_real_control"}:
        raise ValueError("Both real control types are required for threshold validation.")
    arrays = {kind: tuple(np.concatenate(v) for v in pair) for kind, pair in controls.items()}
    result = {}
    for target in targets:
        cutoffs = {kind: weighted_control_cutoff(p, w, target) for kind, (p, w) in arrays.items()}
        threshold = max(cutoffs.values())
        fprs = {kind: float(w[p >= threshold].sum() / w.sum()) for kind, (p, w) in arrays.items()}
        if any(value > target + 1e-15 for value in fprs.values()):
            raise ValueError("Weighted threshold failed its explicit control constraint.")
        result[f"fpr_{round(target * 100):02d}"] = {
            "target_sample_fpr": target,
            "threshold": threshold,
            "oof_control_sample_fpr": fprs,
            "per_control_cutoff": cutoffs,
            "control_valid_samples": {kind: int(w.sum()) for kind, (_, w) in arrays.items()},
            "selection_scope": "train_group_out_of_fold_only",
            "constraint": "each_control_separately",
        }
    return result


def weighted_roc(y, score, weight):
    y, score, weight = np.asarray(y), np.asarray(score), np.asarray(weight)
    good = weight > 0
    if set(y[good]) != {0, 1}:
        return {"auroc": None, "directional_eer": None}
    fpr, tpr, _ = roc_curve(
        y[good], score[good], sample_weight=weight[good], drop_intermediate=False
    )
    d = fpr - (1 - tpr)
    i = int(np.searchsorted(d, 0))
    eer = (
        fpr[i]
        if d[i] == 0
        else fpr[i - 1] + (-d[i - 1]) / (d[i] - d[i - 1]) * (fpr[i] - fpr[i - 1])
    )
    return {
        "auroc": float(roc_auc_score(y[good], score[good], sample_weight=weight[good])),
        "directional_eer": float(eer),
    }


def common_mask_ranking(records, scores, config, score_guard):
    # Each cell may contain both classes; use one weighted observation for each
    # class. This equals expanding every scored sample without millions of rows.
    yy, ss, ww = [], [], []
    for record, score in zip(records, scores, strict=True):
        counts = sample_cells(record, config, score_guard)
        for label, weights in enumerate(counts):
            yy.append(np.full(len(score), label))
            ss.append(score)
            ww.append(weights)
    return weighted_roc(np.concatenate(yy), np.concatenate(ss), np.concatenate(ww))


def evaluate_context(records, probabilities, config, threshold, score_guard, *, detailed=True):
    if not np.isfinite(threshold) or not 0 <= threshold <= REJECT_ALL_THRESHOLD:
        raise ValueError("Invalid declared threshold.")
    # The all-negative sentinel retains raw margins/ranking, with no positive
    # decisions. Ordinary operating points use the unchanged Phase 3A evaluator.
    effective = probabilities if threshold <= 1 else [np.zeros_like(p) for p in probabilities]
    result = evaluate_localization(
        records, effective, config, min(threshold, 1), 3, 0, score_guard, detailed=detailed
    )
    result["threshold"] = threshold
    result["all_negative_operating_point"] = threshold > 1
    if not detailed:
        return result
    group_values = list(result["sample_by_source_group"].values())
    result["source_group_macro_sample"] = {
        key: float(np.mean([r[key] if r[key] is not None else 0 for r in group_values]))
        for key in ("precision", "recall", "f1", "iou")
    }
    result["macro_undefined_policy"] = (
        "zero_for_undefined_per_group_precision_or_recall_no_groups_dropped"
    )
    return result
