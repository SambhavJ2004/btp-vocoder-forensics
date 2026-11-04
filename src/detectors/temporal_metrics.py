"""Source-grouped development localization metrics; no window-wise confidence intervals."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from data.invariants import ARCHIVE_SR
from detectors.eer import compute_eer
from detectors.temporal_features import (
    guarded_labels,
    predicted_intervals,
    sample_labels,
    smooth_probabilities,
)


def confusion(y, prediction) -> dict:
    y, prediction = np.asarray(y), np.asarray(prediction, dtype=bool)
    return {
        "tp": int(np.sum((y == 1) & prediction)),
        "fp": int(np.sum((y == 0) & prediction)),
        "fn": int(np.sum((y == 1) & ~prediction)),
        "tn": int(np.sum((y == 0) & ~prediction)),
    }


def add_counts(target: dict, counts: dict):
    for name in ("tp", "fp", "fn", "tn"):
        target[name] = target.get(name, 0) + counts[name]


def rates(counts: dict) -> dict:
    tp, fp, fn, tn = (counts.get(k, 0) for k in ("tp", "fp", "fn", "tn"))

    def ratio(a, b):
        return float(a / b) if b else None

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "false_positive_rate": ratio(fp, fp + tn),
        "valid_count": tp + fp + fn + tn,
    }


def binary_metrics(y, margin, prediction) -> dict:
    y, margin = np.asarray(y), np.asarray(margin)
    valid = y >= 0
    yy, mm = y[valid], margin[valid]
    result = rates(confusion(yy, np.asarray(prediction)[valid]))
    result.update(ignored_count=int(np.sum(~valid)), raw_auroc=None, raw_directional_eer=None)
    if set(yy) == {0, 1}:
        result.update(
            raw_auroc=float(roc_auc_score(yy, mm)),
            raw_directional_eer=compute_eer(-mm[yy == 0], -mm[yy == 1])[0],
        )
    return result


def evaluate_localization(
    records: list[dict],
    probabilities: list[np.ndarray],
    config,
    threshold: float,
    smoothing: int,
    merge_samples: int,
    score_guard: int,
    *,
    detailed: bool = True,
) -> dict:
    """Score complete audio predictions; labels are used only after postprocessing.

    Each mixed recording has one pure synthetic reference. Event matching uses
    highest valid-mask IoU, with earliest predicted segment breaking ties.
    """
    groups, classes = defaultdict(dict), defaultdict(dict)
    total, controls = {}, {}
    windows_y, windows_m, windows_p = [], [], []
    window_groups = defaultdict(lambda: ([], [], []))
    window_classes = defaultdict(lambda: ([], [], []))
    rows, best_ious, errors = [], [], []
    near_total, away_total = {}, {}
    event_counts = {"0.5": {}, "0.75": {}}
    for record, probability in zip(records, probabilities, strict=True):
        row, starts = record["row"], record["starts"]
        smoothed = smooth_probabilities(probability, smoothing)
        intervals = predicted_intervals(
            starts, smoothed, row["num_samples"], config, threshold, merge_samples
        )
        truth = record.get("score_labels")
        if truth is None:
            truth = guarded_labels(
                sample_labels(row["interval_labels"], row["num_samples"]), score_guard
            )
        prediction = np.zeros(len(truth), dtype=bool)
        for s, e in intervals:
            prediction[s:e] = True
        counts = confusion(truth, prediction)
        add_counts(total, counts)
        add_counts(groups[row["group_id"]], counts)
        add_counts(classes[row["example_class"]], counts)
        if not row["has_synthetic"]:
            add_counts(controls, counts)
        if not detailed:
            continue
        valid = truth >= 0
        scored_intervals = [[s, e] for s, e in intervals if np.any(valid[s:e])]
        positive = (truth == 1) & valid
        reference_samples = int(positive.sum())
        ious = []
        for s, e in scored_intervals:
            intersection = int(positive[s:e].sum())
            union = reference_samples + int(valid[s:e].sum()) - intersection
            ious.append(intersection / union if union else 0)
        best = max(ious, default=0.0) if row["has_synthetic"] else None
        if best is not None:
            best_ious.append(best)
        for level in (0.5, 0.75):
            matched = int(row["has_synthetic"] and best >= level)
            ec = event_counts[str(level)]
            ec["tp"] = ec.get("tp", 0) + matched
            ec["fp"] = ec.get("fp", 0) + len(scored_intervals) - matched
            ec["fn"] = ec.get("fn", 0) + int(row["has_synthetic"]) - matched
        if best is not None and best >= 0.5:
            matched = scored_intervals[ious.index(best)]
            core = row["pure_synthetic_ranges"][0]
            errors.append(
                [abs(matched[0] - core[0]) / ARCHIVE_SR, abs(matched[1] - core[1]) / ARCHIVE_SR]
            )
        positions = np.arange(len(truth))
        distance = np.minimum(
            abs(positions - row["start_sample"]), abs(positions - row["end_sample"])
        )
        near = distance <= round(0.25 * ARCHIVE_SR)
        add_counts(near_total, confusion(truth[near], prediction[near]))
        add_counts(away_total, confusion(truth[~near], prediction[~near]))
        wy, margin = record["window_y"], record["margin"]
        decision = smoothed >= threshold
        windows_y.append(wy)
        windows_m.append(margin)
        windows_p.append(decision)
        for collection, key in (
            (window_groups, row["group_id"]),
            (window_classes, row["example_class"]),
        ):
            for target, values in zip(collection[key], (wy, margin, decision), strict=True):
                target.append(values)
        rows.append(
            {
                "example_id": row["example_id"],
                "group_id": row["group_id"],
                "split": row["split"],
                "example_class": row["example_class"],
                "sample_metrics": rates(counts),
                "best_reference_iou": best,
                "valid_predicted_events": len(scored_intervals),
                "ground_truth_edit_samples": [row["start_sample"], row["end_sample"]],
                "ground_truth_pure_synthetic_ranges": row["pure_synthetic_ranges"],
                "ground_truth_transition_ignore_ranges": row["transition_ignore_ranges"],
                "model_predicted_intervals": [
                    {
                        "start_sample": s,
                        "end_sample": e,
                        "start_seconds": s / ARCHIVE_SR,
                        "end_seconds": e / ARCHIVE_SR,
                    }
                    for s, e in intervals
                ],
                "development_only": True,
                "citable": False,
            }
        )
    group_rates = {g: rates(c) for g, c in sorted(groups.items())}
    f1s = [m["f1"] for m in group_rates.values() if m["f1"] is not None]
    result = {
        "sample_micro": rates(total),
        "source_group_macro_f1": float(np.mean(f1s)) if f1s else None,
        "source_groups": len(groups),
        "sample_by_class": {k: rates(v) for k, v in sorted(classes.items())},
        "control_sample_false_positive_rate": rates(controls)["false_positive_rate"],
    }
    if not detailed:
        return result

    def window_summary(collection):
        return {
            key: binary_metrics(*(np.concatenate(values) for values in lists))
            for key, lists in sorted(collection.items())
        }

    class_windows = window_summary(window_classes)
    group_windows = window_summary(window_groups)

    def macro(values, key):
        usable = [m[key] for m in values if m[key] is not None]
        return float(np.mean(usable)) if usable else None

    result.update(
        window_micro=binary_metrics(
            np.concatenate(windows_y), np.concatenate(windows_m), np.concatenate(windows_p)
        ),
        window_by_source_group=group_windows,
        window_by_class=class_windows,
        window_source_macro={
            key: macro(group_windows.values(), key)
            for key in ("raw_auroc", "raw_directional_eer", "precision", "recall", "f1")
        },
        sample_by_source_group=group_rates,
        per_vocoder={
            key.removeprefix("mixed_"): {
                "window": class_windows[key],
                "sample_micro": rates(value),
            }
            for key, value in classes.items()
            if key.startswith("mixed_")
        },
        mean_best_reference_iou=float(np.mean(best_ious)) if best_ious else None,
        events={
            key: {
                k: v
                for k, v in rates({**counts, "tn": 0}).items()
                if k in {"tp", "fp", "fn", "precision", "recall", "f1"}
            }
            for key, counts in event_counts.items()
        },
        matched_onset_mae_seconds=float(np.mean(errors, axis=0)[0]) if errors else None,
        matched_offset_mae_seconds=float(np.mean(errors, axis=0)[1]) if errors else None,
        matched_references_at_iou_0_5=len(errors),
        reference_events=len(best_ious),
        boundary_diagnostics={
            "near_width_seconds": 0.25,
            "near": rates(near_total),
            "away": rates(away_total),
        },
        recordings=rows,
    )
    for vocoder, metrics in result["per_vocoder"].items():
        members = [r for r in rows if r["example_class"] == f"mixed_{vocoder}"]
        metrics["source_macro_sample_f1"] = macro([r["sample_metrics"] for r in members], "f1")
        metrics["mean_best_reference_iou"] = float(
            np.mean([r["best_reference_iou"] for r in members])
        )
    for kind in ("null_control", "real_to_real_control"):
        members = [r for r in rows if r["example_class"] == kind]
        valid_count = result["sample_by_class"][kind]["valid_count"]
        events = sum(r["valid_predicted_events"] for r in members)
        result.setdefault("controls", {})[kind] = {
            "window_false_positive_rate": class_windows[kind]["false_positive_rate"],
            "sample_false_positive_rate": result["sample_by_class"][kind]["false_positive_rate"],
            "recording_false_alarm_fraction": float(
                np.mean([r["valid_predicted_events"] > 0 for r in members])
            ),
            "events_per_valid_minute": events / (valid_count / ARCHIVE_SR / 60)
            if valid_count
            else None,
            "recordings": len(members),
        }
    return result
