"""Separate closed-set, genuine false-attribution and withheld-source metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from data.invariants import GLOBAL_SEED
from detectors.source_verification import FLAGS, SCORES


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def matrix_metrics(cm):
    cm = np.asarray(cm)
    k = cm.shape[0]
    tp = np.diag(cm[:, :k])
    rows, cols = cm.sum(axis=1), cm[:, :k].sum(axis=0)
    return {
        "macro_f1": float(
            np.mean(np.divide(2 * tp, rows + cols, out=np.zeros(k), where=rows + cols > 0))
        ),
        "per_class": [
            {
                "precision": ratio(tp[i], cols[i]),
                "recall": ratio(tp[i], rows[i]),
                "f1": ratio(2 * tp[i], rows[i] + cols[i]) or 0.0,
                "support": int(rows[i]),
            }
            for i in range(k)
        ],
    }


def score_metrics(conditions, groups, raw, predicted, states, known, *, bootstrap=2000):
    conditions, groups = np.asarray(conditions), np.asarray(groups)
    predicted, states = np.asarray(predicted), np.asarray(states)
    k = len(known)
    n = len(conditions)
    if (
        groups.shape != (n,)
        or predicted.shape != (n,)
        or states.shape != (n,)
        or not np.isin(predicted, np.arange(-1, k)).all()
        or not np.isin(
            states, ["known_source", "unknown_or_inconclusive", "no_supported_synthetic_source"]
        ).all()
        or not np.array_equal(predicted >= 0, states == "known_source")
    ):
        raise ValueError("Invalid source decision/metadata geometry.")
    truth = np.array([known.index(c) if c in known else -1 for c in conditions])
    is_known, real = truth >= 0, conditions == "real"
    unknown = ~is_known & ~real
    accepted = predicted >= 0
    correct = is_known & (predicted == truth)
    original = np.asarray(raw["source_index"])
    closed_cm, selective_cm = np.zeros((k, k), dtype=int), np.zeros((k, k + 1), dtype=int)
    for i in np.flatnonzero(is_known):
        closed_cm[truth[i], original[i]] += 1
        selective_cm[truth[i], predicted[i] if accepted[i] else k] += 1
    counts = {
        "known": int(is_known.sum()),
        "genuine": int(real.sum()),
        "withheld": int(unknown.sum()),
        "known_accepted": int((is_known & accepted).sum()),
        "known_correct_accepted": int(correct.sum()),
        "genuine_false_attributed": int((real & accepted).sum()),
        "withheld_false_attributed": int((unknown & accepted).sum()),
    }
    result = {
        "counts": counts,
        "known_classes": list(known),
        "source_groups": len(set(groups)),
        "closed_set": {
            "confusion_matrix": closed_cm.tolist(),
            "labels": list(known),
            "accuracy": ratio(np.trace(closed_cm), is_known.sum()),
            **matrix_metrics(closed_cm),
        },
        "selective_known": {
            "confusion_matrix": selective_cm.tolist(),
            "columns": [*known, "no_source_label"],
            **matrix_metrics(selective_cm),
        },
        "known_coverage": ratio((is_known & accepted).sum(), is_known.sum()),
        "known_accepted_accuracy": ratio(correct.sum(), (is_known & accepted).sum()),
        "known_correct_source_fraction": ratio(correct.sum(), is_known.sum()),
        "source_label_coverage_all_inputs": ratio(accepted.sum(), n),
        "source_label_precision_all_inputs": ratio(correct.sum(), accepted.sum()),
        "all_input_precision_scope": (
            "observed_known_genuine_withheld_mix_not_deployment_prevalence"
        ),
        "known_false_rejection_rate": ratio((is_known & ~accepted).sum(), is_known.sum()),
        "genuine_false_attribution_rate": ratio((real & accepted).sum(), real.sum()),
        "withheld_rejection_rate": ratio((unknown & ~accepted).sum(), unknown.sum()),
        "withheld_rejection_is_not_generator_identification": True,
        "decision_population": {
            key: {
                state: int(np.sum(mask & (states == state)))
                for state in (
                    "known_source",
                    "no_supported_synthetic_source",
                    "unknown_or_inconclusive",
                )
            }
            for key, mask in (("known", is_known), ("genuine", real), ("withheld", unknown))
        },
        "per_vocoder": {},
        "source_group_counts": {},
        "ranking": {},
        **FLAGS,
    }
    for c in sorted(set(conditions) - {"real"}):
        mask = conditions == c
        result["per_vocoder"][str(c)] = {
            "role": "known" if c in known else "withheld",
            "recordings": int(mask.sum()),
            "source_label_fraction": ratio((mask & accepted).sum(), mask.sum()),
            "correct_source_fraction": ratio((mask & correct).sum(), mask.sum())
            if c in known
            else None,
            "accepted_accuracy": ratio((mask & correct).sum(), (mask & accepted).sum())
            if c in known
            else None,
            "rejection_rate": ratio((mask & ~accepted).sum(), mask.sum()),
        }
    group_counts, group_closed, group_selective = [], [], []
    for g in sorted(set(groups)):
        mask = groups == g
        gc = [
            int((mask & is_known).sum()),
            int((mask & is_known & accepted).sum()),
            int((mask & correct).sum()),
            int((mask & real).sum()),
            int((mask & real & accepted).sum()),
            int((mask & unknown).sum()),
            int((mask & unknown & accepted).sum()),
        ]
        result["source_group_counts"][str(g)] = dict(
            zip(
                (
                    "known",
                    "accepted_known",
                    "correct_known",
                    "genuine",
                    "false_known_genuine",
                    "withheld",
                    "false_known_withheld",
                ),
                gc,
                strict=True,
            )
        )
        group_counts.append(gc)
        cc, sc = np.zeros_like(closed_cm), np.zeros_like(selective_cm)
        for i in np.flatnonzero(mask & is_known):
            cc[truth[i], original[i]] += 1
            sc[truth[i], predicted[i] if accepted[i] else k] += 1
        group_closed.append(cc)
        group_selective.append(sc)
    for name in SCORES:
        s = np.asarray(raw[name])
        ranking = {}
        for label, negative in (("known_vs_withheld", unknown), ("known_vs_genuine", real)):
            use = is_known | negative
            ranking[label + "_auroc"] = (
                float(roc_auc_score(is_known[use], s[use]))
                if (is_known.any() and negative.any())
                else None
            )
        result["ranking"][name] = ranking
    use = real | is_known
    result["ranking"]["binary_real_vs_known_auroc"] = (
        float(roc_auc_score(real[use], np.asarray(raw["real_margin"])[use]))
        if real.any() and is_known.any()
        else None
    )
    if bootstrap:
        rng = np.random.default_rng(GLOBAL_SEED + 404)
        draws = rng.integers(0, len(group_counts), size=(bootstrap, len(group_counts)))
        totals = np.asarray(group_counts)[draws].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            series = {
                "known_coverage": totals[:, 1] / totals[:, 0],
                "known_accepted_accuracy": totals[:, 2] / totals[:, 1],
                "known_correct_source_fraction": totals[:, 2] / totals[:, 0],
                "genuine_false_attribution_rate": totals[:, 4] / totals[:, 3],
                "withheld_rejection_rate": 1 - totals[:, 6] / totals[:, 5],
                "source_label_coverage_all_inputs": (
                    (totals[:, 1] + totals[:, 4] + totals[:, 6])
                    / (totals[:, 0] + totals[:, 3] + totals[:, 5])
                ),
                "source_label_precision_all_inputs": (
                    totals[:, 2] / (totals[:, 1] + totals[:, 4] + totals[:, 6])
                ),
            }
        for label, matrices in (
            ("closed_macro_f1", group_closed),
            ("selective_known_macro_f1", group_selective),
        ):
            cm = np.asarray(matrices)[draws].sum(axis=1)
            tp = np.diagonal(cm[:, :, :k], axis1=1, axis2=2)
            den = cm.sum(axis=2) + cm[:, :, :k].sum(axis=1)
            series[label] = np.divide(
                2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den > 0
            ).mean(axis=1)
        result["source_bootstrap"] = {
            "resamples": bootstrap,
            "scope": "conditional_frozen_scores_thresholds_no_refit_or_selection_repeat",
            "intervals": {
                key: {
                    "ci95": np.quantile(v[np.isfinite(v)], [0.025, 0.975]).tolist()
                    if np.isfinite(v).any()
                    else None,
                    "undefined_replicates": int((~np.isfinite(v)).sum()),
                }
                for key, v in series.items()
            },
        }
    for label in ("known_correct_source_fraction", "known_coverage"):
        values = [
            ratio(gc[2 if label == "known_correct_source_fraction" else 1], gc[0])
            for gc in group_counts
            if gc[0]
        ]
        result.setdefault("source_variability", {})[label] = {
            "median": float(np.median(values)) if values else None,
            "p05": float(np.quantile(values, 0.05)) if values else None,
            "p95": float(np.quantile(values, 0.95)) if values else None,
        }
    return result
