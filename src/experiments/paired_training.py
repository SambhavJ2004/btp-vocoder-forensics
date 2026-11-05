"""Fixed paired-negative weighting study; waveform-only inference uses existing bundles."""

from __future__ import annotations

import argparse
import warnings
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from data.invariants import GLOBAL_SEED
from data.partial_audio import VOCODERS
from detectors.forensic_pilot import LinearHead, sha256_file, write_json
from detectors.temporal_context import (
    CANDIDATES,
    load_context_bundle,
    predict_context_file,
    save_context_bundle,
)
from detectors.temporal_features import (
    analysis_comparison,
    extract_windows,
    guarded_labels,
    sample_labels,
    smooth_probabilities,
    window_starts,
)
from detectors.temporal_model import load_temporal_bundle
from detectors.temporal_operating import (
    control_operating_points,
    evaluate_context,
    grouped_folds,
    sample_cells,
)
from experiments.acoustic_diagnostics import activity_bins, group_summary, window_rms
from experiments.paired_signal import reference_activity_threshold
from experiments.partial_audio import FLAGS, artifact_inventory, read_input, source_frame
from experiments.phase3d import seal_input
from experiments.temporal_ablation import (
    bundle_for,
    evaluate_points,
    load_features,
    save_predictions,
)
from experiments.temporal_localization import code_provenance, fit_temporal_head
from experiments.temporal_review import SCORE_GUARD, read_json, review_rows

CONFIG = CANDIDATES["A64"]
CANDIDATES_WEIGHTS = {"A_original": None, "B_paired25": 0.25, "C_paired50": 0.50}


def fixed_plan():
    return {
        "phase": "3E",
        "candidates": CANDIDATES_WEIGHTS,
        "representation": "unchanged_spectral31_64ms_hop16_below8k_PCM16",
        "seed": GLOBAL_SEED,
        "folds": 3,
        "C": 1.0,
        "pairing": "same_source_same_start_end_null_original_real_verified",
        "active_pair": "intersection_real_and_all_three_synthetic_64ms_RMS_above_real_threshold",
        "activity_reference": "full_real_512_128_max_minus30dB_floor_two_PCM16_steps",
        "minimum_shared_active_pairs_each_source": 8,
        "all_original_pure_windows_retained": True,
        "positive_weights": "unchanged_original_balanced_class_weights",
        "negative_weights": "per_source_target_matched_share_rest_proportional_to_original_counts",
        "class_and_source_control": "preserve_original_per_fold_source_by_class_total_mass",
        "normalization": "reuse_exact_unweighted_fit_group_original_A_scaler_all_pure_windows",
        "no_duplicate_training_rows": True,
        "primary_score_guard_samples": SCORE_GUARD,
        "guard_derivation": "3308+ceil(1411/2)+353+ceil(353/2)=4544_same_frozen_A64",
        "secondary_score_guard_samples": 5249,
        "operating_points": [0.5, "OOF_1pct", "OOF_5pct", "OOF_10pct"],
        "selection": (
            "maximize_OOF_5pct_synthetic_TP_then_group_F1_"
            "then_lower_max_control_FPR_then_lower_paired_mass"
        ),
        "control_constraint": "each_null_R2R_sample_FPR_le_0.05",
        "meaningful_development_heuristic": {
            "minimum_OOF_absolute_recall_gain": 0.02,
            "paired_recall_gain_CI_lower_gt_zero": True,
            "group_F1_not_lower": True,
            "not_a_scientific_admission_gate": True,
        },
        "research_recommendation": (
            "C_if_no_meaningful_OOF_gain;B_if_gain_but_dev_control_FPR_gt5pct_or_recall_below_A;"
            "A_otherwise;never_reselect_model_or_threshold_on_dev"
        ),
        "secondary_signal_audit": "OOF_margin_rank_correlation_and_same_time_pair_differences",
        "bootstrap": "2000_paired_source_resamples_fixed_predictions_and_thresholds_no_refits",
        "matched_empirical_FPR": (
            "evaluation_only_no_cutoff_export_no_candidate_selection_dev_after_lock"
        ),
        "dev_role": (
            "previously_exposed_benchmark_only_after_selection_and_all_models_thresholds_locked"
        ),
        "eval_used": False,
        "neural_training": False,
        **FLAGS,
    }


def build_pair_masks(records, rms_state):
    """Truth/activity select supervised pairs; only acoustic feature arrays enter fit."""
    identifiers = rms_state["example_ids"].tolist()
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate RMS evidence identities.")
    by_id = {key: i for i, key in enumerate(identifiers)}
    masks = [np.zeros(len(r["starts"]), dtype=bool) for r in records]
    evidence, pairs = [], []
    offsets = np.r_[0, np.cumsum([len(r["starts"]) for r in records])]
    for group in sorted({r["row"]["group_id"] for r in records}):
        indices = [i for i, r in enumerate(records) if r["row"]["group_id"] == group]
        own = {records[i]["row"]["example_class"]: i for i in indices}
        if set(own) != {"null_control", "real_to_real_control", *(f"mixed_{v}" for v in VOCODERS)}:
            raise ValueError("Require all five conditions for each paired training source.")
        null_index = own["null_control"]
        null = records[null_index]
        n = len(null["starts"])
        shared = np.ones(n, dtype=bool)
        levels = {}
        threshold = None
        for i in indices:
            record = records[i]
            if record["row"]["split"] != "train":
                raise ValueError("Paired construction is restricted to train.")
            np.testing.assert_array_equal(record["starts"], null["starts"])
            row = record["row"]
            if row["source_real_id"] != null["row"]["source_real_id"]:
                raise ValueError("Paired conditions do not share the original real source.")
            if row["example_id"] not in by_id:
                raise ValueError("Missing measured RMS evidence.")
            j = by_id[row["example_id"]]
            a, b = rms_state["offsets"][j : j + 2]
            if b - a != n:
                raise ValueError("RMS/window evidence geometry differs.")
            current = float(rms_state["thresholds"][j])
            if threshold is not None and threshold != current:
                raise ValueError("Conditions must share the real-reference activity threshold.")
            threshold = current
            levels[i] = rms_state["own_rms"][a:b]
            if row["has_synthetic"]:
                shared &= (record["window_y"] == 1) & (levels[i] >= current)
            elif row["example_class"] == "null_control":
                shared &= (record["window_y"] == 0) & (levels[i] >= current)
        if shared.sum() < 8:
            raise ValueError(
                f"Insufficient shared speech-active pairs in {group}: {shared.sum()}; "
                "no repair/drop."
            )
        masks[null_index] = shared
        for v in VOCODERS:
            i = own[f"mixed_{v}"]
            for position in np.flatnonzero(shared):
                start = int(null["starts"][position])
                pairs.append(
                    {
                        "group_id": group,
                        "vocoder_annotation_not_feature": v,
                        "synthetic_example_id": records[i]["row"]["example_id"],
                        "real_example_id": null["row"]["example_id"],
                        "source_real_id": null["row"]["source_real_id"],
                        "start_sample": start,
                        "end_sample": start + CONFIG.window_samples,
                        "positive_global_window": int(offsets[i] + position),
                        "negative_global_window": int(offsets[null_index] + position),
                    }
                )
        evidence.append(
            {
                "group_id": group,
                "shared_active_coordinates": int(shared.sum()),
                "positive_pair_links": int(3 * shared.sum()),
                "threshold": threshold,
            }
        )
    return masks, {"sources": evidence, "pairs": pairs, **FLAGS}


def controlled_weights(y, groups, paired, matched_share):
    """Redistribute only negative mass, preserving A's source-by-class budgets."""
    y, groups, paired = np.asarray(y), np.asarray(groups), np.asarray(paired)
    if (
        y.ndim != 1
        or groups.shape != y.shape
        or paired.shape != y.shape
        or paired.dtype.kind != "b"
        or not np.isin(y, (0, 1)).all()
        or not len(y)
        or set(y) != {0, 1}
        or (paired & (y != 0)).any()
        or matched_share not in (None, 0.25, 0.50)
    ):
        raise ValueError("Invalid controlled-weight inputs or undeclared weighting level.")
    counts = np.bincount(y.astype(int), minlength=2)
    class_weight = len(y) / (2 * counts)
    base = class_weight[y.astype(int)]
    weights = base.copy()
    summaries = []
    for group in sorted(set(groups)):
        own = groups == group
        real, positive = own & (y == 0), own & (y == 1)
        hard, other = real & paired, real & ~paired
        if not hard.any() or not other.any() or not positive.any():
            raise ValueError(
                "Each fit source needs both classes, paired negatives and other controls."
            )
        mass = float(base[real].sum())
        if matched_share is not None:
            weights[hard] = mass * matched_share / hard.sum()
            weights[other] = mass * (1 - matched_share) / other.sum()
        for label in (0, 1):
            use = own & (y == label)
            np.testing.assert_allclose(weights[use].sum(), base[use].sum(), rtol=3e-14, atol=1e-10)
        summaries.append(
            {
                "group_id": str(group),
                "real_windows": int(real.sum()),
                "synthetic_windows": int(positive.sum()),
                "paired_negative_windows": int(hard.sum()),
                "negative_mass": mass,
                "positive_mass": float(weights[positive].sum()),
                "matched_negative_mass": float(weights[hard].sum()),
                "matched_share": float(weights[hard].sum() / mass),
            }
        )
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Every original pure window must retain positive finite weight.")
    np.testing.assert_array_equal(weights[y == 1], base[y == 1])
    np.testing.assert_allclose(
        [weights[y == label].sum() for label in (0, 1)], len(y) / 2, rtol=3e-14
    )
    return weights, {
        "sources": summaries,
        "windows": len(y),
        "class_mass": {str(i): float(weights[y == i].sum()) for i in (0, 1)},
        "class_weights_original": class_weight.tolist(),
        "effective_sample_size": float(weights.sum() ** 2 / (weights @ weights)),
        "matched_negative_mass_fraction": float(weights[paired].sum() / weights[y == 0].sum()),
        **FLAGS,
    }


def training_arrays(records):
    if any(r["row"]["split"] != "train" for r in records):
        raise ValueError("No dev/eval row may enter fitting.")
    x = np.vstack([r["features"] for r in records])
    y = np.concatenate([r["window_y"] for r in records])
    groups = np.concatenate([np.repeat(r["row"]["group_id"], len(r["starts"])) for r in records])
    paired = np.concatenate([r["paired_negative"] for r in records])
    take = y >= 0
    return x[take], y[take], groups[take], paired[take], np.flatnonzero(take)


def fit_weighted(records, share, normalization):
    x, y, groups, paired, indices = training_arrays(records)
    weights, evidence = controlled_weights(y, groups, paired, share)
    estimator = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=2000, tol=1e-6, random_state=GLOBAL_SEED
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit((x - normalization.mean) / normalization.scale, y, sample_weight=weights)
    head = LinearHead(
        normalization.mean.copy(),
        normalization.scale.copy(),
        estimator.coef_.copy(),
        estimator.intercept_.copy(),
    )
    head.validate(31, 1)
    return head, evidence, indices, weights


def admitted_inputs(dataset, source, baseline, ablation, review):
    selected, identity = review_rows(dataset, source, baseline)
    for root in (ablation, review):
        seal_input(root)
        if read_json(root / "provenance.json")["data"] != identity:
            raise ValueError("Reviewed data/model identity differs.")
    validation = read_json(review / "validation.json")
    audit = read_json(review / "source_audit.json")
    if (
        validation.get("status") != "passed_structural_diagnostic_integrity"
        or not validation.get("alignment_gates_all_passed")
        or any(not r["passed"] or r["peak_lag"] != 0 for r in audit["alignment"].values())
    ):
        raise ValueError("Unchanged source alignment/reconstruction gates must pass.")
    for path, digest in audit["source_hashes"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("Reviewed original training source changed.")
    rows = selected["train"]
    if {r["example_id"] for r in audit["outputs"]} != {r["example_id"] for r in rows}:
        raise ValueError("Reviewed training output inventory differs.")
    return selected, identity, audit["source_hashes"]


def verify_real_counterparts(records, rms, source, dataset, work):
    frame = source_frame(source)
    real_paths = {
        str(r.utt_id): Path(r.path)
        for r in frame[(frame.split == "train") & (frame.condition == "real")].itertuples()
    }
    indices = {key: i for i, key in enumerate(rms["example_ids"].tolist())}
    evidence = []
    for record in records:
        row = record["row"]
        if row["example_class"] != "null_control":
            continue
        real = read_input(real_paths[row["group_id"]])
        null = read_input(dataset / row["output_filename"])
        np.testing.assert_array_equal(real, null)
        processed = analysis_comparison({"real": real, "null": null}, work)
        starts, x = extract_windows(processed["real"], CONFIG)
        np.testing.assert_array_equal(starts, record["starts"])
        np.testing.assert_array_equal(x, record["features"])
        np.testing.assert_array_equal(processed["real"], processed["null"])
        j = indices[row["example_id"]]
        a, b = rms["offsets"][j : j + 2]
        np.testing.assert_array_equal(
            window_rms(processed["real"], starts), rms["reference_rms"][a:b]
        )
        if reference_activity_threshold(processed["real"]) != rms["thresholds"][j]:
            raise ValueError("Real-reference activity threshold differs.")
        evidence.append(
            {
                "group_id": row["group_id"],
                "original_real_sha256": sha256_file(real_paths[row["group_id"]]),
                "null_identity_and_features_exact": True,
            }
        )
        print(f"Verified paired real counterpart {row['group_id']}", flush=True)
    return evidence


def attach_pairs(records, rms):
    masks, evidence = build_pair_masks(records, rms)
    lookup = {key: i for i, key in enumerate(rms["example_ids"].tolist())}
    for r, mask in zip(records, masks, strict=True):
        r["paired_negative"] = mask
        j = lookup[r["row"]["example_id"]]
        a, b = rms["offsets"][j : j + 2]
        r["activity_bins"] = activity_bins(rms["own_rms"][a:b], rms["thresholds"][j])
    return evidence


def roc_control_recall(
    scores, positive_weights, null_weights, r2r_weights, targets=(0.01, 0.05, 0.10)
):
    """Post-lock empirical ROC statistic only; never returns a deployment cutoff."""
    scores = np.asarray(scores, dtype=float)
    weights = np.column_stack((positive_weights, null_weights, r2r_weights)).astype(float)
    if (
        scores.ndim != 1
        or weights.shape != (len(scores), 3)
        or not len(scores)
        or not np.isfinite(scores).all()
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or (weights.sum(axis=0) <= 0).any()
        or any(not 0 <= t <= 1 for t in targets)
    ):
        raise ValueError("Invalid weighted control ROC inputs.")
    order = np.argsort(-scores, kind="stable")
    ss = scores[order]
    ww = weights[order]
    beginnings = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]
    cumulative = np.vstack(
        [np.zeros(3), np.cumsum(np.add.reduceat(ww, beginnings, axis=0), axis=0)]
    )
    rates = cumulative / weights.sum(axis=0)
    max_fpr = rates[:, 1:].max(axis=1)
    return {
        f"fpr_{round(target * 100):02d}": {
            "recall": float(rates[np.flatnonzero(max_fpr <= target + 1e-15)[-1], 0]),
            "evaluation_only_no_cutoff_export": True,
        }
        for target in targets
    }


def empirical_roc_summary(records):
    values = []
    weights = []
    for r in records:
        real, syn = sample_cells(r, CONFIG, SCORE_GUARD)
        kind = r["row"]["example_class"]
        values.append(smooth_probabilities(expit(r["margin"]), 3))
        weights.append(
            np.column_stack(
                (
                    syn,
                    real if kind == "null_control" else np.zeros(len(real)),
                    real if kind == "real_to_real_control" else np.zeros(len(real)),
                )
            )
        )
    w = np.concatenate(weights)
    s = np.concatenate(values)
    use = w.sum(axis=1) > 0
    return roc_control_recall(s[use], *w[use].T)


def paired_improvements(reference, candidate):
    groups = sorted(reference["sample_by_source_group"])
    if set(groups) != set(candidate["sample_by_source_group"]):
        raise ValueError("Paired comparison requires identical source groups.")
    a = np.array(
        [
            [reference["sample_by_source_group"][g][k] for k in ("tp", "fp", "fn", "tn")]
            for g in groups
        ]
    )
    b = np.array(
        [
            [candidate["sample_by_source_group"][g][k] for k in ("tp", "fp", "fn", "tn")]
            for g in groups
        ]
    )
    rng = np.random.default_rng(GLOBAL_SEED + 305)
    draws = rng.integers(0, len(groups), size=(2000, len(groups)))
    aa, bb = a[draws].sum(axis=1), b[draws].sum(axis=1)
    delta = bb[:, 0] / (bb[:, 0] + bb[:, 2]) - aa[:, 0] / (aa[:, 0] + aa[:, 2])
    ref = reference["sample_micro"]["recall"]
    new = candidate["sample_micro"]["recall"]
    f1_delta = [
        candidate["sample_by_source_group"][g]["f1"] - reference["sample_by_source_group"][g]["f1"]
        for g in groups
    ]
    return {
        "absolute_sample_recall_change": new - ref,
        "relative_sample_recall_change": new / ref - 1 if ref else None,
        "absolute_group_F1_change": candidate["source_group_macro_f1"]
        - reference["source_group_macro_f1"],
        "relative_group_F1_change": candidate["source_group_macro_f1"]
        / reference["source_group_macro_f1"]
        - 1
        if reference["source_group_macro_f1"]
        else None,
        "paired_source_recall_change_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
        "paired_source_group_F1_change": group_summary(f1_delta),
        "uncertainty_scope": (
            "conditional_on_frozen_OOF_or_exposed_dev_scores_and_selected_thresholds_no_refits"
        ),
    }


def choose_candidate(metrics):
    for m in metrics.values():
        if any(v["sample_false_positive_rate"] > 0.05 + 1e-15 for v in m["controls"].values()):
            raise ValueError("Selection candidate violates a separate control FPR cap.")
    return sorted(
        metrics,
        key=lambda name: (
            -metrics[name]["sample_micro"]["tp"],
            -metrics[name]["source_group_macro_f1"],
            max(c["sample_false_positive_rate"] for c in metrics[name]["controls"].values()),
            CANDIDATES_WEIGHTS[name] or 0,
        ),
    )[0]


def compare_heads(actual, expected):
    for key in ("mean", "scale", "coefficients", "intercept"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))


def discrimination_audit(candidates, pair_evidence):
    flat = {name: np.concatenate([r["margin"] for r in rows]) for name, rows in candidates.items()}
    reference = flat["A_original"]
    result = {}
    for name, margins in flat.items():
        grouped = defaultdict(list)
        for pair in pair_evidence["pairs"]:
            grouped[(pair["vocoder_annotation_not_feature"], pair["group_id"])].append(
                margins[pair["positive_global_window"]] - margins[pair["negative_global_window"]]
            )
        result[name] = {
            "margin_spearman_vs_A": (
                float(spearmanr(reference, margins).statistic)
                if np.ptp(reference) and np.ptp(margins)
                else None
            ),
            "same_time_training_OOF_oracle_diagnostic": {},
        }
        for vocoder in VOCODERS:
            source_values = [
                np.asarray(values) for (v, g), values in sorted(grouped.items()) if v == vocoder
            ]
            result[name]["same_time_training_OOF_oracle_diagnostic"][vocoder] = {
                "source_mean_paired_margin_difference": group_summary(
                    [v.mean() for v in source_values]
                ),
                "source_mean_pair_win_fraction": group_summary(
                    [np.mean((v > 0) + 0.5 * (v == 0)) for v in source_values]
                ),
            }
    return {"models": result, "not_an_inference_adjustment": True, **FLAGS}


def research_recommendation(preferred, primary, improvements, dev_primary):
    change = improvements[preferred]
    useful = (
        preferred != "A_original"
        and change["absolute_sample_recall_change"] >= 0.02
        and change["paired_source_recall_change_ci95"][0] > 0
        and change["absolute_group_F1_change"] >= 0
    )
    benchmark_transport = (
        all(
            c["sample_false_positive_rate"] <= 0.05
            for c in dev_primary[preferred]["controls"].values()
        )
        and dev_primary[preferred]["sample_micro"]["recall"]
        >= dev_primary["A_original"]["sample_micro"]["recall"]
    )
    decision = "C" if not useful else ("A" if benchmark_transport else "B")
    return {
        "recommendation": decision,
        "meaningful_train_OOF_gain": useful,
        "exposed_dev_transport_check": benchmark_transport,
        "preferred_configuration_unchanged": preferred,
        "not_a_scientific_or_publication_gate": True,
        **FLAGS,
    }


def score_and_save(records, points, root, *, diagnostics):
    save_predictions(root, records, CONFIG)
    metrics = evaluate_points(records, CONFIG, points, root, diagnostics=diagnostics)
    write_json(
        root / "empirical_control_ROC.json",
        {
            "operating_statistics": empirical_roc_summary(records),
            "not_used_to_select_model_or_bundle_thresholds": True,
            **FLAGS,
        },
    )
    masked = [
        {
            **r,
            "score_labels": guarded_labels(
                sample_labels(r["row"]["interval_labels"], r["row"]["num_samples"]), 5249
            ),
        }
        for r in records
    ]
    for label, point in points.items():
        secondary = evaluate_context(
            masked,
            [expit(r["margin"]) for r in masked],
            CONFIG,
            point["threshold"],
            5249,
            detailed=False,
        )
        write_json(
            root / f"{label}.secondary5249.json",
            {**secondary, "threshold_reselected": False, **FLAGS},
        )
    return metrics


def run(dataset_dir, source_manifest, baseline_dir, ablation_dir, review_dir, out_dir):
    dataset, source, baseline, ablation, review, target = (
        Path(p).resolve()
        for p in (dataset_dir, source_manifest, baseline_dir, ablation_dir, review_dir, out_dir)
    )
    staging = target.with_name(target.name + ".staging")
    if target.exists() or staging.exists():
        raise FileExistsError("Fresh Phase 3E run/staging required.")
    if any(
        p.is_relative_to(r) or r.is_relative_to(p)
        for p in (target, staging)
        for r in (dataset, source.parent, baseline, ablation, review)
    ):
        raise ValueError("Paired-training output must be separate from immutable inputs.")
    begin = perf_counter()
    with threadpool_limits(limits=1):
        selected, identity, source_hashes = admitted_inputs(
            dataset, source, baseline, ablation, review
        )
        records = load_features(ablation / "features/A64/train", selected["train"])
        for r in records:
            np.testing.assert_array_equal(
                r["starts"], window_starts(r["row"]["num_samples"], CONFIG)
            )
            if (
                r["features"].shape != (len(r["starts"]), 31)
                or not np.isfinite(r["features"]).all()
            ):
                raise ValueError("Invalid frozen A64 feature geometry.")
        with np.load(review / "window_rms.npz", allow_pickle=False) as archive:
            rms = {k: archive[k] for k in archive.files}
        staging.mkdir(parents=True)
        write_json(staging / "plan.json", fixed_plan())
        code = code_provenance()
        repo = Path(__file__).resolve().parents[2]
        code["source_sha256"].update(
            {p.relative_to(repo).as_posix(): sha256_file(p) for p in (repo / "src").rglob("*.py")}
        )
        write_json(
            staging / "provenance.json",
            {
                "code": code,
                "data": identity,
                "review_inventory_sha256": sha256_file(review / "artifacts.sha256.json"),
                **FLAGS,
            },
        )
        work = staging / "_analysis"
        work.mkdir()
        try:
            real_check = verify_real_counterparts(records, rms, source, dataset, work)
            write_json(
                staging / "original_real_counterparts.json", {"sources": real_check, **FLAGS}
            )
            pair_evidence = attach_pairs(records, rms)
            write_json(staging / "paired_examples.json", pair_evidence)
            assignments = grouped_folds([r["row"]["group_id"] for r in records], GLOBAL_SEED, 3)
            write_json(staging / "folds.json", {"assignments": assignments, **FLAGS})
            candidates = {name: [{**r} for r in records] for name in CANDIDATES_WEIGHTS}
            fold_info = defaultdict(list)
            for fold in range(3):
                training = [r for r in records if assignments[r["row"]["group_id"]] != fold]
                held = [
                    i for i, r in enumerate(records) if assignments[r["row"]["group_id"]] == fold
                ]
                reference_head, _ = fit_temporal_head(training, 1.0, GLOBAL_SEED)
                original = load_context_bundle(ablation / f"candidates/A64/fold-{fold}/bundle").head
                compare_heads(reference_head, original)
                for name, share in CANDIDATES_WEIGHTS.items():
                    if share is None:
                        head = reference_head
                        x, y, g, p, indices = training_arrays(training)
                        weights, evidence = controlled_weights(y, g, p, None)
                    else:
                        head, evidence, indices, weights = fit_weighted(
                            training, share, reference_head
                        )
                    root = staging / "candidates" / name / f"fold-{fold}"
                    info = {
                        "fold": fold,
                        "fit_groups": sorted({r["row"]["group_id"] for r in training}),
                        "held_out_groups": sorted({records[i]["row"]["group_id"] for i in held}),
                        "weighting": evidence,
                        "scaler_scope": "same_exact_A_fit_group_scaler",
                        **FLAGS,
                    }
                    save_context_bundle(
                        bundle_for(head, CONFIG, {"fixed_05": 0.5}, **info), root / "bundle"
                    )
                    np.savez_compressed(
                        root / "training_weights.npz",
                        pure_window_indices_in_fit_records=indices,
                        weights=weights,
                    )
                    restored = load_context_bundle(root / "bundle")
                    for i in held:
                        candidates[name][i]["margin"] = restored.margins(records[i]["features"])
                        candidates[name][i]["fold"] = fold
                    write_json(root / "fit.json", info)
                    fold_info[name].append(info)
                print(f"Paired training outer fold {fold} completed", flush=True)
            point_maps, metrics = {}, {}
            for name, scored in candidates.items():
                points = {
                    "fixed_05": {"threshold": 0.5, "selection_scope": "predeclared_reference"},
                    **control_operating_points(
                        scored, [r["margin"] for r in scored], CONFIG, SCORE_GUARD
                    ),
                }
                point_maps[name] = points
                metrics[name] = score_and_save(
                    scored, points, staging / "candidates" / name / "oof", diagnostics=True
                )
                write_json(
                    staging / "candidates" / name / "cross_validation.json",
                    {"folds": fold_info[name], "points": points, **FLAGS},
                )
            primary = {k: v["fpr_05"] for k, v in metrics.items()}
            preferred = choose_candidate(primary)
            improvements = {
                name: paired_improvements(primary["A_original"], m) for name, m in primary.items()
            }
            write_json(
                staging / "OOF_comparison.json",
                {
                    "preferred": preferred,
                    "improvements": improvements,
                    "primary_metrics": primary,
                    "selection_scope": "train_OOF_only",
                    **FLAGS,
                },
            )
            write_json(
                staging / "discrimination_audit.json",
                discrimination_audit(candidates, pair_evidence),
            )
            original_final = load_temporal_bundle(baseline / "bundle").head
            full_A, _ = fit_temporal_head(records, 1.0, GLOBAL_SEED)
            compare_heads(full_A, original_final)
            for name, share in CANDIDATES_WEIGHTS.items():
                if share is None:
                    head = full_A
                    x, y, g, p, indices = training_arrays(records)
                    weights, evidence = controlled_weights(y, g, p, None)
                else:
                    head, evidence, indices, weights = fit_weighted(records, share, full_A)
                root = staging / "candidates" / name
                save_context_bundle(
                    bundle_for(
                        head,
                        CONFIG,
                        {k: v["threshold"] for k, v in point_maps[name].items()},
                        fit_groups=sorted(assignments),
                        paired_negative_mass=share,
                        training_weights=evidence,
                    ),
                    root / "bundle",
                )
                np.savez_compressed(
                    root / "training_weights.npz", pure_window_indices=indices, weights=weights
                )
                write_json(
                    root / "training_distribution.json",
                    weight_distribution(records, weights, indices, evidence),
                )
            lock = {
                "preferred": preferred,
                "points": point_maps,
                "OOF_improvements": improvements,
                "bundle_inventories": {
                    name: artifact_inventory(staging / "candidates" / name / "bundle")
                    for name in candidates
                },
                "selection_completed_before_dev_features": True,
                **FLAGS,
            }
            write_json(staging / "selection_locked.json", lock)
            # No dev cache parsing, model choice or threshold fitting occurs above this point.
            development = load_features(ablation / "features/A64/dev", selected["dev"])
            dev_metrics = {}
            for name in candidates:
                restored = load_context_bundle(staging / "candidates" / name / "bundle")
                scored = [{**r, "margin": restored.margins(r["features"])} for r in development]
                dev_metrics[name] = score_and_save(
                    scored,
                    point_maps[name],
                    staging / "candidates" / name / "dev",
                    diagnostics=True,
                )
            dev_primary = {k: v["fpr_05"] for k, v in dev_metrics.items()}
            write_json(
                staging / "dev_comparison.json",
                {
                    "preferred_unchanged": preferred,
                    "benchmark": "previously_exposed_17_source_dev",
                    "improvements": {
                        name: paired_improvements(dev_primary["A_original"], m)
                        for name, m in dev_primary.items()
                    },
                    "primary_metrics": dev_primary,
                    **FLAGS,
                },
            )
            write_json(
                staging / "research_decision.json",
                research_recommendation(preferred, primary, improvements, dev_primary),
            )
            # Single-input inference receives neither source manifests nor paired real audio.
            example = development[0]
            prediction = predict_context_file(
                staging / "candidates" / preferred / "bundle",
                dataset / example["row"]["output_filename"],
                work,
            )
            restored = load_context_bundle(staging / "candidates" / preferred / "bundle")
            np.testing.assert_array_equal(
                prediction["synthetic_margins"], restored.margins(example["features"])
            )
            write_json(
                staging / "label_free_inference.json",
                {"example_id": example["row"]["example_id"], **prediction},
            )
            for name in candidates:
                if (
                    artifact_inventory(staging / "candidates" / name / "bundle")
                    != lock["bundle_inventories"][name]
                ):
                    raise ValueError("Locked bundle changed during dev benchmarking.")
            work.rmdir()
            _, after = review_rows(dataset, source, baseline)
            if after != identity:
                raise ValueError("Input dataset identity changed.")
            for path, digest in source_hashes.items():
                if sha256_file(Path(path)) != digest:
                    raise ValueError("Reviewed training source changed.")
            write_json(
                staging / "validation.json",
                {
                    "status": "passed",
                    "train_groups": len(assignments),
                    "dev_benchmark_groups": len({r["row"]["group_id"] for r in development}),
                    "eval_audio_bytes_opened": 0,
                    "eval_annotation_records_used": 0,
                    "new_neural_training": False,
                    "baseline_all_numeric_arrays_exact": True,
                    "identical_normalization_across_candidates": True,
                    "no_inference_pair_labels_or_source_adjustment": True,
                    "preferred": preferred,
                    "elapsed_seconds": perf_counter() - begin,
                    **FLAGS,
                },
            )
            write_json(
                staging / "artifacts.sha256.json", {"files": artifact_inventory(staging), **FLAGS}
            )
            staging.rename(target)
            return {"output": str(target), "preferred": preferred, **FLAGS}
        except Exception as exc:
            write_json(staging / "failure.json", {"status": "failed", "error": str(exc), **FLAGS})
            raise


def weight_distribution(records, weights, indices, evidence):
    y = np.concatenate([r["window_y"] for r in records])[indices]
    bins = np.concatenate([r["activity_bins"] for r in records])[indices]
    kinds = np.concatenate(
        [np.repeat(r["row"]["example_class"], len(r["starts"])) for r in records]
    )[indices]
    paired = np.concatenate([r["paired_negative"] for r in records])[indices]
    strata = {
        "matched_real": paired,
        "null_other": (y == 0) & (kinds == "null_control") & ~paired,
        "real_to_real": (y == 0) & (kinds == "real_to_real_control"),
        "ordinary_real_mixed": (y == 0) & np.char.startswith(kinds, "mixed_"),
    }
    np.testing.assert_array_equal(
        sum(mask.astype(int) for mask in strata.values()), (y == 0).astype(int)
    )
    return {
        "weighting": evidence,
        "negative_strata": {
            key: {"windows": int(mask.sum()), "mass": float(weights[mask].sum())}
            for key, mask in strata.items()
        },
        "activity_class_mass": {
            f"{label}:{level}": {
                "windows": int(np.sum((y == label) & (bins == level))),
                "mass": float(weights[(y == label) & (bins == level)].sum()),
            }
            for label in (0, 1)
            for level in range(3)
        },
        "condition_mass": {
            str(kind): {
                "windows": int(np.sum(kinds == kind)),
                "mass": float(weights[kinds == kind].sum()),
            }
            for kind in sorted(set(kinds))
        },
        **FLAGS,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dataset-dir",
        "source-manifest",
        "baseline-dir",
        "ablation-dir",
        "review-dir",
        "out-dir",
    ):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    print(
        run(
            args.dataset_dir,
            args.source_manifest,
            args.baseline_dir,
            args.ablation_dir,
            args.review_dir,
            args.out_dir,
        )
    )


if __name__ == "__main__":
    main()
