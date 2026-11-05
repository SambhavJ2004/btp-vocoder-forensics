"""Training-only acoustic/dataset failure investigation; no neural training or eval."""

from __future__ import annotations

import argparse
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from data.alignment import assert_aligned, check_vocoder_alignment
from data.invariants import (
    ARCHIVE_SR,
    GLOBAL_SEED,
    HIGH_BAND_ANALYSIS,
    InvariantViolation,
    exclusion_for,
)
from data.partial_audio import VOCODERS
from detectors.forensic_pilot import LinearHead, sha256_file, write_json
from detectors.temporal_cnn import GEOMETRY, load_cnn, loss_weights
from detectors.temporal_features import (
    analysis_comparison,
    guarded_labels,
    sample_labels,
    smooth_probabilities,
)
from detectors.temporal_metrics import binary_metrics, confusion, rates
from detectors.temporal_operating import (
    common_mask_ranking,
    control_operating_points,
    evaluate_context,
    grouped_folds,
)
from experiments.acoustic_diagnostics import (
    ACTIVITY_NAMES,
    BANDS,
    BOOTSTRAPS,
    GRID,
    activity_bins,
    balanced_indices,
    correlation,
    group_summary,
    rms_controlled_features,
    spectral_pair,
    window_rms,
)
from experiments.learned_temporal import SUPPORT_GUARD, sampled_training
from experiments.paired_signal import frame_rms, reference_activity_threshold
from experiments.partial_audio import (
    FLAGS,
    artifact_inventory,
    read_input,
    source_frame,
    validate_written_example,
)
from experiments.temporal_ablation import load_features, save_predictions
from experiments.temporal_localization import code_provenance
from experiments.temporal_review import (
    center_edges,
    checked_inventory,
    read_json,
    review_rows,
    saved_records,
)

VARIANTS = ("balanced", "activity_matched", "rms_normalized", "activity_rms")


def fixed_plan():
    return {
        "phase": "3D",
        "analysis_split": "train_only",
        "eval_used": False,
        "neural_training": False,
        "dev_used": False,
        "seed": GLOBAL_SEED,
        "folds": 3,
        "reuse_phase3b_folds": True,
        "C": 1.0,
        "linear_ablations": list(VARIANTS),
        "balanced_cap_each_source_class": 72,
        "matched_cap_each_source_class_activity_bin": 24,
        "fit_weights": "equal_source_and_binary_class",
        "scaler": "fit_groups_only_unweighted",
        "rms_normalization": "0.03_per_64ms_frame_in_RAM_no_clipping_silence_le_2LSB_unchanged",
        "activity_bins": [
            "below_real_ref_max_minus30dB",
            "minus30_to_minus10dB",
            "at_least_minus10dB",
        ],
        "activity_assignment": "real_reference_same_position_primary_own_RMS_secondary",
        "score_guard_samples": SUPPORT_GUARD,
        "operating_points": [0.5, "train_OOF_5pct_each_control_sample_FPR"],
        "uncertainty": {
            "bootstrap_source_statistics": BOOTSTRAPS,
            "percentile": [2.5, 97.5],
            "minimum_source_groups": 10,
            "refit_models": False,
        },
        "spectral": {
            "window_samples": GRID.window_samples,
            "hop_samples": GRID.hop_samples,
            "fft": 2048,
            "PSD_floor": 1e-12,
            "envelope_lifter_samples": 22,
            "paired_frame_mask": "real_reference_RMS_active_identical_positions",
            "bands": BANDS,
            "periodicity_proxy_lag_hz": [50, 500],
            "HNR_or_F0_claim": False,
        },
        "full_band": "exploratory_archive_processing_diagnostic_not_generalizable_detection",
        "INV18": "exclude_GL_Vocos_from_full_high_band_distances_keep_bandwidth_profiles_only",
        "alignment": "unchanged_INV16_probe_all_eligible_train_sources_no_shift_or_repair",
        "frozen_scores": ["A64_OOF", "CNN_OOF", "A64_final_in_sample", "CNN_final_in_sample"],
        "oracle_centering": "subtract_known_null_valid_mean_source_offset_ranking_diagnostic_only",
        "far_real": "null_real_at_least_1sec_from_raw_target_boundaries",
        "selection_bias": "compare_123_admitted_and_17_excluded_real_train_sources_only",
        **FLAGS,
    }


def seal_input(root):
    seal = read_json(root / "artifacts.sha256.json")
    checked_inventory(
        root,
        {**seal["files"], "artifacts.sha256.json": sha256_file(root / "artifacts.sha256.json")},
    )


def attach_scores(records, prediction_file):
    with np.load(prediction_file, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    if arrays["record_example_ids"].tolist() != [r["row"]["example_id"] for r in records]:
        raise ValueError("Frozen prediction record identities differ from training cache.")
    for i, record in enumerate(records):
        a, b = arrays["record_offsets"][i : i + 2]
        np.testing.assert_array_equal(record["starts"], arrays["window_start_samples"][a:b])
        np.testing.assert_array_equal(record["window_y"], arrays["window_labels"][a:b])
        record["margin"] = arrays["synthetic_margins"][a:b].copy()


def frozen_inputs(dataset, source, baseline, ablation, cnn):
    selected, identity = review_rows(dataset, source, baseline)
    for root in (ablation, cnn):
        seal_input(root)
        if read_json(root / "provenance.json")["data"] != identity:
            raise ValueError("Frozen model/data identity mismatch.")
    rows = selected["train"]
    expected = grouped_folds([r["group_id"] for r in rows], GLOBAL_SEED, 3)
    for receipt in (
        ablation / "candidates/A64/cross_validation.json",
        cnn / "cross_validation/cross_validation.json",
    ):
        for fold in read_json(receipt)["folds"]:
            held = {g for g, f in expected.items() if f == fold["fold"]}
            if (
                set(fold["held_out_groups"]) != held
                or set(fold["fit_groups"]) != set(expected) - held
            ):
                raise ValueError("Frozen fold/source-group membership differs.")
    linear = load_features(ablation / "features/A64/train", rows)
    attach_scores(linear, ablation / "candidates/A64/oof/window_predictions.npz")
    neural = load_features(cnn / "features/train", rows)
    attach_scores(neural, cnn / "cross_validation/oof/window_predictions.npz")
    model, _ = load_cnn(cnn / "bundle")
    final_cnn = [{**r, "margin": model.margins(r["features"])} for r in neural]
    final_a = saved_records(baseline, selected, "train")
    models = {
        "A64_OOF": (linear, GRID),
        "CNN_OOF": (neural, GEOMETRY),
        "A64_final_in_sample": (final_a, GRID),
        "CNN_final_in_sample": (final_cnn, GEOMETRY),
    }
    # Audit the actual frozen CNN sampler using position tokens, never new model inputs.
    tokens, offset = [], 0
    for record in neural:
        tokens.append(
            {**record, "features": np.arange(offset, offset + len(record["starts"]))[:, None]}
        )
        offset += len(record["starts"])
    chosen, _, _ = sampled_training(tokens)
    offset = 0
    for record in neural:
        record["sampled_mask"] = np.isin(
            np.arange(offset, offset + len(record["starts"])), chosen[:, 0]
        )
        offset += len(record["starts"])
    thresholds = {
        "A64": read_json(ablation / "candidates/A64/cross_validation.json")["points"]["fpr_05"][
            "threshold"
        ],
        "CNN": read_json(cnn / "selection_locked.json")["points"]["fpr_05"]["threshold"],
    }
    return rows, identity, models, thresholds


def dense(record, config):
    edges = center_edges(record["starts"], record["row"]["num_samples"], config.window_samples)
    return (
        np.repeat(record["margin"], np.diff(edges)),
        np.repeat(smooth_probabilities(expit(record["margin"]), 3), np.diff(edges)),
    )


def mean_or_missing(values):
    return float(np.mean(values)) if len(values) else None


def guarded_core_bounds(row):
    """Exclude the first sample of the closed transition-mask dilation."""
    start = row["start_sample"] + row["crossfade_samples_per_boundary"] + SUPPORT_GUARD + 1
    end = row["end_sample"] - row["crossfade_samples_per_boundary"] - SUPPORT_GUARD
    truth = guarded_labels(sample_labels(row["interval_labels"], row["num_samples"]), SUPPORT_GUARD)
    expected = 1 if row["has_synthetic"] else 0
    if start >= end or not np.all(truth[start:end] == expected):
        raise ValueError("Diagnostic core must contain only valid pure-source samples.")
    return start, end


def score_group(
    members, model_records, config, threshold, reference_rms, own_rms, activity_threshold
):
    """Dense origin-label diagnostics; truth and identities never alter predictions."""
    by_class = {r["example_class"]: r for r in members}
    null = by_class["null_control"]
    n = null["num_samples"]
    linear_starts = reference_rms[0]
    edges = center_edges(linear_starts, n, GRID.window_samples)
    reference_bins = np.repeat(activity_bins(reference_rms[1], activity_threshold), np.diff(edges))
    outputs, activity = [], []
    score_arrays = {}
    for record in model_records:
        row = record["row"]
        margin, probability = dense(record, config)
        truth = guarded_labels(sample_labels(row["interval_labels"], n), SUPPORT_GUARD)
        prediction = probability >= threshold
        own_bins = np.repeat(
            activity_bins(own_rms[row["example_class"]], activity_threshold), np.diff(edges)
        )
        scores = {
            "group_id": row["group_id"],
            "example_class": row["example_class"],
            **rates(confusion(truth, prediction)),
            "real_margin": distribution_finite(margin[truth == 0]),
            "synthetic_margin": distribution_finite(margin[truth == 1]),
        }
        scores["synthetic_minus_real_margin"] = (
            scores["synthetic_margin"]["mean"] - scores["real_margin"]["mean"]
            if scores["synthetic_margin"]["count"]
            else None
        )
        outputs.append(scores)
        score_arrays[row["example_class"]] = (margin, probability, truth)
        for assignment, bins in (
            ("reference_matched", reference_bins),
            ("condition_own", own_bins),
        ):
            for i, name in enumerate(ACTIVITY_NAMES):
                keep = bins == i
                activity.append(
                    {
                        "group_id": row["group_id"],
                        "example_class": row["example_class"],
                        "assignment": assignment,
                        "activity": name,
                        **confusion(truth[keep], prediction[keep]),
                        "real_margin_sum": float(margin[keep & (truth == 0)].sum()),
                        "synthetic_margin_sum": float(margin[keep & (truth == 1)].sum()),
                    }
                )
    core_s, core_e = guarded_core_bounds(null)
    for row in members:
        if guarded_core_bounds(row) != (core_s, core_e):
            raise ValueError("Paired hard-negative core coordinates differ.")
    r2r = by_class["real_to_real_control"]
    ds = r2r["donor_start_sample"] + core_s - null["start_sample"]
    de = ds + core_e - core_s
    null_margin, _, null_truth = score_arrays["null_control"]
    position = np.arange(n)
    far = (
        np.minimum(abs(position - null["start_sample"]), abs(position - null["end_sample"]))
        >= ARCHIVE_SR
    ) & (null_truth == 0)
    negatives = {
        "matched_real": float(null_margin[core_s:core_e].mean()),
        "real_to_real_core": float(score_arrays["real_to_real_control"][0][core_s:core_e].mean()),
        "null_full": float(null_margin[null_truth == 0].mean()),
        "far_real": mean_or_missing(null_margin[far]),
        "real_away": float(null_margin[ds:de].mean()),
    }
    cores = [
        {
            "group_id": null["group_id"],
            "vocoder": v,
            "synthetic_margin": float(score_arrays[f"mixed_{v}"][0][core_s:core_e].mean()),
            "negative_margins": negatives,
        }
        for v in VOCODERS
    ]
    return outputs, activity, cores


def distribution_finite(values):
    from experiments.temporal_review import distribution

    return {**distribution(values), "std": float(np.std(values)) if len(values) else None}


def aggregate_scores(score_rows, activity_rows, core_rows, models, covariates):
    summary = {}
    for name, (records, config) in models.items():
        subset = [r for r in score_rows if r["model"] == name]
        by_group = defaultdict(Counter)
        for row in subset:
            by_group[row["group_id"]].update({k: row[k] for k in ("tp", "fp", "fn", "tn")})
        source = {g: rates(c) for g, c in sorted(by_group.items())}
        null_rows = [r for r in subset if r["example_class"] == "null_control"]
        offsets = {r["group_id"]: r["real_margin"]["mean"] for r in null_rows}
        masked = [
            {
                **r,
                "score_labels": guarded_labels(
                    sample_labels(r["row"]["interval_labels"], r["row"]["num_samples"]),
                    SUPPORT_GUARD,
                ),
            }
            for r in records
        ]
        raw_rank = common_mask_ranking(
            masked, [r["margin"] for r in records], config, SUPPORT_GUARD
        )
        centered = [r["margin"] - offsets[r["row"]["group_id"]] for r in records]
        centered_rank = common_mask_ranking(masked, centered, config, SUPPORT_GUARD)
        top = sorted(source, key=lambda g: -source[g]["fp"])[:10]
        top_tp = sorted(source, key=lambda g: -source[g]["tp"])[:10]
        total_fp = sum(r["fp"] for r in source.values())
        total_tp = sum(r["tp"] for r in source.values())
        item = {
            "scope": "OOF" if name.endswith("OOF") else "in_sample_resubstitution_not_validation",
            "source_groups": source,
            "pooled_ranking": raw_rank,
            "oracle_null_source_centered_ranking": centered_rank,
            "source_macro": {
                k: group_summary([r[k] for r in source.values()])
                for k in ("f1", "recall", "false_positive_rate")
            },
            "null_source_score_means": group_summary(list(offsets.values())),
            "top10_FP_groups": top,
            "top10_FP_share": sum(source[g]["fp"] for g in top) / total_fp if total_fp else None,
            "top10_TP_groups": top_tp,
            "top10_TP_share": sum(source[g]["tp"] for g in top_tp) / total_tp if total_tp else None,
            "top10_real_exposure_share": sum(source[g]["fp"] + source[g]["tn"] for g in top)
            / sum(r["fp"] + r["tn"] for r in source.values()),
            "activity": {},
            "hard_negatives": {},
            "vocoder_margin_contrasts": {},
        }
        ordered = sorted(offsets)
        item["source_covariates"] = {
            "negative_score_vs_real_rms": correlation(
                [offsets[g] for g in ordered], [covariates[g]["rms"] for g in ordered]
            ),
            "negative_score_vs_real_activity": correlation(
                [offsets[g] for g in ordered], [covariates[g]["active_fraction"] for g in ordered]
            ),
            "FPR_vs_real_rms": correlation(
                [source[g]["false_positive_rate"] for g in ordered],
                [covariates[g]["rms"] for g in ordered],
            ),
            "FPR_vs_real_activity": correlation(
                [source[g]["false_positive_rate"] for g in ordered],
                [covariates[g]["active_fraction"] for g in ordered],
            ),
        }
        between = float(np.var(list(offsets.values())))
        within = (
            float(np.mean([r["real_margin"]["std"] ** 2 for r in null_rows]))
            if "std" in null_rows[0]["real_margin"]
            else None
        )
        item["between_source_negative_mean_variance"] = between
        item["within_source_real_margin_variance"] = within
        fold_map = grouped_folds(list(offsets), GLOBAL_SEED, 3)
        item["fold_negative_means"] = {
            str(f): group_summary([offsets[g] for g in ordered if fold_map[g] == f])
            for f in range(3)
        }
        for v in VOCODERS:
            row_values = [r for r in subset if r["example_class"] == f"mixed_{v}"]
            item["vocoder_margin_contrasts"][v] = group_summary(
                [r["synthetic_minus_real_margin"] for r in row_values]
            )
            cores = [r for r in core_rows if r["model"] == name and r["vocoder"] == v]
            for kind in cores[0]["negative_margins"]:
                valid = [r for r in cores if r["negative_margins"][kind] is not None]
                pos = np.array([r["synthetic_margin"] for r in valid])
                neg = np.array([r["negative_margins"][kind] for r in valid])
                contrast = pos - neg
                metric = binary_metrics(
                    np.r_[np.zeros(len(neg)), np.ones(len(pos))],
                    np.r_[neg, pos],
                    np.r_[neg, pos] >= 0,
                )
                item["hard_negatives"].setdefault(v, {})[kind] = {
                    "source_groups": len(valid),
                    "pooled_core_mean_auroc": metric["raw_auroc"],
                    "within_source_win_probability": group_summary(
                        (contrast > 0) + 0.5 * (contrast == 0)
                    ),
                    "paired_margin_difference": group_summary(contrast),
                    "oracle_diagnostic_only": True,
                }
        for assignment in ("reference_matched", "condition_own"):
            for activity in ACTIVITY_NAMES:
                a = [
                    r
                    for r in activity_rows
                    if r["model"] == name
                    and r["assignment"] == assignment
                    and r["activity"] == activity
                ]
                grouped = defaultdict(Counter)
                for r in a:
                    grouped[r["group_id"]].update({k: r[k] for k in ("tp", "fp", "fn", "tn")})
                stats = [rates(c) for c in grouped.values()]
                all_counts = Counter()
                for c in grouped.values():
                    all_counts.update(c)
                item["activity"][f"{assignment}:{activity}"] = {
                    "sample_micro": rates(all_counts),
                    "source_FPR": group_summary([r["false_positive_rate"] for r in stats]),
                    "source_recall": group_summary([r["recall"] for r in stats]),
                }
        summary[name] = item
    return summary


def nuisance_cv(records, destination):
    assignments = grouped_folds([r["row"]["group_id"] for r in records], GLOBAL_SEED, 3)
    results = {}
    for variant in VARIANTS:
        indices = balanced_indices(
            records, activity_matched=variant in {"activity_matched", "activity_rms"}
        )
        key = "rms_features" if variant in {"rms_normalized", "activity_rms"} else "features"
        features = np.concatenate([r[key][take] for r, take in zip(records, indices, strict=True)])
        labels = np.concatenate(
            [r["window_y"][take] for r, take in zip(records, indices, strict=True)]
        )
        groups = np.concatenate(
            [
                np.repeat(r["row"]["group_id"], len(take))
                for r, take in zip(records, indices, strict=True)
            ]
        )
        predictions = [None] * len(records)
        fits = []
        for fold in range(3):
            take = np.array([assignments[g] != fold for g in groups])
            pipeline = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=1, solver="lbfgs", max_iter=2000, tol=1e-6, random_state=GLOBAL_SEED
                        ),
                    ),
                ]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                pipeline.fit(
                    features[take],
                    labels[take],
                    classifier__sample_weight=loss_weights(labels[take], groups[take]),
                )
            head = LinearHead.from_pipeline(pipeline)
            root = destination / variant / f"fold-{fold}"
            root.mkdir(parents=True)
            np.savez_compressed(
                root / "model.npz",
                **{k: getattr(head, k) for k in ("mean", "scale", "coefficients", "intercept")},
            )
            fit = {
                "fold": fold,
                "fit_groups": sorted(set(groups[take])),
                "held_out_groups": sorted(g for g, f in assignments.items() if f == fold),
                "sampled_fit_windows": int(take.sum()),
                **FLAGS,
            }
            write_json(root / "fit.json", fit)
            fits.append(fit)
            for i, r in enumerate(records):
                if assignments[r["row"]["group_id"]] == fold:
                    predictions[i] = head.decision_function(r[key])[:, 0]
        if any(p is None for p in predictions):
            raise ValueError("Incomplete nuisance OOF predictions.")
        scored = [
            {
                **r,
                "margin": p,
                "score_labels": guarded_labels(
                    sample_labels(r["row"]["interval_labels"], r["row"]["num_samples"]),
                    SUPPORT_GUARD,
                ),
            }
            for r, p in zip(records, predictions, strict=True)
        ]
        points = {
            "fixed_05": {"threshold": 0.5},
            **control_operating_points(scored, predictions, GRID, SUPPORT_GUARD, targets=(0.05,)),
        }
        save_predictions(destination / variant / "oof", scored, GRID)
        result = {
            "sampled_windows": len(labels),
            "sampled_by_activity_class": {},
            "folds": fits,
            "points": points,
            "metrics": {},
        }
        for i, r in enumerate(records):
            for j in indices[i]:
                token = f"{r['window_y'][j]}:{ACTIVITY_NAMES[r['own_activity'][j]]}"
                result["sampled_by_activity_class"][token] = (
                    result["sampled_by_activity_class"].get(token, 0) + 1
                )
        for point, decision in points.items():
            metrics = evaluate_context(
                scored, [expit(p) for p in predictions], GRID, decision["threshold"], SUPPORT_GUARD
            )
            result["metrics"][point] = {k: v for k, v in metrics.items() if k != "recordings"}
            result["metrics"][point]["source_F1_uncertainty"] = group_summary(
                [v["f1"] for v in metrics["sample_by_source_group"].values()]
            )
            write_json(destination / variant / f"{point}.json", {**metrics, **FLAGS})
        write_json(destination / variant / "summary.json", {**result, **FLAGS})
        results[variant] = result
        print(f"Nuisance ablation {variant}: {len(labels)} training windows", flush=True)
    return results


def run(dataset_dir, source_manifest, baseline_dir, ablation_dir, cnn_dir, out_dir):
    dataset, source, baseline, ablation, cnn, target = (
        Path(p).resolve()
        for p in (dataset_dir, source_manifest, baseline_dir, ablation_dir, cnn_dir, out_dir)
    )
    staging = target.with_name(target.name + ".staging")
    if target.exists() or staging.exists():
        raise FileExistsError("Fresh Phase 3D output required.")
    if any(
        p.is_relative_to(r) or r.is_relative_to(p)
        for p in (target, staging)
        for r in (dataset, source.parent, baseline, ablation, cnn)
    ):
        raise ValueError("Phase 3D outputs must be separate from immutable inputs.")
    begin = perf_counter()
    with threadpool_limits(limits=1):
        rows, identity, models, thresholds = frozen_inputs(dataset, source, baseline, ablation, cnn)
        staging.mkdir(parents=True)
        write_json(staging / "plan.json", fixed_plan())
        code = code_provenance()
        repo = Path(__file__).resolve().parents[2]
        code["source_sha256"].update(
            {p.relative_to(repo).as_posix(): sha256_file(p) for p in (repo / "src").rglob("*.py")}
        )
        write_json(staging / "provenance.json", {"code": code, "data": identity, **FLAGS})
        work = staging / "_analysis"
        work.mkdir()
        try:
            frame = source_frame(source)
            train_frame = frame[frame.split == "train"]
            paths = {
                (str(r.utt_id), str(r.condition)): Path(r.path) for r in train_frame.itertuples()
            }
            catalog = read_json(dataset / "sources.json")["sources"]
            excluded_train = {
                r["utt_id"]: r
                for r in read_json(dataset / "selection.json")["excluded"]
                if r["selection_split"] == "train"
            }
            groups = sorted({r["group_id"] for r in rows})
            acoustic, score_rows, activities, core_rows, window_rows, sampling = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            validation, source_hashes, selection = [], {}, []
            alignment_pairs = {v: [] for v in VOCODERS}
            for group in groups:
                members = [r for r in rows if r["group_id"] == group]
                by_class = {r["example_class"]: r for r in members}
                null = by_class["null_control"]
                if any(r["split"] != "train" for r in members):
                    raise ValueError("Train-only analysis.")
                original = {}
                for condition in ("real", *VOCODERS):
                    path = paths[(group, condition)]
                    source_id = (
                        null["source_real_id"]
                        if condition == "real"
                        else by_class[f"mixed_{condition}"]["source_donor_id"]
                    )
                    digest = sha256_file(path)
                    if digest != catalog[source_id]["sha256"]:
                        raise ValueError("Original source hash mismatch.")
                    source_hashes[str(path)] = digest
                    original[condition] = read_input(path)
                for v in VOCODERS:
                    alignment_pairs[v].append((original["real"], original[v]))
                for row in members:
                    donor = original[row["source_donor_condition"]]
                    validation.append(
                        {
                            "example_id": row["example_id"],
                            **validate_written_example(dataset, row, original["real"], donor),
                        }
                    )
                wavs = {
                    **original,
                    **{
                        f"partial_{r['example_class']}": read_input(dataset / r["output_filename"])
                        for r in members
                    },
                }
                analysis = analysis_comparison(wavs, work)
                np.testing.assert_array_equal(analysis["real"], analysis["partial_null_control"])
                threshold = reference_activity_threshold(analysis["real"])
                starts = next(r for r in models["A64_OOF"][0] if r["row"]["group_id"] == group)[
                    "starts"
                ]
                ref_levels = window_rms(analysis["real"], starts)
                own = {}
                for row in members:
                    kind = row["example_class"]
                    wave = analysis[f"partial_{kind}"]
                    levels = window_rms(wave, starts)
                    own[kind] = levels
                    record = next(
                        r
                        for r in models["A64_OOF"][0]
                        if r["row"]["example_id"] == row["example_id"]
                    )
                    gain_features, gain_evidence = rms_controlled_features(wave, starts)
                    record["rms_features"] = gain_features
                    record["own_activity"] = activity_bins(levels, threshold)
                    window_rows.append(
                        {
                            "example_id": row["example_id"],
                            "own_rms": levels,
                            "reference_rms": ref_levels,
                            "activity_threshold": threshold,
                        }
                    )
                    sampling.append(
                        {
                            "group_id": group,
                            "example_class": kind,
                            "sampler": "A64_all_pure",
                            "counts": sampling_counts(record["window_y"], record["own_activity"]),
                            "rms_control": gain_evidence,
                        }
                    )
                    neural = next(
                        r
                        for r in models["CNN_OOF"][0]
                        if r["row"]["example_id"] == row["example_id"]
                    )
                    nl = window_rms(wave, neural["starts"], GEOMETRY.window_samples)
                    sampling.append(
                        {
                            "group_id": group,
                            "example_class": kind,
                            "sampler": "CNN_cap24",
                            "counts": sampling_counts(
                                neural["window_y"],
                                activity_bins(nl, threshold),
                                neural["sampled_mask"],
                            ),
                            "available_counts": sampling_counts(
                                neural["window_y"], activity_bins(nl, threshold)
                            ),
                        }
                    )
                s = null["start_sample"] + null["crossfade_samples_per_boundary"] + 3308
                e = null["end_sample"] - null["crossfade_samples_per_boundary"] - 3308
                for v in VOCODERS:
                    excluded = exclusion_for(HIGH_BAND_ANALYSIS, v)
                    for band, source_wavs in (
                        ("primary_below8k", analysis),
                        ("exploratory_archive_fullband", original),
                    ):
                        a = spectral_pair(
                            source_wavs["real"][s:e],
                            source_wavs[v][s:e],
                            threshold=threshold,
                            allow_fullband_distance=band != "primary_below8k" and excluded is None,
                        )
                        acoustic.append(
                            {
                                "group_id": group,
                                "vocoder": v,
                                "analysis": band,
                                **a,
                                "full_high_band_distance_exclusion": excluded.render()
                                if excluded
                                else None,
                                "fullband_profiles_are_processing_diagnostics_only": True,
                                **FLAGS,
                            }
                        )
                for name, (records, config) in models.items():
                    current = [r for r in records if r["row"]["group_id"] == group]
                    outputs, activity, cores = score_group(
                        members,
                        current,
                        config,
                        thresholds["CNN" if name.startswith("CNN") else "A64"],
                        (starts, ref_levels),
                        own,
                        threshold,
                    )
                    score_rows.extend({"model": name, **r} for r in outputs)
                    activities.extend({"model": name, **r} for r in activity)
                    core_rows.extend({"model": name, **r} for r in cores)
                selection.append(selection_record(group, analysis["real"], "eligible"))
                print(f"Phase 3D train source {group}", flush=True)
            alignment = {}
            for v, pairs in alignment_pairs.items():
                report = check_vocoder_alignment(pairs, v)
                try:
                    assert_aligned(report)
                    failure = None
                except InvariantViolation as exc:
                    failure = str(exc)
                alignment[v] = {
                    **report.to_dict(),
                    "gate_failure": failure,
                    "no_alignment_repair": True,
                }
            del alignment_pairs
            for group in sorted(set(train_frame.utt_id) - set(groups)):
                path = paths[(group, "real")]
                source_hashes[str(path)] = sha256_file(path)
                if (
                    source_hashes[str(path)]
                    != excluded_train[group]["observed_source_files"]["real"]["sha256"]
                ):
                    raise ValueError("Excluded original training source changed.")
                wave = read_input(path)
                filtered = analysis_comparison({"real": wave}, work)["real"]
                selection.append(
                    {
                        **selection_record(group, filtered, "excluded_from_partial_dataset"),
                        "exclusion_reason": excluded_train[group]["reason"],
                    }
                )
            write_json(
                staging / "source_audit.json",
                {
                    "outputs": validation,
                    "alignment": alignment,
                    "source_hashes": source_hashes,
                    "reconstructed_training_outputs": len(validation),
                    **FLAGS,
                },
            )
            write_json(staging / "acoustics.json", {"observations": acoustic, **FLAGS})
            write_json(
                staging / "source_scores.json",
                {"recordings": score_rows, "activity": activities, "cores": core_rows, **FLAGS},
            )
            write_json(staging / "sampling.json", {"records": sampling, **FLAGS})
            write_json(staging / "selection_bias.json", {"sources": selection, **FLAGS})
            np.savez_compressed(
                staging / "window_rms.npz",
                example_ids=np.array([r["example_id"] for r in window_rows]),
                offsets=np.r_[0, np.cumsum([len(r["own_rms"]) for r in window_rows])],
                own_rms=np.concatenate([r["own_rms"] for r in window_rows]),
                reference_rms=np.concatenate([r["reference_rms"] for r in window_rows]),
                thresholds=np.array([r["activity_threshold"] for r in window_rows]),
            )
            covariates = {r["group_id"]: r for r in selection if r["status"] == "eligible"}
            scores = aggregate_scores(score_rows, activities, core_rows, models, covariates)
            write_json(staging / "score_summary.json", {**scores, **FLAGS})
            write_json(staging / "acoustic_summary.json", summarize_acoustics(acoustic))
            # A failed unchanged alignment gate is a blocker, never waived for a new fit.
            if all(r["passed"] for r in alignment.values()):
                ablations = nuisance_cv(models["A64_OOF"][0], staging / "nuisance")
                write_json(staging / "nuisance_summary.json", {**ablations, **FLAGS})
            else:
                write_json(
                    staging / "nuisance_summary.json",
                    {
                        "status": "not_run_alignment_gate_failed",
                        "blockers": [v for v, r in alignment.items() if not r["passed"]],
                        **FLAGS,
                    },
                )
            work.rmdir()
            for path, digest in source_hashes.items():
                if sha256_file(Path(path)) != digest:
                    raise ValueError("Original source changed during diagnosis.")
            _, after = review_rows(dataset, source, baseline)
            if after != identity:
                raise ValueError("Immutable dataset identity changed.")
            write_json(
                staging / "validation.json",
                {
                    "status": "passed_structural_diagnostic_integrity",
                    "alignment_gates_all_passed": all(r["passed"] for r in alignment.values()),
                    "train_groups": len(groups),
                    "original_train_groups": len(set(train_frame.utt_id)),
                    "training_outputs_reconstructed": len(validation),
                    "eval_waveforms_opened": 0,
                    "dev_waveforms_decoded": 0,
                    "new_neural_training": False,
                    "elapsed_seconds": perf_counter() - begin,
                    **FLAGS,
                },
            )
            write_json(
                staging / "artifacts.sha256.json", {"files": artifact_inventory(staging), **FLAGS}
            )
            staging.rename(target)
            return {"output": str(target), "status": "completed", **FLAGS}
        except Exception as exc:
            write_json(staging / "failure.json", {"status": "failed", "error": str(exc), **FLAGS})
            raise


def sampling_counts(y, activity, selected=None):
    keep = (y >= 0) if selected is None else (y >= 0) & selected
    return {
        f"{label}:{name}": int(np.sum(keep & (y == label) & (activity == i)))
        for label in (0, 1)
        for i, name in enumerate(ACTIVITY_NAMES)
    }


def selection_record(group, wave, status):
    _, levels = frame_rms(wave)
    threshold = max(float(levels.max()) * 10 ** (-30 / 20), 2 / 32768) if len(levels) else 2 / 32768
    return {
        "group_id": group,
        "status": status,
        "duration_seconds": len(wave) / ARCHIVE_SR,
        "rms": float(np.sqrt(np.mean(wave.astype(float) ** 2))),
        "active_fraction": float(np.mean(levels >= threshold)) if len(levels) else None,
    }


def summarize_acoustics(rows):
    summary = {}
    for band in sorted({r["analysis"] for r in rows}):
        summary[band] = {}
        for v in VOCODERS:
            take = [r for r in rows if r["analysis"] == band and r["vocoder"] == v]
            summary[band][v] = {
                key: group_summary([r[key] for r in take])
                for key in (
                    "rms_ratio_db",
                    "lsd_below8k_db",
                    "shape_lsd_below8k_db",
                    "envelope_below8k_db",
                    "periodicity_difference",
                    "flatness_difference",
                    "zero_lag_waveform_correlation",
                    "relative_waveform_error_rms",
                    "fullband_lsd_db",
                    "real_high_band_fraction",
                    "synthetic_high_band_fraction",
                )
            }
            summary[band][v]["bands"] = [
                group_summary([r["band_energy_ratio_db"][i] for r in take])
                for i in range(len(BANDS))
            ]
    return {"by_analysis": summary, **FLAGS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dataset-dir",
        "source-manifest",
        "baseline-dir",
        "ablation-dir",
        "cnn-dir",
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
            args.cnn_dir,
            args.out_dir,
        )
    )


if __name__ == "__main__":
    main()
