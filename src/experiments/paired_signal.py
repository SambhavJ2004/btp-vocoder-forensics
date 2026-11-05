"""Training-only paired/oracle diagnostics; never a deployed inference path.

Ground truth selects diagnostic intervals only. Oracle classification is not
localization. All existing inputs/artifacts are read-only and eval is unused.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from data.invariants import ARCHIVE_SR, GLOBAL_SEED, TRIM_FRAME_LENGTH, TRIM_HOP_LENGTH
from data.partial_audio import VOCODERS
from detectors.forensic_pilot import LinearHead, sha256_file, write_json
from detectors.temporal_features import WindowConfig, analysis_comparison, extract_windows
from detectors.temporal_metrics import binary_metrics
from detectors.temporal_operating import grouped_folds, weighted_control_cutoff
from experiments.forensic_pilot import fit_head
from experiments.partial_audio import FLAGS, artifact_inventory, read_input, source_frame
from experiments.temporal_localization import code_provenance
from experiments.temporal_review import distribution, read_json, review_rows

FEATURE_CONFIG = WindowConfig(window_ms=64, hop_ms=16)
KINDS = ("real_target", "real_away", "real_to_real_control", *VOCODERS)
VARIANTS = ("raw", "rms", "active", "rms_active")


@dataclass(frozen=True)
class PairedConfig:
    seed: int = GLOBAL_SEED
    rms_target: float = 0.03
    min_active_frames: int = 8
    activity_top_db: float = 30.0
    core_guard_samples: int = FEATURE_CONFIG.guard_samples
    oracle_c: float = 1.0
    signal_auroc_floor: float = 0.60
    signal_recall_floor: float = 0.10
    oracle_fpr_cap: float = 0.05

    def validate(self):
        if asdict(self) != asdict(PairedConfig()):
            raise ValueError("Require the single predeclared Phase 3C diagnostic recipe.")


DEFAULT_PAIRED_CONFIG = PairedConfig()


def phase3c_plan():
    return {
        "paired": asdict(PairedConfig()),
        "oracle_variants": list(VARIANTS),
        "oracle_representation": "mean_std_of_31_64ms_spectral_features",
        "oracle_negatives": ["real_target", "real_away", "real_to_real_control"],
        "activity_matching": (
            "raw_below8k_target_intersection_real_and_all_vocoders_away_intersection_real_and_r2r"
        ),
        "folds": 3,
        "fold_policy": "reuse_phase3b_source_hash_folds_train_only",
        "learned_decision": (
            "rms_active_oof_auroc_ge_0.60_and_recall_ge_0.10_at_5pct_each_negative_core_fpr"
        ),
        "learned_candidate_if_signal": "one_numpy_cnn_logmel24_128ms_hop16ms",
        "cnn_channels": 8,
        "cnn_kernel": [3, 3],
        "cnn_pool": "frequency_pairs",
        "cnn_optimizer": "Adam_lr0.001_batch256_l2_0.0001_dropout0.10",
        "cnn_epochs_max": 20,
        "cnn_patience": 4,
        "cnn_min_delta": 0.0001,
        "cnn_training_sample_cap_each_record_label": 24,
        "cnn_loss_weighting": "equal_source_and_binary_class",
        "cnn_epoch_selection": (
            "inner_group_holdout_within_outer_train_then_outer_refit_median_epoch_final"
        ),
        "cnn_fpr_targets": [0.01, 0.05, 0.10],
        "cnn_smoothing": 3,
        "cnn_merge_ms": 0,
        "localization_score_guard_samples": 4544,
        "eval_used": False,
        "dev_role": "benchmark_only_after_train_choices_locked",
        **FLAGS,
    }


def rms(wav):
    if np.iscomplexobj(wav):
        raise ValueError("Diagnostic samples must be real.")
    values = np.asarray(wav, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Require nonempty finite mono diagnostic samples.")
    return float(np.sqrt(np.mean(values**2)))


def frame_rms(wav, length=TRIM_FRAME_LENGTH, hop=TRIM_HOP_LENGTH):
    if np.iscomplexobj(wav):
        raise ValueError("Activity samples must be real.")
    values = np.asarray(wav, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or length <= 0 or hop <= 0:
        raise ValueError("Invalid activity samples/geometry.")
    starts = np.arange(0, max(0, len(values) - length + 1), hop, dtype=np.int64)
    prefix = np.r_[0.0, np.cumsum(values**2)]
    return starts, np.sqrt(np.maximum((prefix[starts + length] - prefix[starts]) / length, 0))


def reference_activity_threshold(real, config=DEFAULT_PAIRED_CONFIG):
    config.validate()
    _, levels = frame_rms(real)
    if not len(levels) or levels.max() <= 2 / 32768:
        raise ValueError("No measurable real-reference activity.")
    return max(float(levels.max()) * 10 ** (-config.activity_top_db / 20), 2 / 32768)


def feature_activity(wav, starts, threshold):
    prefix = np.r_[0.0, np.cumsum(np.asarray(wav, dtype=float) ** 2)]
    level = np.sqrt(
        np.maximum(
            (prefix[starts + FEATURE_CONFIG.window_samples] - prefix[starts])
            / FEATURE_CONFIG.window_samples,
            0,
        )
    )
    return level >= threshold


def measures(wav, threshold):
    _, levels = frame_rms(wav)
    _, features = extract_windows(wav, FEATURE_CONFIG)
    # Mean frame PSD from log features, kept below the shared analysis band.
    band_psd = (10 ** features[:, :24]).mean(axis=0)
    return {
        "rms": rms(wav),
        "sample_peak": float(np.max(np.abs(wav))),
        "activity_fraction": float(np.mean(levels >= threshold)) if len(levels) else None,
        "activity_threshold": threshold,
        "frame_rms": distribution(levels),
        "inband_power_mean": float(np.mean(10 ** features[:, 24])),
        "mean_psd_bands24": band_psd.tolist(),
        "bands_1khz_mean_psd": band_psd.reshape(8, 3).mean(axis=1).tolist(),
        "feature_mean31": features.mean(axis=0).tolist(),
        "feature_std31": features.std(axis=0).tolist(),
    }


def oracle_vectors(cores: dict, threshold, config=DEFAULT_PAIRED_CONFIG):
    """One observation per core; truth-based cropping is explicitly ORACLE.

    Activity masks are frozen before gain normalization. The paired target
    mask is shared across real/all three vocoders; away/R2R share their own
    mask. RMS control uses the union of selected frame support, avoiding a
    residual gain difference when active/inactive proportions differ.
    """
    config.validate()
    if set(cores) != set(KINDS) or len({len(w) for w in cores.values()}) != 1:
        raise ValueError("Oracle cores require six paired, equally long roles.")
    raw = {key: extract_windows(wav, FEATURE_CONFIG) for key, wav in cores.items()}
    starts = raw["real_target"][0]
    if len(cores["real_target"]) < FEATURE_CONFIG.window_samples:
        return None, {"reason": "core_shorter_than_feature_window"}
    masks = {key: feature_activity(wav, starts, threshold) for key, wav in cores.items()}
    target_mask = np.logical_and.reduce([masks[key] for key in ("real_target", *VOCODERS)])
    away_mask = masks["real_away"] & masks["real_to_real_control"]
    if min(target_mask.sum(), away_mask.sum()) < config.min_active_frames:
        return None, {
            "reason": "insufficient_common_active_frames",
            "target_active_frames": int(target_mask.sum()),
            "away_active_frames": int(away_mask.sum()),
        }
    result, evidence = (
        {},
        {
            "target_active_frames": int(target_mask.sum()),
            "away_active_frames": int(away_mask.sum()),
            "all_frames": len(starts),
            "gains": {},
        },
    )
    for variant in VARIANTS:
        result[variant] = {}
        for key, wav in cores.items():
            selected = (
                (away_mask if key in ("real_away", "real_to_real_control") else target_mask)
                if "active" in variant
                else np.ones(len(starts), dtype=bool)
            )
            values = np.asarray(wav, dtype=float)
            gain = 1.0
            if "rms" in variant:
                support = np.zeros(len(values), dtype=bool)
                for start in starts[selected]:
                    support[start : start + FEATURE_CONFIG.window_samples] = True
                level = rms(values[support])
                if level <= 2 / 32768:
                    return None, {
                        "reason": "unmeasurable_rms_for_normalization",
                        "kind": key,
                        "variant": variant,
                        "measured_rms": level,
                    }
                gain = config.rms_target / level
            _, features = extract_windows(values * gain, FEATURE_CONFIG) if gain != 1 else raw[key]
            take = features[selected]
            result[variant][key] = np.r_[take.mean(axis=0), take.std(axis=0)]
            evidence["gains"][f"{variant}:{key}"] = {
                "gain": gain,
                "scaled_sample_peak": float(np.max(np.abs(values * gain))),
            }
    return result, evidence


def oracle_metrics(observations, margin, cap=0.05):
    y = np.array([r["label"] for r in observations])
    p = expit(margin)
    cutoffs = {}
    for kind in KINDS[:3]:
        keep = np.array([r["kind"] == kind for r in observations])
        cutoffs[kind] = weighted_control_cutoff(p[keep], np.ones(int(keep.sum())), cap)
    threshold = max(cutoffs.values())
    result = binary_metrics(y, margin, p >= threshold)
    result.update(
        threshold=threshold,
        negative_core_fpr_cap=cap,
        negative_core_fpr={
            key: float(np.mean(p[[r["kind"] == key for r in observations]] >= threshold))
            for key in KINDS[:3]
        },
        per_vocoder={},
        per_group={},
        per_recording_scores=[
            {**r, "synthetic_margin": float(s), "uncalibrated_probability": float(pp)}
            for r, s, pp in zip(observations, margin, p, strict=True)
        ],
    )
    for vocoder in VOCODERS:
        for scope, negatives in (
            ("all_negatives", KINDS[:3]),
            ("same_time_real", ("real_target",)),
        ):
            keep = np.array([r["kind"] in (*negatives, vocoder) for r in observations])
            result["per_vocoder"].setdefault(vocoder, {})[scope] = binary_metrics(
                y[keep], margin[keep], p[keep] >= threshold
            )
    for group in sorted({r["group_id"] for r in observations}):
        keep = np.array([r["group_id"] == group for r in observations])
        result["per_group"][group] = binary_metrics(y[keep], margin[keep], p[keep] >= threshold)
    result["group_auroc_distribution"] = distribution(
        [r["raw_auroc"] for r in result["per_group"].values()]
    )
    result["group_macro_f1"] = float(
        np.mean([r["f1"] if r["f1"] is not None else 0 for r in result["per_group"].values()])
    )
    result["oracle_only_not_localization"] = True
    return result


def fit_oracle(vectors, assignments, out):
    groups = sorted(vectors)
    observations = [
        {"group_id": group, "kind": kind, "label": int(kind in VOCODERS)}
        for group in groups
        for kind in KINDS
    ]
    all_results = {}
    with threadpool_limits(limits=1):
        for variant in VARIANTS:
            x = np.array([vectors[r["group_id"]][variant][r["kind"]] for r in observations])
            y = np.array([r["label"] for r in observations])
            margin = np.full(len(y), np.nan)
            for fold in sorted(set(assignments.values())):
                held = np.array([assignments[r["group_id"]] == fold for r in observations])
                if not held.any() or not (~held).any():
                    raise ValueError("Empty eligible oracle fold.")
                pipeline = fit_head(x[~held], y[~held], 1.0, GLOBAL_SEED, balanced=True)
                head = LinearHead.from_pipeline(pipeline)
                head.validate(62, 1)
                margin[held] = head.decision_function(x[held])[:, 0]
                root = out / variant / f"fold-{fold}"
                root.mkdir(parents=True)
                np.savez_compressed(
                    root / "numeric_model.npz",
                    **{k: getattr(head, k) for k in ("mean", "scale", "coefficients", "intercept")},
                )
                write_json(
                    root / "fit.json",
                    {
                        "fit_groups": sorted(
                            {
                                r["group_id"]
                                for r, h in zip(observations, held, strict=True)
                                if not h
                            }
                        ),
                        "held_out_groups": sorted(
                            {r["group_id"] for r, h in zip(observations, held, strict=True) if h}
                        ),
                        "scaler_scope": "fold_fit_groups_only",
                        "C": 1.0,
                        "score_direction": "higher_is_synthetic",
                        **FLAGS,
                    },
                )
            if not np.isfinite(margin).all():
                raise ValueError("Incomplete oracle OOF coverage.")
            metrics = oracle_metrics(observations, margin)
            write_json(out / variant / "metrics.json", {**metrics, **FLAGS})
            np.savez_compressed(
                out / variant / "features_predictions.npz",
                features=x,
                labels=y,
                synthetic_margins=margin,
                group_ids=np.array([r["group_id"] for r in observations]),
                kinds=np.array([r["kind"] for r in observations]),
            )
            all_results[variant] = {
                k: v for k, v in metrics.items() if k not in {"per_recording_scores", "per_group"}
            }
    controlled = all_results["rms_active"]
    proceed = controlled["raw_auroc"] >= 0.60 and controlled["recall"] >= 0.10
    return {
        "variants": all_results,
        "learned_model_signal_decision": {
            "proceed": bool(proceed),
            "based_on": "rms_active_train_group_oof_only",
            "pooled_auroc": controlled["raw_auroc"],
            "core_recall_at_each_negative_5pct": controlled["recall"],
            "criterion": (
                "AUROC >= .60 and recall >= .10; diagnostic decision, "
                "not a scientific/publication gate"
            ),
            "oracle_not_localization": True,
        },
        **FLAGS,
    }


def boundary_measures(real, full_synthetic, partial, row):
    """Paired edit/synthesis differences around raw edit boundaries, no repair."""
    result = []
    for edge_name, edge in (("onset", row["start_sample"]), ("offset", row["end_sample"])):
        for lo, hi in ((0, 0.05), (0.05, 0.15), (0.15, 0.25), (0.25, 0.40)):
            for side, sign in (("before", -1), ("after", 1)):
                s, e = (
                    (edge - round(hi * ARCHIVE_SR), edge - round(lo * ARCHIVE_SR))
                    if sign < 0
                    else (edge + round(lo * ARCHIVE_SR), edge + round(hi * ARCHIVE_SR))
                )
                s, e = max(0, s), min(len(real), e)
                if e <= s:
                    continue
                result.append(
                    {
                        "boundary": edge_name,
                        "side": side,
                        "distance_seconds": [lo, hi],
                        "samples": [s, e],
                        "partial_minus_real_rms": rms(partial[s:e] - real[s:e]),
                        "partial_minus_full_synthetic_rms": rms(partial[s:e] - full_synthetic[s:e]),
                        "partial_rms": rms(partial[s:e]),
                        "real_rms": rms(real[s:e]),
                        "full_synthetic_rms": rms(full_synthetic[s:e]),
                    }
                )
    return result


def run_paired_diagnostics(dataset_dir, source_manifest, baseline_dir, out_dir):
    dataset, source, baseline, target = (
        Path(p).resolve() for p in (dataset_dir, source_manifest, baseline_dir, out_dir)
    )
    staging = target.with_name(target.name + ".staging")
    if target.exists() or staging.exists():
        raise FileExistsError("Fresh diagnostic output required.")
    if any(
        p.is_relative_to(root) or root.is_relative_to(p)
        for p in (target, staging)
        for root in (dataset, source.parent, baseline)
    ):
        raise ValueError("Diagnostic outputs must be separate from all inputs.")
    selected, identity = review_rows(dataset, source, baseline)
    rows = selected["train"]
    assignments = grouped_folds([r["group_id"] for r in rows], GLOBAL_SEED, 3)
    staging.mkdir(parents=True)
    write_json(staging / "plan.json", phase3c_plan())
    code = code_provenance()
    repo = Path(__file__).resolve().parents[2]
    for path in sorted((repo / "src").rglob("*.py")):
        code["source_sha256"][path.relative_to(repo).as_posix()] = sha256_file(path)
    write_json(staging / "provenance.json", {"code": code, "data": identity, **FLAGS})
    write_json(staging / "folds.json", {"assignments": assignments, **FLAGS})
    work = staging / "_analysis"
    work.mkdir()
    catalog = read_json(dataset / "sources.json")["sources"]
    frame = source_frame(source)
    paths = {(str(r.utt_id), str(r.condition)): Path(r.path) for r in frame.itertuples()}
    all_measures, paired_deltas, boundaries, exclusions, vectors = [], [], [], [], {}
    core_residuals = []
    source_hashes = {}
    start_time = perf_counter()
    try:
        with threadpool_limits(limits=1):
            for group in sorted(assignments):
                members = [r for r in rows if r["group_id"] == group]
                by_class = {r["example_class"]: r for r in members}
                null = by_class["null_control"]
                r2r = by_class["real_to_real_control"]
                if any(r["split"] != "train" for r in members):
                    raise ValueError("Diagnostics cannot use dev/eval.")
                keys = ("start_sample", "end_sample", "crossfade_samples_per_boundary")
                if any(tuple(r[k] for k in keys) != tuple(null[k] for k in keys) for r in members):
                    raise ValueError("Paired oracle intervals must share actual edit coordinates.")
                raw_core = [
                    null["start_sample"] + null[keys[2]],
                    null["end_sample"] - null[keys[2]],
                ]
                if any(
                    by_class[f"mixed_{v}"]["pure_synthetic_ranges"] != [raw_core] for v in VOCODERS
                ):
                    raise ValueError("Oracle core differs from generator pure-synthetic labels.")
                if r2r["donor_end_sample"] - r2r["donor_start_sample"] != null["end_sample"] - null[
                    "start_sample"
                ] or not (
                    r2r["donor_end_sample"] <= null["start_sample"]
                    or r2r["donor_start_sample"] >= null["end_sample"]
                ):
                    raise ValueError("Matched away control must be a disjoint real donor.")
                wavs = {}
                for condition in ("real", *VOCODERS):
                    path = paths[(group, condition)]
                    key = (
                        null["source_real_id"]
                        if condition == "real"
                        else by_class[f"mixed_{condition}"]["source_donor_id"]
                    )
                    digest = sha256_file(path)
                    if digest != catalog[key]["sha256"]:
                        raise ValueError("Paired original source hash changed.")
                    source_hashes[str(path)] = digest
                    wavs[condition] = read_input(path)
                for kind, row in by_class.items():
                    wavs[f"partial_{kind}"] = read_input(dataset / row["output_filename"])
                if len({len(w) for w in wavs.values()}) != 1:
                    raise ValueError("Paired source lengths disagree.")
                analysis = analysis_comparison(wavs, work)
                np.testing.assert_array_equal(analysis["real"], analysis["partial_null_control"])
                threshold = reference_activity_threshold(analysis["real"])
                s = (
                    null["start_sample"]
                    + null["crossfade_samples_per_boundary"]
                    + FEATURE_CONFIG.guard_samples
                )
                e = (
                    null["end_sample"]
                    - null["crossfade_samples_per_boundary"]
                    - FEATURE_CONFIG.guard_samples
                )
                ds = r2r["donor_start_sample"] + (s - null["start_sample"])
                de = ds + e - s
                if not (
                    0 <= s < e <= len(analysis["real"]) and 0 <= ds < de <= len(analysis["real"])
                ):
                    raise ValueError("Invalid oracle bounds.")
                cores = {
                    "real_target": analysis["real"][s:e],
                    "real_away": analysis["real"][ds:de],
                    "real_to_real_control": analysis["partial_real_to_real_control"][s:e],
                    **{v: analysis[v][s:e] for v in VOCODERS},
                }
                for condition in (*VOCODERS, "real_to_real_control"):
                    actual = (
                        analysis[f"partial_mixed_{condition}"][s:e]
                        if condition in VOCODERS
                        else cores[condition]
                    )
                    original = cores[condition] if condition in VOCODERS else cores["real_away"]
                    delta = actual - original
                    core_residuals.append(
                        {
                            "group_id": group,
                            "condition": condition,
                            "rms_difference": rms(delta),
                            "max_pcm16_step_difference": float(np.max(np.abs(delta))) * 32768,
                            "changed_samples": int(np.count_nonzero(delta)),
                            "samples": len(delta),
                            **FLAGS,
                        }
                    )
                measured = {key: measures(w, threshold) for key, w in cores.items()}
                for kind, value in measured.items():
                    all_measures.append(
                        {
                            "group_id": group,
                            "split": "train",
                            "kind": kind,
                            "oracle_interval_samples": [ds, de] if kind == "real_away" else [s, e],
                            "ground_truth_edit_samples": [null["start_sample"], null["end_sample"]],
                            **value,
                            **FLAGS,
                        }
                    )
                real = measured["real_target"]
                for kind in (*VOCODERS, "real_away", "real_to_real_control"):
                    item = measured[kind]
                    paired_deltas.append(
                        {
                            "group_id": group,
                            "kind": kind,
                            "rms_ratio_db": 20 * np.log10(item["rms"] / real["rms"]),
                            "sample_peak_difference": item["sample_peak"] - real["sample_peak"],
                            "activity_fraction_difference": item["activity_fraction"]
                            - real["activity_fraction"],
                            "inband_energy_ratio_db": 10
                            * np.log10(item["inband_power_mean"] / real["inband_power_mean"]),
                            "bands_1khz_ratio_db": (
                                10
                                * np.log10(
                                    np.maximum(item["bands_1khz_mean_psd"], 1e-12)
                                    / np.maximum(real["bands_1khz_mean_psd"], 1e-12)
                                )
                            ).tolist(),
                            **FLAGS,
                        }
                    )
                for v in VOCODERS:
                    row = by_class[f"mixed_{v}"]
                    boundaries.append(
                        {
                            "group_id": group,
                            "condition": v,
                            "measurements": boundary_measures(
                                analysis["real"], analysis[v], analysis[f"partial_mixed_{v}"], row
                            ),
                            **FLAGS,
                        }
                    )
                for kind in ("null_control", "real_to_real_control"):
                    row = by_class[kind]
                    boundaries.append(
                        {
                            "group_id": group,
                            "condition": kind,
                            "measurements": boundary_measures(
                                analysis["real"], analysis["real"], analysis[f"partial_{kind}"], row
                            ),
                            **FLAGS,
                        }
                    )
                value, evidence = oracle_vectors(cores, threshold)
                if value is None:
                    exclusions.append({"group_id": group, "split": "train", **evidence, **FLAGS})
                else:
                    vectors[group] = value
                    write_json(
                        staging / f"oracle-evidence-{group}.json",
                        {"group_id": group, **evidence, **FLAGS},
                    )
                status = "eligible" if value is not None else evidence["reason"]
                print(f"Paired diagnostics {group}: {status}", flush=True)
        work.rmdir()
        write_json(staging / "paired_measures.json", {"observations": all_measures, **FLAGS})
        write_json(staging / "paired_deltas.json", {"observations": paired_deltas, **FLAGS})
        write_json(staging / "boundary_measures.json", {"groups_conditions": boundaries, **FLAGS})
        write_json(
            staging / "editing_core_residuals.json", {"observations": core_residuals, **FLAGS}
        )
        write_json(staging / "exclusions.json", {"exclusions": exclusions, **FLAGS})
        summary = {
            kind: {
                key: distribution([r[key] for r in paired_deltas if r["kind"] == kind])
                for key in (
                    "rms_ratio_db",
                    "sample_peak_difference",
                    "activity_fraction_difference",
                    "inband_energy_ratio_db",
                )
            }
            for kind in (*VOCODERS, "real_away", "real_to_real_control")
        }
        write_json(
            staging / "paired_summary.json",
            {"by_condition": summary, "source_groups": len(assignments), **FLAGS},
        )
        if len(vectors) < 6 or len({assignments[g] for g in vectors}) != 3:
            raise ValueError("Insufficient measured oracle coverage; do not fabricate folds.")
        oracle = fit_oracle(vectors, assignments, staging / "oracle")
        write_json(staging / "oracle_summary.json", oracle)
        _, after = review_rows(dataset, source, baseline)
        if after != identity:
            raise ValueError("Immutable inputs changed during paired diagnostics.")
        write_json(
            staging / "validation.json",
            {
                "status": "passed",
                "training_source_groups": len(assignments),
                "oracle_eligible_groups": len(vectors),
                "oracle_excluded_groups": len(exclusions),
                "oracle_fold_counts": dict(Counter(assignments[g] for g in vectors)),
                "original_train_source_sha256": source_hashes,
                "eval_waveforms_opened": 0,
                "eval_annotation_records_used": 0,
                "dev_predictive_processing": False,
                "elapsed_seconds": perf_counter() - start_time,
                **FLAGS,
            },
        )
        write_json(
            staging / "artifacts.sha256.json", {"files": artifact_inventory(staging), **FLAGS}
        )
        staging.rename(target)
        return {
            "output": str(target),
            "learned_model_signal_decision": oracle["learned_model_signal_decision"],
            **FLAGS,
        }
    except Exception as exc:
        write_json(staging / "failure.json", {"error": str(exc), "status": "failed", **FLAGS})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-dir", "source-manifest", "baseline-dir", "out-dir"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    print(
        run_paired_diagnostics(
            args.dataset_dir, args.source_manifest, args.baseline_dir, args.out_dir
        )
    )


if __name__ == "__main__":
    main()
