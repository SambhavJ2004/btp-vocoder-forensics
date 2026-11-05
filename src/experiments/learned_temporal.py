"""Conditional single-CNN development experiment; train groups first, dev benchmark last."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from data.invariants import ARCHIVE_SR, GLOBAL_SEED
from detectors.forensic_pilot import sha256_file, write_json
from detectors.temporal_cnn import (
    FIELDS,
    GEOMETRY,
    extract_logmel,
    load_cnn,
    predict_cnn_file,
    save_cnn,
    train_cnn,
)
from detectors.temporal_features import (
    analysis_comparison,
    assign_window_labels,
    filter_support_receipt,
    guarded_labels,
    sample_labels,
)
from detectors.temporal_operating import (
    common_mask_ranking,
    control_operating_points,
    evaluate_context,
    grouped_folds,
)
from experiments.paired_signal import phase3c_plan
from experiments.partial_audio import FLAGS, artifact_inventory, read_input
from experiments.temporal_ablation import (
    evaluate_points,
    load_features,
    save_features,
    save_predictions,
)
from experiments.temporal_localization import code_provenance
from experiments.temporal_review import SCORE_GUARD, checked_inventory, read_json, review_rows

SUPPORT_GUARD = (
    GEOMETRY.guard_samples
    + (GEOMETRY.window_samples + 1) // 2
    + GEOMETRY.hop_samples
    + (GEOMETRY.hop_samples + 1) // 2
)


def learned_plan():
    return {
        **phase3c_plan(),
        "execution": "single_fixed_numpy_cnn_after_paired_signal_criterion_passes",
        "inner_holdout_fraction": 0.20,
        "inner_holdout_count": "ceil_20pct_of_outer_fit_source_groups",
        "inner_group_order": "sha256_seed_phase3c_inner_group",
        "trainable_parameters": 521,
        "patch_shape": [24, 7],
        "subframe_samples": 706,
        "subframe_hop_samples": 353,
        "subframe_final_anchor": True,
        "fft_samples": 1024,
        "mel_definition": "reuse_phase3b_HTK_discrete_PSD_mean_weights_below_8k",
        "log_floor": 1e-12,
        "feature_storage": "float32_log10_psd",
        "numeric_training": "float64_numpy_scipy",
        "normalization": "per_mel_band_mean_std_over_fit_sample_patches_and_time_only",
        "sampling": "up_to_24_pure_windows_each_record_and_binary_label_hash_seeded",
        "loss": "weighted_BCE_mean_plus_L2_half_squared_kernel_and_head",
        "loss_weights": "N_over_2G_n_source_class_mean_weight_one",
        "early_stop_loss": "inner_source_class_balanced_BCE_without_dropout_or_L2",
        "early_stop_scope": "inner_fit_only_then_outer_refit_from_scratch",
        "final_epochs": "integer_median_of_three_outer_inner_selected_epochs",
        "seed_policy": "inner_GLOBAL_plus_100fold_plus_11_outer_plus_21_final_plus_1000",
        "no_oracle_crops_or_gain_or_activity_masks_at_inference": True,
        "training_membership": "all_frozen_train_groups_not_just_oracle_eligible",
        "primary_operating_point": "fpr_05_from_train_group_OOF",
        "secondary_support_guard_samples": SUPPORT_GUARD,
        "secondary_support_scope": (
            "conservative_filter_window_smoothing_projection_guard_5249_"
            "primary_fixed_4544_is_origin_label_comparison_not_full_support_isolation"
        ),
    }


def support_metrics(records, points, destination):
    """Secondary origin mask with the larger CNN effective support, no reselection."""
    destination.mkdir(parents=True, exist_ok=False)
    masked = [
        {
            **r,
            "score_labels": guarded_labels(
                sample_labels(r["row"]["interval_labels"], r["row"]["num_samples"]), SUPPORT_GUARD
            ),
        }
        for r in records
    ]
    probabilities = [expit(r["margin"]) for r in masked]
    ranking = common_mask_ranking(masked, [r["margin"] for r in masked], GEOMETRY, SUPPORT_GUARD)
    for name, point in points.items():
        metrics = evaluate_context(
            masked, probabilities, GEOMETRY, point["threshold"], SUPPORT_GUARD
        )
        write_json(
            destination / f"{name}.json",
            {
                **metrics,
                "score_guard_samples": SUPPORT_GUARD,
                "common_mask_ranking": ranking,
                "threshold_reselected": False,
                **FLAGS,
            },
        )


def sampled_training(records):
    patches, labels, groups = [], [], []
    for record in records:
        row = record["row"]
        if row["split"] != "train":
            raise ValueError("CNN fitting sampler requires frozen train records.")
        for label in (0, 1):
            indices = np.flatnonzero(record["window_y"] == label)
            seed_text = f"{GLOBAL_SEED}:phase3c_sample:{row['example_id']}:{label}"
            seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            take = np.sort(rng.choice(indices, min(24, len(indices)), replace=False))
            if len(take):
                patches.append(record["features"][take])
                labels.append(np.full(len(take), label, dtype=np.int8))
                groups.extend([row["group_id"]] * len(take))
    if not patches:
        raise ValueError("No pure training windows; do not weaken masks.")
    return np.concatenate(patches), np.concatenate(labels), np.asarray(groups)


def inner_groups(groups):
    ordered = sorted(
        set(groups),
        key=lambda g: hashlib.sha256(f"{GLOBAL_SEED}:phase3c_inner:{g}".encode()).hexdigest(),
    )
    count = max(1, int(np.ceil(len(ordered) * 0.20)))
    if count >= len(ordered):
        raise ValueError("Insufficient groups for disjoint inner validation.")
    return set(ordered[count:]), set(ordered[:count])


def extract_split(dataset, rows, destination, work, split):
    if split not in {"train", "dev"}:
        raise ValueError("Eval extraction is forbidden.")
    records = []
    begin = perf_counter()
    for group in sorted({r["group_id"] for r in rows}):
        members = [r for r in rows if r["group_id"] == group]
        keys = {
            r["example_id"]: "real" if r["example_class"] == "null_control" else r["example_id"]
            for r in members
        }
        wavs = {}
        for row in members:
            if row["split"] != split:
                raise ValueError("Source partition mismatch.")
            wav = read_input(dataset / row["output_filename"])
            if len(wav) != row["num_samples"] or not 0 < len(wav) <= 60 * ARCHIVE_SR:
                raise ValueError("CNN source duration differs from manifest.")
            wavs[keys[row["example_id"]]] = wav
        analysis = analysis_comparison(wavs, work)
        for row in members:
            starts, patches = extract_logmel(analysis[keys[row["example_id"]]])
            truth = guarded_labels(sample_labels(row["interval_labels"], row["num_samples"]), 3308)
            records.append(
                {
                    "row": row,
                    "starts": starts,
                    "features": patches,
                    "window_y": assign_window_labels(truth, starts, GEOMETRY.window_samples),
                }
            )
        print(f"CNN {split} features {group}", flush=True)
    # Keep the same canonical order as the inherited integrity checker.
    records.sort(
        key=lambda r: (r["row"]["group_id"], r["row"]["example_class"], r["row"]["example_id"])
    )
    save_features(destination, records)
    labels = np.concatenate([r["window_y"] for r in records])
    return records, {
        "elapsed_seconds": perf_counter() - begin,
        "recordings": len(records),
        "source_groups": len({r["row"]["group_id"] for r in records}),
        "windows": len(labels),
        "real": int(np.sum(labels == 0)),
        "synthetic": int(np.sum(labels == 1)),
        "ignore": int(np.sum(labels == -1)),
        "feature_bytes": sum(r["features"].nbytes for r in records),
    }


def cross_validate(records, assignments, root):
    x, y, groups = sampled_training(records)
    fit_info = []
    for fold in range(3):
        outer_fit = {g for g, f in assignments.items() if f != fold}
        held = set(assignments) - outer_fit
        inner_fit, inner_val = inner_groups(outer_fit)
        take_fit, take_val = np.isin(groups, list(inner_fit)), np.isin(groups, list(inner_val))
        start = perf_counter()
        _, stopping = train_cnn(
            x[take_fit],
            y[take_fit],
            groups[take_fit],
            GLOBAL_SEED + 100 * fold + 11,
            validation=(x[take_val], y[take_val], groups[take_val]),
        )
        selected_epoch = stopping["selected_epoch"]
        take = np.isin(groups, list(outer_fit))
        model, refit = train_cnn(
            x[take], y[take], groups[take], GLOBAL_SEED + 100 * fold + 21, epochs=selected_epoch
        )
        bundle = root / f"fold-{fold}" / "bundle"
        save_cnn(
            model,
            bundle,
            {"fixed_05": 0.5},
            fit_groups=sorted(outer_fit),
            held_out_groups=sorted(held),
            selected_epochs=selected_epoch,
            fold=fold,
        )
        restored, _ = load_cnn(bundle)
        for record in records:
            if record["row"]["group_id"] in held:
                if "margin" in record:
                    raise ValueError("Repeated CNN outer validation prediction.")
                record["margin"] = model.margins(record["features"])
                np.testing.assert_array_equal(
                    record["margin"], restored.margins(record["features"])
                )
                record["fold"] = fold
        info = {
            "fold": fold,
            "fit_groups": sorted(outer_fit),
            "held_out_groups": sorted(held),
            "inner_fit_groups": sorted(inner_fit),
            "inner_validation_groups": sorted(inner_val),
            "early_stopping": stopping,
            "outer_refit": refit,
            "sampled_fit_windows": int(take.sum()),
            "sampled_inner_fit_windows": int(take_fit.sum()),
            "sampled_inner_validation_windows": int(take_val.sum()),
            "elapsed_seconds": perf_counter() - start,
        }
        fit_info.append(info)
        write_json(root / f"fold-{fold}" / "fit.json", {**info, **FLAGS})
        print(f"CNN outer fold {fold}: selected {selected_epoch} epochs", flush=True)
    if any("margin" not in r for r in records):
        raise ValueError("Incomplete CNN train OOF coverage.")
    points = {
        "fixed_05": {"threshold": 0.5, "selection_scope": "predeclared_reference"},
        **control_operating_points(records, [r["margin"] for r in records], GEOMETRY, SCORE_GUARD),
    }
    save_predictions(root / "oof", records, GEOMETRY)
    metrics = evaluate_points(records, GEOMETRY, points, root / "oof", diagnostics=True)
    support_metrics(records, points, root / "oof/support_guard")
    write_json(root / "cross_validation.json", {"folds": fit_info, "points": points, **FLAGS})
    return points, fit_info, metrics, (x, y, groups)


def run_learned(dataset_dir, source_manifest, baseline_dir, diagnostics_dir, out_dir):
    dataset, source, baseline, diagnostics, target = (
        Path(p).resolve()
        for p in (dataset_dir, source_manifest, baseline_dir, diagnostics_dir, out_dir)
    )
    staging = target.with_name(target.name + ".staging")
    if target.exists() or staging.exists():
        raise FileExistsError("Fresh CNN experiment output required.")
    if any(
        p.is_relative_to(r) or r.is_relative_to(p)
        for p in (target, staging)
        for r in (dataset, source.parent, baseline, diagnostics)
    ):
        raise ValueError("CNN outputs must be separate from all immutable inputs.")
    selected, identity = review_rows(dataset, source, baseline)
    seal = read_json(diagnostics / "artifacts.sha256.json")
    checked_inventory(
        diagnostics,
        {
            **seal["files"],
            "artifacts.sha256.json": sha256_file(diagnostics / "artifacts.sha256.json"),
        },
    )
    summary = read_json(diagnostics / "oracle_summary.json")
    controlled = summary["variants"]["rms_active"]
    if (
        read_json(diagnostics / "plan.json") != phase3c_plan()
        or read_json(diagnostics / "provenance.json")["data"] != identity
        or read_json(diagnostics / "validation.json")["status"] != "passed"
        or not summary["learned_model_signal_decision"]["proceed"]
        or controlled["raw_auroc"] < 0.60
        or controlled["recall"] < 0.10
        or any(fpr > 0.05 for fpr in controlled["negative_core_fpr"].values())
        or read_json(diagnostics / "provenance.json")["code"]["source_sha256"][
            "src/experiments/paired_signal.py"
        ]
        != sha256_file(Path(__file__).with_name("paired_signal.py"))
    ):
        raise ValueError("Validated training-only signal criterion/identity not satisfied.")
    staging.mkdir(parents=True)
    start = perf_counter()
    assignments = grouped_folds([r["group_id"] for r in selected["train"]], GLOBAL_SEED, 3)
    write_json(staging / "plan.json", learned_plan())
    write_json(staging / "folds.json", {"assignments": assignments, **FLAGS})
    code = code_provenance()
    repo = Path(__file__).resolve().parents[2]
    code["source_sha256"].update(
        {
            p.relative_to(repo).as_posix(): sha256_file(p)
            for p in sorted((repo / "src").rglob("*.py"))
        }
    )
    write_json(
        staging / "provenance.json",
        {
            "code": code,
            "data": identity,
            "diagnostic_receipt_sha256": sha256_file(diagnostics / "artifacts.sha256.json"),
            **FLAGS,
        },
    )
    write_json(staging / "filter_support.json", filter_support_receipt(GEOMETRY))
    work = staging / "_analysis"
    work.mkdir()
    try:
        with threadpool_limits(limits=1):
            train, train_cost = extract_split(
                dataset, selected["train"], staging / "features/train", work, "train"
            )
            points, folds, oof_metrics, (x, y, groups) = cross_validate(
                train, assignments, staging / "cross_validation"
            )
            epochs = int(np.median([f["early_stopping"]["selected_epoch"] for f in folds]))
            before_fit = perf_counter()
            final, fit = train_cnn(x, y, groups, GLOBAL_SEED + 1000, epochs=epochs)
            final_seconds = perf_counter() - before_fit
            thresholds = {k: v["threshold"] for k, v in points.items()}
            save_cnn(
                final,
                staging / "bundle",
                thresholds,
                fit_groups=sorted(assignments),
                selected_epochs=epochs,
                loss_weighting=learned_plan()["loss_weights"],
                threshold_fit_split="train_group_oof_only",
                final_fit=fit,
            )
            # Freeze both numeric weights and operating decisions before any dev decode.
            lock = {
                "points": points,
                "final_epochs": epochs,
                "bundle_inventory": artifact_inventory(staging / "bundle"),
                "train_oof_primary": oof_metrics["fpr_05"],
                "dev_decode_started": False,
                "eval_used": False,
                **FLAGS,
            }
            write_json(staging / "selection_locked.json", lock)
            dev, dev_cost = extract_split(
                dataset, selected["dev"], staging / "features/dev", work, "dev"
            )
            restored, _ = load_cnn(staging / "bundle")
            for record in dev:
                record["margin"] = restored.margins(record["features"])
            save_predictions(staging / "dev", dev, GEOMETRY)
            evaluate_points(dev, GEOMETRY, points, staging / "dev", diagnostics=True)
            support_metrics(dev, points, staging / "dev/support_guard")
            # Cache reload and label-free single-file inference must reproduce saved outputs.
            cached = load_features(staging / "features/dev", selected["dev"])
            for a, b in zip(dev, cached, strict=True):
                np.testing.assert_array_equal(a["features"], b["features"])
            example = dev[0]
            inference_start = perf_counter()
            prediction = predict_cnn_file(
                staging / "bundle", dataset / example["row"]["output_filename"], work
            )
            np.testing.assert_array_equal(prediction["synthetic_margins"], example["margin"])
            write_json(
                staging / "label_free_inference_check.json",
                {
                    "example_id": example["row"]["example_id"],
                    "saved_margins_match_exactly": True,
                    "elapsed_seconds": perf_counter() - inference_start,
                    **prediction,
                },
            )
        work.rmdir()
        _, after = review_rows(dataset, source, baseline)
        if after != identity or artifact_inventory(staging / "bundle") != lock["bundle_inventory"]:
            raise ValueError("Inputs or locked model changed during dev benchmarking.")
        write_json(
            staging / "validation.json",
            {
                "status": "passed",
                "training_source_groups": len(assignments),
                "development_benchmark_groups": len({r["group_id"] for r in selected["dev"]}),
                "eval_waveforms_opened": 0,
                "eval_annotation_records_used": 0,
                "dev_used_for_model_selection": False,
                "all_outer_fold_numeric_loads_exact": True,
                "label_free_inference_exact": True,
                "train_cost": train_cost,
                "dev_cost": dev_cost,
                "final_fit_seconds": final_seconds,
                "total_elapsed_seconds": perf_counter() - start,
                "final_fit_sampled_windows": len(y),
                "final_epochs": epochs,
                "trainable_parameters": sum(getattr(final, k).size for k in FIELDS[:4]),
                **FLAGS,
            },
        )
        write_json(
            staging / "artifacts.sha256.json", {"files": artifact_inventory(staging), **FLAGS}
        )
        staging.rename(target)
        return {"output": str(target), "status": "passed", **FLAGS}
    except Exception as exc:
        write_json(staging / "failure.json", {"status": "failed", "error": str(exc), **FLAGS})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    for name in ("dataset-dir", "source-manifest", "baseline-dir", "diagnostics-dir", "out-dir"):
        run.add_argument(f"--{name}", required=True)
    predict = commands.add_parser("predict")
    for name in ("bundle", "wav", "work-dir"):
        predict.add_argument(f"--{name}", required=True)
    predict.add_argument("--operating-point", default="fpr_05")
    args = parser.parse_args(argv)
    if args.command == "run":
        print(
            run_learned(
                args.dataset_dir,
                args.source_manifest,
                args.baseline_dir,
                args.diagnostics_dir,
                args.out_dir,
            )
        )
    else:
        import json

        print(
            json.dumps(predict_cnn_file(args.bundle, args.wav, args.work_dir, args.operating_point))
        )


if __name__ == "__main__":
    main()
