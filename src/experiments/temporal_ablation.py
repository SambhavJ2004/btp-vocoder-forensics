"""Fixed train-group cross-validation followed by locked, descriptive dev checks."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from data.invariants import ARCHIVE_SR, GLOBAL_SEED
from detectors.forensic_pilot import sha256_file, write_json
from detectors.temporal_context import (
    CANDIDATES,
    PROTOCOL,
    ContextBundle,
    extract_context,
    load_context_bundle,
    save_context_bundle,
)
from detectors.temporal_features import (
    analysis_comparison,
    assign_window_labels,
    filter_support_receipt,
    guarded_labels,
    sample_labels,
    smooth_probabilities,
    window_starts,
)
from detectors.temporal_model import load_temporal_bundle
from detectors.temporal_operating import (
    common_mask_ranking,
    control_operating_points,
    evaluate_context,
    grouped_folds,
)
from experiments.partial_audio import FLAGS, artifact_inventory, read_input
from experiments.temporal_localization import code_provenance, fit_temporal_head
from experiments.temporal_review import (
    SCORE_GUARD,
    diagnostic_predictions,
    distribution,
    read_json,
    review_rows,
    saved_records,
)


def fixed_plan():
    return {
        "candidates": {
            name: {"window_ms": c.window_ms, "representation": c.representation}
            for name, c in CANDIDATES.items()
        },
        "hop_ms": 16,
        "C": 1.0,
        "smoothing": 3,
        "merge_ms": 0,
        "seed": GLOBAL_SEED,
        "folds": 3,
        "fold_assignment": "sha256_seed_group_sort_round_robin",
        "control_fpr_targets": [0.01, 0.05, 0.10],
        "threshold_constraint": "both_control_types_separately_sample_weighted_OOF",
        "selection": (
            "OOF_group_macro_sample_f1_at_5pct_then_max_control_fpr_then_shorter_context_then_name"
        ),
        "score_guard_samples": SCORE_GUARD,
        "pure_window_filter_guard_samples": 3308,
        "dev_selection": False,
        "eval_waveforms_opened": 0,
        "eval_annotation_records_used": 0,
        "fixed_dev_diagnostic_thresholds": [0.5, 0.75, 0.9, 0.95, 0.99],
        **FLAGS,
    }


def save_features(root, records):
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "records.json", {"rows": [r["row"] for r in records], **FLAGS})
    np.savez_compressed(
        root / "features.npz",
        features=np.vstack([r["features"] for r in records]),
        starts=np.concatenate([r["starts"] for r in records]),
        window_y=np.concatenate([r["window_y"] for r in records]),
        offsets=np.r_[0, np.cumsum([len(r["starts"]) for r in records])],
    )
    write_json(
        root / "checksums.json", {p.name: sha256_file(p) for p in root.iterdir() if p.is_file()}
    )


def load_features(root, expected_rows):
    checksums = read_json(root / "checksums.json")
    if set(checksums) != {"records.json", "features.npz"}:
        raise ValueError("Incomplete feature cache integrity record.")
    for name, digest in checksums.items():
        if sha256_file(root / name) != digest:
            raise ValueError("Feature cache integrity mismatch.")
    rows = read_json(root / "records.json")["rows"]
    if rows != expected_rows:
        raise ValueError("Feature cache source grouping differs from selected manifest rows.")
    records = []
    with np.load(root / "features.npz", allow_pickle=False) as arrays:
        # NpzFile inflates a member on every access; materialize it only once.
        state = {name: arrays[name] for name in ("features", "starts", "window_y", "offsets")}
        for i, row in enumerate(rows):
            begin, end = state["offsets"][i : i + 2]
            records.append(
                {
                    "row": row,
                    "starts": state["starts"][begin:end].copy(),
                    "features": state["features"][begin:end].copy(),
                    "window_y": state["window_y"][begin:end].copy(),
                    "score_labels": guarded_labels(
                        sample_labels(row["interval_labels"], row["num_samples"]), SCORE_GUARD
                    ),
                }
            )
    return records


def reuse_train_features(previous_run, source_snapshot, identity, rows, destination):
    """Copy verified completed train features into a fresh run, never resume.

    Match data/config/dependencies and archived driver bytes. The extraction
    and writer AST must be unchanged; a reader fix needs no repeat DSP pass.
    """
    import ast

    from detectors.forensic_pilot import runtime_versions

    prior, snapshot = Path(previous_run).resolve(), Path(source_snapshot).resolve()
    provenance = read_json(prior / "provenance.json")
    code = provenance["code"]
    repository = Path(__file__).resolve().parents[2]
    if (
        provenance["data"] != identity
        or code["environment"] != runtime_versions()
        or read_json(prior / "config.json")["candidates"]
        != {k: asdict(v) for k, v in CANDIDATES.items()}
        or sha256_file(snapshot) != code["source_sha256"]["src/experiments/temporal_ablation.py"]
    ):
        raise ValueError("Reusable training feature identity/config/source snapshot differs.")
    for name, digest in code["source_sha256"].items():
        if name not in {
            "src/experiments/temporal_ablation.py",
            "src/experiments/temporal_review.py",
        }:
            if sha256_file(repository / name) != digest:
                raise ValueError("Reusable feature processing dependency changed.")

    def functions(path):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(Path(path).read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef)
        }

    old, new = functions(snapshot), functions(__file__)
    if any(old[name] != new[name] for name in ("extract_split", "save_features")):
        raise ValueError("Reusable feature extraction/writing behavior changed.")
    counts, loads, hashes = {}, {}, {}
    for name, config in CANDIDATES.items():
        start = perf_counter()
        cache = prior / "features" / name / "train"
        before = artifact_inventory(cache)
        records = load_features(cache, rows)
        for record in records:
            row = record["row"]
            if row["split"] != "train" or record["features"].shape != (
                len(record["starts"]),
                len(config.feature_names),
            ):
                raise ValueError("Reusable features cross training scope or dimensions.")
            if not np.isfinite(record["features"]).all():
                raise ValueError("Reusable features contain nonfinite values.")
            np.testing.assert_array_equal(
                record["starts"], window_starts(row["num_samples"], config)
            )
            pure = guarded_labels(
                sample_labels(row["interval_labels"], row["num_samples"]), config.guard_samples
            )
            np.testing.assert_array_equal(
                record["window_y"],
                assign_window_labels(pure, record["starts"], config.window_samples),
            )
        save_features(destination / name / "train", records)
        if (
            artifact_inventory(cache) != before
            or artifact_inventory(destination / name / "train") != before
        ):
            raise ValueError("Reusable feature copy changed its sealed numeric bytes.")
        labels = np.concatenate([r["window_y"] for r in records])
        counts[name] = {
            "real": int(np.sum(labels == 0)),
            "synthetic": int(np.sum(labels == 1)),
            "ignore": int(np.sum(labels == -1)),
            "all": len(labels),
            "recordings": len(records),
            "source_groups": len({r["row"]["group_id"] for r in records}),
            "feature_dimensions": len(config.feature_names),
            "feature_matrix_bytes": sum(r["features"].nbytes for r in records),
        }
        loads[name] = perf_counter() - start
        hashes[name] = before
    return {
        "feature_extraction_seconds": None,
        "shared_decode_filter_pcm16_seconds": None,
        "window_counts": counts,
        "cache_reuse_seconds": loads,
        "cache_source": str(prior),
        "archived_source_sha256": sha256_file(snapshot),
        "verified_source_files": hashes,
        "cache_numeric_and_label_integrity": "exact",
    }


def extract_split(dataset, rows, root, work, split):
    records = {name: [] for name in CANDIDATES}
    seconds = {name: 0.0 for name in CANDIDATES}
    filtering_seconds = 0.0
    for group in sorted({r["group_id"] for r in rows}):
        members = [r for r in rows if r["group_id"] == group]
        keys = {
            r["example_id"]: "real" if r["example_class"] == "null_control" else r["example_id"]
            for r in members
        }
        before = perf_counter()
        wavs = {}
        for row in members:
            if row["split"] != split or split not in {"train", "dev"}:
                raise ValueError("Extraction is restricted to the selected train/dev scope.")
            wav = read_input(dataset / row["output_filename"])
            if len(wav) != row["num_samples"] or not 0 < len(wav) <= 60 * ARCHIVE_SR:
                raise ValueError("Context source duration/manifest mismatch.")
            wavs[keys[row["example_id"]]] = wav
        analysis = analysis_comparison(wavs, work)
        filtering_seconds += perf_counter() - before
        for row in members:
            truth = guarded_labels(sample_labels(row["interval_labels"], row["num_samples"]), 3308)
            for name, config in CANDIDATES.items():
                before = perf_counter()
                starts, features = extract_context(analysis[keys[row["example_id"]]], config)
                seconds[name] += perf_counter() - before
                records[name].append(
                    {
                        "row": row,
                        "starts": starts,
                        "features": features,
                        "window_y": assign_window_labels(truth, starts, config.window_samples),
                    }
                )
    counts = {}
    for name, values in records.items():
        labels = np.concatenate([r["window_y"] for r in values])
        counts[name] = {
            "real": int(np.sum(labels == 0)),
            "synthetic": int(np.sum(labels == 1)),
            "ignore": int(np.sum(labels == -1)),
            "all": len(labels),
            "recordings": len(values),
            "source_groups": len({r["row"]["group_id"] for r in values}),
            "feature_dimensions": len(CANDIDATES[name].feature_names),
            "feature_matrix_bytes": sum(r["features"].nbytes for r in values),
        }
        save_features(root / name / split, values)
    return {
        "shared_decode_filter_pcm16_seconds": filtering_seconds,
        "feature_extraction_seconds": seconds,
        "window_counts": counts,
    }


def bundle_for(head, config, thresholds, **extra):
    return ContextBundle(
        head,
        config,
        thresholds,
        {
            "protocol": PROTOCOL,
            "score_direction": "higher_is_synthetic",
            "probability_semantics": "uncalibrated_class_balanced_logistic_sigmoid",
            "feature_names": config.feature_names,
            "threshold_fit_split": "train_group_oof_only",
            "offline_lookahead": "zero_phase_filter_and_centered_smoothing_not_streaming_causal",
            **extra,
            **FLAGS,
        },
    )


def save_predictions(root, records, config):
    root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        root / "window_predictions.npz",
        record_example_ids=np.asarray([r["row"]["example_id"] for r in records]),
        record_source_groups=np.asarray([r["row"]["group_id"] for r in records]),
        record_offsets=np.r_[0, np.cumsum([len(r["starts"]) for r in records])],
        record_validation_fold=np.asarray([r.get("fold", -1) for r in records]),
        window_start_samples=np.concatenate([r["starts"] for r in records]),
        window_end_samples=np.concatenate([r["starts"] + config.window_samples for r in records]),
        window_labels=np.concatenate([r["window_y"] for r in records]),
        synthetic_margins=np.concatenate([r["margin"] for r in records]),
        uncalibrated_probabilities=np.concatenate([expit(r["margin"]) for r in records]),
        smoothed_probabilities=np.concatenate(
            [smooth_probabilities(expit(r["margin"]), 3) for r in records]
        ),
    )


def evaluate_points(records, config, points, root, *, diagnostics):
    probabilities = [expit(r["margin"]) for r in records]
    ranking = {
        "raw_margin": common_mask_ranking(
            records, [r["margin"] for r in records], config, SCORE_GUARD
        ),
        "smoothed_probability": common_mask_ranking(
            records, [smooth_probabilities(p, 3) for p in probabilities], config, SCORE_GUARD
        ),
        "mask_policy": "identical_fixed_4544_sample_guard_all_candidates_all_operating_points",
    }
    result = {}
    for label, point in points.items():
        metrics = evaluate_context(records, probabilities, config, point["threshold"], SCORE_GUARD)
        outputs = metrics.pop("recordings")
        metrics["common_mask_sample_ranking"] = ranking
        metrics["operating_point"] = point
        metrics["source_variability"] = {
            key: distribution(
                [
                    r[key] if r[key] is not None else 0
                    for r in metrics["sample_by_source_group"].values()
                ]
            )
            for key in ("precision", "recall", "f1", "false_positive_rate")
        }
        write_json(root / f"{label}.metrics.json", {**metrics, **FLAGS})
        write_json(root / f"{label}.recordings.json", {"recordings": outputs, **FLAGS})
        if diagnostics:
            diag_records = [
                {**r, "smoothed": smooth_probabilities(p, 3)}
                for r, p in zip(records, probabilities, strict=True)
            ]
            write_json(
                root / f"{label}.diagnostics.json",
                diagnostic_predictions(diag_records, config, point["threshold"]),
            )
        result[label] = metrics
    return result


def cross_validate(records, config, assignments, root):
    margins = [None] * len(records)
    folds, fit_seconds = [], 0.0
    for fold in sorted(set(assignments.values())):
        train = [r for r in records if assignments[r["row"]["group_id"]] != fold]
        indices = [i for i, r in enumerate(records) if assignments[r["row"]["group_id"]] == fold]
        train_groups = sorted({r["row"]["group_id"] for r in train})
        held_groups = sorted({records[i]["row"]["group_id"] for i in indices})
        if set(train_groups) & set(held_groups):
            raise ValueError("Cross-validation source leakage.")
        start = perf_counter()
        head, fit = fit_temporal_head(train, 1.0, GLOBAL_SEED)
        fit_seconds += perf_counter() - start
        bundle = bundle_for(
            head,
            config,
            {"fixed_05": 0.5},
            fold=fold,
            fit_groups=train_groups,
            held_out_groups=held_groups,
            scaler_fit_scope="fold_training_groups_only",
        )
        save_context_bundle(bundle, root / f"fold-{fold}" / "bundle")
        restored = load_context_bundle(root / f"fold-{fold}" / "bundle")
        for i in indices:
            if margins[i] is not None:
                raise ValueError("Repeated validation prediction.")
            margins[i] = head.decision_function(records[i]["features"])[:, 0]
            np.testing.assert_array_equal(margins[i], restored.margins(records[i]["features"]))
            records[i]["fold"] = fold
        folds.append(
            {"fold": fold, "fit_groups": train_groups, "held_out_groups": held_groups, "fit": fit}
        )
    if any(m is None for m in margins):
        raise ValueError("Incomplete source-group OOF coverage.")
    for record, margin in zip(records, margins, strict=True):
        record["margin"] = margin
    points = {
        "fixed_05": {"threshold": 0.5, "selection_scope": "predeclared_reference"},
        **control_operating_points(records, margins, config, SCORE_GUARD),
    }
    save_predictions(root / "oof", records, config)
    metrics = evaluate_points(records, config, points, root / "oof", diagnostics=False)
    fold_metrics = {}
    for fold in sorted(set(assignments.values())):
        subset = [r for r in records if r["fold"] == fold]
        fold_metrics[str(fold)] = {
            "raw_common_mask_ranking": common_mask_ranking(
                subset, [r["margin"] for r in subset], config, SCORE_GUARD
            ),
            "fpr_05": evaluate_context(
                subset,
                [expit(r["margin"]) for r in subset],
                config,
                points["fpr_05"]["threshold"],
                SCORE_GUARD,
                detailed=False,
            ),
        }
    write_json(
        root / "cross_validation.json",
        {
            "folds": folds,
            "fold_metrics": fold_metrics,
            "points": points,
            "fit_seconds": fit_seconds,
            **FLAGS,
        },
    )
    return points, metrics, {"fold_fit_seconds": fit_seconds, "folds": folds}


def run_ablation(
    dataset_dir,
    source_manifest,
    baseline_dir,
    audit_dir,
    out_dir,
    train_cache_run=None,
    cache_source_snapshot=None,
):
    dataset, source, baseline, audit = (
        Path(p).resolve() for p in (dataset_dir, source_manifest, baseline_dir, audit_dir)
    )
    target = Path(out_dir).resolve()
    staging = target.with_name(target.name + ".staging")
    for protected in (
        dataset,
        source.parent,
        baseline,
        audit,
        *(Path(p).resolve() for p in (train_cache_run, cache_source_snapshot) if p is not None),
    ):
        if any(
            p.is_relative_to(protected) or protected.is_relative_to(p) for p in (target, staging)
        ):
            raise ValueError("New ablation output must be separate from all input/audit artifacts.")
    if target.exists() or staging.exists():
        raise FileExistsError("Ablation outputs never overwrite/resume existing runs.")
    if (train_cache_run is None) != (cache_source_snapshot is None):
        raise ValueError("Training cache reuse requires its archived source snapshot.")
    correctness = read_json(audit / "correctness.json")
    plan = read_json(audit / "predeclared.json")
    if (
        correctness.get("status") != "passed"
        or correctness.get("correctness_defects_found") != []
        or correctness.get("original_numeric_model_reproduction") != "all_four_arrays_exact"
        or plan != fixed_plan()
    ):
        raise ValueError("Require the audited baseline and fixed predeclared Phase 3B plan.")
    selected, identity = review_rows(dataset, source, baseline)
    assignments = grouped_folds([r["group_id"] for r in selected["train"]], GLOBAL_SEED, 3)
    staging.mkdir(parents=True)
    work = staging / "_analysis"
    work.mkdir()
    code = code_provenance()
    repository = Path(__file__).resolve().parents[2]
    for path in sorted((repository / "src").rglob("*.py")):
        code["source_sha256"][path.relative_to(repository).as_posix()] = sha256_file(path)
    write_json(
        staging / "config.json",
        {
            "plan": plan,
            "candidates": {k: asdict(v) for k, v in CANDIDATES.items()},
            "fold_assignments": assignments,
            "filter_support": filter_support_receipt(),
            **FLAGS,
        },
    )
    write_json(
        staging / "provenance.json",
        {
            "code": code,
            "data": identity,
            "audit_correctness_sha256": sha256_file(audit / "correctness.json"),
            "predeclared_plan_sha256": sha256_file(audit / "predeclared.json"),
            "training_cache_reuse": None
            if train_cache_run is None
            else {
                "prior_run": str(Path(train_cache_run).resolve()),
                "prior_provenance_sha256": sha256_file(Path(train_cache_run) / "provenance.json"),
                "source_snapshot_sha256": sha256_file(cache_source_snapshot),
            },
            **FLAGS,
        },
    )
    try:
        with threadpool_limits(limits=1):
            print("Preparing train features; dev not yet extracted/scored", flush=True)
            costs = {
                "train": extract_split(
                    dataset, selected["train"], staging / "features", work, "train"
                )
                if train_cache_run is None
                else reuse_train_features(
                    train_cache_run,
                    cache_source_snapshot,
                    identity,
                    selected["train"],
                    staging / "features",
                )
            }
            comparisons, final_points, fit_info = {}, {}, {}
            for name, config in CANDIDATES.items():
                print(f"Train-only grouped validation {name}", flush=True)
                records = load_features(staging / "features" / name / "train", selected["train"])
                root = staging / "candidates" / name
                points, metrics, info = cross_validate(records, config, assignments, root)
                point = metrics["fpr_05"]
                comparisons[name] = {
                    "oof_group_macro_f1_at_05": point["source_group_macro_sample"]["f1"],
                    "max_oof_control_fpr": max(
                        r["sample_false_positive_rate"] for r in point["controls"].values()
                    ),
                    "window_auroc": point["window_micro"]["raw_auroc"],
                    "common_mask_raw_auroc": point["common_mask_sample_ranking"]["raw_margin"][
                        "auroc"
                    ],
                }
                final_points[name], fit_info[name] = points, info
            chosen = min(
                comparisons,
                key=lambda name: (
                    -comparisons[name]["oof_group_macro_f1_at_05"],
                    comparisons[name]["max_oof_control_fpr"],
                    CANDIDATES[name].window_ms,
                    name,
                ),
            )
            write_json(
                staging / "selection.json",
                {
                    "selected_candidate": chosen,
                    "comparisons": comparisons,
                    "points": final_points,
                    "dev_used_for_selection": False,
                    "eval_used": False,
                    "selection_completed_before_dev_extraction": True,
                    **FLAGS,
                },
            )
            selection_hash = sha256_file(staging / "selection.json")
            print(f"Selection locked from train OOF: {chosen}", flush=True)
            for name, config in CANDIDATES.items():
                records = load_features(staging / "features" / name / "train", selected["train"])
                start = perf_counter()
                head, fit = fit_temporal_head(records, 1.0, GLOBAL_SEED)
                fit_info[name]["full_train_fit_seconds"] = perf_counter() - start
                fit_info[name]["full_train_fit"] = fit
                if name == "A64":
                    old = load_temporal_bundle(baseline / "bundle")
                    for field in ("mean", "scale", "coefficients", "intercept"):
                        np.testing.assert_array_equal(
                            getattr(head, field), getattr(old.head, field)
                        )
                bundle = bundle_for(
                    head,
                    config,
                    {k: v["threshold"] for k, v in final_points[name].items()},
                    fit_groups=sorted(assignments),
                    selected_from_train_oof=name == chosen,
                    selection_sha256=selection_hash,
                )
                save_context_bundle(bundle, staging / "candidates" / name / "bundle")
                restored = load_context_bundle(staging / "candidates" / name / "bundle")
                for record in records:
                    np.testing.assert_array_equal(
                        bundle.margins(record["features"]), restored.margins(record["features"])
                    )
            print("Extracting descriptive dev features after selection lock", flush=True)
            costs["dev"] = extract_split(
                dataset, selected["dev"], staging / "features", work, "dev"
            )
            summary = {}
            for name, config in CANDIDATES.items():
                print(f"Locked dev reporting {name}", flush=True)
                records = load_features(staging / "features" / name / "dev", selected["dev"])
                bundle = load_context_bundle(staging / "candidates" / name / "bundle")
                start = perf_counter()
                for record in records:
                    record["margin"] = bundle.margins(record["features"])
                fit_info[name]["dev_linear_prediction_seconds"] = perf_counter() - start
                if name == "A64":
                    old = {
                        r["row"]["example_id"]: r for r in saved_records(baseline, selected, "dev")
                    }
                    for record in records:
                        np.testing.assert_array_equal(
                            record["margin"], old[record["row"]["example_id"]]["margin"]
                        )
                root = staging / "candidates" / name / "dev"
                save_predictions(root, records, config)
                metrics = evaluate_points(
                    records, config, final_points[name], root, diagnostics=True
                )
                if name == "A64":
                    assert (
                        metrics["fixed_05"]["source_group_macro_f1"]
                        == read_json(baseline / "metrics.json")["metrics"]["dev"][
                            "source_group_macro_f1"
                        ]
                    )
                summary[name] = {
                    point: {
                        k: v
                        for k, v in result.items()
                        if k not in {"sample_by_source_group", "window_by_source_group"}
                    }
                    for point, result in metrics.items()
                }
            if sha256_file(staging / "selection.json") != selection_hash:
                raise ValueError("Selection changed during descriptive dev checks.")
        work.rmdir()
        _, after = review_rows(dataset, source, baseline)
        if after != identity:
            raise ValueError("Consumed input identity changed during ablation.")
        write_json(
            staging / "metrics.json", {"selected_candidate": chosen, "dev": summary, **FLAGS}
        )
        write_json(
            staging / "costs.json",
            {
                "extraction": costs,
                "fitting": fit_info,
                "timings_are_observations_not_deterministic_artifacts": True,
                **FLAGS,
            },
        )
        write_json(
            staging / "validation.json",
            {
                "status": "passed",
                "phase3a_model_and_dev_margins_reproduced": "exact",
                "bundle_roundtrips": "exact",
                "selection_sha256": selection_hash,
                "train_groups": len(assignments),
                "fold_group_counts": dict(Counter(assignments.values())),
                "dev_used_for_selection": False,
                "eval_waveforms_opened": 0,
                "eval_annotation_records_used": 0,
                "consumed_input_hashes_unchanged": True,
                **FLAGS,
            },
        )
        write_json(
            staging / "artifacts.sha256.json", {"files": artifact_inventory(staging), **FLAGS}
        )
        staging.rename(target)
        return {"selected_candidate": chosen, "output": str(target), **FLAGS}
    except Exception as exc:
        write_json(staging / "failure.json", {"status": "failed", "error": str(exc), **FLAGS})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-dir", "source-manifest", "baseline-dir", "audit-dir", "out-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--train-cache-run")
    parser.add_argument("--cache-source-snapshot")
    args = parser.parse_args(argv)
    print(
        run_ablation(
            args.dataset_dir,
            args.source_manifest,
            args.baseline_dir,
            args.audit_dir,
            args.out_dir,
            args.train_cache_run,
            args.cache_source_snapshot,
        )
    )


if __name__ == "__main__":
    main()
