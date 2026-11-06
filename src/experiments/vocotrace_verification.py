"""Train-group source verification benchmark; no frozen eval waveform access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from data.invariants import ARCHIVE_TIER, GLOBAL_SEED, PRIMARY_ANALYSIS
from data.manifest import (
    analysis_frame,
    read_manifest,
    require_primary,
    select_tier,
    validate_manifest,
)
from detectors.forensic_features import FeatureConfig
from detectors.forensic_pilot import KNOWN_SOURCES, LinearHead, load_bundle, sha256_file, write_json
from detectors.source_verification import (
    FLAGS,
    LEVELS,
    PROTOCOL,
    SCORES,
    SourceVerifier,
    confidence_cutoff,
    genuine_gate,
    load_verifier,
    predict_file,
    save_verifier,
)
from detectors.verification_metrics import matrix_metrics, score_metrics
from experiments.forensic_pilot import (
    code_provenance,
    extract_dataset_features,
    fit_head,
    validate_pilot_frame,
)
from experiments.partial_audio import artifact_inventory

CASES = {
    "closed3": KNOWN_SOURCES,
    **{f"without_{u}": tuple(k for k in KNOWN_SOURCES if k != u) for u in KNOWN_SOURCES},
}


def fixed_plan():
    return {
        "protocol": PROTOCOL,
        "features": asdict(FeatureConfig()),
        "cases": CASES,
        "seed": GLOBAL_SEED,
        "C_binary_and_attribution": 1.0,
        "covariance": "0.9*pooled_within_class+0.1*I",
        "scores": SCORES,
        "energy_temperature": 1.0,
        "confidence_reject_budgets": LEVELS,
        "genuine_calibration_false_gate_budget": 0.05,
        "source_folds": "3_SHA256_seed_vocotrace4b_round_robin",
        "outer_calibration": "hash_order_remaining_train_every_third;rest_fit",
        "final_calibration": "source_fold_2;final_fit_folds_0_1;no_refit_after_calibration",
        "selection": (
            "retain95_only_max_correct_known_fraction_then_accuracy_"
            "then_lower_genuine_FAR_then_score_order"
        ),
        "selection_constraint": "OOF_genuine_false_attribution_le_0.05_and_nonzero_known_coverage",
        "fallback": "reject_all_source_labels_if_no_feasible_score",
        "withheld_scores_before_lock": False,
        "dev_scores_before_lock": False,
        "normalization": "binary_fit_genuine_and_known;attribution_fit_known_synthetic_only",
        "confidence_calibration": (
            "known_calibration_only;no_withheld_class;not_probability_calibration"
        ),
        "bootstrap": "2000_source_group_resamples_conditional_fixed_scores_and_thresholds",
        "eval_used": False,
        **FLAGS,
    }


def ordered(groups, namespace):
    return sorted(
        set(groups),
        key=lambda g: hashlib.sha256(
            f"{GLOBAL_SEED}:vocotrace4b:{namespace}:{g}".encode()
        ).hexdigest(),
    )


def group_folds(groups):
    values = ordered(groups, "outer")
    if len(values) < 9:
        raise ValueError("At least nine training source groups required for nested calibration.")
    return {g: i % 3 for i, g in enumerate(values)}


def outer_partition(assignments, fold):
    held = sorted(g for g, f in assignments.items() if f == fold)
    remaining = ordered([g for g, f in assignments.items() if f != fold], f"cal-{fold}")
    cal = sorted(remaining[::3])
    fit = sorted(set(remaining) - set(cal))
    if set(fit) & set(cal) or set(fit) & set(held) or set(cal) & set(held):
        raise ValueError("Cross-validation source leakage.")
    return fit, cal, held


def fit_system(frame, x, calibration, cal_x, known, *, held_out=None):
    known = tuple(known)
    for data in (frame, calibration):
        if (
            set(data.split) != {"train"}
            or not set(data.condition) <= {"real", *known}
            or data.duplicated(["utt_id", "condition"]).any()
            or any(set(g.condition) != {"real", *known} for _, g in data.groupby("utt_id"))
        ):
            raise ValueError("Only complete genuine/known training groups may fit/calibrate.")
    if (
        set(frame.utt_id) & set(calibration.utt_id)
        or held_out in known
        or len(set(frame.utt_id)) < 2
        or calibration.empty
    ):
        raise ValueError("Invalid fit/calibration/withheld partition.")
    if x.shape != (len(frame), 64) or cal_x.shape != (len(calibration), 64):
        raise ValueError("Feature/row geometry mismatch.")
    y = (frame.condition.to_numpy() == "real").astype(int)
    syn = y == 0
    ay = np.array([known.index(c) for c in frame.condition.to_numpy()[syn]])
    binary = LinearHead.from_pipeline(fit_head(x, y, 1.0, GLOBAL_SEED, balanced=True))
    attribution = LinearHead.from_pipeline(fit_head(x[syn], ay, 1.0, GLOBAL_SEED, balanced=False))
    z = (x[syn] - attribution.mean) / attribution.scale
    centers = np.vstack([z[ay == i].mean(axis=0) for i in range(len(known))])
    residuals = z - centers[ay]
    covariance = residuals.T @ residuals / (len(z) - len(known))
    precision = np.linalg.inv(0.9 * covariance + 0.1 * np.eye(64))
    meta = {
        "format_version": 1,
        "protocol": PROTOCOL,
        "features": asdict(FeatureConfig()),
        "known_classes": list(known),
        "held_out": held_out,
        "selected_score": None,
        "fit_groups": sorted(set(frame.utt_id)),
        "calibration_groups": sorted(set(calibration.utt_id)),
        "fit_rows": len(frame),
        "attribution_fit_rows": int(syn.sum()),
        "fit_conditions": sorted(set(frame.condition)),
        "calibration_conditions": sorted(set(calibration.condition)),
        "binary_threshold": 0.0,
        "confidence_thresholds": {s: {level: 0.0 for level in LEVELS} for s in SCORES},
        "confidence_semantics": "uncalibrated_higher_is_known_like",
        **FLAGS,
    }
    bundle = SourceVerifier(binary, attribution, centers, precision, meta)
    raw = bundle.raw_scores(cal_x)
    real = calibration.condition.to_numpy() == "real"
    threshold = genuine_gate(raw["real_margin"][real], 0.05)
    meta["binary_threshold"] = threshold
    meta["confidence_thresholds"] = {
        s: {level: confidence_cutoff(raw[s][~real], budget) for level, budget in LEVELS.items()}
        for s in SCORES
    }
    meta["calibration_evidence"] = {
        "genuine_rows": int(real.sum()),
        "known_rows": int((~real).sum()),
        "genuine_gate_false_rate": float(np.mean(raw["real_margin"][real] <= threshold)),
        "known_confidence_rejection": {
            s: {
                level: float(np.mean(raw[s][~real] < meta["confidence_thresholds"][s][level]))
                for level in LEVELS
            }
            for s in SCORES
        },
    }
    bundle.validate()
    return bundle


def choose_score(metrics):
    feasible = [
        s
        for s in SCORES
        if metrics[s]["genuine_false_attribution_rate"] <= 0.05 and metrics[s]["known_coverage"] > 0
    ]
    return (
        min(
            feasible,
            key=lambda s: (
                -metrics[s]["known_correct_source_fraction"],
                -metrics[s]["known_accepted_accuracy"],
                metrics[s]["genuine_false_attribution_rate"],
                SCORES.index(s),
            ),
        )
        if feasible
        else "reject_all"
    )


def baseline_audit(root):
    root = Path(root)
    expected = json.loads((root / "artifacts.sha256.json").read_text())
    if artifact_inventory(root) != expected:
        raise ValueError("Historical pilot artifacts changed.")
    bundle = load_bundle(root / "bundle")
    metrics = json.loads((root / "metrics.json").read_text())
    cm = metrics["attribution"]["confusion_matrix"]
    if not np.isclose(
        matrix_metrics(cm)["macro_f1"], metrics["attribution"]["macro_f1"], atol=1e-15, rtol=0
    ):
        raise ValueError("Historical aggregate attribution F1 inconsistent.")
    return bundle, {
        "historical_aggregate_only_no_new_eval_scoring": True,
        "metrics": metrics,
        "confusion_arithmetic_reproduced": True,
        "model_sha256": sha256_file(root / "bundle/models.npz"),
        "prior_inventory_sha256": sha256_file(root / "artifacts.sha256.json"),
        **FLAGS,
    }


def admitted_data(manifest, baseline):
    historical, audit = baseline_audit(baseline)
    full = read_manifest(manifest)
    validate_manifest(full)
    frame = (
        analysis_frame(
            require_primary(select_tier(full, ARCHIVE_TIER), why=PROTOCOL),
            PRIMARY_ANALYSIS,
            why=PROTOCOL,
        )
        .sort_values(["utt_id", "condition"])
        .reset_index(drop=True)
    )
    validate_pilot_frame(frame)
    old = historical.metadata["dataset_identity"]
    groups = frame[["utt_id", "speaker", "split"]].drop_duplicates().to_dict("records")
    split_hash = hashlib.sha256(
        json.dumps(groups, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        sha256_file(manifest) != old["manifest_sha256"]
        or split_hash != old["split_sha256"]
        or groups != old["utterance_groups"]
    ):
        raise ValueError("Historical source manifest/frozen groups differ.")
    for name, digest in old["source_artifacts"].items():
        if sha256_file(Path(manifest).parent / name) != digest:
            raise ValueError("Historical source provenance artifact changed.")
    selected = frame[frame.split.isin(["train", "dev"])].copy().reset_index(drop=True)
    selected["audio_sha256"] = [sha256_file(p) for p in selected.path]
    if (selected.groupby("audio_sha256").split.nunique() > 1).any():
        raise ValueError("Train/dev byte-copy leakage.")
    with (Path(baseline) / "feature_rows.csv").open(encoding="utf-8", newline="") as stream:
        old_rows = list(islice(csv.DictReader(stream), len(selected)))
    current = selected[["utt_id", "condition", "split", "tier", "audio_sha256"]].to_dict("records")
    if old_rows != current:
        raise ValueError("Consumed train/dev rows or audio hashes differ from historical pilot.")
    return (
        selected,
        historical,
        audit,
        {
            "manifest_sha256": sha256_file(manifest),
            "split_sha256": split_hash,
            "train_dev_audio": current,
            "all_frozen_group_metadata": groups,
            "source_artifacts": old["source_artifacts"],
            "legacy_generation_provenance_unresolved": True,
            "fresh_signal_gates_rerun": False,
            "eval_audio_hashes_refreshed": False,
            **FLAGS,
        },
    )


def validate_feature_cache(cache, identity):
    receipt = json.loads((cache / "artifacts.sha256.json").read_text())
    if artifact_inventory(cache) != receipt["files"]:
        raise ValueError("Feature-cache run integrity mismatch.")
    for p in [cache, *cache.rglob("*")]:
        if (
            p.is_symlink()
            or getattr(p, "is_junction", lambda: False)()
            or (p.is_file() and p.stat().st_nlink != 1)
        ):
            raise ValueError("Unsafe feature-cache path.")
    provenance = json.loads((cache / "provenance.json").read_text())
    plan = json.loads((cache / "plan.json").read_text())
    validation = json.loads((cache / "validation.json").read_text())
    if (
        provenance["data"] != identity
        or plan["features"] != asdict(FeatureConfig())
        or validation["status"] != "passed"
    ):
        raise ValueError("Feature-cache data/configuration/admission differs.")
    repo = Path(__file__).resolve().parents[2]
    for dependency in (
        "data/audio_io.py",
        "data/invariants.py",
        "data/preprocess.py",
        "detectors/base.py",
        "detectors/forensic_features.py",
        "experiments/forensic_pilot.py",
    ):
        name = "src/" + dependency
        if sha256_file(repo / name) != provenance["code"]["source_sha256"][name]:
            raise ValueError("Cached feature processing dependency changed.")
    return {
        "run": str(cache),
        "inventory_sha256": sha256_file(cache / "artifacts.sha256.json"),
        "reused_only_features_and_row_identity": True,
        "dev_array_loaded_after_lock": True,
    }


def extract_split(frame, split, root, feature_cache=None):
    if split not in ("train", "dev") or set(frame.split) != {split}:
        raise ValueError("Feature extraction may consume only the declared train/dev partition.")
    if feature_cache is not None:
        columns = ["utt_id", "condition", "split", "audio_sha256"]
        rows = pd.read_csv(feature_cache / f"{split}_rows.csv", dtype=str)
        if rows.to_dict("records") != frame[columns].astype(str).to_dict("records"):
            raise ValueError("Cached feature row identities differ.")
        with np.load(feature_cache / f"{split}_features.npz", allow_pickle=False) as archive:
            if archive.files != ["features"]:
                raise ValueError("Unexpected feature-cache numeric schema.")
            x = archive["features"]
        if x.shape != (len(frame), 64) or not np.isfinite(x).all():
            raise ValueError("Invalid cached feature geometry.")
        for name in (f"{split}_features.npz", f"{split}_rows.csv"):
            shutil.copyfile(feature_cache / name, root / name)
        return x
    x = extract_dataset_features(frame, FeatureConfig(), root)
    np.savez_compressed(root / f"{split}_features.npz", features=x)
    frame[["utt_id", "condition", "split", "audio_sha256"]].to_csv(
        root / f"{split}_rows.csv", index=False
    )
    return x


def policy(bundle, raw, name, level):
    if name == "unconditional":
        return raw["source_index"], np.repeat("known_source", len(raw["source_index"]))
    if name == "binary_only":
        accepted = raw["real_margin"] <= bundle.metadata["binary_threshold"]
        return np.where(accepted, raw["source_index"], -1), np.where(
            accepted, "known_source", "no_supported_synthetic_source"
        )
    return bundle.decisions(raw, name, level)


def collect(frame, scored):
    """Each source-condition receives exactly one out-of-fold result."""
    indices = np.concatenate([i for i, _, _ in scored])
    if len(indices) != len(set(indices)):
        raise ValueError("Repeated OOF source-condition.")
    order = np.argsort(indices)
    data = frame.iloc[indices[order]].reset_index(drop=True)
    raw = {key: np.concatenate([r[key] for _, _, r in scored])[order] for key in scored[0][2]}
    return data, raw, indices[order]


def evaluate_scored(frame, scored, known, root, *, selected):
    root.mkdir(parents=True, exist_ok=False)
    policies = [(s, level) for s in SCORES for level in LEVELS] + [
        (s, "retain95") for s in ("unconditional", "binary_only", "reject_all")
    ]
    data, raw, indices = collect(frame, scored)
    predictions = {}
    for name, level in policies:
        parts = [policy(b, r, name, level) for _, b, r in scored]
        order = np.argsort(np.concatenate([i for i, _, _ in scored]))
        predicted = np.concatenate([p for p, _ in parts])[order]
        states = np.concatenate([s for _, s in parts])[order]
        metrics = score_metrics(
            data.condition.to_numpy(), data.utt_id.to_numpy(), raw, predicted, states, known
        )
        metrics.update(
            {"score": name, "level": level, "selected": name == selected and level == "retain95"}
        )
        write_json(root / f"{name}.{level}.metrics.json", metrics)
        predictions[f"{name}.{level}.source"] = [known[p] if p >= 0 else "" for p in predicted]
        predictions[f"{name}.{level}.decision"] = states
    output = data[["utt_id", "condition", "split", "audio_sha256"]].copy()
    output["row_index"] = indices
    fold_by_index = {
        int(i): b.metadata.get("outer_fold", -1) for inds, b, _ in scored for i in inds
    }
    output["validation_fold"] = [fold_by_index[int(i)] for i in indices]
    output["source_role"] = [
        "genuine" if c == "real" else "known" if c in known else "withheld" for c in data.condition
    ]
    output["real_margin"] = raw["real_margin"]
    output["unconditional_source"] = [known[p] for p in raw["source_index"]]
    for s in SCORES:
        output[s] = raw[s]
    for j, c in enumerate(known):
        output[f"source_logit_{c}"] = raw["source_logits"][:, j]
    for key, value in predictions.items():
        output[key] = value
    output["citable"] = False
    output.to_csv(root / "scores_predictions.csv", index=False)


def run(manifest, baseline_dir, out_dir, feature_cache=None):
    manifest, baseline, target = (Path(p).resolve() for p in (manifest, baseline_dir, out_dir))
    staging = target.with_name(target.name + ".staging")
    cache = Path(feature_cache).resolve() if feature_cache is not None else None
    if target.exists() or staging.exists():
        raise FileExistsError("Fresh verification output/staging required.")
    if any(
        p.is_relative_to(r) or r.is_relative_to(p)
        for p in (target, staging)
        for r in (manifest.parent, baseline, *((cache,) if cache is not None else ()))
    ):
        raise ValueError("Verification outputs must be separate from inputs.")
    begin = perf_counter()
    with threadpool_limits(limits=1):
        frame, historical, audit, identity = admitted_data(manifest, baseline)
        train = frame[frame.split == "train"].reset_index(drop=True)
        dev = frame[frame.split == "dev"].reset_index(drop=True)
        assignments = group_folds(train.utt_id)
        cache_evidence = validate_feature_cache(cache, identity) if cache is not None else None
        staging.mkdir(parents=True)
        write_json(staging / "plan.json", fixed_plan())
        write_json(
            staging / "provenance.json",
            {"code": code_provenance(), "data": identity, "feature_cache": cache_evidence, **FLAGS},
        )
        write_json(
            staging / "folds.json",
            {
                "assignments": assignments,
                "outer": {
                    str(f): dict(
                        zip(
                            ("fit", "calibration", "held"),
                            outer_partition(assignments, f),
                            strict=True,
                        )
                    )
                    for f in range(3)
                },
                **FLAGS,
            },
        )
        try:
            x = extract_split(train, "train", staging, cache)
            for head, mask in (
                (historical.binary, np.ones(len(train), dtype=bool)),
                (historical.attribution, train.condition.to_numpy() != "real"),
            ):
                np.testing.assert_allclose(head.mean, x[mask].mean(axis=0), atol=1e-12, rtol=0)
                scale = x[mask].std(axis=0)
                scale[scale == 0] = 1
                np.testing.assert_allclose(head.scale, scale, atol=1e-12, rtol=0)
            audit["current_cold_train_features_reproduce_original_train_scalers"] = True
            write_json(staging / "historical_baseline_audit.json", audit)
            all_scored, final, choices, comparisons = {}, {}, {}, {}
            for case, known in CASES.items():
                root = staging / case
                scored = []
                withheld = next((u for u in KNOWN_SOURCES if u not in known), None)
                permitted = train.condition.isin(["real", *known]).to_numpy()
                for fold in range(3):
                    fit, cal, held = outer_partition(assignments, fold)
                    fm = permitted & train.utt_id.isin(fit).to_numpy()
                    cm = permitted & train.utt_id.isin(cal).to_numpy()
                    hm = permitted & train.utt_id.isin(held).to_numpy()
                    bundle = fit_system(
                        train[fm], x[fm], train[cm], x[cm], known, held_out=withheld
                    )
                    bundle.metadata.update({"outer_fold": fold, "held_groups": held})
                    save_verifier(bundle, root / f"fold-{fold}/bundle")
                    restored = load_verifier(root / f"fold-{fold}/bundle")
                    scored.append((np.flatnonzero(hm), restored, restored.raw_scores(x[hm])))
                data, raw, _ = collect(train, scored)
                candidate_metrics = {}
                for s in SCORES:
                    pp = [policy(b, r, s, "retain95") for _, b, r in scored]
                    order = np.argsort(np.concatenate([i for i, _, _ in scored]))
                    predicted = np.concatenate([p for p, _ in pp])[order]
                    states = np.concatenate([p for _, p in pp])[order]
                    candidate_metrics[s] = score_metrics(
                        data.condition.to_numpy(),
                        data.utt_id.to_numpy(),
                        raw,
                        predicted,
                        states,
                        known,
                        bootstrap=0,
                    )
                choice = choose_score(candidate_metrics)
                choices[case] = choice
                comparisons[case] = {
                    "selected_score": choice,
                    "known_genuine_OOF_only": candidate_metrics,
                    "withheld_rows_used_for_selection": 0,
                    **FLAGS,
                }
                fit = sorted(g for g, f in assignments.items() if f != 2)
                cal = sorted(g for g, f in assignments.items() if f == 2)
                fm = permitted & train.utt_id.isin(fit).to_numpy()
                cm = permitted & train.utt_id.isin(cal).to_numpy()
                bundle = fit_system(train[fm], x[fm], train[cm], x[cm], known, held_out=withheld)
                bundle.metadata["selected_score"] = choice
                save_verifier(bundle, root / "bundle")
                final[case] = load_verifier(root / "bundle")
                all_scored[case] = scored
                print(f"Locked train-only score selection for {case}: {choice}", flush=True)
            write_json(staging / "selection_comparison.json", comparisons)
            lock = {
                "choices": choices,
                "bundles": {c: artifact_inventory(staging / c / "bundle") for c in CASES},
                "withheld_or_dev_scores_used_for_selection": False,
                **FLAGS,
            }
            write_json(staging / "selection_locked.json", lock)
            # Unknown-class OOF predictions are produced only after every choice/model is locked.
            for case, known in CASES.items():
                scored = all_scored[case]
                if len(known) == 2:
                    for fold in range(3):
                        held = outer_partition(assignments, fold)[2]
                        um = (
                            ~train.condition.isin(["real", *known]).to_numpy()
                            & train.utt_id.isin(held).to_numpy()
                        )
                        bundle = load_verifier(staging / case / f"fold-{fold}/bundle")
                        scored.append((np.flatnonzero(um), bundle, bundle.raw_scores(x[um])))
                evaluate_scored(
                    train, scored, known, staging / case / "oof", selected=choices[case]
                )
            dev_x = extract_split(dev, "dev", staging, cache)
            for case, known in CASES.items():
                bundle = final[case]
                evaluate_scored(
                    dev,
                    [(np.arange(len(dev)), bundle, bundle.raw_scores(dev_x))],
                    known,
                    staging / case / "dev",
                    selected=choices[case],
                )
            # Reproduce only the historical model's previously selected dev benchmark, not eval.
            old_logits = historical.attribution.decision_function(dev_x)
            old_real = historical.binary.decision_function(dev_x)[:, 0]
            from sklearn.metrics import f1_score, roc_auc_score

            from detectors.eer import compute_eer

            dy = dev.condition.to_numpy() == "real"
            truth = np.array([KNOWN_SOURCES.index(c) for c in dev.condition.to_numpy()[~dy]])
            write_json(
                staging / "historical_dev_replay.json",
                {
                    "binary_auroc": float(roc_auc_score(dy, old_real)),
                    "binary_eer": compute_eer(old_real[dy], old_real[~dy])[0],
                    "oracle_synthetic_macro_f1": float(
                        f1_score(truth, old_logits[~dy].argmax(axis=1), average="macro")
                    ),
                    "previously_selected_exposed_dev_not_new_selection": True,
                    **FLAGS,
                },
            )
            example = train.iloc[0]
            prediction = predict_file(staging / "closed3/bundle", example.path, work_dir=staging)
            expected = final["closed3"].predict_features(x[0])
            if prediction != expected:
                raise ValueError("Single-input inference differs from cached feature prediction.")
            write_json(staging / "label_free_inference.json", prediction)
            for case in CASES:
                if artifact_inventory(staging / case / "bundle") != lock["bundles"][case]:
                    raise ValueError("Locked model changed during benchmarking.")
            if artifact_inventory(baseline) != json.loads(
                (baseline / "artifacts.sha256.json").read_text()
            ):
                raise ValueError("Historical pilot changed.")
            for row in frame.itertuples():
                if sha256_file(row.path) != row.audio_sha256:
                    raise ValueError("Consumed audio changed.")
            write_json(
                staging / "validation.json",
                {
                    "status": "passed",
                    "train_groups": len(assignments),
                    "dev_groups": int(dev.utt_id.nunique()),
                    "eval_audio_opened": 0,
                    "eval_features_used": 0,
                    "research_logistic_fits": 32,
                    "selection_before_withheld_and_dev_scoring": True,
                    "choices": choices,
                    "genuine_authenticity_claim_supported": False,
                    "general_open_set_claim_supported": False,
                    "elapsed_seconds": perf_counter() - begin,
                    **FLAGS,
                },
            )
            write_json(
                staging / "artifacts.sha256.json", {"files": artifact_inventory(staging), **FLAGS}
            )
            staging.rename(target)
            return {"output": str(target), "choices": choices, **FLAGS}
        except Exception as exc:
            write_json(staging / "failure.json", {"error": str(exc), "status": "failed", **FLAGS})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--feature-cache")
    args = parser.parse_args(argv)
    print(run(args.manifest, args.baseline_dir, args.out_dir, args.feature_cache))


if __name__ == "__main__":
    main()
