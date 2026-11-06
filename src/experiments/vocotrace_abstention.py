"""Read-only 4B decision replay and predetermined train-fold robustness diagnostics.

No fitting, threshold selection, eval reads or historical artifact writes occur here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from data.audio_io import read_processed, write_audio
from data.invariants import ARCHIVE_SR, GLOBAL_SEED, ZEROSHOT_SR, check_band_limit
from data.manifest import read_manifest
from data.preprocess import band_limit_comparison_set, resample_once
from detectors.forensic_features import (
    FeatureConfig,
    extract_comparison_features,
    read_archive_input,
)
from detectors.forensic_pilot import KNOWN_SOURCES, runtime_versions, sha256_file, write_json
from detectors.source_verification import FLAGS, LEVELS, SCORES, load_verifier
from detectors.verification_metrics import score_metrics
from experiments.partial_audio import artifact_inventory
from experiments.vocotrace_verification import CASES, policy

CELLS = ("clean", "gain_minus6", "gain_minus12", "lowpass6000", "lowpass4000", "first1s", "first2s")
POLICIES = [(s, level) for s in SCORES for level in LEVELS] + [
    (s, "retain95") for s in ("unconditional", "binary_only", "reject_all")
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_history(root):
    root = Path(root)
    for p in [root, *root.rglob("*")]:
        if (
            p.is_symlink()
            or getattr(p, "is_junction", lambda: False)()
            or (p.is_file() and p.stat().st_nlink != 1)
        ):
            raise ValueError("Unsafe historical artifact path.")
    if artifact_inventory(root) != read_json(root / "artifacts.sha256.json")["files"]:
        raise ValueError("Historical run integrity mismatch.")
    if read_json(root / "validation.json")["status"] != "passed":
        raise ValueError("Historical run did not pass validation.")


def predetermined_groups(assignments, per_fold=4):
    if set(assignments.values()) != {0, 1, 2}:
        raise ValueError("Expected the frozen three training folds.")
    selected = {}
    for fold in range(3):
        groups = sorted(
            (g for g, f in assignments.items() if f == fold),
            key=lambda g: hashlib.sha256(
                f"{GLOBAL_SEED}:vocotrace4c:robustness:{g}".encode()
            ).hexdigest(),
        )
        if len(groups) < per_fold:
            raise ValueError("Insufficient training sources; do not fabricate fold coverage.")
        selected.update({g: fold for g in groups[:per_fold]})
    return selected


def fixed_plan(root):
    assignments = read_json(root / "folds.json")["assignments"]
    return {
        "protocol": "vocotrace_phase4c_abstention_v1",
        "historical_run": str(root),
        "historical_inventory_sha256": sha256_file(root / "artifacts.sha256.json"),
        "selected_train_groups": predetermined_groups(assignments),
        "cells": list(CELLS),
        "gains_db": [-6, -12],
        "filter_stopbands_hz": [6000, 4000],
        "duration_support_samples": [ARCHIVE_SR, 2 * ARCHIVE_SR],
        "duration_policy": "first_min_available_support_then_legacy_repeat_to_89027",
        "all_cells": "whole_real_BigVGAN_GriffinLim_Vocos_groups;no_partial_recordings",
        "processing_order": "analysis_stress_then_mandatory_8k_filter_then_PCM16_then_features",
        "lowpass_order": "common_lower_filter_then_common_8k_filter;PCM16_lower_band_check",
        "resampling": "single_soxr_hq_22050_to_16000_PCM16_preflight_only;no_model_scoring",
        "resampling_limit": "current_22050_feature_contract;no_roundtrip_or_header_relabel",
        "models": "frozen_4B_outer_fold_models;source_disjoint_fit_calibration_and_scoring",
        "thresholds": "unchanged_4B;all_three_scores_two_levels_and_references_descriptive",
        "selection": "none;4B_primary_method_already_selected_on_train_OOF",
        "uncertainty": (
            "2000_paired_source_bootstrap_of_fixed_scores;not_model_selection_uncertainty"
        ),
        "research_fits": 0,
        "dev_audio_reads": 0,
        "eval_access": False,
        **FLAGS,
    }


def categorize(condition, known, predicted, state):
    if condition == "real":
        return "genuine_no_attribution" if not predicted else "genuine_false_attribution"
    if condition in known:
        if not predicted:
            return "known_rejected"
        return "known_correct" if predicted == condition else "known_wrong"
    if predicted:
        return "withheld_wrong_attribution"
    return (
        "withheld_rejected_real_like"
        if state == "no_supported_synthetic_source"
        else "withheld_rejected_uncertain"
    )


def decision_rows(frame, raw, bundle, selected):
    """Expand every declared policy, with independent gate and support diagnostics."""
    output = []
    for name, level in POLICIES:
        labels, states = policy(bundle, raw, name, level)
        for i, row in enumerate(frame.to_dict("records")):
            gate = bool(raw["real_margin"][i] <= bundle.metadata["binary_threshold"])
            has_score = name in SCORES
            cutoff = bundle.metadata["confidence_thresholds"][name][level] if has_score else None
            value = float(raw[name][i]) if has_score else None
            support = value >= cutoff if has_score else name != "reject_all"
            pred = bundle.known[labels[i]] if labels[i] >= 0 else ""
            reason = (
                "accepted_source_label"
                if pred
                else "policy_reject_all"
                if name == "reject_all"
                else "binary_real_like_gate"
                if not gate
                else "source_support_below_cutoff"
            )
            output.append(
                {
                    "utt_id": row["utt_id"],
                    "condition": row["condition"],
                    "split": row["split"],
                    "audio_sha256": row["audio_sha256"],
                    "policy": name,
                    "level": level,
                    "primary": name == selected and level == "retain95",
                    "real_margin_higher_real": float(raw["real_margin"][i]),
                    "binary_threshold_le": bundle.metadata["binary_threshold"],
                    "binary_margin_minus_threshold": (
                        float(raw["real_margin"][i]) - bundle.metadata["binary_threshold"]
                    ),
                    "binary_synthetic_route": gate,
                    "candidate_source": bundle.known[raw["source_index"][i]],
                    **{
                        f"logit_{k}": float(raw["source_logits"][i, j])
                        for j, k in enumerate(bundle.known)
                    },
                    **{s: float(raw[s][i]) for s in SCORES},
                    "confidence_value": value,
                    "confidence_cutoff_ge": cutoff,
                    "confidence_margin": value - cutoff if has_score else None,
                    "confidence_pass_counterfactual": bool(support),
                    "binary_gate_applied": name not in ("unconditional", "reject_all"),
                    "confidence_gate_applied": has_score,
                    "accepted": bool(pred),
                    "source": pred,
                    "decision": str(states[i]),
                    "reason": reason,
                    "category": categorize(row["condition"], bundle.known, pred, states[i]),
                    **FLAGS,
                }
            )
    return pd.DataFrame(output)


def summarize_paths(paths):
    primary = paths[paths.primary]
    counts = primary.category.value_counts().to_dict()
    conditions = {}
    for c, d in primary.groupby("condition"):
        conditions[c] = {
            "n": len(d),
            "accepted": int(d.accepted.sum()),
            "binary_gate_fail": int((~d.binary_synthetic_route).sum()),
            "support_fail_all_inputs": int((~d.confidence_pass_counterfactual).sum()),
            "support_fail_after_binary_pass": int(
                (d.binary_synthetic_route & ~d.confidence_pass_counterfactual).sum()
            ),
            "binary_and_support_fail": int(
                (~d.binary_synthetic_route & ~d.confidence_pass_counterfactual).sum()
            ),
            "categories": d.category.value_counts().to_dict(),
            "quantiles_min_p25_median_p75_max": {
                s: np.quantile(d[s], [0, 0.25, 0.5, 0.75, 1]).tolist()
                for s in (
                    "real_margin_higher_real",
                    "binary_margin_minus_threshold",
                    "distance",
                    "msp",
                    "negative_energy",
                    "confidence_margin",
                )
            },
        }
    groups = {}
    for g, d in primary.groupby("utt_id"):
        groups[g] = {
            "accepted": int(d.accepted.sum()),
            "correct": int((d.category == "known_correct").sum()),
            "categories": d.category.value_counts().to_dict(),
        }
    return {"counts": counts, "conditions": conditions, "groups": groups, **FLAGS}


def audit_partition(root, split, output):
    """Replay saved feature arrays; no input audio or frozen eval members are needed."""
    if split not in ("dev", "oof"):
        raise ValueError("Only historical dev and training OOF may be audited.")
    partition = "train" if split == "oof" else "dev"
    rows = pd.read_csv(root / f"{partition}_rows.csv", keep_default_na=False)
    if set(rows.split) != {partition} or rows.duplicated(["utt_id", "condition"]).any():
        raise ValueError("Invalid historical source grouping.")
    if any(set(g.condition) != {"real", *KNOWN_SOURCES} for _, g in rows.groupby("utt_id")):
        raise ValueError("Incomplete comparison groups.")
    with np.load(root / f"{partition}_features.npz", allow_pickle=False) as data:
        x = data["features"]
    if x.shape != (len(rows), 64) or not np.isfinite(x).all():
        raise ValueError("Invalid historical feature geometry.")
    choices = read_json(root / "selection_locked.json")["choices"]
    summary, tradeoffs, all_paths = {}, [], []
    for case, known in CASES.items():
        saved = pd.read_csv(root / case / split / "scores_predictions.csv", keep_default_na=False)
        if saved[["utt_id", "condition", "split", "audio_sha256"]].to_dict("records") != (
            rows.to_dict("records")
        ):
            raise ValueError("Historical prediction row identities differ.")
        pieces = []
        for fold, indices in saved.groupby("validation_fold").groups.items():
            bundle = load_verifier(
                root / case / (f"fold-{fold}/bundle" if split == "oof" else "bundle")
            )
            idx = np.asarray(list(indices))
            selected_rows = rows.iloc[idx]
            if split == "oof":
                assert_disjoint(bundle, selected_rows)
            raw = bundle.raw_scores(x[idx])
            for key in ("real_margin", *SCORES):
                np.testing.assert_allclose(raw[key], saved.iloc[idx][key], atol=2e-12, rtol=2e-13)
            for j, k in enumerate(known):
                np.testing.assert_allclose(
                    raw["source_logits"][:, j],
                    saved.iloc[idx][f"source_logit_{k}"],
                    atol=2e-12,
                    rtol=2e-13,
                )
            for name, level in POLICIES:
                p, s = policy(bundle, raw, name, level)
                expected = [known[v] if v >= 0 else "" for v in p]
                if (
                    expected != saved.iloc[idx][f"{name}.{level}.source"].tolist()
                    or s.tolist() != saved.iloc[idx][f"{name}.{level}.decision"].tolist()
                ):
                    raise ValueError("Saved decision differs from reconstructed gates.")
            pieces.append(decision_rows(selected_rows, raw, bundle, choices[case]))
        paths = pd.concat(pieces, ignore_index=True)
        paths.insert(0, "case", case)
        summary[case] = summarize_paths(paths)
        all_paths.append(paths)
        for name, level in POLICIES:
            m = read_json(root / case / split / f"{name}.{level}.metrics.json")
            d = paths[(paths.policy == name) & (paths.level == level)]
            known_mask = d.condition.isin(known)
            counts = {
                "known": int(known_mask.sum()),
                "genuine": int((d.condition == "real").sum()),
                "withheld": int((~known_mask & (d.condition != "real")).sum()),
                "known_accepted": int((known_mask & d.accepted).sum()),
                "known_correct_accepted": int((d.category == "known_correct").sum()),
                "genuine_false_attributed": int((d.category == "genuine_false_attribution").sum()),
                "withheld_false_attributed": int(
                    (d.category == "withheld_wrong_attribution").sum()
                ),
            }
            if counts != m["counts"]:
                raise ValueError("Historical metric count discrepancy.")
            tradeoffs.append(
                {
                    "case": case,
                    "policy": name,
                    "level": level,
                    "primary": name == choices[case] and level == "retain95",
                    **{
                        k: m[k]
                        for k in (
                            "known_coverage",
                            "known_accepted_accuracy",
                            "known_correct_source_fraction",
                            "genuine_false_attribution_rate",
                            "withheld_rejection_rate",
                            "source_label_coverage_all_inputs",
                            "source_label_precision_all_inputs",
                        )
                    },
                }
            )
    pd.concat(all_paths, ignore_index=True).to_csv(
        output / f"{split}_decision_paths.csv", index=False
    )
    pd.DataFrame(tradeoffs).to_csv(output / f"{split}_fixed_policy_tradeoffs.csv", index=False)
    write_json(output / f"{split}_path_summary.json", summary)
    return summary


def assert_disjoint(bundle, rows):
    groups = set(rows.utt_id)
    if (
        set(rows.split) != {"train"}
        or not groups <= set(bundle.metadata["held_groups"])
        or groups
        & (set(bundle.metadata["fit_groups"]) | set(bundle.metadata["calibration_groups"]))
    ):
        raise ValueError("Robustness scoring must be source-disjoint training OOF.")


def transform_group(wavs, cell):
    if (
        set(wavs) != {"real", *KNOWN_SOURCES}
        or len({len(w) for w in wavs.values()}) != 1
        or any(w.ndim != 1 or not np.isfinite(w).all() for w in wavs.values())
    ):
        raise ValueError("Expected a complete, equal-length finite source comparison group.")
    if cell not in CELLS:
        raise ValueError("Undeclared stress condition.")
    if cell.startswith("gain_"):
        gain = 10 ** ((-6 if cell == "gain_minus6" else -12) / 20)
        return {c: (w * gain).astype(np.float32) for c, w in wavs.items()}
    if cell.startswith("lowpass"):
        return band_limit_comparison_set(wavs, sr=ARCHIVE_SR, cutoff_hz=int(cell[7:]))
    if cell.startswith("first"):
        n = ARCHIVE_SR * int(cell[5])
        return {c: w[:n].copy() for c, w in wavs.items()}
    return {c: w.copy() for c, w in wavs.items()}


def paired_changes(clean, changed, *, resamples=2000):
    """Paired source resampling, including descendants together on both sides."""
    groups = sorted(clean["source_group_counts"])
    if groups != sorted(changed["source_group_counts"]):
        raise ValueError("Paired robustness comparisons need identical source groups.")
    fields = (
        "known",
        "accepted_known",
        "correct_known",
        "genuine",
        "false_known_genuine",
        "withheld",
        "false_known_withheld",
    )
    samples = np.random.default_rng(GLOBAL_SEED + 405).integers(
        0, len(groups), (resamples, len(groups))
    )

    def values(metrics):
        counts = np.array([[metrics["source_group_counts"][g][k] for k in fields] for g in groups])
        totals = counts[samples].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return {
                "known_coverage": totals[:, 1] / totals[:, 0],
                "known_correct_source_fraction": totals[:, 2] / totals[:, 0],
                "known_accepted_accuracy": totals[:, 2] / totals[:, 1],
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

    a, b = values(clean), values(changed)
    output = {}
    for k in a:
        delta = b[k] - a[k]
        good = np.isfinite(delta)
        output[k] = {
            "changed_minus_clean": changed[k] - clean[k]
            if changed[k] is not None and clean[k] is not None
            else None,
            "ci95": np.quantile(delta[good], [0.025, 0.975]).tolist() if good.any() else None,
            "undefined_replicates": int((~good).sum()),
        }
    return {
        "source_groups": len(groups),
        "resamples": resamples,
        "metrics": output,
        "scope": "paired_conditional_fixed_models_and_cutoffs;exploratory_subset",
        **FLAGS,
    }


def resampling_preflight(wavs, work):
    records = []
    for i, (condition, wav) in enumerate(wavs.items()):
        derived = resample_once(wav, ARCHIVE_SR, ZEROSHOT_SR)
        path = Path(work) / f"resampled_{i}.wav"
        write_audio(path, derived, ZEROSHOT_SR)
        decoded = read_processed(path, expected_sr=ZEROSHOT_SR)
        try:
            read_archive_input(path, FeatureConfig())
        except ValueError as exc:
            unsupported = str(exc)
        else:
            raise ValueError("Archive model unexpectedly accepted derived 16-kHz input.")
        if len(decoded) != int(np.ceil(len(wav) * ZEROSHOT_SR / ARCHIVE_SR)):
            raise ValueError("Unexpected single-step resampling length.")
        records.append(
            {
                "condition": condition,
                "input_samples": len(wav),
                "output_samples": len(decoded),
                "sample_rate": ZEROSHOT_SR,
                "duration_error_s": len(decoded) / ZEROSHOT_SR - len(wav) / ARCHIVE_SR,
                "wav_sha256": sha256_file(path),
                "model_score_produced": False,
                "unsupported_reason": unsupported,
            }
        )
    return records


def run_robustness(root, manifest, output, plan):
    provenance = read_json(root / "provenance.json")["data"]
    if sha256_file(manifest) != provenance["manifest_sha256"]:
        raise ValueError("Source manifest identity differs.")
    # The resolver reads structural rows only. Discard non-train rows before any audio I/O.
    frame = read_manifest(manifest)
    selected = plan["selected_train_groups"]
    frame = frame[
        (frame.split == "train")
        & (frame.tier == "archive")
        & frame.utt_id.isin(selected)
        & frame.condition.isin(["real", *KNOWN_SOURCES])
    ]
    rows = pd.read_csv(root / "train_rows.csv", keep_default_na=False)
    rows = rows[rows.utt_id.isin(selected)].copy().reset_index(drop=True)
    paths = {(r.utt_id, r.condition): r.path for r in frame.itertuples()}
    if len(paths) != len(rows) or set(frame.utt_id) != set(selected):
        raise ValueError("Selected training source coverage differs.")
    choices = read_json(root / "selection_locked.json")["choices"]
    scored = {cell: {case: [] for case in CASES} for cell in CELLS}
    feature_blocks, waveform_receipts, resampled = [], [], []
    bundles = {
        (case, fold): load_verifier(root / case / f"fold-{fold}/bundle")
        for case in CASES
        for fold in range(3)
    }
    with np.load(root / "train_features.npz", allow_pickle=False) as a:
        reference = a["features"]
    full_rows = pd.read_csv(root / "train_rows.csv", keep_default_na=False)
    for g, data in rows.groupby("utt_id", sort=True):
        wavs = {}
        for row in data.itertuples():
            path = paths[g, row.condition]
            if sha256_file(path) != row.audio_sha256:
                raise ValueError("Selected source audio integrity mismatch.")
            wavs[row.condition] = read_archive_input(path, FeatureConfig())
        with TemporaryDirectory(prefix="4c-analysis-", dir=output) as work:
            resampled.extend({"utt_id": g, **r} for r in resampling_preflight(wavs, work))
            for cell in CELLS:
                changed = transform_group(wavs, cell)
                features = extract_comparison_features(changed, work)
                x = np.vstack([features[c] for c in data.condition])
                if cell == "clean":
                    idx = full_rows.index[full_rows.utt_id == g].to_numpy()
                    np.testing.assert_allclose(x, reference[idx], atol=1e-12, rtol=0)
                feature_blocks.append(x)
                for i, row in enumerate(data.itertuples()):
                    path = Path(work) / f"analysis_{i}.wav"
                    decoded = read_processed(path, expected_sr=ARCHIVE_SR)
                    if cell.startswith("lowpass"):
                        check_band_limit(
                            decoded,
                            ARCHIVE_SR,
                            cutoff_hz=int(cell[7:]),
                            where="4C delivered PCM16 lower-band stress",
                        )
                    waveform_receipts.append(
                        {
                            "utt_id": g,
                            "condition": row.condition,
                            "cell": cell,
                            "split": "train",
                            "fold": selected[g],
                            "parent_sha256": row.audio_sha256,
                            "input_samples": len(wavs[row.condition]),
                            "support_samples": len(changed[row.condition]),
                            "sample_rate": ARCHIVE_SR,
                            "wav_sha256": sha256_file(path),
                            "feature_sha256": hashlib.sha256(
                                x[i].astype("<f8").tobytes()
                            ).hexdigest(),
                            "decoded_peak": float(np.max(np.abs(decoded))),
                            "zero_fraction": float(np.mean(decoded == 0)),
                            **FLAGS,
                        }
                    )
                for case in CASES:
                    bundle = bundles[case, selected[g]]
                    assert_disjoint(bundle, data)
                    raw = bundle.raw_scores(x)
                    scored[cell][case].append((data.copy(), raw, bundle))
        for row in data.itertuples():
            if sha256_file(paths[g, row.condition]) != row.audio_sha256:
                raise ValueError("Consumed source changed.")
        print(f"Completed fixed robustness cells for training source {g}", flush=True)
    metrics, prediction_rows = {}, []
    for cell in CELLS:
        metrics[cell] = {}
        for case, known in CASES.items():
            parts = scored[cell][case]
            d = pd.concat([p[0] for p in parts], ignore_index=True)
            raw = {k: np.concatenate([p[1][k] for p in parts]) for k in parts[0][1]}
            decisions = [policy(b, r, choices[case], "retain95") for _, r, b in parts]
            p = np.concatenate([v[0] for v in decisions])
            s = np.concatenate([v[1] for v in decisions])
            metrics[cell][case] = score_metrics(d.condition, d.utt_id, raw, p, s, known)
            for data, raw_part, bundle in parts:
                detail = decision_rows(data, raw_part, bundle, choices[case])
                detail.insert(0, "case", case)
                detail.insert(0, "cell", cell)
                prediction_rows.append(detail)
    write_json(output / "robustness_metrics.json", metrics)
    write_json(
        output / "robustness_paired_changes.json",
        {
            cell: {
                case: paired_changes(metrics["clean"][case], metrics[cell][case]) for case in CASES
            }
            for cell in CELLS
            if cell != "clean"
        },
    )
    write_json(output / "resampling_preflight.json", {"rows": resampled, **FLAGS})
    pd.DataFrame(waveform_receipts).to_csv(output / "analysis_waveform_receipts.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        output / "robustness_decisions.csv", index=False
    )
    np.savez_compressed(output / "robustness_features.npz", features=np.vstack(feature_blocks))
    return {
        "groups": len(selected),
        "source_recordings": len(rows),
        "feature_rows": sum(len(b) for b in feature_blocks),
        "resampled_preflight_rows": len(resampled),
        "research_fits": 0,
        **FLAGS,
    }


def run(historical_run, manifest, out_dir):
    root, manifest, target = [Path(p).resolve() for p in (historical_run, manifest, out_dir)]
    if target.exists():
        raise FileExistsError("Fresh analysis output required.")
    if any(target.is_relative_to(p) or p.is_relative_to(target) for p in (root, manifest.parent)):
        raise ValueError("Analysis output overlaps protected inputs.")
    verify_history(root)
    plan = fixed_plan(root)
    target.mkdir(parents=True)
    write_json(target / "plan_locked.json", plan)  # Before any dev prediction inspection.
    repo = Path(__file__).resolve().parents[2]
    write_json(
        target / "provenance.json",
        {
            "head": subprocess.check_output(
                ["git", "--no-optional-locks", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "environment": runtime_versions(),
            "source_sha256": {
                p.relative_to(repo).as_posix(): sha256_file(p)
                for p in sorted((repo / "src").rglob("*.py"))
            },
            **FLAGS,
        },
    )
    with threadpool_limits(limits=1):
        audit_partition(root, "dev", target)
        audit_partition(root, "oof", target)
        result = run_robustness(root, manifest, target, plan)
    verify_history(root)
    write_json(
        target / "validation.json",
        {
            "status": "passed",
            **result,
            "dev_prediction_policy_rows_replayed": 4 * 80 * len(POLICIES),
            "oof_prediction_policy_rows_replayed": 4 * 560 * len(POLICIES),
            "eval_audio_opened": 0,
            "eval_features_used": 0,
            "dev_audio_opened": 0,
        },
    )
    write_json(target / "artifacts.sha256.json", {"files": artifact_inventory(target), **FLAGS})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-run", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.historical_run, args.manifest, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
