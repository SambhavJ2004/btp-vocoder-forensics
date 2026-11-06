"""Synthetic CPU checks for decision replay, paired stress tests and protected splits."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data.audio_io import read_processed, write_audio
from data.invariants import ARCHIVE_SR, check_band_limit
from detectors.forensic_features import FeatureConfig, extract_comparison_features
from detectors.forensic_pilot import KNOWN_SOURCES, LinearHead, sha256_file, write_json
from detectors.source_verification import FLAGS, LEVELS, PROTOCOL, SCORES, SourceVerifier
from detectors.verification_metrics import score_metrics
from experiments import vocotrace_abstention as analysis
from experiments.vocotrace_verification import evaluate_scored


def bundle(known=KNOWN_SOURCES):
    coefficients = np.zeros((len(known) if len(known) == 3 else 1, 64))
    coefficients[:, 1] = np.arange(len(coefficients)) + 1
    binary = np.zeros((1, 64))
    binary[0, 0] = 1

    def head(c):
        return LinearHead(np.zeros(64), np.ones(64), c, np.zeros(len(c)))

    return SourceVerifier(
        head(binary),
        head(coefficients),
        np.zeros((len(known), 64)),
        np.eye(64),
        {
            "format_version": 1,
            "protocol": PROTOCOL,
            "features": asdict(FeatureConfig()),
            "known_classes": list(known),
            "selected_score": "msp",
            "binary_threshold": 0.0,
            "confidence_thresholds": {
                s: {level: 0.8 if s == "msp" else -1.0 for level in LEVELS} for s in SCORES
            },
            "held_groups": ["h"],
            "fit_groups": ["f"],
            "calibration_groups": ["c"],
            **FLAGS,
        },
    )


def test_all_error_categories_and_counterfactual_gate_reasons():
    b = bundle(KNOWN_SOURCES[:2])
    truth = [
        "real",
        "bigvgan_112m",
        "griffin_lim",
        "bigvgan_112m",
        "griffin_lim",
        "vocos",
        "vocos",
        "vocos",
    ]
    frame = pd.DataFrame(
        {"utt_id": ["h"] * 8, "condition": truth, "split": "dev", "audio_sha256": "synthetic"}
    )
    raw = {
        "real_margin": np.array([1, 0, -1, 1, -1, 1, -1, -1.0]),
        "source_index": np.array([0, 0, 0, 0, 1, 0, 0, 1]),
        "source_logits": np.zeros((8, 2)),
        "distance": np.zeros(8),
        "msp": np.array([0.1, 0.8, 0.9, 0.9, 0.7, 0.1, 0.7, 0.9]),
        "negative_energy": np.ones(8),
    }
    all_rows = analysis.decision_rows(frame, raw, b, "msp")
    d = all_rows[all_rows.primary]
    assert d.category.tolist() == [
        "genuine_no_attribution",
        "known_correct",
        "known_wrong",
        "known_rejected",
        "known_rejected",
        "withheld_rejected_real_like",
        "withheld_rejected_uncertain",
        "withheld_wrong_attribution",
    ]
    assert (
        d.reason.iloc[5] == "binary_real_like_gate"
    )  # Both gates fail; first cause remains binary.
    assert not d.confidence_pass_counterfactual.iloc[5]
    assert d.reason.iloc[6] == "source_support_below_cutoff"
    assert d.accepted.iloc[1]  # Both exact ties pass: real <= 0 and MSP >= .8.
    assert len(all_rows) == 8 * 9
    assert set(all_rows.citable) == {False}
    summary = analysis.summarize_paths(all_rows)
    assert summary["conditions"]["vocos"]["binary_and_support_fail"] == 1
    assert summary["conditions"]["vocos"]["support_fail_after_binary_pass"] == 1
    reject = all_rows[all_rows.policy == "reject_all"]
    assert not reject.accepted.any() and set(reject.reason) == {"policy_reject_all"}


def test_genuine_false_attribution_is_not_counted_as_correct():
    assert analysis.categorize("real", KNOWN_SOURCES, "vocos", "known_source") == (
        "genuine_false_attribution"
    )


def test_predetermined_selection_is_order_independent_and_exactly_balanced():
    assignments = {f"u{i:02d}": i % 3 for i in range(30)}
    a = analysis.predetermined_groups(assignments)
    assert a == analysis.predetermined_groups(dict(reversed(list(assignments.items()))))
    assert len(a) == 12 and all(list(a.values()).count(f) == 4 for f in range(3))
    with pytest.raises(ValueError, match="Insufficient"):
        analysis.predetermined_groups({"a": 0, "b": 1, "c": 2})
    with pytest.raises(ValueError, match="three"):
        analysis.predetermined_groups({"a": 0})


@pytest.mark.parametrize("bad", ["dev", "eval", "fit", "calibration", "not_held"])
def test_scoring_forbids_fit_calibration_dev_and_eval_groups(bad):
    b = bundle()
    rows = pd.DataFrame({"utt_id": ["h"], "split": ["train"]})
    if bad in ("dev", "eval"):
        rows["split"] = bad
    elif bad in ("fit", "calibration"):
        b.metadata[f"{bad}_groups"] = ["h"]
    else:
        rows["utt_id"] = "other"
    with pytest.raises(ValueError, match="source-disjoint"):
        analysis.assert_disjoint(b, rows)


def waves(seconds=2.2):
    t = np.arange(int(ARCHIVE_SR * seconds)) / ARCHIVE_SR
    w = 0.08 * np.sin(2 * np.pi * 1000 * t) + 0.03 * np.sin(2 * np.pi * 7000 * t)
    return {c: w.astype(np.float32) for c in ("real", *KNOWN_SOURCES)}


@pytest.mark.parametrize("cell", analysis.CELLS)
def test_stress_is_symmetric_deterministic_and_never_mutates_parents(tmp_path, cell):
    wavs = waves()
    original = {c: w.copy() for c, w in wavs.items()}
    a = analysis.transform_group(wavs, cell)
    b = analysis.transform_group(wavs, cell)
    for c in wavs:
        np.testing.assert_array_equal(wavs[c], original[c])
        np.testing.assert_array_equal(a[c], a["real"])
        np.testing.assert_array_equal(a[c], b[c])
    if cell.startswith("gain"):
        gain = 10 ** ((-6 if cell == "gain_minus6" else -12) / 20)
        np.testing.assert_array_equal(a["real"], original["real"] * gain)
    if cell.startswith("first"):
        n = ARCHIVE_SR * int(cell[5])
        np.testing.assert_array_equal(a["real"], original["real"][:n])
    x = extract_comparison_features(a, tmp_path)
    assert all(v.shape == (64,) and np.isfinite(v).all() for v in x.values())
    if cell.startswith("lowpass"):
        check_band_limit(
            read_processed(tmp_path / "analysis_0.wav", expected_sr=ARCHIVE_SR),
            ARCHIVE_SR,
            cutoff_hz=int(cell[7:]),
        )


def test_incomplete_unequal_nonfinite_and_undeclared_cells_fail():
    w = waves(0.2)
    for bad in (
        {k: v for k, v in w.items() if k != "real"},
        {**w, "real": w["real"][:-1]},
        {**w, "real": w["real"] * np.nan},
    ):
        with pytest.raises(ValueError, match="comparison group"):
            analysis.transform_group(bad, "clean")
    with pytest.raises(ValueError, match="Undeclared"):
        analysis.transform_group(w, "gain_plus99")


def test_single_step_resampling_checks_rate_length_and_keeps_source_unchanged(tmp_path):
    w = waves(0.1001)
    source = tmp_path / "source.wav"
    write_audio(source, w["real"], ARCHIVE_SR)
    digest = sha256_file(source)
    rows = analysis.resampling_preflight(w, tmp_path)
    assert len(rows) == 4 and all(not r["model_score_produced"] for r in rows)
    assert all(0 <= r["duration_error_s"] < 1 / 16000 for r in rows)
    assert all(r["sample_rate"] == 16000 and "22050" in r["unsupported_reason"] for r in rows)
    assert digest == sha256_file(source)
    assert rows == analysis.resampling_preflight(w, tmp_path)


def test_paired_changes_use_same_sources_and_include_undefined_denominators():
    raw = {
        "source_index": np.array([0, 1, 0, 1]),
        "real_margin": np.zeros(4),
        **{s: np.arange(4.0) for s in SCORES},
    }
    truth, groups = ["a", "real", "a", "real"], ["g1", "g1", "g2", "g2"]
    before = score_metrics(
        truth,
        groups,
        raw,
        [0, -1, 0, -1],
        ["known_source", "unknown_or_inconclusive"] * 2,
        ("a", "b"),
        bootstrap=0,
    )
    after = score_metrics(
        truth, groups, raw, [-1] * 4, ["unknown_or_inconclusive"] * 4, ("a", "b"), bootstrap=0
    )
    result = analysis.paired_changes(before, after)
    assert result["metrics"]["known_coverage"]["changed_minus_clean"] == -1
    assert result["metrics"]["known_coverage"]["ci95"] == [-1, -1]
    assert result["metrics"]["known_accepted_accuracy"]["undefined_replicates"] == 2000
    assert result == analysis.paired_changes(before, after)
    after["source_group_counts"].pop("g1")
    with pytest.raises(ValueError, match="identical"):
        analysis.paired_changes(before, after)


def test_saved_prediction_replay_detects_tampering_without_audio_access(tmp_path, monkeypatch):
    root = tmp_path / "prior"
    output = tmp_path / "audit"
    root.mkdir()
    output.mkdir()
    rows = pd.DataFrame(
        {
            "utt_id": ["h"] * 4,
            "condition": ["real", *KNOWN_SOURCES],
            "split": "dev",
            "audio_sha256": "synthetic",
        }
    )
    rows.to_csv(root / "dev_rows.csv", index=False)
    x = np.zeros((4, 64))
    x[:, 0] = [1, -1, -2, -3]
    x[:, 1] = [0, 1, 2, 3]
    np.savez_compressed(root / "dev_features.npz", features=x)
    b = bundle()
    write_json(root / "selection_locked.json", {"choices": {"closed3": "msp"}})
    evaluate_scored(
        rows,
        [(np.arange(4), b, b.raw_scores(x))],
        KNOWN_SOURCES,
        root / "closed3/dev",
        selected="msp",
    )
    monkeypatch.setattr(analysis, "CASES", {"closed3": KNOWN_SOURCES})
    monkeypatch.setattr(analysis, "load_verifier", lambda _: b)

    def forbidden(*args, **kwargs):
        pytest.fail("Replay must not decode any audio")

    monkeypatch.setattr(analysis, "read_archive_input", forbidden)
    summary = analysis.audit_partition(root, "dev", output)
    assert summary["closed3"]["counts"]["genuine_no_attribution"] == 1
    replay = pd.read_csv(output / "dev_decision_paths.csv")
    assert len(replay) == 36
    predictions = root / "closed3/dev/scores_predictions.csv"
    saved = pd.read_csv(predictions, keep_default_na=False)
    saved.loc[0, "msp.retain95.source"] = "vocos"
    saved.to_csv(predictions, index=False)
    with pytest.raises(ValueError, match="Saved decision"):
        analysis.audit_partition(root, "dev", output)
    with pytest.raises(ValueError, match="Only historical"):
        analysis.audit_partition(root, "eval", output)


def test_run_refuses_overwrite_and_protected_output_overlap(tmp_path):
    root = tmp_path / "history"
    root.mkdir()
    with pytest.raises(FileExistsError):
        analysis.run(root, tmp_path / "data/manifest.csv", root)
    with pytest.raises(ValueError, match="overlaps"):
        analysis.run(root, tmp_path / "data/manifest.csv", root / "child")


def test_fresh_plan_precedes_any_prediction_or_waveform_read(tmp_path, monkeypatch):
    root, data, target = tmp_path / "history", tmp_path / "data/manifest.csv", tmp_path / "output"
    monkeypatch.setattr(analysis, "verify_history", lambda _: None)
    monkeypatch.setattr(analysis, "fixed_plan", lambda _: {"locked": True})
    monkeypatch.setattr(analysis.subprocess, "check_output", lambda *a, **k: "synthetic-head")
    events = []

    def audit(r, split, out):
        assert analysis.read_json(out / "plan_locked.json") == {"locked": True}
        events.append(split)

    def robust(r, manifest, out, plan):
        assert events == ["dev", "oof"]
        return {"research_fits": 0}

    monkeypatch.setattr(analysis, "audit_partition", audit)
    monkeypatch.setattr(analysis, "run_robustness", robust)
    analysis.run(root, data, target)
    assert analysis.read_json(target / "validation.json")["eval_audio_opened"] == 0


def test_group_guard_accepts_only_correct_training_role():
    analysis.assert_disjoint(
        bundle(), SimpleNamespace(utt_id=pd.Series(["h"]), split=pd.Series(["train"]))
    )


def test_robustness_driver_reads_only_selected_train_audio_and_repeats_exactly(
    tmp_path, monkeypatch
):
    root = tmp_path / "prior"
    root.mkdir()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("synthetic metadata", encoding="utf-8")
    write_json(root / "provenance.json", {"data": {"manifest_sha256": sha256_file(manifest)}})
    write_json(root / "selection_locked.json", {"choices": {"closed3": "msp"}})
    rows, feature_parts, frames = [], [], []
    for fold in range(3):
        wavs = waves(0.2)
        decoded = {}
        for j, (condition, wav) in enumerate(wavs.items()):
            path = tmp_path / f"source_{fold}_{j}.wav"
            write_audio(path, wav, ARCHIVE_SR)
            decoded[condition] = read_processed(path, expected_sr=ARCHIVE_SR)
            row = {
                "utt_id": f"h{fold}",
                "condition": condition,
                "split": "train",
                "audio_sha256": sha256_file(path),
            }
            rows.append(row)
            frames.append({**row, "tier": "archive", "path": str(path)})
        work = tmp_path / f"reference{fold}"
        work.mkdir()
        features = extract_comparison_features(decoded, work)
        feature_parts.extend(features[c] for c in decoded)
    pd.DataFrame(rows).to_csv(root / "train_rows.csv", index=False)
    np.savez_compressed(root / "train_features.npz", features=np.vstack(feature_parts))
    # Structural eval metadata is present; a forbidden path must never reach the decoder.
    frames.append(
        {**frames[0], "utt_id": "eval-source", "split": "eval", "path": "FORBIDDEN_EVAL.wav"}
    )
    monkeypatch.setattr(analysis, "read_manifest", lambda _: pd.DataFrame(frames))
    monkeypatch.setattr(analysis, "CASES", {"closed3": KNOWN_SOURCES})
    monkeypatch.setattr(analysis, "CELLS", ("clean", "first1s"))

    def load(path):
        b = bundle()
        b.metadata["held_groups"] = [f"h{path.parent.name[-1]}"]
        return b

    monkeypatch.setattr(analysis, "load_verifier", load)
    original = analysis.read_archive_input
    calls = []

    def guarded(path, config):
        assert path != "FORBIDDEN_EVAL.wav"
        calls.append(str(path))
        return original(path, config)

    monkeypatch.setattr(analysis, "read_archive_input", guarded)
    plan = {"selected_train_groups": {f"h{f}": f for f in range(3)}}
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    result = analysis.run_robustness(root, manifest, a, plan)
    assert result == analysis.run_robustness(root, manifest, b, plan)
    assert result["source_recordings"] == 12 and result["feature_rows"] == 24
    assert result["research_fits"] == 0
    assert len(calls) == 2 * (12 + 12)  # Source reads plus rate-contract preflight calls.
    for name in (
        "robustness_decisions.csv",
        "analysis_waveform_receipts.csv",
        "resampling_preflight.json",
        "robustness_metrics.json",
        "robustness_paired_changes.json",
    ):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    for frame in frames[:-1]:
        assert sha256_file(frame["path"]) == frame["audio_sha256"]
