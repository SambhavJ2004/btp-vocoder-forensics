"""CPU numerical and leakage contracts; all fitted audio here is synthetic fixture data."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logsumexp

from data.audio_io import write_audio
from data.invariants import ARCHIVE_SR
from data.manifest import ManifestRow, write_manifest
from detectors.attribution import energy_score
from detectors.forensic_pilot import KNOWN_SOURCES
from detectors.source_verification import (
    SCORES,
    confidence_cutoff,
    genuine_gate,
    load_verifier,
    save_verifier,
)
from detectors.verification_metrics import score_metrics
from experiments.forensic_pilot import train_pilot
from experiments.partial_audio import artifact_inventory
from experiments.vocotrace_verification import (
    CASES,
    choose_score,
    fit_system,
    group_folds,
    outer_partition,
    run,
)


@pytest.mark.parametrize("temperature", [0, -1, np.nan, np.inf, -np.inf, True, [1], "1"])
def test_invalid_energy_temperature_rejected(temperature):
    with pytest.raises(ValueError):
        energy_score([[0, 1, 2]], temperature)


@pytest.mark.parametrize("logits", [[], np.empty((2, 0)), 0, [1, np.inf], [np.nan, 1]])
def test_invalid_energy_logits_rejected(logits):
    with pytest.raises(ValueError):
        energy_score(logits)


@pytest.mark.parametrize("temperature", [0.01, 1, 100])
def test_energy_semantics_orientation_and_constant_shift_preserved(temperature):
    z = np.array([[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]])
    np.testing.assert_allclose(
        energy_score(z, temperature),
        -temperature * logsumexp(z / temperature, axis=-1),
        atol=2e-14,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        energy_score(z + 5, temperature), energy_score(z, temperature) - 5, atol=2e-14, rtol=1e-14
    )
    assert energy_score([0.0, 0.0], temperature) == pytest.approx(-temperature * np.log(2))


def test_extreme_energy_is_stable_or_explicitly_unrepresentable():
    for t in (1e-300, 1.0, 1e300):
        result = energy_score([[1e308, -1e308]], t)
        np.testing.assert_allclose(result, [-1e308], rtol=1e-14)
    # Product T*log(10) overflows, although the sum with max=-1e308 is representable.
    np.testing.assert_allclose(
        energy_score(np.full((1, 10), -1e308), 1e308), [(1 - np.log(10)) * 1e308], rtol=1e-14
    )
    with pytest.raises(ValueError, match="representable"):
        energy_score(np.full((1, 10), 1e308), 1e308)


def test_calibration_boundaries_are_attainable_whole_ties_and_conservative():
    assert confidence_cutoff([0, 0, 1, 2], 0.25) == 0
    assert confidence_cutoff([0, 1, 2, 3], 0.25) == 1
    gate = genuine_gate([0, 1, 2, 3], 0.25)
    assert gate < 1 and gate > 0.99 and np.mean(np.array([0, 1, 2, 3]) <= gate) == 0.25
    tie = genuine_gate([0, 0, 2, 3], 0.25)
    assert tie < 0  # May not split the two tied genuine observations.
    assert genuine_gate([2, 3], 0.05) < 2


@pytest.mark.parametrize("fn", [confidence_cutoff, genuine_gate])
def test_invalid_calibration_data_rejected(fn):
    for values, budget in [([], 0.05), ([np.nan], 0.05), ([1], -1), ([1], 1)]:
        with pytest.raises(ValueError):
            fn(values, budget)


def numeric_rows(groups, seed=4):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        [
            {"utt_id": g, "condition": c, "split": "train"}
            for g in groups
            for c in ("real", *KNOWN_SOURCES)
        ]
    )
    x = rng.normal(size=(len(frame), 64))
    for j, c in enumerate(KNOWN_SOURCES):
        x[frame.condition == c, j] += 2
    return frame, x


def fitted(known=KNOWN_SOURCES):
    frame, x = numeric_rows([f"fit{i}" for i in range(6)])
    cal, cx = numeric_rows([f"cal{i}" for i in range(3)], seed=5)
    keep = frame.condition.isin(["real", *known]).to_numpy()
    ck = cal.condition.isin(["real", *known]).to_numpy()
    unknown = next((c for c in KNOWN_SOURCES if c not in known), None)
    return (
        fit_system(frame[keep], x[keep], cal[ck], cx[ck], known, held_out=unknown),
        frame,
        x,
        cal,
        cx,
    )


def test_binary_source_logits_and_support_scores_have_independent_expected_values():
    b, _, x, _, _ = fitted(KNOWN_SOURCES[:2])
    raw = b.raw_scores(x)
    margin = b.attribution.decision_function(x)[:, 0]
    np.testing.assert_allclose(
        expit(margin), 1 / (1 + np.exp(raw["source_logits"][:, 0] - raw["source_logits"][:, 1]))
    )
    b.centroids[:] = 0
    b.precision[:] = np.eye(64)
    expected = -np.sum(((x - b.attribution.mean) / b.attribution.scale) ** 2, axis=1)
    np.testing.assert_allclose(b.raw_scores(x)["distance"], expected, atol=1e-12)


def test_withheld_class_cannot_enter_any_fit_scaler_or_calibration():
    known = KNOWN_SOURCES[:2]
    b, frame, x, cal, cx = fitted(known)
    assert KNOWN_SOURCES[2] not in b.metadata["fit_conditions"]
    assert KNOWN_SOURCES[2] not in b.metadata["calibration_conditions"]
    with pytest.raises(ValueError):
        fit_system(frame, x, cal, cx, known, held_out=KNOWN_SOURCES[2])
    # Poisoning excluded rows leaves the fitted numeric state and thresholds identical.
    x[frame.condition == KNOWN_SOURCES[2]] = 1e200
    cx[cal.condition == KNOWN_SOURCES[2]] = -1e200
    m = frame.condition.isin(["real", *known]).to_numpy()
    cm = cal.condition.isin(["real", *known]).to_numpy()
    again = fit_system(frame[m], x[m], cal[cm], cx[cm], known, held_out=KNOWN_SOURCES[2])
    for name in ("binary", "attribution"):
        for field in ("mean", "scale", "coefficients", "intercept"):
            np.testing.assert_array_equal(
                getattr(getattr(b, name), field), getattr(getattr(again, name), field)
            )
    assert b.metadata == again.metadata
    train_syn = m & frame.condition.ne("real").to_numpy()
    np.testing.assert_allclose(b.attribution.mean, x[train_syn].mean(axis=0), atol=1e-14)


def test_train_calibration_and_outer_source_groups_remain_disjoint():
    assignments = group_folds([f"u{i}" for i in range(140)])
    assert sorted(list(assignments.values()).count(i) for i in range(3)) == [46, 47, 47]
    for fold in range(3):
        fit, cal, held = outer_partition(assignments, fold)
        assert not set(fit) & set(cal) and not set(fit) & set(held) and not set(cal) & set(held)
        assert set(fit) | set(cal) | set(held) == set(assignments)
    b, frame, x, cal, cx = fitted()
    cal.loc[0, "split"] = "dev"
    with pytest.raises(ValueError):
        fit_system(frame, x, cal, cx, KNOWN_SOURCES)
    with pytest.raises(ValueError):
        group_folds(["few"])


def test_selection_rejects_cap_violation_and_reject_all_is_not_success():
    metrics = {
        s: {
            "genuine_false_attribution_rate": 0.06,
            "known_coverage": 0.9,
            "known_correct_source_fraction": 0.8,
            "known_accepted_accuracy": 8 / 9,
        }
        for s in SCORES
    }
    assert choose_score(metrics) == "reject_all"
    metrics["distance"]["genuine_false_attribution_rate"] = 0.05
    assert choose_score(metrics) == "distance"
    metrics["msp"] = {**metrics["distance"], "known_correct_source_fraction": 0.81}
    assert choose_score(metrics) == "msp"


def test_metrics_separate_genuine_unknown_coverage_and_correctness():
    truth = np.array(["a", "b", "a", "b", "real", "real", "u"])
    prediction = np.array([0, 1, -1, 0, 0, -1, -1])
    states = np.where(prediction >= 0, "known_source", "unknown_or_inconclusive")
    raw = {"source_index": np.array([0, 1, 1, 1, 0, 0, 1]), "real_margin": np.arange(7.0)}
    raw.update({s: np.arange(7.0) for s in SCORES})
    m = score_metrics(truth, np.arange(7), raw, prediction, states, ("a", "b"), bootstrap=200)
    assert m["closed_set"]["confusion_matrix"] == [[1, 1], [0, 2]]
    assert m["closed_set"]["macro_f1"] == pytest.approx(11 / 15)
    assert m["selective_known"]["confusion_matrix"] == [[1, 0, 1], [1, 1, 0]]
    assert m["selective_known"]["macro_f1"] == pytest.approx(7 / 12)
    assert m["known_coverage"] == 0.75 and m["known_accepted_accuracy"] == pytest.approx(2 / 3)
    assert m["genuine_false_attribution_rate"] == 0.5 and m["withheld_rejection_rate"] == 1
    assert m["source_label_coverage_all_inputs"] == pytest.approx(4 / 7)
    assert m["source_label_precision_all_inputs"] == 0.5
    m = score_metrics(
        truth,
        np.arange(7),
        raw,
        np.full(7, -1),
        np.repeat("unknown_or_inconclusive", 7),
        ("a", "b"),
        bootstrap=0,
    )
    assert m["known_coverage"] == 0 and m["known_accepted_accuracy"] is None
    assert m["source_label_precision_all_inputs"] is None
    assert m["selective_known"]["macro_f1"] == 0


@pytest.mark.parametrize("damage", ["extra", "corrupt", "class_map", "scale", "precision"])
def test_numeric_bundle_integrity_and_validation(tmp_path, damage):
    b, _, x, _, _ = fitted()
    b.metadata["selected_score"] = "distance"
    root = tmp_path / "bundle"
    save_verifier(b, root)
    restored = load_verifier(root)
    assert restored.predict_features(x[0]) == b.predict_features(x[0])
    if damage == "extra":
        (root / "unrelated.txt").write_text("unexpected")
        with pytest.raises(ValueError):
            load_verifier(root)
    elif damage == "corrupt":
        (root / "model.npz").write_bytes(b"changed")
        with pytest.raises(ValueError):
            load_verifier(root)
    else:
        if damage == "class_map":
            b.metadata["known_classes"] = ["made_up", *KNOWN_SOURCES[:2]]
        if damage == "scale":
            b.binary.scale[0] = 0
        if damage == "precision":
            b.precision[0, 0] = -1
        with pytest.raises(ValueError):
            b.validate()


def test_label_free_decision_categories_never_certify_genuine():
    b, _, x, _, _ = fitted()
    b.metadata["selected_score"] = "distance"
    b.metadata["binary_threshold"] = 1e6
    b.metadata["confidence_thresholds"]["distance"]["retain95"] = -1e12
    p = b.predict_features(x[0])
    assert p["decision"] == "known_source" and p["known_source"] in KNOWN_SOURCES
    b.metadata["confidence_thresholds"]["distance"]["retain95"] = 1e12
    assert b.predict_features(x[0])["decision"] == "unknown_or_inconclusive"
    b.metadata["binary_threshold"] = -1e6
    p = b.predict_features(x[0])
    assert p["decision"] == "no_supported_synthetic_source" and p["known_source"] is None
    assert p["genuine_claim_supported"] is False and p["ground_truth_used"] is False


def test_complete_benchmark_no_eval_decoder_and_lock_before_dev_or_unknown(tmp_path, monkeypatch):
    import experiments.forensic_pilot as pilot
    import experiments.vocotrace_verification as module

    rows = []
    n = int(0.2 * ARCHIVE_SR)
    t = np.arange(n) / ARCHIVE_SR
    for i, split in enumerate(["train"] * 9 + ["dev"] * 2 + ["eval"]):
        for j, c in enumerate(("real", *KNOWN_SOURCES)):
            wav = 0.05 * np.sin(
                2 * np.pi * (300 + j * 1200 + i) * t
            ) + 0.001 * np.random.default_rng(i * 10 + j).normal(size=n)
            path = tmp_path / "source" / "archive" / c / f"u{i:02d}.wav"
            write_audio(path, wav.astype(np.float32), ARCHIVE_SR)
            rows.append(
                ManifestRow(
                    utt_id=f"u{i:02d}",
                    condition=c,
                    path=str(path),
                    speaker="fixture",
                    split=split,
                    duration_s=n / ARCHIVE_SR,
                    lufs_out=-27.0,
                    trim_start=0,
                    trim_end=n,
                    trim_ref_length=n,
                    lufs_in=-27.0,
                    lufs_in_band=-27.0,
                    lufs_out_band=-27.0,
                    loudness_gain_db=0.0,
                    loudness_target_lufs=-27.0,
                    alignment_checked=True,
                    alignment_peak_lag=0,
                    high_band_fraction=0.0,
                )
            )
    manifest = write_manifest(rows, tmp_path / "source/manifest.csv")
    baseline = tmp_path / "old-pilot"
    train_pilot(manifest, baseline, dataset_id="synthetic-fixture")
    before = {p: artifact_inventory(p) for p in (manifest.parent, baseline)}
    out = tmp_path / "verification"
    staging = out.with_name(out.name + ".staging")
    original_read = pilot.read_archive_input

    def guarded_read(path, config):
        assert "u11" not in str(path), "eval decode forbidden"
        if any(u in str(path) for u in ("u09", "u10")):
            assert (staging / "selection_locked.json").is_file()
        return original_read(path, config)

    monkeypatch.setattr(pilot, "read_archive_input", guarded_read)
    original_extract = module.extract_split

    def guarded_extract(frame, split, root, feature_cache=None):
        if split == "dev":
            lock = json.loads((root / "selection_locked.json").read_text())
            for case, inventory in lock["bundles"].items():
                assert artifact_inventory(root / case / "bundle") == inventory
        return original_extract(frame, split, root, feature_cache)

    monkeypatch.setattr(module, "extract_split", guarded_extract)
    original_eval = module.evaluate_scored

    def guarded_eval(frame, scored, known, root, **kwargs):
        assert (root.parents[1] / "selection_locked.json").is_file()
        return original_eval(frame, scored, known, root, **kwargs)

    monkeypatch.setattr(module, "evaluate_scored", guarded_eval)
    result = run(manifest, baseline, out)
    assert set(result["choices"]) == set(CASES)
    for case in CASES:
        bundle = load_verifier(out / case / "bundle")
        assert not set(bundle.metadata["fit_groups"]) & set(bundle.metadata["calibration_groups"])
        if case.startswith("without_"):
            assert case.removeprefix("without_") not in bundle.metadata["fit_conditions"]
        predictions = pd.read_csv(out / case / "oof/scores_predictions.csv")
        assert len(predictions) == 36 and set(predictions.split) == {"train"}
    assert json.loads((out / "label_free_inference.json").read_text())["ground_truth_used"] is False
    assert json.loads((out / "validation.json").read_text())["eval_audio_opened"] == 0
    assert (
        artifact_inventory(out) == json.loads((out / "artifacts.sha256.json").read_text())["files"]
    )
    for p, expected in before.items():
        assert artifact_inventory(p) == expected
    replay = tmp_path / "cache-replay"
    run(manifest, baseline, replay, feature_cache=out)
    for case in CASES:
        with np.load(out / case / "bundle/model.npz", allow_pickle=False) as first:
            with np.load(replay / case / "bundle/model.npz", allow_pickle=False) as second:
                for key in first.files:
                    np.testing.assert_array_equal(first[key], second[key])
        assert (out / case / "oof/scores_predictions.csv").read_bytes() == (
            replay / case / "oof/scores_predictions.csv"
        ).read_bytes()
    bad = tmp_path / "bad-cache"
    shutil.copytree(out, bad)
    (bad / "train_features.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        run(manifest, baseline, tmp_path / "bad-replay", feature_cache=bad)
    with pytest.raises(FileExistsError):
        run(manifest, baseline, out)
