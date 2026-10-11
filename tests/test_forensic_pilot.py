"""CPU pilot contracts using deliberately small synthetic fixtures, not research data."""

from __future__ import annotations

import json
import shutil
import wave
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from data.audio_io import read_processed, write_audio
from data.invariants import ARCHIVE_SR, InvariantViolation
from data.manifest import ManifestRow, write_manifest
from detectors.forensic_features import (
    FeatureConfig,
    extract_comparison_features,
    extract_file_features,
    read_archive_input,
)
from detectors.forensic_pilot import (
    KNOWN_SOURCES,
    SOURCE_CLASSES,
    load_bundle,
    predict_file,
    sha256_file,
    write_json,
)
from experiments.forensic_pilot import (
    attribution_labels,
    binary_labels,
    evaluate_pilot,
    evaluate_predictions,
    fit_head,
    load_pilot_dataset,
    select_heads,
    select_operating_threshold,
    train_pilot,
    validate_pilot_frame,
)


@pytest.fixture(scope="module")
def pilot_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("pilot-fixture")
    rows = []
    splits = ["train", "train", "dev", "eval", "eval"]
    conditions = ("real", *KNOWN_SOURCES)
    n = int(0.4 * ARCHIVE_SR)
    t = np.arange(n) / ARCHIVE_SR
    for i, split in enumerate(splits):
        for j, condition in enumerate(conditions):
            rng = np.random.default_rng(100 * i + j)
            wav = (0.06 * np.sin(2 * np.pi * (400 + j * 1100 + i) * t)
                   + 0.002 * rng.standard_normal(n)).astype(np.float32)
            path = root / "archive" / condition / f"u{i}.wav"
            write_audio(path, wav, ARCHIVE_SR)
            # Manifest contract fixture declarations; these are not measured gate receipts.
            rows.append(ManifestRow(
                utt_id=f"u{i}", condition=condition, path=str(path), speaker="fixture",
                split=split, duration_s=n / ARCHIVE_SR, lufs_out=-27.0,
                trim_start=0, trim_end=n, trim_ref_length=n,
                lufs_in=-27.0, lufs_in_band=-27.0, lufs_out_band=-27.0,
                loudness_gain_db=0.0, loudness_target_lufs=-27.0,
                alignment_checked=True, alignment_peak_lag=0, high_band_fraction=0.0,
            ))
    manifest = write_manifest(rows, root / "manifest.csv")
    return root, manifest


@pytest.fixture(scope="module")
def trained(pilot_data):
    root, manifest = pilot_data
    output = root / "trained"
    metrics = train_pilot(manifest, output, dataset_id="synthetic-unit-fixture", seed=7)
    return root, manifest, output, metrics


def test_frozen_groups_and_no_real_duplication(pilot_data):
    _, manifest = pilot_data
    frame, identity = load_pilot_dataset(manifest)
    assert len(frame) == 20
    assert frame.groupby("utt_id").split.nunique().eq(1).all()
    assert identity["split_counts"] == {"train": 2, "dev": 1, "eval": 2}
    assert frame[frame.condition == "real"].utt_id.is_unique
    assert identity["citable"] is False and identity["signal_gates_rerun"] is False


@pytest.mark.parametrize("damage", [
    "duplicate", "shared_path", "different_split", "missing_condition", "unknown_condition",
    "unknown_split", "missing_id", "different_speaker", "speaker_overlap",
])
def test_rejects_split_and_identity_leaks(pilot_data, damage):
    _, manifest = pilot_data
    frame, _ = load_pilot_dataset(manifest)
    frame = frame.copy()
    if damage == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif damage == "shared_path":
        frame.loc[1, "path"] = frame.loc[0, "path"]
    elif damage == "different_split":
        frame.loc[0, "split"] = "dev"
    elif damage == "missing_condition":
        frame = frame.drop(index=0)
    elif damage == "unknown_condition":
        frame.loc[0, "condition"] = "unseen"
    elif damage == "unknown_split":
        frame.loc[0, "split"] = "validation"
    elif damage == "missing_id":
        frame.loc[0, "utt_id"] = None
    elif damage == "different_speaker":
        frame.loc[0, "speaker"] = "other"
    else:
        frame.loc[frame.utt_id == "u0", "speaker"] = "other"
    with pytest.raises(InvariantViolation):
        validate_pilot_frame(frame)


def test_multispeaker_disjoint_existing_splits_allowed(pilot_data):
    _, manifest = pilot_data
    frame, _ = load_pilot_dataset(manifest)
    frame["speaker"] = frame.split
    validate_pilot_frame(frame)


def test_cross_split_identical_audio_rejected(pilot_data, tmp_path):
    root, _ = pilot_data
    copy = tmp_path / "copy"
    shutil.copytree(root / "archive", copy / "archive")
    shutil.copyfile(root / "manifest.csv", copy / "manifest.csv")
    shutil.copyfile(copy / "archive/real/u0.wav", copy / "archive/real/u3.wav")
    with pytest.raises(InvariantViolation, match="Identical audio"):
        load_pilot_dataset(copy / "manifest.csv")


def test_label_mappings_are_explicit_and_stable():
    assert binary_labels(["real", "vocos", "griffin_lim", "bigvgan_112m"]).tolist() == [1, 0, 0, 0]
    assert attribution_labels(["vocos", "bigvgan_112m", "griffin_lim"]).tolist() == [2, 0, 1]
    assert SOURCE_CLASSES == {"0": "bigvgan_112m", "1": "griffin_lim", "2": "vocos"}


@pytest.mark.parametrize("mapper,conditions", [
    (binary_labels, ["unknown"]), (attribution_labels, ["real"]),
    (attribution_labels, ["unknown"]),
])
def test_invalid_labels_rejected(mapper, conditions):
    with pytest.raises(ValueError):
        mapper(conditions)


def test_training_and_inference_feature_identity(pilot_data, tmp_path):
    root, _ = pilot_data
    paths = {name: root / "archive" / name / "u0.wav" for name in ("real", *KNOWN_SOURCES)}
    wavs = {name: read_processed(path, expected_sr=ARCHIVE_SR) for name, path in paths.items()}
    before = {name: wav.copy() for name, wav in wavs.items()}
    vectors = extract_comparison_features(wavs, tmp_path)
    for name, path in paths.items():
        np.testing.assert_array_equal(vectors[name], extract_file_features(path, work_dir=tmp_path))
        np.testing.assert_array_equal(wavs[name], before[name])
        assert vectors[name].shape == (64,) and np.isfinite(vectors[name]).all()


def test_renamed_file_has_identical_features(pilot_data, tmp_path):
    root, _ = pilot_data
    original = root / "archive/real/u0.wav"
    renamed = tmp_path / "vocos_fake_name.wav"
    shutil.copyfile(original, renamed)
    np.testing.assert_array_equal(
        extract_file_features(original, work_dir=tmp_path),
        extract_file_features(renamed, work_dir=tmp_path),
    )


def test_unequal_group_lengths_rejected(tmp_path):
    with pytest.raises(InvariantViolation, match="unequal sample lengths"):
        extract_comparison_features({"real": np.zeros(5000), "vocos": np.zeros(5001)}, tmp_path)


@pytest.mark.parametrize("field,value", [
    ("sample_rate", 16000), ("band_limit_hz", 10000.0), ("n_fft", 512),
    ("crop_policy", "zero_pad"), ("analysis_encoding", "FLOAT"), ("version", "v2"),
])
def test_unsupported_feature_protocol_rejected(field, value):
    with pytest.raises(ValueError, match="Unsupported"):
        replace(FeatureConfig(), **{field: value}).validate()


@pytest.mark.parametrize("kind", ["missing", "text", "extension", "rate", "stereo",
                                  "bit_depth", "empty", "short", "long", "clipped"])
def test_invalid_or_unsupported_audio_rejected(tmp_path, kind):
    path = tmp_path / ("input.mp3" if kind == "extension" else "input.wav")
    if kind == "text":
        path.write_text("not audio", encoding="utf-8")
    elif kind != "missing":
        # Deliberately invalid inputs bypass sanctioned writer using stdlib WAV.
        width = 4 if kind == "bit_depth" else 2
        channels = 2 if kind == "stereo" else 1
        n = {"empty": 0, "short": 10, "long": ARCHIVE_SR * 61}.get(kind, 5000)
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(channels)
            stream.setsampwidth(width)
            stream.setframerate(16000 if kind == "rate" else ARCHIVE_SR)
            stream.writeframes((b"\xff\x7f" if kind == "clipped" else b"\0" * width)
                               * n * channels)
    with pytest.raises((ValueError, InvariantViolation)):
        read_archive_input(path, FeatureConfig())


def test_scaler_and_balancing_use_training_rows_only():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(12, 64))
    conditions = np.array(["real", *KNOWN_SOURCES] * 3)
    labels = binary_labels(conditions)
    pipeline = fit_head(x, labels, 0.1, 7, balanced=True)
    np.testing.assert_allclose(pipeline.named_steps["scale"].mean_, x.mean(axis=0))
    assert pipeline.named_steps["scale"].n_samples_seen_ == 12
    assert pipeline.named_steps["classifier"].class_weight == "balanced"
    _, _, _, selection = select_heads(x, conditions, x + 100, conditions, c_grid=(0.1,))
    assert selection["train_binary_counts"] == {"real": 3, "synthetic": 9}
    assert selection["binary_class_weights"] == pytest.approx({"real": 2, "synthetic": 2 / 3})
    assert selection["refit_on_dev"] is False
    binary, attribution, _, _ = select_heads(x, conditions, x + 500, conditions, c_grid=(0.1,))
    np.testing.assert_allclose(binary.mean, x.mean(axis=0))
    np.testing.assert_allclose(attribution.mean, x[labels == 0].mean(axis=0))


@pytest.mark.parametrize("grid", [(), (0.0,), (-1,), (np.inf,), (np.nan,)])
def test_invalid_regularization_grid(grid):
    with pytest.raises(ValueError, match="C grid"):
        select_heads(np.zeros((4, 64)), ["real", *KNOWN_SOURCES],
                     np.zeros((4, 64)), ["real", *KNOWN_SOURCES], c_grid=grid)


def test_threshold_is_attainable_dev_policy_and_preserves_ties():
    threshold, point = select_operating_threshold(np.array([2, 3]), np.array([0, 1]))
    assert threshold == 1 and point["balanced_error"] == 0
    threshold, point = select_operating_threshold(np.zeros(2), np.zeros(6))
    assert point["balanced_error"] == 0.5
    assert (point["frr"], point["far"]) in ((0, 1), (1, 0))
    assert np.isfinite(threshold)


def test_independently_derived_metrics():
    conditions = ["real", "real", *KNOWN_SOURCES, *KNOWN_SOURCES]
    p = pd.DataFrame({
        "condition": conditions, "binary_label": binary_labels(conditions),
        "real_score": [3, 4, -1, -2, -3, 0, -1, -2],
        "binary_prediction": ["real"] * 2 + ["synthetic"] * 6,
        "known_source_prediction": ["vocos"] * 2 + list(KNOWN_SOURCES) * 2,
    })
    m = evaluate_predictions(p)
    assert m["binary"]["eer"] == 0 and m["binary"]["auroc"] == 1
    assert m["binary"]["confusion_matrix"] == [[6, 0], [0, 2]]
    assert all(item["eer"] == 0 for item in m["binary"]["per_vocoder"].values())
    assert m["attribution"]["macro_f1"] == 1
    assert m["attribution"]["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    p["real_score"] = 0.0
    assert evaluate_predictions(p)["binary"]["eer"] == 0.5
    p["real_score"] = [-3, -4, 1, 2, 3, 4, 5, 6]
    assert evaluate_predictions(p)["binary"]["eer"] == 1


def test_save_load_reproduce_evaluation_and_inference(trained, tmp_path):
    root, manifest, output, metrics = trained
    bundle = load_bundle(output / "bundle")
    assert bundle.metadata["source_classes"] == SOURCE_CLASSES
    assert bundle.metadata["features"] == asdict(FeatureConfig())
    repeat = evaluate_pilot(output / "bundle", manifest, tmp_path / "repeat")
    assert repeat == metrics
    original = pd.read_csv(output / "predictions.csv")
    pd.testing.assert_frame_equal(original, pd.read_csv(tmp_path / "repeat/predictions.csv"))
    for row in original.itertuples():
        result = predict_file(bundle, row.path, work_dir=tmp_path)
        assert result["real_score"] == pytest.approx(row.real_score, abs=1e-12)
        assert result["prediction"] == row.binary_prediction
        assert result["citable"] is False
        if result["prediction"] == "synthetic":
            assert result["known_source"] == row.known_source_prediction
            assert set(result["known_source_scores"]) == set(KNOWN_SOURCES)
    receipt = json.loads((output / "artifacts.sha256.json").read_text())
    assert all(sha256_file(output / name) == digest for name, digest in receipt.items())
    assert len(original) == 8
    assert (root / "archive").is_dir()


def test_numeric_state_matches_sklearn_and_reproducible():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(20, 64))
    conditions = np.array(["real", *KNOWN_SOURCES] * 5)
    from detectors.forensic_pilot import LinearHead

    sklearn_head = fit_head(x, binary_labels(conditions), 0.1, 7, balanced=True)
    numeric = LinearHead.from_pipeline(sklearn_head)
    np.testing.assert_allclose(numeric.decision_function(x)[:, 0],
                               sklearn_head.decision_function(x), atol=1e-12)
    first = select_heads(x, conditions, x, conditions, c_grid=(0.1,))
    second = select_heads(x, conditions, x, conditions, c_grid=(0.1,))
    np.testing.assert_array_equal(first[0].coefficients, second[0].coefficients)
    np.testing.assert_array_equal(first[1].coefficients, second[1].coefficients)
    assert first[2:] == second[2:]


@pytest.mark.parametrize("damage", ["hash", "feature", "mapping", "environment",
                                  "preprocessing", "threshold", "citable", "shape"])
def test_incompatible_or_corrupt_bundles_rejected(trained, tmp_path, damage):
    _, _, output, _ = trained
    root = tmp_path / "bundle"
    shutil.copytree(output / "bundle", root)
    meta = json.loads((root / "bundle.json").read_text())
    if damage == "hash":
        (root / "models.npz").write_bytes(b"broken")
    elif damage == "shape":
        with np.load(root / "models.npz", allow_pickle=False) as arrays:
            state = {name: arrays[name].copy() for name in arrays.files}
        state["binary_coefficients"] = np.zeros((1, 63))
        np.savez_compressed(root / "models.npz", **state)
        meta["model_sha256"] = sha256_file(root / "models.npz")
    else:
        changes = {
            "feature": ("features", dict(meta["features"], sample_rate=16000)),
            "mapping": ("source_classes", {"0": "vocos"}),
            "environment": ("environment", {}),
            "preprocessing": ("preprocessing_identity", {}),
            "threshold": ("operating_threshold", "not a number"),
            "citable": ("citable", True),
        }
        key, value = changes[damage]
        meta[key] = value
    if damage != "hash":
        write_json(root / "bundle.json", meta)
        write_json(root / "bundle.sha256.json", {
            name: sha256_file(root / name) for name in ("bundle.json", "models.npz")
        })
    with pytest.raises(ValueError):
        load_bundle(root)


def test_different_dataset_refused_and_outputs_not_overwritten(trained, tmp_path):
    root, manifest, output, _ = trained
    with pytest.raises(FileExistsError):
        train_pilot(manifest, output)
    copied = tmp_path / "changed.csv"
    copied.write_bytes(manifest.read_bytes() + b"\n")
    shutil.copytree(root / "archive", tmp_path / "archive")
    with pytest.raises(ValueError, match="identity differs"):
        evaluate_pilot(output / "bundle", copied, tmp_path / "new-eval")


def test_cli_pilot_dispatch(trained, tmp_path, capsys):
    from experiments.cli import main

    _, manifest, output, _ = trained
    assert main(["pilot", "evaluate", "--bundle", str(output / "bundle"),
                 "--manifest", str(manifest), "--out-dir", str(tmp_path / "evaluation")]) == 0
    assert json.loads(capsys.readouterr().out)["citable"] is False
    path = pd.read_csv(output / "predictions.csv").path.iloc[0]
    assert main(["pilot", "predict", "--bundle", str(output / "bundle"),
                 "--wav", path, "--work-dir", str(tmp_path)]) == 0
    assert "real_score" in json.loads(capsys.readouterr().out)


def test_eval_features_accessed_after_bundle_freeze_and_repeat_training(
    trained, tmp_path, monkeypatch,
):
    import experiments.forensic_pilot as pilot

    _, manifest, original, _ = trained
    events = []
    original_extract, original_save = pilot.extract_dataset_features, pilot.save_bundle

    def extract(frame, config, work_dir):
        events.append(("extract", set(frame.split)))
        if set(frame.split) == {"eval"}:
            assert events[-2][0] == "save"
        return original_extract(frame, config, work_dir)

    def save(bundle, out_dir):
        events.append(("save", None))
        return original_save(bundle, out_dir)

    monkeypatch.setattr(pilot, "extract_dataset_features", extract)
    monkeypatch.setattr(pilot, "save_bundle", save)
    repeated = tmp_path / "repeated"
    pilot.train_pilot(manifest, repeated, dataset_id="synthetic-unit-fixture", seed=7)
    assert events == [("extract", {"train", "dev"}), ("save", None), ("extract", {"eval"})]
    first, second = load_bundle(original / "bundle"), load_bundle(repeated / "bundle")
    np.testing.assert_array_equal(first.binary.coefficients, second.binary.coefficients)
    np.testing.assert_array_equal(first.attribution.coefficients, second.attribution.coefficients)
    assert first.metadata["operating_threshold"] == second.metadata["operating_threshold"]


def test_cli_train_dispatch(pilot_data, tmp_path, monkeypatch, capsys):
    import experiments.forensic_pilot as pilot
    from experiments.cli import main

    _, manifest = pilot_data
    captured = {}

    def train(manifest_path, output, *, dataset_id, seed):
        captured.update(manifest=manifest_path, output=output, dataset_id=dataset_id, seed=seed)
        return {"citable": False}

    monkeypatch.setattr(pilot, "train_pilot", train)
    assert main(["pilot", "train", "--manifest", str(manifest), "--out-dir", str(tmp_path),
                 "--dataset-id", "fixture", "--seed", "9"]) == 0
    assert captured == {"manifest": str(manifest), "output": str(tmp_path),
                        "dataset_id": "fixture", "seed": 9}
    assert json.loads(capsys.readouterr().out) == {"citable": False}
