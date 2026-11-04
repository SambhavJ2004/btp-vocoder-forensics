"""CPU temporal tests use synthetic sources, never production/eval audio or final metrics."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from data.audio_io import write_audio
from data.invariants import ARCHIVE_SR
from data.manifest import ManifestRow, write_manifest
from data.partial_audio import VOCODERS, PartialAudioConfig
from detectors.forensic_pilot import LinearHead, sha256_file
from detectors.temporal_features import (
    WindowConfig,
    analysis_comparison,
    assign_window_labels,
    extract_windows,
    filter_support_receipt,
    guarded_labels,
    predicted_intervals,
    sample_labels,
    smooth_probabilities,
    window_starts,
)
from detectors.temporal_metrics import binary_metrics, evaluate_localization
from detectors.temporal_model import load_temporal_bundle, predict_temporal_file
from experiments.partial_audio import FLAGS, artifact_inventory, build_partial_dataset, write_json
from experiments.temporal_localization import (
    TemporalExperimentConfig,
    fit_temporal_head,
    load_temporal_rows,
    train_temporal,
)


@pytest.mark.parametrize(
    "changes",
    [
        {"hop_ms": 0},
        {"hop_ms": 65},
        {"window_ms": 300},
        {"window_ms": float("nan")},
        {"window_ms": True},
        {"filter_guard_ms": 0},
        {"sample_rate": 16000},
        {"n_bands": 32},
    ],
)
def test_invalid_window_config(changes):
    with pytest.raises(ValueError):
        replace(WindowConfig(), **changes).validate()


def test_window_grid_tail_and_short_policy():
    cfg = WindowConfig()
    assert not len(window_starts(0, cfg))
    assert window_starts(4, cfg).tolist() == [0]
    starts = window_starts(2500, cfg)
    assert starts[0] == 0 and starts[-1] + cfg.window_samples == 2500
    assert np.all(np.diff(starts) > 0)
    y = assign_window_labels(np.zeros(4), np.array([0]), cfg.window_samples)
    assert y.tolist() == [-1]


def test_spectral_features_have_physical_energy_and_no_metadata_inputs():
    cfg = WindowConfig()
    time = np.arange(3000) / ARCHIVE_SR
    wav = 0.12 * np.sin(2 * np.pi * 1000 * time)
    starts, x = extract_windows(wav, cfg)
    assert x.shape == (len(starts), 31)
    assert np.isfinite(x).all()
    np.testing.assert_allclose(x[:, 25], 1000 / 8000, atol=0.002)
    np.testing.assert_allclose(x[:, 24], np.log10(0.12**2 / 2), atol=0.003)
    np.testing.assert_array_equal(x, extract_windows(wav.copy(), cfg)[1])


@pytest.mark.parametrize("n", [0, 1, 50, 2048])
def test_empty_silent_and_short_feature_arrays_are_explicit(n):
    cfg = WindowConfig()
    starts, x = extract_windows(np.zeros(n), cfg)
    assert x.shape == (len(starts), 31) and np.isfinite(x).all()
    if n:
        assert np.all(x[:, -1] == 0)
        assert np.all(x[:, 24] == -12)


@pytest.mark.parametrize(
    "wav", [np.zeros((2, 2)), np.array([float("nan")]), np.array([float("inf")]), np.array([1j])]
)
def test_invalid_audio_rejected(wav):
    with pytest.raises(ValueError):
        extract_windows(wav, WindowConfig())


def test_pure_mixed_ignore_and_padding_labels():
    labels = np.r_[np.zeros(40), np.ones(40), [-1] * 20].astype(np.int8)
    starts = np.array([0, 20, 30, 40, 60, 80, 95])
    assert assign_window_labels(labels, starts, 20).tolist() == [0, 0, -1, 1, 1, -1, -1]
    with pytest.raises(ValueError):
        sample_labels([{"start_sample": 1, "end_sample": 4, "label": 0}], 4)


def test_guard_handles_both_fades_and_zero_fade_class_transitions():
    labels = np.r_[np.zeros(10), np.ones(10), np.zeros(10)].astype(np.int8)
    guarded = guarded_labels(labels, 2)
    assert np.all(guarded[8:13] == -1) and np.all(guarded[18:23] == -1)
    assert np.all(guarded[:2] == -1) and np.all(guarded[-2:] == -1)
    assert guarded[15] == 1 and guarded[5] == 0
    np.testing.assert_array_equal(guarded_labels(labels, 0), labels)


def test_fixed_filter_support_meets_declared_effective_tolerance():
    receipt = filter_support_receipt()
    assert receipt["finite_support"] is False
    assert 0 < receipt["bounded_delta_pcm16_steps"] < 0.01
    assert receipt["filter_guard_samples"] == round(0.15 * ARCHIVE_SR)


def test_common_filter_materialization_and_short_silence(tmp_path):
    wav = np.zeros(40, dtype=np.float32)
    result = analysis_comparison({"real": wav, "mixed": wav.copy()}, tmp_path)
    np.testing.assert_array_equal(result["real"], wav)
    np.testing.assert_array_equal(result["mixed"], wav)
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError):
        analysis_comparison({"real": np.empty(0)}, tmp_path)


def test_center_cells_and_gap_merge_do_not_consult_ground_truth():
    cfg = WindowConfig(window_ms=4, hop_ms=4)
    length = cfg.window_samples
    starts = np.arange(5) * length
    p = np.array([0, 1, 0, 1, 0], dtype=float)
    assert predicted_intervals(starts, p, 5 * length, cfg, 0.5, 0) == [
        [length, 2 * length],
        [3 * length, 4 * length],
    ]
    assert predicted_intervals(starts, p, 5 * length, cfg, 0.5, length) == [[length, 4 * length]]
    smoothed = smooth_probabilities(np.array([0.0, 1.0, 0.0]), 3)
    np.testing.assert_allclose(smoothed, [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_array_equal(p, [0, 1, 0, 1, 0])


@pytest.mark.parametrize("margin,auc,eer", [([-2, 2], 1, 0), ([2, -2], 0, 1), ([0, 0], 0.5, 0.5)])
def test_window_auroc_eer_orientation(margin, auc, eer):
    metrics = binary_metrics(np.array([0, 1]), np.array(margin), np.array(margin) >= 0)
    assert metrics["raw_auroc"] == auc and metrics["raw_directional_eer"] == eer
    assert binary_metrics(np.zeros(2), np.zeros(2), np.zeros(2))["raw_auroc"] is None


def test_group_metric_is_not_pooled_window_or_sample_metric():
    records = []
    for group, n, positives in (("a", 10, 2), ("b", 100, 80)):
        records.append(
            {
                "row": {
                    "group_id": group,
                    "num_samples": n,
                    "example_class": "mixed_vocos",
                    "has_synthetic": True,
                    "interval_labels": [
                        {"start_sample": 0, "end_sample": positives, "label": 1},
                        {"start_sample": positives, "end_sample": n, "label": 0},
                    ],
                },
                "starts": np.array([0]),
            }
        )
    result = evaluate_localization(
        records,
        [np.ones(1), np.ones(1)],
        WindowConfig(window_ms=4, hop_ms=4),
        0.5,
        1,
        0,
        0,
        detailed=False,
    )
    assert result["source_group_macro_f1"] == pytest.approx((1 / 3 + 8 / 9) / 2)
    assert result["sample_micro"]["f1"] == pytest.approx(164 / 192)


def test_train_only_normalization_and_explicit_class_balance():
    x = np.arange(124, dtype=float).reshape(4, 31) / 100
    records = [{"features": x.copy(), "window_y": np.array([0, 0, -1, 1])}]
    head, info = fit_temporal_head(records, 0.1, 7)
    np.testing.assert_allclose(head.mean, x[[0, 1, 3]].mean(axis=0))
    assert info["class_weights"] == {"0": 0.75, "1": 1.5}
    assert info["scaler_fit_split"] == "train_only"
    np.testing.assert_array_equal(records[0]["features"], x)
    assert isinstance(head, LinearHead)


@pytest.fixture(scope="module")
def temporal_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("temporal-source")
    rows = []
    n = 4 * ARCHIVE_SR
    time = np.arange(n) / ARCHIVE_SR
    for i, split in enumerate(("train", "train", "dev", "dev", "eval")):
        for j, condition in enumerate(("real", *VOCODERS)):
            wave = (0.12 * np.sin(2 * np.pi * (173 + i * 7 + j * 521) * time)).astype(np.float32)
            path = root / "archive" / condition / f"u{i}.wav"
            write_audio(path, wave, ARCHIVE_SR)
            rows.append(
                ManifestRow(
                    utt_id=f"u{i}",
                    condition=condition,
                    path=str(path),
                    speaker="fixture",
                    split=split,
                    duration_s=4,
                    lufs_out=-27,
                    trim_start=0,
                    trim_end=n,
                    trim_ref_length=n,
                    lufs_in=-27,
                    lufs_in_band=-27,
                    lufs_out_band=-27,
                    loudness_gain_db=0,
                    loudness_target_lufs=-27,
                    alignment_checked=True,
                    alignment_peak_lag=0,
                    high_band_fraction=0,
                )
            )
    source = write_manifest(rows, root / "manifest.csv")
    dataset = tmp_path_factory.mktemp("temporal-dataset") / "dataset"
    build_partial_dataset(
        source,
        dataset,
        PartialAudioConfig(
            n_utterances=5,
            all_utterances=True,
            real_to_real_controls=True,
            segment_seconds=0.7,
            boundary_margin_seconds=0.1,
            demo_examples=3,
        ),
    )
    return source, dataset


@pytest.fixture(scope="module")
def temporal_run(temporal_data, tmp_path_factory):
    source, dataset = temporal_data
    root = tmp_path_factory.mktemp("temporal-run") / "run"
    config = TemporalExperimentConfig(
        settings=((32.0, 8.0), (64.0, 16.0)),
        c_grid=(0.1,),
        thresholds=(0.35, 0.5, 0.65),
        smoothing_widths=(1, 3),
        merge_ms=(0.0,),
    )
    before = artifact_inventory(dataset)
    result = train_temporal(dataset, source, root, config)
    assert before == artifact_inventory(dataset)
    return source, dataset, root, config, result


def test_train_dev_run_bundle_prediction_labels_and_numeric_reproduction(temporal_run, tmp_path):
    source, dataset, root, config, result = temporal_run
    assert result["eval_used"] is False
    assert result["window_counts"]["train"]["recordings"] == 10
    assert result["window_counts"]["dev"]["source_groups"] == 2
    for split in ("train", "dev"):
        with np.load(root / split / "window_predictions.npz", allow_pickle=False) as state:
            assert (state["window_labels"] == -1).any()
            assert set(state["record_source_groups"]) <= (
                {"u0", "u1"} if split == "train" else {"u2", "u3"}
            )
        outputs = json.loads((root / split / "recordings.json").read_text())["recordings"]
        assert all(
            "ground_truth_edit_samples" in r and "model_predicted_intervals" in r for r in outputs
        )
    repeated = tmp_path / "repeated"
    train_temporal(dataset, source, repeated, config)
    assert artifact_inventory(root) == artifact_inventory(repeated)


def test_eval_waveforms_never_decoded_or_scored(temporal_data, tmp_path, monkeypatch):
    import experiments.temporal_localization as module

    source, dataset = temporal_data
    rows = json.loads((dataset / "manifest.json").read_text())["examples"]
    forbidden = {r["example_id"] for r in rows if r["split"] == "eval"}
    original = module.read_input
    seen = []

    def guarded(path):
        assert Path(path).stem not in forbidden
        seen.append(Path(path).stem)
        return original(path)

    monkeypatch.setattr(module, "read_input", guarded)
    result = train_temporal(
        dataset,
        source,
        tmp_path / "no-eval",
        TemporalExperimentConfig(
            settings=((32.0, 8.0),),
            c_grid=(0.1,),
            thresholds=(0.5,),
            smoothing_widths=(1,),
            merge_ms=(0.0,),
        ),
    )
    assert len(seen) == 20 and result["eval_used"] is False


def test_frozen_split_corruption_cannot_be_resealed_into_training(temporal_data, tmp_path):
    source, original = temporal_data
    root = tmp_path / "bad-split"
    shutil.copytree(original, root)
    doc = json.loads((root / "manifest.json").read_text())
    row = next(r for r in doc["examples"] if r["split"] == "dev")
    row["split"] = "train"
    write_json(root / "manifest.json", doc)
    write_json(root / "artifacts.sha256.json", {"files": artifact_inventory(root), **FLAGS})
    with pytest.raises(ValueError, match="frozen split"):
        load_temporal_rows(root, source)


def test_single_file_inference_ignores_filename_and_needs_no_ground_truth(temporal_run, tmp_path):
    _, dataset, root, _, _ = temporal_run
    row = json.loads((dataset / "manifest.json").read_text())["examples"][0]
    a, b = tmp_path / "real.wav", tmp_path / "vocos.wav"
    shutil.copyfile(dataset / row["output_filename"], a)
    shutil.copyfile(a, b)
    first = predict_temporal_file(root / "bundle", a, tmp_path)
    second = predict_temporal_file(root / "bundle", b, tmp_path)
    assert first == second and first["ground_truth_used"] is False


def test_bundle_corruption_and_object_arrays_fail_closed(temporal_run, tmp_path):
    _, _, original, _, _ = temporal_run
    root = tmp_path / "bundle"
    shutil.copytree(original / "bundle", root)
    with np.load(root / "model.npz", allow_pickle=False) as state:
        arrays = {key: state[key] for key in state.files}
    arrays["mean"] = np.array([{"unsafe": True}], dtype=object)
    np.savez_compressed(root / "model.npz", **arrays)
    checks = json.loads((root / "checksums.json").read_text())
    checks["model.npz"] = sha256_file(root / "model.npz")
    write_json(root / "checksums.json", checks)
    with pytest.raises(ValueError):
        load_temporal_bundle(root)


@pytest.mark.parametrize(
    "config",
    [
        TemporalExperimentConfig(settings=()),
        TemporalExperimentConfig(settings=((32.0, 8.0), (32.0, 8.0))),
        TemporalExperimentConfig(c_grid=(0.0,)),
        TemporalExperimentConfig(thresholds=(2.0,)),
    ],
)
def test_invalid_search_grid_fails_before_run_creation(config):
    with pytest.raises(ValueError):
        config.validate()


def test_probability_from_head_is_explicitly_logistic():
    head = LinearHead(np.zeros(2), np.ones(2), np.array([[2.0, -1.0]]), np.array([0.0]))
    np.testing.assert_allclose(
        expit(head.decision_function(np.array([[1.0, 0.0]]))), [[1 / (1 + np.exp(-2))]]
    )
