"""CPU fixtures for training-only, explicitly oracle acoustic diagnostics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from data.audio_io import write_audio
from data.invariants import ARCHIVE_SR
from data.manifest import ManifestRow, write_manifest
from data.partial_audio import VOCODERS, PartialAudioConfig
from experiments.paired_signal import (
    KINDS,
    PairedConfig,
    frame_rms,
    measures,
    oracle_metrics,
    oracle_vectors,
    reference_activity_threshold,
    rms,
    run_paired_diagnostics,
)
from experiments.partial_audio import artifact_inventory, build_partial_dataset
from experiments.temporal_localization import TemporalExperimentConfig, train_temporal
from experiments.temporal_review import read_json


def waves():
    time = np.arange(15000) / ARCHIVE_SR
    return {kind: 0.1 * np.sin(2 * np.pi * (200 + 101 * i) * time) for i, kind in enumerate(KINDS)}


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf], np.zeros((2, 2)), [1j]])
def test_invalid_rms_input_rejected(values):
    with pytest.raises(ValueError):
        rms(values)


def test_rms_activity_and_band_energy_use_explicit_physical_units():
    t = np.arange(15000) / ARCHIVE_SR
    wav = 0.1 * np.sin(2 * np.pi * 1000 * t)
    assert rms(wav) == pytest.approx(0.1 / np.sqrt(2), rel=1e-3)
    starts, levels = frame_rms(wav)
    assert len(starts) == len(levels) and starts[0] == 0 and starts[-1] + 512 <= len(wav)
    assert not len(frame_rms(np.zeros(12))[0])
    threshold = reference_activity_threshold(wav)
    assert threshold >= 2 / 32768 and threshold < rms(wav)
    result = measures(wav, threshold)
    assert result["activity_fraction"] == 1
    assert result["inband_power_mean"] == pytest.approx(0.005, rel=0.005)
    assert len(result["mean_psd_bands24"]) == 24 and len(result["bands_1khz_mean_psd"]) == 8
    assert len(result["feature_mean31"]) == 31
    with pytest.raises(ValueError):
        reference_activity_threshold(np.zeros(15000))


@pytest.mark.parametrize(
    "changes",
    [{"rms_target": 0.1}, {"min_active_frames": 1}, {"core_guard_samples": 0}, {"seed": 2}],
)
def test_unplanned_oracle_settings_rejected(changes):
    with pytest.raises(ValueError):
        replace(PairedConfig(), **changes).validate()


def test_rms_control_removes_only_gain_sensitivity_and_preserves_shapes():
    cores = waves()
    vectors, evidence = oracle_vectors(cores, 0.001)
    assert vectors is not None and evidence["target_active_frames"] >= 8
    gains = [0.25, 0.5, 1, 2, 4, 8]
    changed = {kind: cores[kind] * gain for kind, gain in zip(KINDS, gains, strict=True)}
    other, _ = oracle_vectors(changed, 0.001)
    for kind in KINDS:
        for variant in ["rms", "rms_active"]:
            assert vectors[variant][kind].shape == (62,)
            np.testing.assert_allclose(
                vectors[variant][kind], other[variant][kind], rtol=0, atol=2e-10
            )
    assert not np.allclose(vectors["raw"]["real_target"], other["raw"]["real_target"])


def test_activity_intersection_cannot_select_class_specific_positive_frames():
    cores = waves()
    cores[VOCODERS[0]][:14000] = 0
    result, evidence = oracle_vectors(cores, 0.001)
    assert result is None and evidence["reason"] == "insufficient_common_active_frames"
    assert evidence["target_active_frames"] < 8
    short = {kind: value[:50] for kind, value in waves().items()}
    result, evidence = oracle_vectors(short, 0.001)
    assert result is None and evidence["reason"] == "core_shorter_than_feature_window"


@pytest.mark.parametrize("direction,auc,eer", [(1, 1, 0), (-1, 0, 1), (0, 0.5, 0.5)])
def test_oracle_score_orientation_ties_and_negative_caps(direction, auc, eer):
    rows = [
        {"group_id": f"g{i}", "kind": kind, "label": int(kind in VOCODERS)}
        for i in range(4)
        for kind in KINDS
    ]
    margins = np.array([direction * (1 if r["label"] else -1) for r in rows], dtype=float)
    metrics = oracle_metrics(rows, margins)
    assert metrics["raw_auroc"] == auc and metrics["raw_directional_eer"] == eer
    assert all(value <= 0.05 for value in metrics["negative_core_fpr"].values())
    assert metrics["oracle_only_not_localization"] is True
    assert len(metrics["per_group"]) == 4


@pytest.fixture(scope="module")
def paired_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("paired3c")
    n = 4 * ARCHIVE_SR
    time = np.arange(n) / ARCHIVE_SR
    rows = []
    for i, split in enumerate(["train"] * 6 + ["dev", "eval"]):
        for j, condition in enumerate(("real", *VOCODERS)):
            path = root / "source" / "archive" / condition / f"u{i}.wav"
            write_audio(
                path,
                (0.1 * np.sin(2 * np.pi * (181 + 7 * i + 501 * j) * time)).astype(np.float32),
                ARCHIVE_SR,
            )
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
    source = write_manifest(rows, root / "source" / "manifest.csv")
    dataset = root / "dataset"
    build_partial_dataset(
        source,
        dataset,
        PartialAudioConfig(
            n_utterances=8,
            all_utterances=True,
            real_to_real_controls=True,
            segment_seconds=1,
            boundary_margin_seconds=0.1,
            demo_examples=0,
        ),
    )
    baseline = root / "baseline"
    train_temporal(
        dataset,
        source,
        baseline,
        TemporalExperimentConfig(
            settings=((64.0, 16.0),),
            c_grid=(1.0,),
            thresholds=(0.5,),
            smoothing_widths=(3,),
            merge_ms=(0.0,),
        ),
    )
    return root, source, dataset, baseline


def test_paired_run_is_train_only_grouped_deterministic_and_read_only(paired_dataset, monkeypatch):
    import experiments.paired_signal as module

    root, source, dataset, baseline = paired_dataset
    before = artifact_inventory(dataset)
    old = artifact_inventory(baseline)
    forbidden = {"u6", "u7"}
    forbidden_ids = {
        r["example_id"]
        for r in read_json(dataset / "manifest.json")["examples"]
        if r["split"] != "train"
    }
    original = module.read_input
    seen = []

    def restricted(path):
        assert Path(path).stem not in forbidden | forbidden_ids
        seen.append(str(path))
        return original(path)

    monkeypatch.setattr(module, "read_input", restricted)
    result = run_paired_diagnostics(dataset, source, baseline, root / "diagnostics")
    assert len(seen) == 6 * 9
    out = root / "diagnostics"
    validation = read_json(out / "validation.json")
    assert validation["training_source_groups"] == 6 and validation["oracle_eligible_groups"] == 6
    assert (
        validation["dev_predictive_processing"] is False
        and validation["eval_waveforms_opened"] == 0
    )
    assert len(read_json(out / "paired_measures.json")["observations"]) == 36
    assert len(read_json(out / "paired_deltas.json")["observations"]) == 30
    assert len(read_json(out / "editing_core_residuals.json")["observations"]) == 24
    assert result["learned_model_signal_decision"]["oracle_not_localization"] is True
    for variant in ["raw", "rms", "active", "rms_active"]:
        for fold in range(3):
            info = read_json(out / "oracle" / variant / f"fold-{fold}" / "fit.json")
            assert not set(info["fit_groups"]) & set(info["held_out_groups"])
        with np.load(
            out / "oracle" / variant / "features_predictions.npz", allow_pickle=False
        ) as state:
            assert state["features"].shape == (36, 62)
            assert set(state["group_ids"]) == {f"u{i}" for i in range(6)}
    assert artifact_inventory(dataset) == before and artifact_inventory(baseline) == old
    repeated = root / "repeat"
    run_paired_diagnostics(dataset, source, baseline, repeated)
    for variant in ["raw", "rms", "active", "rms_active"]:
        a = artifact_inventory(out / "oracle" / variant)
        b = artifact_inventory(repeated / "oracle" / variant)
        assert a == b
    with pytest.raises(FileExistsError):
        run_paired_diagnostics(dataset, source, baseline, out)


def test_output_must_be_separate_from_sources(paired_dataset):
    _, source, dataset, baseline = paired_dataset
    with pytest.raises(ValueError, match="separate"):
        run_paired_diagnostics(dataset, source, baseline, dataset / "experiment")
