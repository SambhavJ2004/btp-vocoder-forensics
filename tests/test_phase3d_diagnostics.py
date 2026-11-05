"""CPU diagnostics: known signals, source uncertainty and train-only audits."""

from pathlib import Path

import numpy as np
import pytest
from test_paired_signal import paired_dataset as paired_dataset

from data.alignment import AlignmentReport
from data.invariants import ARCHIVE_SR, HIGH_BAND_ANALYSIS, exclusion_for
from detectors.temporal_cnn import GEOMETRY
from detectors.temporal_features import (
    analysis_comparison,
    assign_window_labels,
    extract_windows,
    guarded_labels,
    sample_labels,
    window_starts,
)
from experiments.acoustic_diagnostics import (
    GRID,
    activity_bins,
    balanced_indices,
    correlation,
    group_summary,
    periodicity,
    psd_frames,
    rms_controlled_features,
    smooth_log_psd,
    spectral_pair,
    window_rms,
)
from experiments.partial_audio import artifact_inventory, read_input
from experiments.phase3d import guarded_core_bounds, run, sampling_counts
from experiments.temporal_review import read_json, review_rows


@pytest.mark.parametrize("wave", [[], [1j], [np.nan], [np.inf], np.ones((2, 2))])
def test_invalid_physical_waveforms_fail(wave):
    with pytest.raises(ValueError):
        window_rms(wave, np.array([0]))


def test_activity_boundaries_and_complete_window_geometry():
    assert activity_bins(np.array([0, 0.0099, 0.01, 0.099, 0.1, 1]), 0.01).tolist() == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    np.testing.assert_allclose(window_rms(np.full(2000, 0.1), np.array([0, 100])), 0.1, atol=1e-14)
    for starts in (np.array([-1]), np.array([1000]), np.array([0.1])):
        with pytest.raises(ValueError):
            window_rms(np.ones(2000), starts)
    with pytest.raises(ValueError):
        activity_bins([np.nan], 0.1)
    with pytest.raises(ValueError):
        activity_bins([0.1], 0)


def test_batched_RMS_control_matches_independent_frames_and_gain_invariance():
    wave = np.random.default_rng(3).normal(0, 0.03, 20000)
    starts = np.arange(0, 15000, 353)
    before = wave.copy()
    actual, evidence = rms_controlled_features(wave, starts)
    for i, start in enumerate(starts):
        frame = wave[start : start + GRID.window_samples]
        _, expected = extract_windows(frame * 0.03 / np.sqrt(np.mean(frame**2)), GRID)
        np.testing.assert_allclose(actual[i], expected[0], rtol=0, atol=2e-10)
    other, _ = rms_controlled_features(wave * 4, starts)
    np.testing.assert_allclose(actual, other, rtol=0, atol=2e-10)
    np.testing.assert_array_equal(wave, before)
    silent, info = rms_controlled_features(np.zeros(20000), starts)
    assert np.isfinite(silent).all() and info["unmeasurable_windows_unchanged"] == len(starts)
    assert evidence["unmeasurable_windows_unchanged"] == 0


def test_spectral_gain_units_cepstral_envelope_and_membership():
    real = np.random.default_rng(43).normal(0, 0.1, 18000)
    psd, _ = psd_frames(real, np.array([0, 353, 706]))
    np.testing.assert_allclose(smooth_log_psd(psd, 1024), np.log10(psd), atol=1e-12)
    assert np.max(np.ptp(smooth_log_psd(psd, 0), axis=1)) < 1e-12
    same = spectral_pair(real, real, threshold=0.001, allow_fullband_distance=True)
    assert same["lsd_below8k_db"] == same["fullband_lsd_db"] == same["envelope_below8k_db"] == 0
    assert same["periodicity_difference"] == 0
    gain = spectral_pair(real, 2 * real, threshold=0.001, allow_fullband_distance=True)
    for key in ("rms_ratio_db", "lsd_below8k_db", "envelope_below8k_db"):
        assert gain[key] == pytest.approx(20 * np.log10(2), abs=0.002)
    assert gain["shape_lsd_below8k_db"] < 0.02
    assert all(
        v == pytest.approx(20 * np.log10(2), abs=0.002) for v in gain["band_energy_ratio_db"]
    )
    assert spectral_pair(real, real, threshold=0.001)["fullband_lsd_db"] is None
    inactive = spectral_pair(real, real, threshold=10)
    assert inactive["matched_real_active_frames"] == 0 and inactive["lsd_below8k_db"] is None
    for v in ("griffin_lim", "vocos"):
        assert exclusion_for(HIGH_BAND_ANALYSIS, v) is not None


def test_periodicity_proxy_separates_tone_and_noise_and_marks_silence():
    t = np.arange(12000) / ARCHIVE_SR
    starts = np.arange(0, 10000, 353)
    assert np.mean(periodicity(0.1 * np.sin(2 * np.pi * 200 * t), starts)) > 0.9
    assert np.mean(periodicity(np.random.default_rng(4).normal(0, 0.1, len(t)), starts)) < 0.3
    assert np.isnan(periodicity(np.zeros(len(t)), starts)).all()


def test_uncertainty_has_one_observation_per_source_and_reports_missing_support():
    values = np.linspace(0, 1, 30)
    result = group_summary(values)
    assert result == group_summary(values.copy())
    assert (
        result["count"] == 30
        and result["source_mean_ci95"][0] < 0.5 < result["source_mean_ci95"][1]
    )
    constant = group_summary([0.2] * 20 + [None] * 3)
    np.testing.assert_allclose(constant["source_mean_ci95"], [0.2, 0.2])
    assert constant["missing_source_groups"] == 3
    assert group_summary([1] * 9)["source_mean_ci95"] is None
    assert group_summary([None] * 20)["count"] == 0
    with pytest.raises(ValueError):
        group_summary([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        group_summary([np.inf])
    assert correlation(values, values)["spearman_rho"] == pytest.approx(1)
    assert correlation(values, np.ones(30))["spearman_rho"] is None


@pytest.mark.parametrize("matched", [False, True])
def test_balance_respects_groups_classes_activity_masks_and_input_order(matched):
    records = []
    for group in ("a", "b"):
        for i in range(3):
            labels = np.r_[np.zeros(90, dtype=int), np.ones(60, dtype=int), np.full(5, -1)]
            records.append(
                {
                    "row": {"group_id": group, "split": "train", "example_id": f"{group}{i}"},
                    "window_y": labels,
                    "own_activity": np.arange(155) % 3,
                }
            )
    selected = balanced_indices(records, activity_matched=matched)
    for group in ("a", "b"):
        counts = np.zeros((2, 3), dtype=int)
        for record, take in zip(records, selected, strict=True):
            if record["row"]["group_id"] == group:
                assert np.all(record["window_y"][take] >= 0)
                for label, activity in zip(
                    record["window_y"][take], record["own_activity"][take], strict=True
                ):
                    counts[label, activity] += 1
        assert counts[0].sum() == counts[1].sum() <= 72
        if matched:
            np.testing.assert_array_equal(counts[0], counts[1])
    by_id = {r["row"]["example_id"]: take for r, take in zip(records, selected, strict=True)}
    reversed_records = list(reversed(records))
    for r, take in zip(
        reversed_records, balanced_indices(reversed_records, activity_matched=matched), strict=True
    ):
        np.testing.assert_array_equal(take, by_id[r["row"]["example_id"]])
    records[0]["row"]["split"] = "dev"
    with pytest.raises(ValueError, match="train"):
        balanced_indices(records)


def test_sampling_counts_exclude_ignore():
    result = sampling_counts(np.array([0, 1, -1]), np.array([0, 1, 2]))
    assert (
        sum(result.values()) == 2 and result["0:low_energy"] == 1 and result["1:active_quiet"] == 1
    )


def test_diagnostic_core_excludes_first_closed_guard_sample():
    from data.partial_audio import interval_labels
    from experiments.temporal_review import independent_guard

    row = {
        "num_samples": 60000,
        "start_sample": 20000,
        "end_sample": 42050,
        "crossfade_samples_per_boundary": 220,
        "has_synthetic": True,
        "interval_labels": interval_labels(60000, 20000, 42050, 220, synthetic=True),
    }
    start, end = guarded_core_bounds(row)
    truth = independent_guard(row, 5249)
    assert end - start == 11111
    assert truth[start - 1] == truth[end] == -1
    assert np.all(truth[start:end] == 1)


@pytest.mark.parametrize("gate_passed", [True, False])
def test_driver_train_only_immutable_and_alignment_failure_blocks_fits(
    paired_dataset, monkeypatch, gate_passed
):
    import experiments.partial_audio as audio_module
    import experiments.phase3d as module

    root, source, dataset, baseline = paired_dataset
    originals = {p: artifact_inventory(p) for p in (source.parent, dataset, baseline)}
    forbidden = {
        r["example_id"]
        for r in read_json(dataset / "manifest.json")["examples"]
        if r["split"] != "train"
    } | {"u6", "u7"}

    def checked(path):
        assert Path(path).stem not in forbidden
        return read_input(path)

    monkeypatch.setattr(module, "read_input", checked)
    monkeypatch.setattr(audio_module, "read_input", checked)

    def frozen_stub(data, manifest, base, ablation, cnn):
        selected, identity = review_rows(data, manifest, base)
        records = []
        neural = []
        work = root / f"stub-work-{gate_passed}"
        work.mkdir()
        for group in sorted({r["group_id"] for r in selected["train"]}):
            rows = [r for r in selected["train"] if r["group_id"] == group]
            wavs = {
                ("real" if r["example_class"] == "null_control" else r["example_id"]): checked(
                    data / r["output_filename"]
                )
                for r in rows
            }
            filtered = analysis_comparison(wavs, work)
            for row in rows:
                key = "real" if row["example_class"] == "null_control" else row["example_id"]
                starts, features = extract_windows(filtered[key], GRID)
                truth = guarded_labels(
                    sample_labels(row["interval_labels"], row["num_samples"]), 3308
                )
                records.append(
                    {
                        "row": row,
                        "starts": starts,
                        "features": features,
                        "window_y": assign_window_labels(truth, starts, GRID.window_samples),
                        "margin": np.sin(starts / 10000),
                    }
                )
                ns = window_starts(row["num_samples"], GEOMETRY)
                neural.append(
                    {
                        "row": row,
                        "starts": ns,
                        "window_y": assign_window_labels(truth, ns, GEOMETRY.window_samples),
                        "margin": np.cos(ns / 10000),
                        "sampled_mask": np.arange(len(ns)) % 5 == 0,
                    }
                )
        models = {
            "A64_OOF": (records, GRID),
            "CNN_OOF": (neural, GEOMETRY),
            "A64_final_in_sample": ([{**r} for r in records], GRID),
            "CNN_final_in_sample": ([{**r} for r in neural], GEOMETRY),
        }
        return selected["train"], identity, models, {"A64": 0.6, "CNN": 0.6}

    monkeypatch.setattr(module, "frozen_inputs", frozen_stub)
    monkeypatch.setattr(
        module,
        "check_vocoder_alignment",
        lambda pairs, v: AlignmentReport(
            condition=v, n_files=len(pairs), peak_lag=0 if gate_passed else 1, passed=gate_passed
        ),
    )
    target = root / f"phase3d-{gate_passed}"
    result = run(dataset, source, baseline, root / "unused-ablation", root / "unused-cnn", target)
    assert result["status"] == "completed"
    receipt = read_json(target / "validation.json")
    assert receipt["eval_waveforms_opened"] == receipt["dev_waveforms_decoded"] == 0
    assert receipt["train_groups"] == 6 and receipt["training_outputs_reconstructed"] == 30
    assert receipt["alignment_gates_all_passed"] == gate_passed
    nuisance = read_json(target / "nuisance_summary.json")
    if gate_passed:
        assert set(nuisance) - {"citable", "development_only"} == {
            "balanced",
            "activity_matched",
            "rms_normalized",
            "activity_rms",
        }
        for variant in ("balanced", "activity_matched", "rms_normalized", "activity_rms"):
            for fold in nuisance[variant]["folds"]:
                assert not set(fold["fit_groups"]) & set(fold["held_out_groups"])
    else:
        assert (
            nuisance["status"] == "not_run_alignment_gate_failed"
            and not (target / "nuisance").exists()
        )
    assert artifact_inventory(target) == read_json(target / "artifacts.sha256.json")["files"]
    for path, inventory in originals.items():
        assert artifact_inventory(path) == inventory
    with pytest.raises(FileExistsError):
        run(dataset, source, baseline, root / "unused-ablation", root / "unused-cnn", target)
