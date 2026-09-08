"""Tests for the confound controls.

These are not unit tests for their own sake: each one asserts that a specific
invariant from CLAUDE.md actually fires. A silently-disabled check is
indistinguishable from a correct pipeline until the results are already wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.invariants import (
    ARCHIVE_SR,
    ARCHIVE_TIER,
    BAND_LIMIT_FILTER_SPEC,
    BAND_LIMIT_MAX_STOPBAND_ENERGY,
    LADDER_FMAX,
    LOUDNESS_TARGET_LUFS,
    ZEROSHOT_SR,
    ZEROSHOT_TIER,
    InvariantViolation,
    check_pairing,
    check_waveform,
    is_band_exempt,
    tier_for_rate,
)
from data.preprocess import TrimSpan, align_length, apply_trim_span, resample_once

REPO_ROOT = Path(__file__).resolve().parents[1]


def _quiet_tone(seconds: float = 1.0, sr: int = ARCHIVE_SR, amp: float = 0.05) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class TestINV01Rates:
    def test_unsanctioned_rate_rejected(self):
        with pytest.raises(InvariantViolation, match="INV-01"):
            check_waveform(_quiet_tone(), 44_100, where="test")

    def test_both_sanctioned_rates_accepted(self):
        check_waveform(_quiet_tone(sr=ARCHIVE_SR), ARCHIVE_SR, where="test")
        check_waveform(_quiet_tone(sr=ZEROSHOT_SR), ZEROSHOT_SR, where="test")

    def test_expected_sr_pins_the_tier(self):
        # A valid zero-shot file is still wrong where the archive was expected.
        with pytest.raises(InvariantViolation, match="INV-01"):
            check_waveform(
                _quiet_tone(sr=ZEROSHOT_SR), ZEROSHOT_SR, where="test", expected_sr=ARCHIVE_SR
            )

    def test_stereo_rejected(self):
        with pytest.raises(InvariantViolation, match="INV-01"):
            check_waveform(np.zeros((100, 2), dtype=np.float32), ARCHIVE_SR, where="test")

    def test_tier_lookup(self):
        assert tier_for_rate(ARCHIVE_SR) == ARCHIVE_TIER
        assert tier_for_rate(ZEROSHOT_SR) == ZEROSHOT_TIER
        with pytest.raises(InvariantViolation, match="INV-01"):
            tier_for_rate(48_000)

    def test_resample_to_archive_is_a_single_operation(self):
        wav = _quiet_tone(0.5, sr=24_000)
        out = resample_once(wav, 24_000)
        assert abs(len(out) - int(0.5 * ARCHIVE_SR)) <= 2

    def test_ljspeech_native_needs_no_resampling_filter(self):
        # The archive rate is chosen so this path applies no filter at all.
        wav = _quiet_tone(0.5, sr=ARCHIVE_SR)
        assert np.array_equal(resample_once(wav, ARCHIVE_SR), wav)


class TestINV01Derivation:
    """Archive high, derive low, exactly one step each."""

    def test_derive_zeroshot_from_archive(self):
        """The production path: normalise at the archive tier, then derive once."""
        from data.preprocess import derive_zeroshot, normalise_loudness

        wav, archive_lufs = normalise_loudness(_quiet_tone(3.0, sr=ARCHIVE_SR), sr=ARCHIVE_SR)
        out, measured = derive_zeroshot(wav, archive_lufs=archive_lufs)
        assert abs(len(out) - 3 * ZEROSHOT_SR) <= 2
        # Not re-normalised: it lands near the parent, but is not forced onto it.
        assert abs(measured - archive_lufs) < 2.0

    def test_derive_zeroshot_refuses_a_non_archive_source(self):
        from data.preprocess import derive_zeroshot

        with pytest.raises(InvariantViolation, match="INV-01"):
            derive_zeroshot(_quiet_tone(sr=24_000), archive_sr=24_000)

    def test_chained_double_resample_is_rejected(self):
        """A record that has been through two resamples must not validate.

        This is the failure the two-tier design is most exposed to: someone
        regenerates the zero-shot set from an already-derived file, or routes
        source through an intermediate rate. Both stack a second anti-aliasing
        imprint that no manifest column describes.
        """
        from data.manifest import validate_manifest

        df = _manifest_frame()
        df.loc[df["tier"] == ZEROSHOT_TIER, "resample_steps"] = 2
        with pytest.raises(InvariantViolation, match="INV-01.*resample step"):
            validate_manifest(df)

    def test_zeroshot_derived_from_source_is_rejected(self):
        from data.manifest import validate_manifest

        df = _manifest_frame()
        df.loc[df["tier"] == ZEROSHOT_TIER, "source_rate"] = 22_050 * 0 + 44_100
        with pytest.raises(InvariantViolation, match="INV-01"):
            validate_manifest(df)

    def test_archive_row_marked_derived_is_rejected(self):
        from data.manifest import validate_manifest

        df = _manifest_frame()
        df.loc[df["tier"] == ARCHIVE_TIER, "derived"] = True
        with pytest.raises(InvariantViolation, match="INV-01"):
            validate_manifest(df)

    def test_tier_rate_disagreement_is_rejected(self):
        from data.manifest import validate_manifest

        df = _manifest_frame()
        df.loc[df["tier"] == ZEROSHOT_TIER, "sr_out"] = ARCHIVE_SR
        with pytest.raises(InvariantViolation, match="INV-01"):
            validate_manifest(df)

    def test_require_primary_refuses_derived_rows(self):
        from data.manifest import require_primary

        df = _manifest_frame()
        with pytest.raises(InvariantViolation, match="INV-01"):
            require_primary(df, why="the band-limited ablation")

    def test_require_primary_accepts_archive_only(self):
        from data.manifest import require_primary, select_tier

        df = _manifest_frame()
        require_primary(select_tier(df, ARCHIVE_TIER), why="band-wise metrics")


class TestINV03Silence:
    def test_reference_span_applied_by_index(self):
        span = TrimSpan(utt_id="u1", start=100, end=900, ref_length=1000)
        wav = np.arange(1000, dtype=np.float32)
        assert np.array_equal(apply_trim_span(wav, span), wav[100:900])

    def test_span_refuses_mismatched_length(self):
        span = TrimSpan(utt_id="u1", start=100, end=900, ref_length=1000)
        with pytest.raises(InvariantViolation, match="INV-03"):
            apply_trim_span(np.zeros(1234, dtype=np.float32), span)


class TestINV04Loudness:
    def test_clipping_file_rejected(self):
        wav = np.ones(ARCHIVE_SR, dtype=np.float32)  # 0 dBFS
        with pytest.raises(InvariantViolation, match="INV-04"):
            check_waveform(wav, ARCHIVE_SR, where="test")

    def test_loudness_drift_rejected(self):
        with pytest.raises(InvariantViolation, match="INV-04"):
            check_waveform(
                _quiet_tone(), ARCHIVE_SR, where="test",
                measured_lufs=LOUDNESS_TARGET_LUFS + 5.0,
            )

    def test_on_target_accepted(self):
        check_waveform(
            _quiet_tone(), ARCHIVE_SR, where="test", measured_lufs=LOUDNESS_TARGET_LUFS
        )

    def test_derived_tier_tolerates_band_limiting_drift(self):
        from data.invariants import check_derived_loudness

        check_derived_loudness(LOUDNESS_TARGET_LUFS - 0.8, where="test")

    def test_derived_tier_rejects_a_second_gain_stage(self):
        from data.invariants import check_derived_loudness

        with pytest.raises(InvariantViolation, match="INV-04"):
            check_derived_loudness(LOUDNESS_TARGET_LUFS - 6.0, where="test")


class TestINV05Encoding:
    def test_non_finite_rejected(self):
        wav = _quiet_tone()
        wav[10] = np.nan
        with pytest.raises(InvariantViolation, match="INV-05"):
            check_waveform(wav, ARCHIVE_SR, where="test")


class TestINV06Pairing:
    def test_unequal_sets_rejected(self):
        with pytest.raises(InvariantViolation, match="INV-06"):
            check_pairing({"real": {"a", "b", "c"}, "hifigan_v1": {"a", "b"}})

    def test_missing_real_condition_rejected(self):
        with pytest.raises(InvariantViolation, match="INV-06"):
            check_pairing({"hifigan_v1": {"a", "b"}})

    def test_paired_sets_accepted(self):
        check_pairing({"real": {"a", "b"}, "hifigan_v1": {"a", "b"}})

    def test_align_length_pads_and_truncates(self):
        assert len(align_length(np.zeros(900, dtype=np.float32), 1000)) == 1000
        assert len(align_length(np.zeros(1100, dtype=np.float32), 1000)) == 1000

    def test_align_length_refuses_large_drift(self):
        with pytest.raises(InvariantViolation, match="INV-06"):
            align_length(np.zeros(ARCHIVE_SR, dtype=np.float32), 100)

    def test_align_tolerance_is_50ms_at_either_rate(self):
        # 40 ms of drift is inside tolerance at both rates; 60 ms is outside.
        for sr in (ARCHIVE_SR, ZEROSHOT_SR):
            ref = sr
            align_length(np.zeros(ref + int(0.04 * sr), dtype=np.float32), ref, sr=sr)
            with pytest.raises(InvariantViolation, match="INV-06"):
                align_length(np.zeros(ref + int(0.06 * sr), dtype=np.float32), ref, sr=sr)


class TestINV14Protocols:
    def test_mismatched_with_training_conditions_rejected(self):
        from detectors.protocols import DetectionResult, Protocol

        with pytest.raises(InvariantViolation, match="INV-14"):
            DetectionResult(
                protocol=Protocol.MISMATCHED, detector="aasist", condition="hifigan_v1",
                eer=0.1, threshold=0.0, n_bonafide=10, n_spoof=10,
                train_conditions=("real", "hifigan_v1"),
            )

    def test_matched_without_training_conditions_rejected(self):
        from detectors.protocols import DetectionResult, Protocol

        with pytest.raises(InvariantViolation, match="INV-14"):
            DetectionResult(
                protocol=Protocol.MATCHED, detector="aasist", condition="hifigan_v1",
                eer=0.1, threshold=0.0, n_bonafide=10, n_spoof=10,
            )

    def test_unknown_tier_rejected(self):
        from detectors.protocols import DetectionResult, Protocol

        with pytest.raises(InvariantViolation, match="INV-01"):
            DetectionResult(
                protocol=Protocol.MISMATCHED, detector="aasist", condition="hifigan_v1",
                eer=0.1, threshold=0.0, n_bonafide=10, n_spoof=10, tier="cd_quality",
            )

    def test_bandlimit_above_tier_nyquist_rejected(self):
        from detectors.protocols import DetectionResult, Protocol

        with pytest.raises(InvariantViolation, match="INV-01"):
            DetectionResult(
                protocol=Protocol.MATCHED, detector="aasist", condition="hifigan_v1",
                eer=0.1, threshold=0.0, n_bonafide=10, n_spoof=10,
                tier=ARCHIVE_TIER, train_conditions=("real", "hifigan_v1"),
                band_limit_hz=12_000.0,
            )

    def test_bandlimit_on_derived_tier_rejected(self):
        from detectors.protocols import DetectionResult, Protocol

        with pytest.raises(InvariantViolation, match="INV-01"):
            DetectionResult(
                protocol=Protocol.MATCHED, detector="aasist", condition="hifigan_v1",
                eer=0.1, threshold=0.0, n_bonafide=10, n_spoof=10,
                tier=ZEROSHOT_TIER, train_conditions=("real", "hifigan_v1"),
                band_limit_hz=6_000.0,
            )

    def test_unlabelled_results_frame_rejected(self):
        from detectors.protocols import assert_protocols_not_pooled

        with pytest.raises(InvariantViolation, match="INV-14"):
            assert_protocols_not_pooled(pd.DataFrame({"eer": [0.1], "condition": ["x"]}))

    def test_results_frame_without_tier_rejected(self):
        from detectors.protocols import assert_protocols_not_pooled

        with pytest.raises(InvariantViolation, match="INV-01"):
            assert_protocols_not_pooled(
                pd.DataFrame({"eer": [0.1], "condition": ["x"], "protocol": ["matched"]})
            )


class TestINV16Alignment:
    """The probe must catch a real delay without false-blocking Griffin-Lim.

    Fixtures matter here more than usual. Two earlier fixture designs produced
    misleading results and are worth naming so they are not reintroduced:
      - a stationary tone has a flat envelope, so envelope cross-correlation is
        degenerate and its peak is decided by noise;
      - whole-signal phase randomisation destroys the temporal envelope, which
        real Griffin-Lim preserves, because Griffin-Lim works per STFT frame.
    So the phase-randomising fixture is actual Griffin-Lim.
    """

    def _speechlike(self, seconds: float = 0.7, seed: int = 0) -> np.ndarray:
        """Harmonic carrier under syllable bursts with sharp onsets.

        Sharp onsets give the envelope autocorrelation a narrow peak; a smooth
        envelope would leave it broad enough for noise to pick the argmax.
        """
        n = int(seconds * ARCHIVE_SR)
        rng = np.random.default_rng(seed)
        t = np.arange(n) / ARCHIVE_SR
        carrier = sum(0.3 / k * np.sin(2 * np.pi * (140 + 3 * seed) * k * t) for k in range(1, 12))
        env = np.zeros(n)
        burst, gap, ramp = int(0.08 * ARCHIVE_SR), int(0.045 * ARCHIVE_SR), int(0.004 * ARCHIVE_SR)
        pos = int(0.02 * ARCHIVE_SR)
        while pos + burst < n:
            seg = np.ones(burst)
            seg[:ramp] = np.linspace(0, 1, ramp)
            seg[-ramp:] = np.linspace(1, 0, ramp)
            env[pos : pos + burst] = seg
            pos += burst + gap
        return (0.2 * carrier * env + 0.002 * rng.standard_normal(n)).astype(np.float32)

    def _griffin_lim(self, x: np.ndarray, n_iter: int = 8) -> np.ndarray:
        """Magnitude preserved, phase reconstructed -- no waveform correspondence."""
        import librosa

        mag = np.abs(librosa.stft(x, n_fft=1024, hop_length=256))
        return librosa.griffinlim(
            mag, n_iter=n_iter, hop_length=256, length=len(x)
        ).astype(np.float32)

    def _identity_pairs(self, shift: int = 0):
        out = []
        for i in range(20):
            ref = self._speechlike(seed=i)
            out.append((ref, np.roll(ref, shift).astype(np.float32)))
        return out

    def _griffin_lim_pairs(self, shift: int = 0):
        out = []
        for i in range(20):
            ref = self._speechlike(seed=i)
            deg = self._griffin_lim(ref)
            out.append((ref, np.roll(deg, shift).astype(np.float32) if shift else deg))
        return out

    def test_aligned_phase_preserving_vocoder_passes(self):
        from data.alignment import check_vocoder_alignment

        report = check_vocoder_alignment(self._identity_pairs(), "aligned_voc")
        assert report.peak_lag == 0
        assert report.waveform_modal_lag == 0
        assert report.phase_preserving
        assert report.waveform_gate_applied  # unambiguous, so it binds
        assert report.passed

    def test_fixed_delay_is_detected(self):
        from data.alignment import check_vocoder_alignment

        report = check_vocoder_alignment(self._identity_pairs(shift=64), "delayed_voc")
        # The waveform measurement recovers the shift exactly where phase is
        # preserved; the envelope peak lands near it, bounded by onset sharpness.
        assert report.waveform_modal_lag == 64
        assert report.peak_lag != 0
        assert not report.passed

    def test_assert_aligned_blocks_a_delayed_condition(self):
        from data.alignment import assert_aligned, check_vocoder_alignment

        report = check_vocoder_alignment(self._identity_pairs(shift=64), "delayed_voc")
        with pytest.raises(InvariantViolation, match="INV-16"):
            assert_aligned(report)

    def test_griffin_lim_passes_on_envelope(self):
        """Discarding phase is Griffin-Lim's definition, not a misalignment.

        Its waveform correlation with the reference is near noise, so a
        waveform-only gate would block it permanently for the property that puts
        it on the quality ladder in the first place.
        """
        from data.alignment import assert_aligned, check_vocoder_alignment

        report = check_vocoder_alignment(self._griffin_lim_pairs(), "griffin_lim")
        # Its waveform lags do not agree across files, so that measurement is
        # noise rather than a delay and the waveform gate does not bind.
        assert not report.waveform_gate_applied
        assert report.waveform_modal_share < 0.6
        assert report.peak_lag == 0
        assert report.envelope_modal_share == 1.0
        assert report.mean_envelope_correlation > 0.9
        assert report.passed
        assert_aligned(report)

    def test_delayed_griffin_lim_is_still_blocked(self):
        """The envelope gate must not become a free pass for phase-free vocoders."""
        from data.alignment import assert_aligned, check_vocoder_alignment

        report = check_vocoder_alignment(self._griffin_lim_pairs(shift=512), "delayed_gl")
        assert not report.waveform_gate_applied
        assert report.peak_lag != 0
        assert not report.passed
        with pytest.raises(InvariantViolation, match="INV-16"):
            assert_aligned(report)

    def test_envelope_window_suppresses_pitch_locking(self):
        """A short envelope window locks onto a pitch period and lies.

        This is the measurement artifact that made the first version of this
        probe raise a false blocker on Griffin-Lim.
        """
        from data.alignment import cross_correlation_lag, energy_envelope

        ref = self._speechlike(seed=3)
        deg = self._griffin_lim(ref)
        short, _ = cross_correlation_lag(
            energy_envelope(ref, ARCHIVE_SR, 2.0), energy_envelope(deg, ARCHIVE_SR, 2.0)
        )
        long, long_r = cross_correlation_lag(
            energy_envelope(ref, ARCHIVE_SR, 50.0), energy_envelope(deg, ARCHIVE_SR, 50.0)
        )
        assert long == 0
        assert long_r > 0.9
        assert abs(short) >= abs(long)

    def test_too_few_probe_files_rejected(self):
        from data.alignment import check_vocoder_alignment

        with pytest.raises(InvariantViolation, match="INV-16"):
            check_vocoder_alignment(self._identity_pairs()[:1], "tiny_probe")

    def test_manifest_gate_requires_an_alignment_record(self):
        from data.manifest import validate_manifest

        df = _manifest_frame()
        df.loc[df["condition"] == "hifigan_v1", "alignment_checked"] = False
        with pytest.raises(InvariantViolation, match="INV-16"):
            validate_manifest(df)

    def test_manifest_gate_rejects_non_zero_lag(self):
        from data.manifest import validate_manifest

        df = _manifest_frame()
        df.loc[df["condition"] == "hifigan_v1", "alignment_peak_lag"] = 3
        with pytest.raises(InvariantViolation, match="INV-16"):
            validate_manifest(df)


class TestEER:
    def test_perfect_separation(self):
        from detectors.eer import compute_eer

        eer, _ = compute_eer(np.ones(100) * 5, np.ones(100) * -5)
        assert eer == pytest.approx(0.0, abs=1e-6)

    def test_chance_on_identical_distributions(self):
        from detectors.eer import compute_eer

        rng = np.random.default_rng(0)
        eer, _ = compute_eer(rng.normal(size=2000), rng.normal(size=2000))
        assert 0.4 < eer < 0.6

    def test_eer_is_a_fraction_not_a_percentage(self):
        from detectors.eer import compute_eer

        rng = np.random.default_rng(1)
        eer, _ = compute_eer(rng.normal(1, 1, 500), rng.normal(-1, 1, 500))
        assert 0.0 <= eer <= 1.0


def _manifest_frame(**overrides) -> pd.DataFrame:
    """A minimal valid two-tier manifest: 3 utterances, real + hifigan_v1."""
    rows = []
    for tier, sr, derived, src, deriv in (
        (ARCHIVE_TIER, ARCHIVE_SR, False, 22_050, "resample_once:22050->22050"),
        (ZEROSHOT_TIER, ZEROSHOT_SR, True, ARCHIVE_SR, "downsample_once:22050->16000"),
    ):
        for cond in ("real", "hifigan_v1"):
            for i in range(3):
                rows.append(
                    {
                        "utt_id": f"u{i}", "condition": cond, "tier": tier,
                        "path": f"{tier}/{cond}/u{i}.wav", "speaker": "LJ", "split": "train",
                        "derived": derived, "source_rate": src, "derivation": deriv,
                        "resample_steps": 1, "sr_out": sr, "subtype": "PCM_16",
                        "duration_s": 1.0, "lufs_out": LOUDNESS_TARGET_LUFS,
                        "trim_start": 10, "trim_end": 900, "trim_ref_length": 1000,
                        "mel_n_fft": 1024, "mel_hop": 256, "mel_win": 1024,
                        "mel_n_mels": 80, "mel_fmin": 0.0, "mel_fmax": LADDER_FMAX,
                        "alignment_checked": cond != "real",
                        "alignment_peak_lag": None if cond == "real" else 0,
                        "band_limit_hz": LADDER_FMAX,
                        "band_filter": BAND_LIMIT_FILTER_SPEC,
                        "band_exempt": False,
                        "band_oob_energy": 1e-7,
                        "vocoder_checkpoint": "", "vocoder_commit": "", "seed": 1,
                    }
                )
    df = pd.DataFrame(rows)
    for k, v in overrides.items():
        df[k] = v
    return df


class TestManifest:
    def test_valid_manifest_passes(self):
        from data.manifest import validate_manifest

        validate_manifest(_manifest_frame())

    def test_per_condition_trim_spans_rejected(self):
        from data.manifest import validate_manifest

        df = _manifest_frame()
        df.loc[df["condition"] == "hifigan_v1", "trim_start"] = 50
        with pytest.raises(InvariantViolation, match="INV-03"):
            validate_manifest(df)

    def test_split_leak_across_conditions_rejected(self):
        from data.manifest import validate_manifest

        df = _manifest_frame()
        df.loc[df["condition"] == "hifigan_v1", "split"] = "eval"
        with pytest.raises(InvariantViolation, match="INV-09"):
            validate_manifest(df)

    def test_missing_zeroshot_tier_rejected_when_required(self):
        from data.manifest import validate_manifest

        df = _manifest_frame()
        df = df[df["tier"] == ARCHIVE_TIER]
        validate_manifest(df)  # fine on its own
        with pytest.raises(InvariantViolation, match="INV-01"):
            validate_manifest(df, require_zeroshot=True)

    def test_select_tier_round_trips(self):
        from data.manifest import select_tier

        df = _manifest_frame()
        assert set(select_tier(df, ARCHIVE_TIER)["sr_out"]) == {ARCHIVE_SR}
        assert set(select_tier(df, ZEROSHOT_TIER)["sr_out"]) == {ZEROSHOT_SR}

    def test_provenance_table_shows_both_derivations(self):
        from data.manifest import provenance_table

        tbl = provenance_table(_manifest_frame())
        assert set(tbl["tier"]) == {ARCHIVE_TIER, ZEROSHOT_TIER}
        assert (tbl["resample_steps"] == 1).all()


class TestLadder:
    def test_registry_covers_every_condition(self):
        from vocoders.registry import ALL_CONDITIONS, SPECS, UNAVAILABLE, get_spec

        assert set(ALL_CONDITIONS) | set(UNAVAILABLE) == set(SPECS)
        for key in SPECS:
            assert get_spec(key).key == key

    def test_eight_generated_conditions_six_rungs_one_unavailable(self):
        from vocoders.registry import (
            ALL_CONDITIONS,
            CONTROL_CONDITIONS,
            LADDER,
            UNAVAILABLE,
        )

        assert len(LADDER) == 6
        assert len(CONTROL_CONDITIONS) == 2
        assert len(ALL_CONDITIONS) == 8
        assert not set(LADDER) & set(CONTROL_CONDITIONS)
        # Unavailable conditions are declared but never generated.
        assert "specdiff_gan" in UNAVAILABLE
        assert "specdiff_gan" not in ALL_CONDITIONS

    def test_vocos_replaced_specdiff_in_the_ladder(self):
        from vocoders.registry import LADDER, UNAVAILABLE

        assert "vocos" in LADDER
        assert "specdiff_gan" not in LADDER
        assert "specdiff_gan" in UNAVAILABLE

    def test_audit_reports_which_pin_mechanism_is_in_force(self):
        from vocoders.registry import checkpoint_audit

        rows = {r["condition"]: r for r in checkpoint_audit()}
        assert rows["griffin_lim"]["pin"] == "n/a"
        assert rows["griffin_lim"]["blocked"] is False

        # HuggingFace-hosted: pinned by revision SHA.
        for key in ("vocos", "bigvgan_base", "bigvgan_112m",
                    "bigvgan_v2_22khz_fullband"):
            assert rows[key]["pin"] == "hf-revision", key
            assert rows[key]["blocked"] is False, key

        # Not on HuggingFace: no revision exists, so still unpinned.
        for key in ("hifigan_v1", "melgan", "melgan_fullband"):
            assert rows[key]["pin"] == "UNPINNED", key
            assert rows[key]["blocked"] is True, key

        # Declared but not generable.
        assert rows["specdiff_gan"]["role"] == "unavailable"
        assert rows["specdiff_gan"]["blocked"] is True

    def test_audit_can_exclude_unavailable(self):
        from vocoders.registry import checkpoint_audit

        keys = {r["condition"] for r in checkpoint_audit(include_unavailable=False)}
        assert "specdiff_gan" not in keys

    def test_exemption_is_declared_consistently(self):
        from data.invariants import BAND_EXEMPT_CONDITIONS
        from vocoders.registry import CONTROL_CONDITIONS, SPECS

        # registry.py cross-checks these at import; assert it stays true.
        assert set(CONTROL_CONDITIONS) == set(BAND_EXEMPT_CONDITIONS)
        assert {k for k, v in SPECS.items() if not v.primary_ladder} == set(
            BAND_EXEMPT_CONDITIONS
        )


class TestBandlimit:
    def test_cutoff_above_archive_nyquist_rejected(self):
        from detectors.bandlimit import lowpass

        with pytest.raises(InvariantViolation):
            lowpass(_quiet_tone(), 12_000.0)

    def test_eight_khz_cutoff_is_now_legal(self):
        """The whole point of the rate change: 8 kHz is inside the band."""
        from detectors.bandlimit import lowpass

        out = lowpass(_quiet_tone(), 8_000.0)
        assert len(out) == len(_quiet_tone())

    def test_sweep_stays_inside_the_retained_band(self):
        from detectors.bandlimit import cutoffs_for_rate, nyquist

        for sr in (ARCHIVE_SR, ZEROSHOT_SR):
            cuts = cutoffs_for_rate(sr)
            assert cuts, f"no cutoffs available at {sr}"
            assert max(cuts) < nyquist(sr)

    def test_ladder_sweep_is_capped_at_the_ladder_band(self):
        """INV-17 consequence: above LADDER_FMAX there is nothing left to remove."""
        from detectors.bandlimit import cutoffs_for_rate

        cuts = cutoffs_for_rate(ARCHIVE_SR)
        assert cuts and max(cuts) < LADDER_FMAX

    def test_exempt_sweep_runs_to_nyquist(self):
        from detectors.bandlimit import cutoffs_for_rate, nyquist

        cuts = cutoffs_for_rate(ARCHIVE_SR, band_exempt=True)
        assert max(cuts) > LADDER_FMAX
        assert max(cuts) < nyquist(ARCHIVE_SR)

    def test_lowpass_attenuates_above_cutoff(self):
        from detectors.bandlimit import lowpass

        t = np.arange(ARCHIVE_SR) / ARCHIVE_SR
        high = (0.1 * np.sin(2 * np.pi * 9000 * t)).astype(np.float32)
        out = lowpass(high, 1000.0)
        # Measured over the steady-state interior: filtfilt leaves a transient at
        # both edges, which is a property of the filter, not of the passband.
        edge = ARCHIVE_SR // 10
        assert np.abs(out[edge:-edge]).max() < 0.01 * np.abs(high).max()


class TestINV17LadderBand:
    """The ladder band: derived, applied to real, and enforced downstream."""

    def _wideband(self, seconds: float = 1.0, sr: int = ARCHIVE_SR) -> np.ndarray:
        """A signal with real energy above LADDER_FMAX, like a 22.05 kHz recording."""
        t = np.arange(int(seconds * sr)) / sr
        low = 0.05 * np.sin(2 * np.pi * 300 * t)
        high = 0.05 * np.sin(2 * np.pi * 9500 * t)  # above LADDER_FMAX
        return (low + high).astype(np.float32)

    # --- derivation -------------------------------------------------------
    def test_ladder_fmax_is_derived_from_the_audit_table(self):
        """Not a literal: it is min(audited fmax) over the constraining rungs."""
        from data.invariants import (
            AUDITED_MEL_FMAX,
            FMAX_FREE_CONDITIONS,
            PRIMARY_LADDER,
        )

        constraining = [
            f
            for c, f in AUDITED_MEL_FMAX.items()
            if c in PRIMARY_LADDER and c not in FMAX_FREE_CONDITIONS
        ]
        assert LADDER_FMAX == min(constraining)

    def test_ladder_fmax_follows_the_table_if_a_checkpoint_is_resourced(self):
        """A ParallelWaveGAN MelGAN (fmax 7600) must drag the ladder band down."""
        from data.invariants import (
            AUDITED_MEL_FMAX,
            FMAX_FREE_CONDITIONS,
            PRIMARY_LADDER,
            _derive_ladder_fmax,
        )

        original = AUDITED_MEL_FMAX["melgan"]
        try:
            AUDITED_MEL_FMAX["melgan"] = 7600.0
            assert _derive_ladder_fmax() == 7600.0
        finally:
            AUDITED_MEL_FMAX["melgan"] = original
        assert _derive_ladder_fmax() == LADDER_FMAX
        assert set(FMAX_FREE_CONDITIONS) <= set(PRIMARY_LADDER)

    def test_griffin_lim_follows_the_band_rather_than_setting_it(self):
        from data.mel import GRIFFIN_LIM_MEL

        assert GRIFFIN_LIM_MEL.fmax == LADDER_FMAX

    def test_mel_configs_come_from_the_audit_table(self):
        from data.invariants import AUDITED_MEL_FMAX
        from vocoders.registry import SPECS

        for key, fmax in AUDITED_MEL_FMAX.items():
            assert SPECS[key].mel.fmax == fmax, key

    def test_ladder_is_fmax_agnostic(self):
        """Vocos entered at fmax=12000 without moving the ladder band."""
        from data.invariants import AUDITED_MEL_FMAX

        assert AUDITED_MEL_FMAX["vocos"] == 12_000.0
        assert AUDITED_MEL_FMAX["vocos"] > LADDER_FMAX
        assert LADDER_FMAX == 8000.0

    def test_unavailable_conditions_do_not_constrain_the_band(self):
        """specdiff_gan is audited but never generated, so it cannot set the band."""
        from data.invariants import (
            AUDITED_MEL_FMAX,
            UNAVAILABLE_CONDITIONS,
            _derive_ladder_fmax,
        )

        assert "specdiff_gan" in UNAVAILABLE_CONDITIONS
        original = AUDITED_MEL_FMAX["specdiff_gan"]
        try:
            AUDITED_MEL_FMAX["specdiff_gan"] = 4000.0
            assert _derive_ladder_fmax() == LADDER_FMAX  # unmoved
        finally:
            AUDITED_MEL_FMAX["specdiff_gan"] = original

    # --- the filter itself ------------------------------------------------
    def test_band_limiting_removes_the_high_band(self):
        from data.invariants import out_of_band_energy
        from data.preprocess import band_limit_to_ladder

        wav = self._wideband()
        before = out_of_band_energy(wav, ARCHIVE_SR, LADDER_FMAX)
        after = out_of_band_energy(
            band_limit_to_ladder(wav, sr=ARCHIVE_SR), ARCHIVE_SR, LADDER_FMAX
        )
        assert before > 0.1
        assert after <= BAND_LIMIT_MAX_STOPBAND_ENERGY

    def test_band_measurement_is_windowed(self):
        """A bare rFFT reports its own spectral leakage as out-of-band energy.

        Regression guard: with a rectangular window, a strong low-frequency tone
        leaks across the spectrum at roughly -50 dB, which is far above the 1e-6
        gate. That made a correctly filtered signal look unfiltered.
        """
        from data.invariants import out_of_band_energy
        from data.preprocess import band_limit_to_ladder

        t = np.arange(ARCHIVE_SR) / ARCHIVE_SR
        tone = (0.05 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        filtered = band_limit_to_ladder(tone, sr=ARCHIVE_SR)

        windowed = out_of_band_energy(filtered, ARCHIVE_SR, LADDER_FMAX)

        x = filtered.astype(np.float64)
        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), d=1.0 / ARCHIVE_SR)
        rectangular = float(spec[freqs > LADDER_FMAX].sum() / spec.sum())

        assert windowed <= BAND_LIMIT_MAX_STOPBAND_ENERGY
        assert rectangular > windowed * 100

    def test_gate_rejects_an_unfiltered_condition(self):
        from data.invariants import check_band_limit

        with pytest.raises(InvariantViolation, match="INV-17"):
            check_band_limit(self._wideband(), ARCHIVE_SR, where="test")

    def test_band_limiting_preserves_sample_alignment(self):
        """Zero-phase, so INV-16 still holds after INV-17 runs."""
        from data.alignment import cross_correlation_lag
        from data.preprocess import band_limit_to_ladder

        wav = self._wideband(2.0)
        lag, _ = cross_correlation_lag(wav, band_limit_to_ladder(wav, sr=ARCHIVE_SR))
        assert lag == 0

    # --- real is filtered identically -------------------------------------
    def test_real_and_vocoded_are_band_limited_identically(self):
        """The central INV-17 claim: filtering only the fakes flips the cliff.

        Two signals that differ only above the ladder band -- as real audio and
        an fmax=8000 vocoder do -- must be indistinguishable on high-band energy
        once both have been through the same filter.
        """
        from data.invariants import out_of_band_energy
        from data.preprocess import band_limit_to_ladder

        t = np.arange(ARCHIVE_SR) / ARCHIVE_SR
        shared = (0.05 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        real = shared + (0.05 * np.sin(2 * np.pi * 9500 * t)).astype(np.float32)
        vocoded = shared.copy()  # an fmax=8000 vocoder: nothing above the band

        assert out_of_band_energy(real, ARCHIVE_SR, LADDER_FMAX) > 0.1
        assert out_of_band_energy(vocoded, ARCHIVE_SR, LADDER_FMAX) < 1e-9

        real_b = band_limit_to_ladder(real, sr=ARCHIVE_SR)
        voc_b = band_limit_to_ladder(vocoded, sr=ARCHIVE_SR)
        for w in (real_b, voc_b):
            assert out_of_band_energy(w, ARCHIVE_SR, LADDER_FMAX) <= (
                BAND_LIMIT_MAX_STOPBAND_ENERGY
            )

    def test_pipeline_order_places_band_limit_before_loudness(self):
        from data.invariants import PIPELINE_ORDER, PIPELINE_ORDER_BAND_EXEMPT

        assert PIPELINE_ORDER.index("band_limit") > PIPELINE_ORDER.index("apply_ref_trim")
        assert PIPELINE_ORDER.index("band_limit") < PIPELINE_ORDER.index(
            "normalise_loudness"
        )
        # The exempt pipeline differs by exactly one step.
        assert set(PIPELINE_ORDER) - set(PIPELINE_ORDER_BAND_EXEMPT) == {"band_limit"}

    def test_process_condition_output_skips_the_filter_when_exempt(self):
        from data.preprocess import TrimSpan, process_condition_output

        n = ARCHIVE_SR
        span = TrimSpan(utt_id="u", start=0, end=n, ref_length=n)
        wav = self._wideband(1.0)

        _, _, oob_ladder = process_condition_output(
            wav, ARCHIVE_SR, span, condition="hifigan_v1"
        )
        assert oob_ladder is not None and oob_ladder <= BAND_LIMIT_MAX_STOPBAND_ENERGY

        _, _, oob_exempt = process_condition_output(
            wav, ARCHIVE_SR, span, condition="bigvgan_v2_22khz_fullband"
        )
        assert oob_exempt is None  # exempt: not filtered, not measured

    def test_band_filter_spec_marks_exemption(self):
        from data.invariants import BAND_LIMIT_FILTER_SPEC
        from data.preprocess import band_filter_spec

        assert band_filter_spec("hifigan_v1") == (LADDER_FMAX, BAND_LIMIT_FILTER_SPEC)
        assert band_filter_spec("bigvgan_v2_22khz_fullband") == (None, "exempt")

    # --- manifest gate ----------------------------------------------------
    def test_manifest_rejects_a_condition_with_a_different_band(self):
        from data.manifest import validate_manifest

        df = _manifest_frame()
        df.loc[df["condition"] == "hifigan_v1", "band_limit_hz"] = 11025.0
        with pytest.raises(InvariantViolation, match="INV-17"):
            validate_manifest(df)

    def test_manifest_rejects_residual_out_of_band_energy(self):
        from data.manifest import validate_manifest

        df = _manifest_frame()
        df.loc[df["condition"] == "hifigan_v1", "band_oob_energy"] = 0.05
        with pytest.raises(InvariantViolation, match="INV-17"):
            validate_manifest(df)

    def test_manifest_rejects_two_different_filters(self):
        from data.manifest import validate_manifest

        df = _manifest_frame()
        df.loc[df["condition"] == "real", "band_filter"] = "chebyshev_order4"
        with pytest.raises(InvariantViolation, match="INV-17"):
            validate_manifest(df)

    def test_manifest_rejects_an_exemption_the_code_does_not_know_about(self):
        from data.manifest import validate_manifest

        df = _manifest_frame()
        df.loc[df["condition"] == "hifigan_v1", "band_exempt"] = True
        with pytest.raises(InvariantViolation, match="INV-17"):
            validate_manifest(df)

    # --- the primary correlation refuses the control ----------------------
    def test_primary_correlation_refuses_the_exempt_condition(self):
        """Step 5: enforced in code, not merely documented."""
        from detectors.protocols import Protocol, spearman_headline

        quality = pd.DataFrame(
            {
                "condition": ["griffin_lim", "melgan", "hifigan_v1", "bigvgan_112m",
                              "bigvgan_v2_22khz_fullband"],
                "utmos_mean": [2.0, 2.5, 3.5, 4.2, 4.3],
            }
        )
        detection = pd.DataFrame(
            {
                "condition": ["griffin_lim", "melgan", "hifigan_v1", "bigvgan_112m",
                              "bigvgan_v2_22khz_fullband"],
                "protocol": ["matched"] * 5,
                "tier": [ARCHIVE_TIER] * 5,
                "detector": ["aasist"] * 5,
                "eer": [0.01, 0.05, 0.12, 0.30, 0.02],
            }
        )
        with pytest.raises(InvariantViolation, match="INV-17"):
            spearman_headline(quality, detection, protocol=Protocol.MATCHED)

    def test_primary_correlation_succeeds_once_the_control_is_filtered(self):
        from data.manifest import primary_ladder_frame
        from detectors.protocols import Protocol, spearman_headline

        quality = pd.DataFrame(
            {
                "condition": ["griffin_lim", "melgan", "hifigan_v1", "bigvgan_112m",
                              "bigvgan_v2_22khz_fullband"],
                "utmos_mean": [2.0, 2.5, 3.5, 4.2, 4.3],
            }
        )
        detection = pd.DataFrame(
            {
                "condition": ["griffin_lim", "melgan", "hifigan_v1", "bigvgan_112m",
                              "bigvgan_v2_22khz_fullband"],
                "protocol": ["matched"] * 5,
                "tier": [ARCHIVE_TIER] * 5,
                "detector": ["aasist"] * 5,
                "eer": [0.01, 0.05, 0.12, 0.30, 0.02],
            }
        )
        stats = spearman_headline(
            primary_ladder_frame(quality, why="test"),
            primary_ladder_frame(detection, why="test"),
            protocol=Protocol.MATCHED,
        )
        assert stats["n_conditions"] == 4
        assert is_band_exempt("bigvgan_v2_22khz_fullband")

    def test_paired_bandwidth_contrast_reports_the_delta(self):
        from detectors.protocols import Protocol, paired_bandwidth_contrast

        detection = pd.DataFrame(
            {
                "condition": ["bigvgan_112m", "bigvgan_v2_22khz_fullband"],
                "protocol": ["matched", "matched"],
                "tier": [ARCHIVE_TIER, ARCHIVE_TIER],
                "detector": ["aasist", "aasist"],
                "eer": [0.30, 0.10],
            }
        )
        out = paired_bandwidth_contrast(detection, protocol=Protocol.MATCHED)
        assert out["delta_eer"] == pytest.approx(-0.20)
        assert out["tier"] == ARCHIVE_TIER

    def test_bandwidth_probe_hit_on_the_control_is_not_a_leak(self):
        """The control differs in bandwidth by design; flagging it trains
        people to ignore the column."""
        from experiments.sanity_checks import _verdict

        leak = _verdict("bandwidth_rolloff", "hifigan_v1", 0.0, ARCHIVE_TIER)
        assert leak["leaked"] is True

        by_design = _verdict(
            "bandwidth_rolloff", "bigvgan_v2_22khz_fullband", 0.0, ARCHIVE_TIER
        )
        assert by_design["leaked"] is False
        assert by_design["expected_by_design"] is True

        # Every other probe still blocks, exempt or not.
        still_blocks = _verdict(
            "silence_duration", "bigvgan_v2_22khz_fullband", 0.0, ARCHIVE_TIER
        )
        assert still_blocks["leaked"] is True


class TestOutOfBandCalibration:
    """Task 1: what does out_of_band_energy actually resolve?

    Injects a tone above the cutoff at known amplitude and checks the reported
    figure against the amplitude that was put in. Without this, a very small
    reported number is indistinguishable from the measurement running out of
    resolution -- and the two call for opposite responses.
    """

    SR = ARCHIVE_SR
    F_HI = 9500.37  # deliberately off-bin; bin-centred tones do not leak

    def _carrier(self, n: int) -> np.ndarray:
        t = np.arange(n) / self.SR
        return np.sin(2 * np.pi * 300.37 * t)  # off-bin too

    def _inject(self, base: np.ndarray, db: float) -> tuple[np.ndarray, float]:
        n = len(base)
        t = np.arange(n) / self.SR
        a = 10 ** (db / 20) * np.sqrt(2)
        x = base + a * np.sin(2 * np.pi * self.F_HI * t)
        expected = (a**2 / 2) / (1 + a**2 / 2)
        return x, expected

    @pytest.mark.parametrize("db", [-40, -60, -80, -100, -120, -140, -160, -180, -200])
    def test_recovers_injected_amplitude(self, db):
        """Linear recovery from -40 dB to -200 dB, well past anything we report."""
        from data.invariants import out_of_band_energy

        base = self._carrier(self.SR * 2)
        base = base / np.sqrt(np.mean(base**2))
        x, expected = self._inject(base, db)
        measured = out_of_band_energy(x, self.SR, LADDER_FMAX)
        assert 0.5 < measured / expected < 2.0, (
            f"{db} dB: expected {expected:.3e}, measured {measured:.3e}"
        )

    def test_floor_is_far_below_anything_we_gate_on(self):
        from data.invariants import out_of_band_energy

        base = self._carrier(self.SR * 2)
        base = base / np.sqrt(np.mean(base**2))
        floor = out_of_band_energy(base, self.SR, LADDER_FMAX)
        assert floor < 1e-20
        assert floor < BAND_LIMIT_MAX_STOPBAND_ENERGY / 1e10

    def test_rectangular_window_would_be_unusable(self):
        """Why the window is not optional: rect floors out above the gate."""
        base = self._carrier(self.SR * 2)
        base = base / np.sqrt(np.mean(base**2))
        spec = np.abs(np.fft.rfft(base)) ** 2
        freqs = np.fft.rfftfreq(len(base), d=1.0 / self.SR)
        rect_floor = float(spec[freqs > LADDER_FMAX].sum() / spec.sum())
        assert rect_floor > BAND_LIMIT_MAX_STOPBAND_ENERGY

    def test_float32_storage_is_the_reporting_floor(self):
        """The 1.78e-16 figure is float32 rounding, not the filter.

        Converting the same array to float64 reproduces it exactly, which is how
        we know the number describes storage rather than signal.
        """
        from data.invariants import MEASUREMENT_FLOOR, out_of_band_energy
        from data.preprocess import band_limit_to_ladder

        t = np.arange(self.SR) / self.SR
        wav = (0.05 * np.sin(2 * np.pi * 300.37 * t)).astype(np.float32)
        f32 = band_limit_to_ladder(wav, sr=self.SR)
        a = out_of_band_energy(f32, self.SR, LADDER_FMAX)
        b = out_of_band_energy(f32.astype(np.float64), self.SR, LADDER_FMAX)
        assert a == pytest.approx(b, rel=1e-9)
        assert a <= MEASUREMENT_FLOOR

    def test_format_oob_refuses_to_quote_noise(self):
        from data.invariants import MEASUREMENT_FLOOR, format_oob

        assert "below measurement floor" in format_oob(1e-16)
        assert "below measurement floor" in format_oob(MEASUREMENT_FLOOR)
        assert format_oob(5.8e-9) == "5.800e-09"

    def test_gate_is_derived_from_the_measured_delivery_floor(self):
        """Not a literal: gate = PCM16 floor x margin."""
        from data.invariants import (
            BAND_LIMIT_FLOOR_MARGIN,
            MEASUREMENT_FLOOR,
            PCM16_OOB_FLOOR,
        )

        assert BAND_LIMIT_MAX_STOPBAND_ENERGY == PCM16_OOB_FLOOR * BAND_LIMIT_FLOOR_MARGIN
        # The measurement floor is not usable as a gate: delivered files sit
        # many orders above it.
        assert MEASUREMENT_FLOOR < PCM16_OOB_FLOOR / 1e5

    def test_pcm16_quantisation_sets_the_delivery_floor(self):
        """Task 2's premise, measured: quantisation raises the residual to ~1e-8."""
        from data.invariants import PCM16_OOB_FLOOR, out_of_band_energy
        from data.preprocess import band_limit_to_ladder, normalise_loudness

        rng = np.random.default_rng(0)
        n = self.SR * 2
        t = np.arange(n) / self.SR
        x = sum(0.3 / k * np.sin(2 * np.pi * 141.7 * k * t) for k in range(1, 70))
        x = ((x * np.hanning(n) + 0.005 * rng.standard_normal(n)) * 0.4).astype(np.float32)
        filtered = band_limit_to_ladder(x, sr=self.SR)
        norm, _ = normalise_loudness(filtered, sr=self.SR)
        quantised = (np.round(np.clip(norm, -1, 1) * 32767) / 32767).astype(np.float32)

        before = out_of_band_energy(filtered, self.SR, LADDER_FMAX)
        after = out_of_band_energy(quantised, self.SR, LADDER_FMAX)
        assert before < 1e-14  # filter output: below the reporting floor
        assert after > before * 1e5  # quantisation dominates
        assert after < PCM16_OOB_FLOOR


class TestINV17QuantisationDependency:
    """Task 2: detector inputs must come from written PCM_16 files.

    INV-17's argument -- that the band residual is destroyed before any detector
    sees it -- depends on this and on nothing in the filter spec. If a detector
    were handed a float32 pipeline array the residual would survive at ~1e-16
    and remain structured.
    """

    def test_read_processed_returns_pcm16_representable_values(self, tmp_path):
        """Whatever comes back from the sanctioned reader is on the 16-bit grid."""
        from data.audio_io import read_processed, write_audio

        t = np.arange(ARCHIVE_SR) / ARCHIVE_SR
        wav = (0.05 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        path = tmp_path / "x.wav"
        write_audio(path, wav, ARCHIVE_SR)

        back = read_processed(path, ARCHIVE_SR)
        grid = back * 32768.0
        assert np.allclose(grid, np.round(grid), atol=1e-3), (
            "read_processed returned values off the PCM_16 grid; the quantisation "
            "step INV-17 depends on did not happen"
        )

    def test_score_condition_reads_from_files(self, tmp_path, monkeypatch):
        """The only call site of score_batch must source audio via read_processed."""
        import data.audio_io as audio_io
        from detectors.protocols import Protocol, score_condition

        seen: list[str] = []
        real_read = audio_io.read_processed

        def spy(path, expected_sr=None):
            seen.append(str(path))
            return real_read(path, expected_sr)

        monkeypatch.setattr(audio_io, "read_processed", spy)

        t = np.arange(ARCHIVE_SR // 2) / ARCHIVE_SR
        wav = (0.05 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        rows = []
        for cond in ("real", "hifigan_v1"):
            for i in range(3):
                p = tmp_path / f"{cond}_{i}.wav"
                audio_io.write_audio(p, wav, ARCHIVE_SR)
                rows.append({"condition": cond, "tier": ARCHIVE_TIER,
                             "split": "eval", "path": str(p), "utt_id": f"u{i}"})
        mf = pd.DataFrame(rows)

        class Dummy:
            spec = type("S", (), {"key": "dummy"})()

            def score_batch(self, wavs):
                # Every array handed to a detector must be on the PCM_16 grid.
                for w in wavs:
                    g = w * 32768.0
                    assert np.allclose(g, np.round(g), atol=1e-3)
                return np.arange(len(wavs), dtype=np.float64)

        score_condition(
            Dummy(), mf, "hifigan_v1", protocol=Protocol.MISMATCHED,
            tier=ARCHIVE_TIER, with_ci=False,
        )
        assert len(seen) == 6, "score_condition did not read via read_processed"


class TestINV08CheckpointDigest:
    """Task 5: sha256 verification for checkpoints with no revision to pin."""

    def _file(self, tmp_path, content=b"weights"):
        p = tmp_path / "gen.pt"
        p.write_bytes(content)
        return p

    def test_digest_is_stable_and_correct(self, tmp_path):
        import hashlib

        from vocoders.checkpoints import sha256_file

        p = self._file(tmp_path)
        assert sha256_file(p) == hashlib.sha256(b"weights").hexdigest()

    def test_verify_accepts_a_matching_digest(self, tmp_path):
        from vocoders.checkpoints import sha256_file, verify_checkpoint

        p = self._file(tmp_path)
        expected = sha256_file(p)
        assert verify_checkpoint(p, expected, condition="melgan") == expected

    def test_verify_fails_loudly_on_mismatch(self, tmp_path):
        from vocoders.checkpoints import verify_checkpoint

        p = self._file(tmp_path)
        with pytest.raises(InvariantViolation, match="INV-08.*mismatch"):
            verify_checkpoint(p, "0" * 64, condition="melgan")

    def test_verify_refuses_an_empty_expectation(self, tmp_path):
        from vocoders.checkpoints import verify_checkpoint

        p = self._file(tmp_path)
        with pytest.raises(InvariantViolation, match="no recorded checkpoint_sha256"):
            verify_checkpoint(p, "", condition="melgan")

    def test_missing_file_raises(self, tmp_path):
        from vocoders.checkpoints import sha256_file

        with pytest.raises(InvariantViolation, match="INV-08"):
            sha256_file(tmp_path / "nope.pt")

    def test_fetch_and_verify_requires_opt_in_for_first_use(self, tmp_path):
        from vocoders.checkpoints import fetch_and_verify
        from vocoders.registry import get_spec

        p = self._file(tmp_path)
        spec = get_spec("melgan")  # no digest recorded yet
        with pytest.raises(InvariantViolation, match="no recorded checkpoint_sha256"):
            fetch_and_verify(p, spec)

        digest = fetch_and_verify(p, spec, allow_first_use=True)
        assert len(digest) == 64

    def test_melgan_pair_must_share_one_digest(self):
        """They are the same file; a divergence voids the controlled contrast."""
        from vocoders.registry import get_spec

        a, b = get_spec("melgan"), get_spec("melgan_fullband")
        assert a.checkpoint == b.checkpoint
        assert a.checkpoint_sha256 == b.checkpoint_sha256


class TestBandwidthPairs:
    """Task 3: two pairs, isolating different things."""

    def _detection(self):
        return pd.DataFrame(
            {
                "condition": ["melgan", "melgan_fullband", "bigvgan_112m",
                              "bigvgan_v2_22khz_fullband"],
                "protocol": ["matched"] * 4,
                "tier": [ARCHIVE_TIER] * 4,
                "detector": ["aasist"] * 4,
                "eer": [0.20, 0.08, 0.30, 0.10],
            }
        )

    def test_melgan_fullband_is_exempt_and_excluded(self):
        from vocoders.registry import CONTROL_CONDITIONS, LADDER

        assert is_band_exempt("melgan_fullband")
        assert "melgan_fullband" in CONTROL_CONDITIONS
        assert "melgan_fullband" not in LADDER

    def test_primary_correlation_refuses_melgan_fullband(self):
        from detectors.protocols import Protocol, spearman_headline

        quality = pd.DataFrame(
            {"condition": ["griffin_lim", "melgan", "hifigan_v1", "melgan_fullband"],
             "utmos_mean": [2.0, 2.5, 3.5, 2.6]}
        )
        detection = pd.DataFrame(
            {"condition": ["griffin_lim", "melgan", "hifigan_v1", "melgan_fullband"],
             "protocol": ["matched"] * 4, "tier": [ARCHIVE_TIER] * 4,
             "detector": ["aasist"] * 4, "eer": [0.01, 0.20, 0.12, 0.08]}
        )
        with pytest.raises(InvariantViolation, match="INV-17"):
            spearman_headline(quality, detection, protocol=Protocol.MATCHED)

    def test_both_pairs_resolve(self):
        from detectors.protocols import BANDWIDTH_PAIRS, paired_bandwidth_contrast

        assert set(BANDWIDTH_PAIRS) == {"bigvgan", "melgan"}
        det = self._detection()
        big = paired_bandwidth_contrast(det, pair="bigvgan")
        mel = paired_bandwidth_contrast(det, pair="melgan")
        assert big["delta_eer"] == pytest.approx(-0.20)
        assert mel["delta_eer"] == pytest.approx(-0.12)

    def test_pairs_report_whether_they_share_weights(self):
        """The reader needs this to interpret delta_eer: shared weights isolate
        the delivery filter, separate checkpoints isolate training bandwidth."""
        from detectors.protocols import paired_bandwidth_contrast

        det = self._detection()
        assert paired_bandwidth_contrast(det, pair="melgan")["shared_weights"] is True
        assert paired_bandwidth_contrast(det, pair="bigvgan")["shared_weights"] is False

    def test_unknown_pair_rejected(self):
        from detectors.protocols import paired_bandwidth_contrast

        with pytest.raises(InvariantViolation, match="unknown bandwidth pair"):
            paired_bandwidth_contrast(self._detection(), pair="nope")


class TestVocosResampleSignature:
    """TASK 1: the Vocos 24 kHz round trip leaves nothing INV-17 does not erase.

    Vocos is the only ladder condition whose output is resampled (24000 ->
    22050) before the archive, and INV-01's rationale says a resampler's rolloff
    is a per-condition signature. The rolloff sits above LADDER_FMAX so INV-17
    should remove it -- but that is an argument, and the same shape of argument
    was wrong for Butterworth, so it is measured here.

    Kept small (8 utterances) for suite runtime; the full 60-utterance run
    including the linear-classifier separability check is recorded in INV-01.
    """

    SR = ARCHIVE_SR
    VOCOS_SR = 24_000

    def _source(self, i: int) -> np.ndarray:
        rng = np.random.default_rng(1000 + i)
        n = int(self.SR * 1.2)
        t = np.arange(n) / self.SR
        f0 = 120 + 7 * i
        x = sum(
            (0.35 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28))
            for k in range(1, int(self.SR / 2 / f0))
        )
        x = x + 0.02 * rng.standard_normal(n)
        x = x * np.hanning(n)
        return (0.35 * x / np.max(np.abs(x))).astype(np.float32)

    @staticmethod
    def _pcm16(w: np.ndarray) -> np.ndarray:
        return (np.round(np.clip(w, -1.0, 1.0) * 32767.0) / 32768.0).astype(np.float32)

    def _paths(self, src):
        """(round-tripped, direct) -- identical source, one extra resample pair."""
        import librosa

        from data.invariants import RESAMPLE_METHOD
        from data.preprocess import band_limit_to_ladder, normalise_loudness

        up = librosa.resample(
            src, orig_sr=self.SR, target_sr=self.VOCOS_SR, res_type=RESAMPLE_METHOD
        )
        down = librosa.resample(
            up, orig_sr=self.VOCOS_SR, target_sr=self.SR, res_type=RESAMPLE_METHOD
        )
        down = down[: len(src)] if len(down) >= len(src) else np.pad(
            down, (0, len(src) - len(down))
        )

        a, _ = normalise_loudness(band_limit_to_ladder(down, sr=self.SR), sr=self.SR)
        b, _ = normalise_loudness(band_limit_to_ladder(src, sr=self.SR), sr=self.SR)
        return self._pcm16(a), self._pcm16(b)

    def test_round_trip_residual_is_below_the_delivery_floor(self):
        from data.invariants import PCM16_OOB_FLOOR, out_of_band_energy

        for i in range(4):
            a, b = self._paths(self._source(i))
            for w in (a, b):
                assert out_of_band_energy(w, self.SR, LADDER_FMAX) < PCM16_OOB_FLOOR

    def test_round_trip_is_erased_to_within_one_lsb(self):
        """The strongest form of the result: the files are almost byte-identical."""
        total = differing = 0
        worst = 0
        for i in range(8):
            a, b = self._paths(self._source(i))
            n = min(len(a), len(b))
            ia = np.round(a[:n] * 32768).astype(np.int64)
            ib = np.round(b[:n] * 32768).astype(np.int64)
            d = np.abs(ia - ib)
            total += n
            differing += int((d > 0).sum())
            worst = max(worst, int(d.max()))

        assert worst <= 1, f"round trip shifted a sample by {worst} LSB"
        assert differing / total < 0.01, (
            f"{100 * differing / total:.2f}% of samples differ; INV-01's exemption "
            "for the Vocos resample assumes the round trip is erased"
        )

    def test_no_per_band_level_shift(self):
        """A resampler rolloff would show as a level difference near the top band."""
        import librosa

        edges = [(0, 2000), (2000, 4000), (4000, 6000), (6000, 7000), (7000, 8000)]

        def band_db(w):
            spec = np.abs(librosa.stft(w, n_fft=2048, hop_length=512)) ** 2
            freqs = librosa.fft_frequencies(sr=self.SR, n_fft=2048)
            return np.array(
                [spec[(freqs >= lo) & (freqs < hi)].mean() for lo, hi in edges]
            )

        a, b = self._paths(self._source(0))
        delta = 10 * np.log10(band_db(a) / band_db(b))
        assert np.abs(delta).max() < 0.01, f"per-band shift {delta} dB"


class TestAblationQuantisationContract:
    """TASK 2: an executable specification for code that does not exist yet.

    `run_bandlimit_ablation` is unimplemented. The obvious implementation --
    reach into Phase A's in-memory arrays to avoid a write -- would hand
    detectors float32 audio in which the INV-17 band residual survives at
    ~1.8e-16 and is still structured, undoing the invariant. PCM_16 is what
    destroys it (INV-17's Why).

    This test states the contract now so the implementation has to satisfy it
    rather than discovering it afterwards. It is expected to fail until the
    ablation exists.
    """

    @pytest.mark.xfail(
        reason=(
            "INV-17: run_bandlimit_ablation is unimplemented. Audio reaching a "
            "detector through the ablation path must be read from written PCM_16 "
            "files, not passed as in-memory float32 from Phase A -- the band "
            "residual survives structured at ~1.8e-16 in float32 and is only "
            "destroyed by 16-bit quantisation."
        ),
        raises=NotImplementedError,
        strict=True,
    )
    def test_ablation_feeds_detectors_only_pcm16_grid_audio(self, tmp_path):
        from data.audio_io import write_audio
        from detectors.bandlimit import band_limited_variant, cutoffs_for_rate
        from experiments.phase_b_detection import run_bandlimit_ablation

        t = np.arange(ARCHIVE_SR) / ARCHIVE_SR
        wav = (0.05 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        path = tmp_path / "real.wav"
        write_audio(path, wav, ARCHIVE_SR)

        seen: list[np.ndarray] = []

        class Recorder:
            spec = type("S", (), {"key": "recorder"})()

            def load(self, checkpoint=None):
                return None

            def fit(self, *a, **k):
                return None

            def score_batch(self, wavs):
                seen.extend(wavs)
                return np.arange(len(wavs), dtype=np.float64)

        # Must not raise: the ablation is expected to exist and to source audio
        # from files. Until it does, this raises NotImplementedError (xfail).
        run_bandlimit_ablation(
            manifest_path=str(tmp_path / "m.csv"),
            detector_factory=Recorder,
            out_path=str(tmp_path / "out.csv"),
            cutoffs=cutoffs_for_rate(ARCHIVE_SR)[:1],
        )

        assert seen, "the ablation fed no audio to a detector"
        for w in seen:
            grid = np.asarray(w, dtype=np.float64) * 32768.0
            assert np.allclose(grid, np.round(grid), atol=1e-3), (
                "INV-17: the ablation handed a detector audio that is not on the "
                "PCM_16 grid, so it never went through a written file. The "
                "band-limit residual survives in float32."
            )

        # The ablation's own filter is applied on top of already-quantised audio.
        filtered = band_limited_variant(wav, 4000.0, ARCHIVE_SR)
        assert filtered is not None


class TestINV07SingleOrdering:
    """D-1: one implementation of the post-trim ordering, two entry points.

    This test is what makes INV-07 true. Before it existed,
    `build_real_condition` open-coded band-limit + loudness while every vocoder
    condition went through `process_condition_output`, and the two agreed only
    by inspection. Real is the reference every trim span and every pairing
    derives from, so a silent divergence would move every condition against real
    at once.
    """

    def _speechlike(self, seconds: float = 1.5, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = int(ARCHIVE_SR * seconds)
        t = np.arange(n) / ARCHIVE_SR
        x = sum(
            (0.3 / k) * np.sin(2 * np.pi * 150 * k * t + rng.uniform(0, 6.28))
            for k in range(1, 60)
        )
        x = x * np.hanning(n) + 0.01 * rng.standard_normal(n)
        # Leading/trailing silence so derive_trim_span has something to find.
        pad = int(0.15 * ARCHIVE_SR)
        x = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
        return (0.35 * x / np.max(np.abs(x))).astype(np.float32)

    def test_both_entry_points_are_byte_identical(self):
        """Same input, same span -> same bytes. The two paths cannot drift."""
        from data.preprocess import (
            derive_trim_span,
            finalise_reference,
            process_condition_output,
        )

        for seed in range(3):
            wav = self._speechlike(seed=seed)
            span = derive_trim_span(wav, f"u{seed}")

            ref_wav, ref_lufs, ref_oob = finalise_reference(
                wav, span, condition="hifigan_v1"
            )
            voc_wav, voc_lufs, voc_oob = process_condition_output(
                wav, ARCHIVE_SR, span, condition="hifigan_v1"
            )

            assert np.array_equal(ref_wav, voc_wav), (
                f"seed {seed}: real and vocoder entry points diverged. INV-07 "
                "requires one implementation of the ordering."
            )
            assert ref_lufs == voc_lufs
            assert ref_oob == voc_oob

    def test_exempt_condition_also_agrees(self):
        """The exemption branch is shared too, not duplicated."""
        from data.preprocess import (
            derive_trim_span,
            finalise_reference,
            process_condition_output,
        )

        wav = self._speechlike(seed=7)
        span = derive_trim_span(wav, "u7")
        cond = "bigvgan_v2_22khz_fullband"

        a, la, oa = finalise_reference(wav, span, condition=cond)
        b, lb, ob = process_condition_output(wav, ARCHIVE_SR, span, condition=cond)
        assert np.array_equal(a, b)
        assert la == lb and oa is None and ob is None

    def test_phase_a_real_and_identity_vocoder_produce_identical_files(
        self, tmp_path, monkeypatch
    ):
        """End to end: an identity vocoder must reproduce the real condition byte
        for byte. If the two code paths ever diverge, this is where it shows."""
        import soundfile as sf

        import vocoders.registry as registry
        from data.corpus import LJSpeech
        from experiments.phase_a_resynthesis import (
            build_real_condition,
            build_vocoder_condition,
        )
        from vocoders.base import Vocoder, VocoderSpec

        # --- a tiny corpus on disk ---
        root = tmp_path / "LJSpeech-1.1"
        (root / "wavs").mkdir(parents=True)
        lines = []
        for i in range(20):
            uid = f"LJ001-{i:04d}"
            sf.write(root / "wavs" / f"{uid}.wav", self._speechlike(seed=i),
                     ARCHIVE_SR, subtype="PCM_16")
            lines.append(f"{uid}|x|x")
        (root / "metadata.csv").write_text("\n".join(lines), encoding="utf-8")
        corpus = LJSpeech(root, subset_size=None)

        # --- an identity vocoder: resynthesis is a no-op ---
        from data.mel import HIFIGAN_V1_MEL

        ident_spec = VocoderSpec(
            key="hifigan_v1", display_name="identity", family="gan", year=2020,
            params_m=1.0, mel=HIFIGAN_V1_MEL, checkpoint="identity",
        )

        class Identity(Vocoder):
            spec = ident_spec

            def load(self):
                return None

            def synthesize(self, mel):
                raise AssertionError("resynthesize is overridden")

            def resynthesize(self, wav, sr):
                return wav.astype(np.float32)

        monkeypatch.setitem(registry.SPECS, "hifigan_v1", ident_spec)
        monkeypatch.setitem(registry._BUILDERS, "hifigan_v1", Identity)

        splits = dict.fromkeys((u.utt_id for u in corpus.utterances()), "train")
        out = tmp_path / "out"
        rows, spans, real_archive, dropped = build_real_condition(corpus, out, splits)
        cond_rows, _ = build_vocoder_condition(
            "hifigan_v1", corpus, out, spans, real_archive, splits, dropped
        )
        assert rows and cond_rows

        # The archive files must be byte-identical.
        n = 0
        for utt_id in spans:
            a = (out / "archive" / "real" / f"{utt_id}.wav").read_bytes()
            b = (out / "archive" / "hifigan_v1" / f"{utt_id}.wav").read_bytes()
            assert a == b, (
                f"{utt_id}: identity vocoder produced different bytes from real. "
                "The real and vocoder code paths have diverged (INV-07)."
            )
            n += 1
        assert n >= 15


class TestINV09FailsClosedOnSpeakers:
    """D-6: a multi-speaker corpus must not be split without an explicit decision."""

    def _utts(self, speakers):
        from data.corpus import Utterance

        return [
            Utterance(utt_id=f"u{i}", path=Path(f"{i}.wav"), speaker=spk)
            for i, spk in enumerate(speakers)
        ]

    def test_multi_speaker_without_a_decision_raises(self):
        from data.corpus import assign_splits

        utts = self._utts([f"s{i % 5}" for i in range(50)])
        with pytest.raises(InvariantViolation, match="INV-09.*speaker_disjoint"):
            assign_splits(utts)

    def test_explicit_true_is_accepted(self):
        from data.corpus import assign_splits

        utts = self._utts([f"s{i % 5}" for i in range(50)])
        splits = assign_splits(utts, speaker_disjoint=True)
        # Speaker-disjoint: no speaker appears in two splits.
        by_speaker: dict[str, set[str]] = {}
        for u in utts:
            by_speaker.setdefault(u.speaker, set()).add(splits[u.utt_id])
        assert all(len(v) == 1 for v in by_speaker.values())

    def test_explicit_false_is_accepted_as_a_recorded_decision(self):
        from data.corpus import assign_splits

        utts = self._utts([f"s{i % 5}" for i in range(50)])
        assert assign_splits(utts, speaker_disjoint=False)

    def test_single_speaker_needs_no_decision(self):
        from data.corpus import assign_splits

        utts = self._utts(["LJ"] * 50)
        assert assign_splits(utts)  # no raise: the two options are identical

    def test_corpus_declares_its_own_shape(self):
        from data.corpus import VCTK, LJSpeech

        assert LJSpeech.is_multi_speaker is False
        assert VCTK.is_multi_speaker is True

    def test_run_rejects_a_corpus_that_misdeclares_itself(self, tmp_path):
        """A declaration that can lie is worse than none, so it is cross-checked."""
        import soundfile as sf

        from data.corpus import LJSpeech
        from experiments.phase_a_resynthesis import run

        root = tmp_path / "LJSpeech-1.1"
        (root / "wavs").mkdir(parents=True)
        t = np.arange(ARCHIVE_SR) / ARCHIVE_SR
        wav = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        lines = []
        for i in range(4):
            uid = f"LJ001-{i:04d}"
            sf.write(root / "wavs" / f"{uid}.wav", wav, ARCHIVE_SR, subtype="PCM_16")
            lines.append(f"{uid}|x|x")
        (root / "metadata.csv").write_text("\n".join(lines), encoding="utf-8")

        class Liar(LJSpeech):
            is_multi_speaker = True  # declares multi, yields one speaker

        with pytest.raises(InvariantViolation, match="INV-09.*is_multi_speaker"):
            run(Liar(root, subset_size=None), [], tmp_path / "o", tmp_path / "m.csv")


class TestINV13LazyImports:
    """D-8: the claim was true but unchecked, so nothing would have caught a
    stray top-level `import torch` until it broke a Kaggle session."""

    HEAVY = ("torch", "torchaudio", "transformers", "pesq", "pyworld")

    def test_importing_the_packages_pulls_no_heavy_dependency(self):
        import subprocess

        code = (
            "import sys; sys.path.insert(0, 'src');"
            "import vocoders, detectors, data, metrics, experiments;"
            f"heavy=[m for m in {self.HEAVY!r} if m in sys.modules];"
            "print(','.join(heavy))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=str(REPO_ROOT), check=True,
        )
        pulled = [m for m in out.stdout.strip().split(",") if m]
        assert not pulled, (
            f"INV-13: importing the packages pulled {pulled}. Adapters must load "
            "nothing until load(); a top-level heavy import makes every condition "
            "share one dependency resolution."
        )


class TestINV11CropPolicy:
    """D-3: the crop is duration-based and reached through one method."""

    def test_crop_is_the_same_duration_at_both_rates(self):
        from detectors.base import ASVSPOOF_CROP_SECONDS, crop_samples_for_rate

        for sr in (ARCHIVE_SR, ZEROSHOT_SR):
            assert crop_samples_for_rate(sr) == pytest.approx(
                ASVSPOOF_CROP_SECONDS * sr, abs=1
            )

    def test_documented_constants_match_the_code(self):
        """The exact integers CLAUDE.md quotes. D-3 was a stale one of these."""
        from detectors.base import crop_samples_for_rate

        assert crop_samples_for_rate(ZEROSHOT_SR) == 64_600
        assert crop_samples_for_rate(ARCHIVE_SR) == 89_027

    def test_crop_for_is_the_shared_entry_point(self):
        """fixed_length_crop is reached through Detector.crop_for, not by hand."""
        from detectors.base import Detector, DetectorSpec, crop_samples_for_rate

        class Dummy(Detector):
            spec = DetectorSpec(
                key="d", display_name="d", params_m=0.0, input_kind="raw",
                max_seconds=4.0375,
            )

            def load(self, checkpoint=None):
                return None

            def score(self, wav):
                return 0.0

        d = Dummy()
        short = np.ones(1000, dtype=np.float32)
        for sr in (ARCHIVE_SR, ZEROSHOT_SR):
            assert len(d.crop_for(short, sr)) == crop_samples_for_rate(sr)

    def test_unsanctioned_rate_has_no_crop(self):
        from detectors.base import crop_samples_for_rate

        with pytest.raises(ValueError, match="INV-01"):
            crop_samples_for_rate(44_100)
