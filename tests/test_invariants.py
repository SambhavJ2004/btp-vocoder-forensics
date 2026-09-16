"""Tests for the confound controls.

These are not unit tests for their own sake: each one asserts that a specific
invariant from CLAUDE.md actually fires. A silently-disabled check is
indistinguishable from a correct pipeline until the results are already wrong.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.invariants import (
    ARCHIVE_SR,
    ARCHIVE_TIER,
    BAND_LIMIT_MAX_STOPBAND_ENERGY,
    HIGH_BAND_ANALYSIS,
    LADDER_FMAX,
    LOUDNESS_TARGET_LUFS,
    PRIMARY_ANALYSIS,
    ZEROSHOT_SR,
    ZEROSHOT_TIER,
    InvariantViolation,
    check_pairing,
    check_waveform,
    is_excluded,
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
                        "archive_band_hz": ARCHIVE_SR / 2.0,
                        "archive_band": "full_band",
                        "high_band_fraction": 0.015,
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

    def test_seven_generated_conditions_six_rungs_one_unavailable(self):
        from vocoders.registry import (
            ALL_CONDITIONS,
            CONTROL_CONDITIONS,
            LADDER,
            UNAVAILABLE,
        )

        assert len(LADDER) == 6
        assert len(CONTROL_CONDITIONS) == 1  # melgan_fullband removed (INV-17)
        assert len(ALL_CONDITIONS) == 7
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
        for key in ("vocos", "bigvgan_base", "bigvgan_v2_22khz_fullband"):
            assert rows[key]["pin"] == "hf-revision", key
            assert rows[key]["blocked"] is False, key
        # bigvgan_112m additionally carries the weight-file digest from meta.json.
        assert rows["bigvgan_112m"]["pin"] == "hf-revision+sha256"
        assert rows["bigvgan_112m"]["blocked"] is False

        # Not on HuggingFace: no revision exists, so still unpinned.
        for key in ("hifigan_v1", "melgan"):
            assert rows[key]["pin"] == "UNPINNED", key
            assert rows[key]["blocked"] is True, key

        # Declared but not generable.
        assert rows["specdiff_gan"]["role"] == "unavailable"
        assert rows["specdiff_gan"]["blocked"] is True

    def test_audit_can_exclude_unavailable(self):
        from vocoders.registry import checkpoint_audit

        keys = {r["condition"] for r in checkpoint_audit(include_unavailable=False)}
        assert "specdiff_gan" not in keys

    def test_controls_are_membership_not_exclusions(self):
        """INV-18 keeps two concepts apart: a paired control is not a rung (a
        membership fact, no measurement needed); an exclusion removes a RUNG
        from one analysis and must cite evidence."""
        from data.invariants import ANALYSIS_EXCLUSIONS, UNAVAILABLE_CONDITIONS
        from vocoders.registry import CONTROL_CONDITIONS, LADDER, SPECS

        declared = {
            k
            for k, v in SPECS.items()
            if not v.primary_ladder and k not in UNAVAILABLE_CONDITIONS
        }
        assert declared == set(CONTROL_CONDITIONS)
        assert not set(CONTROL_CONDITIONS) & set(LADDER)
        for table in ANALYSIS_EXCLUSIONS.values():
            assert not set(table) & set(CONTROL_CONDITIONS), (
                "a control appears as an exclusion; controls are refused by "
                "membership and need no measurement"
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

    def test_sweep_crosses_the_analysis_band(self):
        """INV-17 reversal: the archive is full-band, so the sweep is no longer
        capped at LADDER_FMAX and can place cutoffs on both sides of it."""
        from detectors.bandlimit import cutoffs_for_rate

        cuts = cutoffs_for_rate(ARCHIVE_SR)
        assert any(c < LADDER_FMAX for c in cuts)
        assert any(c > LADDER_FMAX for c in cuts)

    def test_sweep_stops_below_nyquist(self):
        """No band_exempt variant any more: every condition is full-band."""
        from detectors.bandlimit import cutoffs_for_rate, nyquist

        cuts = cutoffs_for_rate(ARCHIVE_SR)
        assert max(cuts) < nyquist(ARCHIVE_SR)

    @staticmethod
    def _sweep_response(cutoff, n=2**17):
        from detectors.bandlimit import lowpass

        imp = np.zeros(n, dtype=np.float32)
        imp[n // 2] = 1.0
        h = lowpass(imp, cutoff).astype(np.float64)
        f = np.fft.rfftfreq(n, 1 / ARCHIVE_SR)
        return f, 20 * np.log10(np.maximum(np.abs(np.fft.rfft(h)), 1e-300))

    def test_sweep_transition_sits_below_every_cutoff(self):
        """TASK 4. Measured directly, at every sweep cutoff: -1 dB at cutoff-85 Hz,
        -3 dB at cutoff-78 Hz, -20 dB at cutoff-60 Hz, nothing at or above it."""
        from detectors.bandlimit import cutoffs_for_rate

        for cut in cutoffs_for_rate(ARCHIVE_SR):
            f, db = self._sweep_response(cut)

            def first(level, f=f, db=db):
                return float(f[np.nonzero(db < level)[0][0]])

            assert first(-1.0) == pytest.approx(cut - 85.0, abs=3.0), cut
            assert first(-3.0) == pytest.approx(cut - 78.0, abs=3.0), cut
            assert first(-20.0) == pytest.approx(cut - 60.5, abs=3.0), cut
            assert db[f >= cut].max() <= -150.0, f"{cut}: leaks across the cutoff"
            assert np.abs(db[f <= cut - 100.0]).max() <= 0.1 + 1e-3, cut

    def test_sweep_residual_above_each_cutoff_is_at_the_storage_floor(self):
        from data.invariants import MEASUREMENT_FLOOR, out_of_band_energy
        from detectors.bandlimit import cutoffs_for_rate, lowpass

        rng = np.random.default_rng(0)
        t = np.arange(2 * ARCHIVE_SR) / ARCHIVE_SR
        x = sum((0.3 / k) * np.sin(2 * np.pi * 140 * k * t + rng.uniform(0, 6.28))
                for k in range(1, 78))
        x = (0.3 * x / np.abs(x).max() + 0.003 * rng.standard_normal(len(t))).astype(np.float32)
        for cut in cutoffs_for_rate(ARCHIVE_SR):
            assert out_of_band_energy(lowpass(x, cut), ARCHIVE_SR, cut) <= MEASUREMENT_FLOOR, cut

    def test_sweep_point_at_ladder_fmax_is_the_analysis_band_limit(self):
        from data.preprocess import band_limit_to_ladder
        from detectors.bandlimit import lowpass

        x = np.random.default_rng(1).standard_normal(ARCHIVE_SR).astype(np.float32)
        assert np.array_equal(lowpass(x, LADDER_FMAX), band_limit_to_ladder(x, sr=ARCHIVE_SR))

    def test_sweep_order_is_derived_per_cutoff(self):
        from detectors.bandlimit import cutoffs_for_rate, filter_order

        orders = {c: filter_order(c) for c in cutoffs_for_rate(ARCHIVE_SR)}
        assert orders == {1000.0: 31, 2000.0: 44, 3000.0: 52, 4000.0: 58, 5000.0: 60,
                          6000.0: 60, 7000.0: 58, 8000.0: 53, 9000.0: 45, 10000.0: 33}

    def test_butterworth_8_would_straddle_every_cutoff(self):
        """Mutation check: the old sweep filter fails the leak assertion above."""
        from scipy.signal import butter, sosfiltfilt

        from detectors.bandlimit import cutoffs_for_rate

        n = 2**17
        imp = np.zeros(n)
        imp[n // 2] = 1.0
        f = np.fft.rfftfreq(n, 1 / ARCHIVE_SR)
        for cut in cutoffs_for_rate(ARCHIVE_SR):
            h = sosfiltfilt(butter(8, cut / (ARCHIVE_SR / 2), output="sos"), imp)
            db = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(h)), 1e-300))
            assert db[np.argmin(np.abs(f - cut))] == pytest.approx(-6.0, abs=0.1), cut
            assert db[f >= cut].max() > -150.0, cut

    def test_lowpass_attenuates_above_cutoff(self):
        from detectors.bandlimit import lowpass

        t = np.arange(ARCHIVE_SR) / ARCHIVE_SR
        high = (0.1 * np.sin(2 * np.pi * 9000 * t)).astype(np.float32)
        out = lowpass(high, 1000.0)
        # Measured over the steady-state interior: filtfilt leaves a transient at
        # both edges, which is a property of the filter, not of the passband.
        edge = ARCHIVE_SR // 10
        assert np.abs(out[edge:-edge]).max() < 0.01 * np.abs(high).max()


class TestINV17FullBandArchive:
    """The archive is full-band; band-limiting is an analysis transform.

    The original INV-17 low-passed at generation on the premise that an
    fmax=8000 mel front-end emits nothing above 8 kHz. Measured on 20 LJSpeech
    files that is false for every condition except Griffin-Lim: real 0.01803,
    BigVGAN 0.01480, Griffin-Lim 0.00000. `fmax` constrains the analysis a
    vocoder consumes, not the synthesis it performs.
    """

    def _wideband(self, seconds: float = 1.0, sr: int = ARCHIVE_SR) -> np.ndarray:
        t = np.arange(int(seconds * sr)) / sr
        low = 0.05 * np.sin(2 * np.pi * 300 * t)
        high = 0.05 * np.sin(2 * np.pi * 9500 * t)  # above LADDER_FMAX
        return (low + high).astype(np.float32)

    # --- derivation is unchanged -----------------------------------------
    def test_ladder_fmax_is_derived_from_the_audit_table(self):
        from data.invariants import (
            AUDITED_MEL_FMAX,
            FMAX_FREE_CONDITIONS,
            PRIMARY_LADDER,
            UNAVAILABLE_CONDITIONS,
        )

        constraining = [
            f
            for c, f in AUDITED_MEL_FMAX.items()
            if c in PRIMARY_LADDER
            and c not in FMAX_FREE_CONDITIONS
            and c not in UNAVAILABLE_CONDITIONS
        ]
        assert LADDER_FMAX == min(constraining)

    def test_ladder_fmax_did_not_move(self):
        """Reclassifying griffin_lim did not change the band: it was already
        excluded from the min as an fmax-free condition."""
        assert LADDER_FMAX == 8000.0

    # --- generation does NOT filter --------------------------------------
    def test_generation_preserves_the_high_band(self):
        """The core reversal: Phase A must not low-pass."""
        from data.invariants import out_of_band_energy
        from data.preprocess import TrimSpan, process_condition_output

        wav = self._wideband(2.0)
        span = TrimSpan(utt_id="u", start=0, end=len(wav), ref_length=len(wav))
        out, _, high_band = process_condition_output(
            wav, ARCHIVE_SR, span, condition="hifigan_v1"
        )
        measured = out_of_band_energy(out, ARCHIVE_SR, LADDER_FMAX)
        assert measured > 0.1, (
            "the archive was band-limited at generation; INV-17 moved that to "
            "analysis time because the >LADDER_FMAX content is hallucinated by "
            "the vocoder and is the most forensically interesting content there is"
        )
        assert high_band == pytest.approx(measured, rel=1e-6)

    def test_high_band_fraction_is_recorded_not_removed(self):
        from data.preprocess import TrimSpan, finalise_reference

        wav = self._wideband(2.0)
        span = TrimSpan(utt_id="u", start=0, end=len(wav), ref_length=len(wav))
        _, _, high_band = finalise_reference(wav, span)
        assert high_band is not None and high_band > 0.1

    def test_pipeline_order_has_no_band_limit_step(self):
        from data.invariants import PIPELINE_ORDER

        assert "band_limit" not in PIPELINE_ORDER
        assert "measure_high_band" in PIPELINE_ORDER
        assert PIPELINE_ORDER.index("measure_high_band") < PIPELINE_ORDER.index(
            "normalise_loudness"
        )

    # --- the analysis-time transform -------------------------------------
    def test_comparison_set_band_limits_every_member(self):
        from data.invariants import out_of_band_energy
        from data.preprocess import band_limit_comparison_set

        wavs = {
            "real": self._wideband(1.0),
            "hifigan_v1": self._wideband(1.0),
        }
        out = band_limit_comparison_set(wavs)
        assert set(out) == set(wavs)
        for w in out.values():
            assert out_of_band_energy(w, ARCHIVE_SR, LADDER_FMAX) <= (
                BAND_LIMIT_MAX_STOPBAND_ENERGY
            )

    def test_comparison_set_refuses_a_set_without_real(self):
        """Band-limiting only the fakes is the same cliff, sign flipped."""
        from data.preprocess import band_limit_comparison_set

        with pytest.raises(InvariantViolation, match="INV-17.*real"):
            band_limit_comparison_set({"hifigan_v1": self._wideband()})

    def test_comparison_set_does_not_mutate_its_input(self):
        from data.invariants import out_of_band_energy
        from data.preprocess import band_limit_comparison_set

        wavs = {"real": self._wideband(), "melgan": self._wideband()}
        before = out_of_band_energy(wavs["real"], ARCHIVE_SR, LADDER_FMAX)
        band_limit_comparison_set(wavs)
        after = out_of_band_energy(wavs["real"], ARCHIVE_SR, LADDER_FMAX)
        assert before == after

    def test_band_limiting_preserves_sample_alignment(self):
        from data.alignment import cross_correlation_lag
        from data.preprocess import band_limit_to_ladder

        wav = self._wideband(2.0)
        lag, _ = cross_correlation_lag(wav, band_limit_to_ladder(wav, sr=ARCHIVE_SR))
        assert lag == 0

    # --- manifest gate is now the inverse --------------------------------
    def test_manifest_requires_a_full_band_archive(self):
        from data.manifest import validate_manifest

        df = _manifest_frame()
        df["archive_band"] = "ladder_band"
        with pytest.raises(InvariantViolation, match="INV-17.*full_band"):
            validate_manifest(df)

    def test_manifest_requires_high_band_fraction_on_archive_rows(self):
        from data.manifest import validate_manifest

        df = _manifest_frame()
        df.loc[df["tier"] == ARCHIVE_TIER, "high_band_fraction"] = None
        with pytest.raises(InvariantViolation, match="INV-17.*high_band_fraction"):
            validate_manifest(df)

    def test_valid_full_band_manifest_passes(self):
        from data.manifest import validate_manifest

        validate_manifest(_manifest_frame())

    # --- the sweep is no longer capped -----------------------------------
    def test_ablation_sweep_now_crosses_the_analysis_band(self):
        from detectors.bandlimit import cutoffs_for_rate, nyquist

        cuts = cutoffs_for_rate(ARCHIVE_SR)
        assert any(c < LADDER_FMAX for c in cuts)
        assert any(c > LADDER_FMAX for c in cuts), (
            "the sweep used to stop below LADDER_FMAX because the archive was "
            "pre-filtered; with a full-band archive it can cross that boundary"
        )
        assert max(cuts) < nyquist(ARCHIVE_SR)


class TestINV18PerAnalysisMembership:
    """Two analyses, different membership, every exclusion backed by evidence."""

    def _frames(self, conditions, *, tier=ARCHIVE_TIER, band_limit_hz=LADDER_FMAX):
        quality = pd.DataFrame(
            {"condition": conditions, "utmos_mean": np.linspace(2.0, 4.5, len(conditions))}
        )
        detection = pd.DataFrame(
            {
                "condition": conditions,
                "protocol": ["matched"] * len(conditions),
                "tier": [tier] * len(conditions),
                "detector": ["aasist"] * len(conditions),
                "eer": np.linspace(0.01, 0.3, len(conditions)),
                "band_limit_hz": [band_limit_hz] * len(conditions),
            }
        )
        return quality, detection

    # --- the evidence ------------------------------------------------------
    def test_every_exclusion_cites_committed_evidence(self):
        """THE INV-18 enforcement: each cited statistic is recomputed from
        data/measurements/ and must match. An exclusion cannot quote a number the
        committed evidence does not contain."""
        from data.invariants import ANALYSIS_EXCLUSIONS
        from data.measurements import verify

        n = 0
        for table in ANALYSIS_EXCLUSIONS.values():
            for exc in table.values():
                assert exc.measurements, f"{exc.condition}: exclusion names no measurement"
                for ref in exc.measurements:
                    verify(ref)
                    n += 1
        assert n >= 4

    def test_a_wrong_citation_is_caught(self):
        from dataclasses import replace

        from data.invariants import ANALYSIS_EXCLUSIONS
        from data.measurements import verify

        ref = ANALYSIS_EXCLUSIONS[HIGH_BAND_ANALYSIS]["griffin_lim"].measurements[0]
        with pytest.raises(InvariantViolation, match="INV-18.*records"):
            verify(replace(ref, value=0.0))

    def test_exclusion_references_resolve_to_real_rows(self):
        from data.invariants import ANALYSIS_EXCLUSIONS
        from data.measurements import load_csv, load_probes

        conditions = set(load_csv()["condition"])
        probes = set(load_probes()["probes"])
        for table in ANALYSIS_EXCLUSIONS.values():
            for exc in table.values():
                for ref in exc.measurements:
                    known = probes if ref.artifact.endswith(".json") else conditions
                    assert ref.condition in known, ref.describe()

    def test_vocos_exclusion_covers_the_pinned_resampler(self):
        """The CSV sliver figures predate the pin; the exclusion must also hold
        for the resampler Vocos is generated with now."""
        from data.invariants import ANALYSIS_EXCLUSIONS
        from data.measurements import verify

        refs = ANALYSIS_EXCLUSIONS[HIGH_BAND_ANALYSIS]["vocos"].measurements
        for probe in ("real_vs_rt_kaiser", "real_vs_rt_kaiser_norm"):
            (pinned,) = [r for r in refs if r.condition == probe]
            assert pinned.metric == "auroc_above"
            assert verify(pinned) >= 0.99, probe

    def test_high_band_evidence_holds_in_the_pipeline_order(self):
        """TASK 3. Under loudness normalisation -- what the pipeline actually does --
        the above-band evidence for the Vocos exclusion and the uniform-round-trip
        rejection is at least as strong as the un-normalised probe showed.

        Deliberately NOT asserted: that normalisation raises it. The rise (kaiser
        0.9935 -> 0.9989, 1x vs 2x 0.9928 -> 0.9941) is within the un-normalised
        fold spread (0.0056, 0.0073) and per-fold values are not recorded.
        """
        from data.invariants import PROBE_CONTROL_MIN_AUROC
        from data.measurements import probe_reading

        for name in ("real_vs_rt_kaiser_norm", "kaiser_1x_vs_2x_norm"):
            r = probe_reading(name)
            assert r.loudness_normalised
            assert r.control_auroc >= 0.99 > PROBE_CONTROL_MIN_AUROC, name

    def test_a_wrong_recorded_probe_value_is_caught(self):
        from data.invariants import MeasurementRef
        from data.measurements import verify

        with pytest.raises(InvariantViolation, match="INV-18.*records"):
            verify(MeasurementRef("probes.json", "real_vs_rt_kaiser", "auroc_below",
                                  "recorded", 0.502))
        with pytest.raises(InvariantViolation, match="recorded"):
            verify(MeasurementRef("probes.json", "real_vs_rt_kaiser", "auroc_below",
                                  "mean", 0.5))

    # --- membership ----------------------------------------------------------
    def test_primary_analysis_excludes_nothing(self):
        from data.invariants import ANALYSIS_EXCLUSIONS

        assert ANALYSIS_EXCLUSIONS[PRIMARY_ANALYSIS] == {}

    def test_griffin_lim_is_back_in_primary_and_out_of_high_band(self):
        """Intended: its cliff lives above LADDER_FMAX, which primary removes."""
        assert not is_excluded(PRIMARY_ANALYSIS, "griffin_lim")
        assert is_excluded(HIGH_BAND_ANALYSIS, "griffin_lim")

    def test_vocos_is_in_primary_and_out_of_high_band(self):
        assert not is_excluded(PRIMARY_ANALYSIS, "vocos")
        assert is_excluded(HIGH_BAND_ANALYSIS, "vocos")

    def test_neural_rungs_are_in_both(self):
        for cond in ("melgan", "hifigan_v1", "bigvgan_base", "bigvgan_112m"):
            assert not is_excluded(PRIMARY_ANALYSIS, cond)
            assert not is_excluded(HIGH_BAND_ANALYSIS, cond)

    def test_analysis_frame_membership_differs_per_analysis(self):
        from data.manifest import analysis_frame
        from vocoders.registry import ALL_CONDITIONS

        df = pd.DataFrame({"condition": list(ALL_CONDITIONS)})
        primary = set(analysis_frame(df, PRIMARY_ANALYSIS, why="t")["condition"])
        high = set(analysis_frame(df, HIGH_BAND_ANALYSIS, why="t")["condition"])
        assert "griffin_lim" in primary and "griffin_lim" not in high
        assert "vocos" in primary and "vocos" not in high
        assert "bigvgan_v2_22khz_fullband" not in primary  # a control, not a rung
        assert "bigvgan_v2_22khz_fullband" not in high

    # --- spearman_headline ----------------------------------------------------
    def test_spearman_includes_griffin_lim_when_band_limited(self):
        from detectors.protocols import Protocol, spearman_headline

        quality, detection = self._frames(
            ["griffin_lim", "melgan", "hifigan_v1", "vocos", "bigvgan_base", "bigvgan_112m"]
        )
        stats = spearman_headline(quality, detection, protocol=Protocol.MATCHED)
        assert stats["n_conditions"] == 6

    def test_spearman_refuses_full_band_rows(self):
        """The load-bearing check: griffin_lim is only safe in the headline
        because the headline is band-limited. Full-band rows bring its cliff back."""
        from detectors.protocols import Protocol, spearman_headline

        quality, detection = self._frames(
            ["griffin_lim", "melgan", "hifigan_v1", "vocos"], band_limit_hz=None
        )
        with pytest.raises(InvariantViolation, match="INV-18.*above LADDER_FMAX"):
            spearman_headline(quality, detection, protocol=Protocol.MATCHED)

    def test_spearman_accepts_zero_shot_rows(self):
        """Zero-shot Nyquist is 8 kHz, so those rows are inherently band-limited."""
        from detectors.protocols import Protocol, spearman_headline

        quality, detection = self._frames(
            ["griffin_lim", "melgan", "hifigan_v1", "vocos"],
            tier=ZEROSHOT_TIER, band_limit_hz=None,
        )
        detection["protocol"] = "mismatched"
        stats = spearman_headline(quality, detection, protocol=Protocol.MISMATCHED)
        assert stats["n_conditions"] == 4

    def test_spearman_refuses_the_paired_control_by_membership(self):
        from detectors.protocols import Protocol, spearman_headline

        quality, detection = self._frames(
            ["melgan", "hifigan_v1", "bigvgan_112m", "bigvgan_v2_22khz_fullband"]
        )
        with pytest.raises(InvariantViolation, match="paired controls"):
            spearman_headline(quality, detection, protocol=Protocol.MATCHED)

    def test_spearman_refuses_a_non_primary_analysis(self):
        from detectors.protocols import Protocol, spearman_headline

        quality, detection = self._frames(["melgan", "hifigan_v1", "bigvgan_112m"])
        with pytest.raises(InvariantViolation, match="INV-18"):
            spearman_headline(
                quality, detection, protocol=Protocol.MATCHED, analysis=HIGH_BAND_ANALYSIS
            )

    def test_refuse_excluded_quotes_the_measurement(self):
        from detectors.protocols import refuse_excluded

        with pytest.raises(InvariantViolation) as exc:
            refuse_excluded(["vocos"], HIGH_BAND_ANALYSIS, where="test")
        msg = str(exc.value)
        assert "resampler signature" in msg
        assert "evidence" in msg and "band_measurements.csv" in msg


class TestHighBandDistance:
    """TASK 3: right AMOUNT of high-band energy is not right CONTENT."""

    def _real(self, n=None):
        n = n or ARCHIVE_SR
        rng = np.random.default_rng(3)
        t = np.arange(n) / ARCHIVE_SR
        x = sum(0.3 / k * np.sin(2 * np.pi * 140 * k * t) for k in range(1, 78))
        x = x * np.hanning(n) + 0.01 * rng.standard_normal(n)
        return (0.3 * x / np.max(np.abs(x))).astype(np.float32)

    def test_identity_is_perfect(self):
        from metrics.spectral import high_band_distance

        r = self._real()
        out = high_band_distance(r, r, condition="bigvgan_112m")
        assert out["hb_lsd"] == pytest.approx(0.0, abs=1e-9)
        assert out["hb_corr"] == pytest.approx(1.0, abs=1e-9)
        assert out["hb_frac_ratio"] == pytest.approx(1.0, abs=1e-9)

    def test_hard_bandwidth_zero_is_visible(self):
        """Griffin-Lim's failure mode: no content at all above the band."""
        from data.preprocess import band_limit_to_ladder
        from metrics.spectral import high_band_distance

        r = self._real()
        gl = band_limit_to_ladder(r, sr=ARCHIVE_SR)
        out = high_band_distance(r, gl, condition="melgan")
        assert out["hb_frac_deg"] == pytest.approx(0.0, abs=1e-12)
        assert out["hb_frac_ratio"] == pytest.approx(0.0, abs=1e-9)
        assert out["hb_lsd"] > 50.0

    def test_right_energy_wrong_structure_is_caught(self):
        """The whole reason this metric exists: an energy fraction cannot see it.

        Replace the high band with noise scaled to the SAME energy. The energy
        ratio reads ~1.0 -- indistinguishable from a perfect reconstruction --
        while the spectral distance and correlation both collapse.
        """
        from scipy.signal import butter, sosfiltfilt

        from metrics.spectral import high_band_distance

        r = self._real()
        nyq = ARCHIVE_SR / 2
        hp = butter(8, LADDER_FMAX / nyq, btype="high", output="sos")
        lp = butter(8, LADDER_FMAX / nyq, btype="low", output="sos")

        low = sosfiltfilt(lp, r)
        real_high = sosfiltfilt(hp, r)
        rng = np.random.default_rng(11)
        noise_high = sosfiltfilt(hp, rng.standard_normal(len(r)))
        noise_high *= np.sqrt(np.mean(real_high**2) / np.mean(noise_high**2))
        fake = (low + noise_high).astype(np.float32)

        out = high_band_distance(r, fake, condition="hifigan_v1")
        assert 0.5 < out["hb_frac_ratio"] < 2.0, "energy should look about right"
        assert out["hb_lsd"] > 3.0, "but the structure should not"
        assert out["hb_corr"] < 0.9

    def test_rejects_unaligned_pairs(self):
        from metrics.spectral import high_band_distance

        with pytest.raises(InvariantViolation, match="INV-06"):
            high_band_distance(self._real(), self._real(ARCHIVE_SR // 2), condition="melgan")

    def test_rejects_a_band_outside_nyquist(self):
        from metrics.spectral import high_band_distance

        r = self._real()
        with pytest.raises(InvariantViolation, match="not inside"):
            high_band_distance(r, r, condition="melgan", lo=12_000.0)

    @pytest.mark.parametrize("cond", ["griffin_lim", "vocos"])
    def test_refuses_high_band_excluded_conditions(self, cond):
        from metrics.spectral import high_band_distance

        r = self._real()
        with pytest.raises(InvariantViolation, match="INV-18.*high_band_distance refuses"):
            high_band_distance(r, r, condition=cond)

    def test_refusal_quotes_the_reason(self):
        from metrics.spectral import high_band_distance

        r = self._real()
        with pytest.raises(InvariantViolation) as exc:
            high_band_distance(r, r, condition="griffin_lim")
        assert "mel-inversion" in str(exc.value)

    def test_condition_is_required(self):
        """No default, so the refusal cannot be bypassed by omitting it."""
        from metrics.spectral import high_band_distance

        r = self._real()
        with pytest.raises(TypeError):
            high_band_distance(r, r)

    def test_runner_records_exclusion_instead_of_scoring(self):
        import inspect

        from metrics import runner

        src = inspect.getsource(runner.evaluate_archive)
        assert "exclusion_for(HIGH_BAND_ANALYSIS" in src
        assert "hb_excluded" in src

    def test_wired_into_the_runner(self):
        import inspect

        from metrics import runner

        src = inspect.getsource(runner.evaluate_archive)
        assert "high_band_distance" in src


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



class TestBandwidthPairs:
    """Task 3: two pairs, isolating different things."""

    def _detection(self):
        return pd.DataFrame(
            {
                "condition": ["melgan", "bigvgan_112m", "bigvgan_v2_22khz_fullband"],
                "protocol": ["matched"] * 3,
                "tier": [ARCHIVE_TIER] * 3,
                "detector": ["aasist"] * 3,
                "eer": [0.20, 0.30, 0.10],
            }
        )

    def test_only_the_bigvgan_pair_remains(self):
        """The melgan pair was one checkpoint with and without the
        generation-time filter; a full-band archive makes both sides identical."""
        from detectors.protocols import BANDWIDTH_PAIRS

        assert set(BANDWIDTH_PAIRS) == {"bigvgan"}

    def test_bigvgan_pair_resolves(self):
        from detectors.protocols import paired_bandwidth_contrast

        out = paired_bandwidth_contrast(self._detection(), pair="bigvgan")
        assert out["delta_eer"] == pytest.approx(-0.20)
        assert out["shared_weights"] is False


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

    def test_control_condition_also_agrees(self):
        """Every condition runs the identical pipeline now -- INV-17 no longer
        filters at generation, so there is no exempt branch to diverge."""
        from data.preprocess import (
            derive_trim_span,
            finalise_reference,
            process_condition_output,
        )

        wav = self._speechlike(seed=7)
        span = derive_trim_span(wav, "u7")
        cond = "bigvgan_v2_22khz_fullband"

        a, la, ha = finalise_reference(wav, span, condition=cond)
        b, lb, hb = process_condition_output(wav, ARCHIVE_SR, span, condition=cond)
        assert np.array_equal(a, b)
        assert la == lb and ha == hb

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


class TestMeasurementArtifacts:
    """TASK 0: the evidence is committed, and the BigVGAN pin traces to it."""

    def _git_ignored(self, rel):
        import subprocess

        r = subprocess.run(
            ["git", "check-ignore", "-q", rel], cwd=str(REPO_ROOT), capture_output=True
        )
        return r.returncode == 0

    def test_measurement_artifacts_are_not_ignored(self):
        for f in ("data/measurements/band_measurements.csv", "data/measurements/meta.json",
                  "data/measurements/probes.json"):
            assert not self._git_ignored(f), f"{f} is ignored; INV-18 cites it by row"

    def test_probes_measured_exactly_the_pinned_resampler(self):
        """The evidence for Vocos's primary membership is about THIS resampler."""
        from data.measurements import load_probes
        from vocoders.registry import get_spec

        meta, pin = load_probes(), get_spec("vocos").resampler
        for k, v in meta["kaiser_params"].items():
            assert getattr(pin, k) == v, f"probes.json {k}={v!r}, pin {getattr(pin, k)!r}"
        assert meta["ladder_fmax"] == LADDER_FMAX
        assert "GroupKFold" in meta["cv"] and "utterance" in meta["cv"]
        assert meta["n_files"] == 200

    def test_probe_figures_quoted_in_the_docs_are_in_the_file(self):
        from data.measurements import load_probes

        p = load_probes()["probes"]
        quoted = {  # ratio, below, below_std, above, above_std, gain delta dB
            "real_vs_rt_default": (0.8005, 0.5291, 0.0024, 1.0, 0.0, None),
            "real_vs_rt_kaiser": (0.9997, 0.5, 0.0, 0.9935, 0.0056, None),
            "real_vs_rt_soxr": (0.9922, 0.5, 0.0, 1.0, 0.0, None),
            "kaiser_1x_vs_2x": (0.9998, 0.5, 0.0, 0.9928, 0.0073, None),
            "real_vs_rt_default_norm": (0.8005, 0.5348, 0.0018, 1.0, 0.0, 0.02986),
            "real_vs_rt_kaiser_norm": (0.9998, 0.5, 0.0, 0.9989, 0.0011, 4.62e-05),
            "real_vs_rt_soxr_norm": (0.9922, 0.5, 0.0, 1.0, 0.0, 0.001534),
            "kaiser_1x_vs_2x_norm": (0.9998, 0.5, 0.0, 0.9941, 0.0065, 4.046e-05),
        }
        assert set(p) == set(quoted)
        for name, (ratio, below, below_std, above, above_std, gain) in quoted.items():
            assert p[name]["ratio_highband"] == pytest.approx(ratio, abs=5e-5), name
            assert p[name]["auroc_below"] == below, name
            assert p[name]["auroc_below_std"] == below_std, name
            assert p[name]["auroc_above"] == above, name
            assert p[name]["auroc_above_std"] == above_std, name
            assert p[name].get("mean_gain_delta_db") == gain, name

    def test_generated_audio_under_data_is_still_ignored(self):
        """The negation must not reopen data/ for generated audio."""
        for f in ("data/resynth/archive/real/x.wav", "data/LJSpeech-1.1/metadata.csv",
                  "data/stray.json", "data/measurements/x.wav"):
            assert self._git_ignored(f), f"{f} should be ignored"

    def test_csv_shape_matches_its_description(self):
        from data.measurements import load_csv

        df = load_csv()
        assert df.shape == (300, 6)
        assert set(df["condition"]) == {
            "real", "bigvgan", "vocos", "griffin_lim", "real_roundtrip"
        }
        assert df["utt_id"].nunique() == 60
        assert (df.groupby("utt_id")["samples"].nunique() == 1).all()

    def test_quoted_default_resampler_ratio_is_in_the_csv(self):
        """The one resampler row the artifacts DO contain: torchaudio default."""
        from data.measurements import load_csv

        df = load_csv()
        m = df.groupby("condition")["hb_8k_nyq"].mean()
        assert m["real_roundtrip"] / m["real"] == pytest.approx(0.8198, abs=5e-5)

    def test_bigvgan_file_digest_comes_from_meta(self):
        from data.measurements import load_meta
        from vocoders.registry import get_spec

        spec = get_spec("bigvgan_112m")
        assert spec.checkpoint_sha256 == load_meta()["bigvgan_sha256"]
        assert spec.checkpoint == load_meta()["bigvgan_repo"]

    def test_file_digest_does_not_overwrite_the_repo_revision(self):
        """Two different identifiers: a 40-hex commit and a 64-hex file digest."""
        from vocoders.registry import get_spec

        spec = get_spec("bigvgan_112m")
        assert re.fullmatch(r"[0-9a-f]{40}", spec.checkpoint_revision)
        assert re.fullmatch(r"[0-9a-f]{64}", spec.checkpoint_sha256)

    def test_bigvgan_mel_comes_from_meta_and_matches_the_audit(self):
        from data.mel import BIGVGAN_V2_22K_FMAX8K_MEL
        from vocoders.registry import get_spec

        mel = get_spec("bigvgan_112m").mel
        assert "meta.json" in mel.source
        for field in ("sample_rate", "n_fft", "hop_length", "win_length",
                      "n_mels", "fmin", "fmax"):
            assert getattr(mel, field) == getattr(BIGVGAN_V2_22K_FMAX8K_MEL, field), field


class TestResamplerPin:
    """TASK 2: Vocos resamples with pinned kaiser parameters, and only those."""

    PINNED = {
        "backend": "torchaudio",
        "resampling_method": "sinc_interp_kaiser",
        "lowpass_filter_width": 64,
        "rolloff": 0.9999,
        "beta": 14.769656459379492,
    }

    def test_vocos_resampler_parameters_are_pinned(self):
        from vocoders.registry import get_spec

        r = get_spec("vocos").resampler
        assert r is not None
        for k, v in self.PINNED.items():
            assert getattr(r, k) == v, f"{k} is {getattr(r, k)!r}, pinned {v!r}"

    def test_pin_is_not_the_torchaudio_default(self):
        """The default (width 6, rolloff 0.99) costs 18% of the high band."""
        from vocoders.registry import get_spec

        r = get_spec("vocos").resampler
        assert r.lowpass_filter_width != 6
        assert r.rolloff != 0.99

    def _cited(self, condition, metric):
        from data.measurements import verify
        from vocoders.registry import get_spec

        (ref,) = [m for m in get_spec("vocos").resampler.justification
                  if (m.condition, m.metric) == (condition, metric)]
        return verify(ref)

    def test_every_justification_resolves(self):
        from data.measurements import verify
        from vocoders.registry import get_spec

        for ref in get_spec("vocos").resampler.justification:
            verify(ref)

    def test_default_resampler_cost_is_a_committed_measurement(self):
        rt_mean = self._cited("real_roundtrip", "hb_8k_nyq")
        real_mean = self._cited("real", "hb_8k_nyq")
        assert 1.0 - rt_mean / real_mean == pytest.approx(0.1802, abs=5e-4)

    def test_pin_benefit_is_a_committed_measurement(self):
        """Formerly 'reported, not committed' (0.9998, 60 files). Now in probes.json."""
        ratio = self._cited("real_vs_rt_kaiser", "ratio_highband")
        assert ratio == pytest.approx(0.9997, abs=5e-5)
        # ...and it is the pin that keeps Vocos admissible below the band.
        assert self._cited("real_vs_rt_kaiser", "auroc_below") == 0.5
        assert self._cited("real_vs_rt_default", "auroc_below") == pytest.approx(0.5291)

    def test_only_vocos_declares_a_resampler(self):
        from vocoders.registry import ALL_CONDITIONS, get_spec

        declared = {c for c in ALL_CONDITIONS if get_spec(c).resampler is not None}
        assert declared == {"vocos"}

    def test_resample_with_spec_passes_exactly_the_pinned_kwargs(self, monkeypatch):
        """Verified against a stand-in, since torchaudio is absent by INV-13."""
        import types

        from data.preprocess import resample_with_spec
        from vocoders.registry import get_spec

        seen = {}

        class FakeTensor:
            def __init__(self, a):
                self.a = a

            def numpy(self):
                return self.a

        def fake_resample(x, orig_freq, new_freq, **kw):
            seen.update(kw, orig_freq=orig_freq, new_freq=new_freq)
            return FakeTensor(np.zeros(4, dtype=np.float32))

        torch_mod = types.ModuleType("torch")
        torch_mod.from_numpy = lambda a: a
        ta = types.ModuleType("torchaudio")
        taf = types.ModuleType("torchaudio.functional")
        taf.resample = fake_resample
        ta.functional = taf
        monkeypatch.setitem(sys.modules, "torch", torch_mod)
        monkeypatch.setitem(sys.modules, "torchaudio", ta)
        monkeypatch.setitem(sys.modules, "torchaudio.functional", taf)

        resample_with_spec(np.ones(8, dtype=np.float32), 24_000, ARCHIVE_SR,
                           get_spec("vocos").resampler)
        assert seen == {
            "lowpass_filter_width": 64,
            "rolloff": 0.9999,
            "resampling_method": "sinc_interp_kaiser",
            "beta": 14.769656459379492,
            "orig_freq": 24_000,
            "new_freq": ARCHIVE_SR,
        }

    def test_fails_closed_without_torchaudio(self, monkeypatch):
        """No silent fallback: another resampler leaves another rolloff."""
        from data.preprocess import resample_with_spec
        from vocoders.registry import get_spec

        monkeypatch.setitem(sys.modules, "torchaudio", None)
        monkeypatch.setitem(sys.modules, "torchaudio.functional", None)
        with pytest.raises(InvariantViolation, match="Do NOT substitute"):
            resample_with_spec(np.ones(8, dtype=np.float32), 24_000, ARCHIVE_SR,
                               get_spec("vocos").resampler)

    def test_same_rate_is_a_no_op(self):
        from data.preprocess import resample_with_spec
        from vocoders.registry import get_spec

        w = np.arange(8, dtype=np.float32)
        out = resample_with_spec(w, ARCHIVE_SR, ARCHIVE_SR, get_spec("vocos").resampler)
        assert np.array_equal(out, w)


class TestINV18ResampleProbe:
    """Where does a resample round trip's signature live?

    Two sources of evidence, kept apart:

    * COMMITTED -- data/measurements/probes.json: the pinned torchaudio kaiser
      resampler on 200 LJSpeech files, GroupKFold(5) by utterance. This is what
      Vocos's primary-analysis membership rests on
      (test_pinned_resampler_is_at_chance_below_the_band).
    * LOCAL -- synthetic audio round-tripped with RESAMPLE_METHOD (soxr_hq), since
      torchaudio is absent by INV-13. It carries loudness normalisation, and
      synthetic flat-spectrum audio loses far more high-band energy to soxr_hq than
      LJSpeech loses to the pin, so it reads 0.55-0.57 below the band: the coupling
      TestINV18LoudnessCoupling measures on real audio. It pins what is robust -- the
      signature is concentrated above the band -- and the grouped-CV trap.
    """

    N = 60

    @classmethod
    def _dataset(cls):
        if getattr(cls, "_cache", None) is not None:
            return cls._cache
        import librosa

        from data.invariants import RESAMPLE_METHOD
        from data.preprocess import band_limit_to_ladder, normalise_loudness

        def src(i):
            rng = np.random.default_rng(1000 + i)
            n = int(ARCHIVE_SR * (1.2 + 0.03 * (i % 10)))
            t = np.arange(n) / ARCHIVE_SR
            f0 = 120 + 7 * (i % 13)
            x = sum((0.35 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28))
                    for k in range(1, int(ARCHIVE_SR / 2 / f0)))
            x = x * (1 + 0.3 * np.sin(2 * np.pi * 3.1 * t)) + 0.02 * rng.standard_normal(n)
            x = x * np.hanning(n)
            return (0.35 * x / np.max(np.abs(x))).astype(np.float32)

        def pcm(w):
            return (np.round(np.clip(w, -1, 1) * 32767) / 32768).astype(np.float32)

        def rt(w):
            up = librosa.resample(w, orig_sr=ARCHIVE_SR, target_sr=24_000,
                                  res_type=RESAMPLE_METHOD)
            d = librosa.resample(up, orig_sr=24_000, target_sr=ARCHIVE_SR,
                                 res_type=RESAMPLE_METHOD)
            return d[:len(w)] if len(d) >= len(w) else np.pad(d, (0, len(w) - len(d)))

        def bands(w, edges):
            spec = np.abs(librosa.stft(w, n_fft=2048, hop_length=512)) ** 2
            f = librosa.fft_frequencies(sr=ARCHIVE_SR, n_fft=2048)
            return np.log10(np.array([spec[(f >= a) & (f < b)].mean()
                                      for a, b in edges]) + 1e-20)

        # The CURRENT pipeline: loudness on the full band, PCM_16, then the
        # analysis-time band limit (INV-17).
        sources = [src(i) for i in range(cls.N)]
        a = [pcm(normalise_loudness(rt(x), sr=ARCHIVE_SR)[0]) for x in sources]
        b = [pcm(normalise_loudness(x, sr=ARCHIVE_SR)[0]) for x in sources]
        lo_edges = [(f, f + 250) for f in range(0, int(LADDER_FMAX), 250)]
        hi_edges = [(f, f + 250) for f in range(int(LADDER_FMAX), 11_000, 250)]
        x_lo = np.vstack([[bands(band_limit_to_ladder(w, sr=ARCHIVE_SR), lo_edges) for w in a],
                          [bands(band_limit_to_ladder(w, sr=ARCHIVE_SR), lo_edges) for w in b]])
        x_hi = np.vstack([[bands(w, hi_edges) for w in a], [bands(w, hi_edges) for w in b]])
        y = np.r_[np.ones(cls.N), np.zeros(cls.N)]
        groups = np.r_[np.arange(cls.N), np.arange(cls.N)]
        cls._cache = (x_lo, x_hi, y, groups)
        return cls._cache

    @staticmethod
    def _auroc(x, y, cv):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        return float(cross_val_score(clf, x, y, cv=cv, scoring="roc_auc").mean())

    def _grouped(self, x, y, groups):
        from sklearn.model_selection import GroupKFold

        return list(GroupKFold(5).split(x, y, groups))

    def test_signature_is_separable_above_the_band(self):
        x_lo, x_hi, y, g = self._dataset()
        assert self._auroc(x_hi, y, self._grouped(x_hi, y, g)) >= 0.95

    def test_signature_is_concentrated_above_the_band(self):
        """What IS robust across every configuration measured: the round trip
        is far more separable above LADDER_FMAX than below it."""
        x_lo, x_hi, y, g = self._dataset()
        cv = self._grouped(x_lo, y, g)
        below, above = self._auroc(x_lo, y, cv), self._auroc(x_hi, y, cv)
        assert below <= 0.75, f"below-band AUROC {below:.3f}"
        assert above - below >= 0.25, f"above {above:.3f} vs below {below:.3f}"

    def test_pinned_resampler_is_at_chance_below_the_band(self):
        """INV-18: Vocos's primary-analysis membership, from committed evidence.

        Was a strict xfail against the local synthetic probe. Now driven by
        probes.json, and read through probe_reading, which refuses a below-band
        figure whose above-band control is not separable.
        """
        from data.measurements import probe_reading

        for name in ("real_vs_rt_kaiser", "real_vs_rt_kaiser_norm"):
            kaiser = probe_reading(name)
            assert kaiser.below_at_chance, f"{name}: deviation {kaiser.below_deviation:.4f}"
            assert kaiser.below_is_degenerate, name
            assert kaiser.control_auroc >= 0.99, name

    def test_the_torchaudio_default_would_not_be_at_chance_below(self):
        """Why the pin is load-bearing for the PRIMARY analysis, not only the high band."""
        from data.measurements import probe_reading

        default = probe_reading("real_vs_rt_default")
        assert not default.below_at_chance
        assert default.auroc_below - 2 * default.auroc_below_std > 0.52

    def test_paired_probe_must_group_by_utterance(self):
        """Shuffled folds split each utterance's two versions across train and
        test. Utterances differ far more than the two paths do, so the classifier
        is systematically inverted. Same data, wildly different answer."""
        from sklearn.model_selection import StratifiedKFold

        x_lo, _, y, g = self._dataset()
        grouped = self._auroc(x_lo, y, self._grouped(x_lo, y, g))
        leaky = self._auroc(x_lo, y, StratifiedKFold(5, shuffle=True, random_state=0))
        assert abs(grouped - leaky) >= 0.2, f"grouped {grouped:.3f} leaky {leaky:.3f}"


class TestINV18RejectedAlternatives:
    """The uniform round trip was considered and rejected; keep it from recurring."""

    def test_uniform_round_trip_rejection_is_recorded(self):
        from data.invariants import REJECTED_ALTERNATIVES

        rec = REJECTED_ALTERNATIVES["uniform_round_trip"]
        # 0.9903 (60 files, never committed) was replaced by the committed figure.
        assert rec.measured_value == pytest.approx(0.9928)
        assert "not idempotent" in rec.reason

    def test_rejection_evidence_is_committed_and_resolves(self):
        from data.invariants import REJECTED_ALTERNATIVES
        from data.measurements import verify

        rec = REJECTED_ALTERNATIVES["uniform_round_trip"]
        assert rec.evidence, "a rejection must cite committed evidence"
        for ref in rec.evidence:
            verify(ref)
        assert rec.in_committed_artifacts is True

    def test_committed_flag_cannot_outlive_its_citation(self):
        """in_committed_artifacts is derived: no citation of the value, no flag."""
        from dataclasses import replace

        from data.invariants import REJECTED_ALTERNATIVES

        rec = REJECTED_ALTERNATIVES["uniform_round_trip"]
        assert replace(rec, evidence=()).in_committed_artifacts is False
        assert replace(rec, measured_value=0.9903).in_committed_artifacts is False


class TestINV18ProbeControls:
    """TASK 2: a below-band 0.5000 is never read without its above-band control.

    probes.json reports exactly 0.5000 +- 0.0000 below LADDER_FMAX for three
    probes. That is a degenerate result -- a decision function that gives both
    members of every pair the same score -- not noisy chance, and it is only
    evidence of an absent artifact because the SAME pipeline separates the SAME
    pairs above the band. See docs/measurements.md, "Why 0.5000 is not a bug".
    """

    def test_every_committed_probe_has_a_separable_above_band_control(self):
        from data.invariants import PROBE_CONTROL_MIN_AUROC
        from data.measurements import load_probes, probe_reading

        for name, fields in load_probes()["probes"].items():
            assert fields["auroc_above"] >= PROBE_CONTROL_MIN_AUROC, name
            probe_reading(name)  # would raise without a separable control

    def test_a_below_band_reading_is_refused_without_its_control(self, tmp_path, monkeypatch):
        import json

        import data.measurements as dm

        broken = {"probes": {"x": {"ratio_highband": 1.0, "auroc_below": 0.5,
                                   "auroc_below_std": 0.0, "auroc_above": 0.61,
                                   "auroc_above_std": 0.02}}}
        (tmp_path / "probes_broken_control.json").write_text(json.dumps(broken))
        monkeypatch.setattr(dm, "MEASUREMENTS_DIR", tmp_path)
        with pytest.raises(InvariantViolation, match="above-band control"):
            dm.probe_reading("x", artifact="probes_broken_control.json")

    def test_same_band_positive_control_reads_live_features(self):
        """The above-band control rules out a pipeline fault in general. This rules
        out one confined to the below-band slice: the same probe, same band, reports
        a non-degenerate 0.5291 for the lossier default resampler."""
        from data.measurements import probe_reading

        default = probe_reading("real_vs_rt_default")
        assert not default.below_is_degenerate
        assert not default.below_at_chance

    @staticmethod
    def _paired_probe(shift, n=200, d=64, seed=0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold

        rng = np.random.default_rng(seed)
        base = (rng.normal(0, 1, (n, 1)) + rng.normal(-4, 1.5, (1, d))
                + rng.normal(0, 0.8, (n, d)))
        x = np.vstack([base + shift, base])
        y = np.r_[np.ones(n), np.zeros(n)]
        g = np.r_[np.arange(n), np.arange(n)]
        aucs, coefs, iters = [], [], []
        for tr, te in GroupKFold(5).split(x, y, g):
            clf = LogisticRegression(max_iter=5000).fit(x[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.decision_function(x[te])))
            coefs.append(float(np.abs(clf.coef_).max()))
            iters.append(int(clf.n_iter_[0]))
        return float(np.mean(aucs)), float(np.std(aucs)), max(coefs), max(iters)

    def test_exact_half_is_a_solver_that_never_left_zero(self):
        """The mechanism, reproduced. lbfgs starts at w = 0, where the gradient of
        the mean log-loss is 0.25 * (mean paired feature difference). Below its
        tolerance (1e-4) it stops at iteration 0: every score is the intercept,
        both members of every pair tie, and AUROC is exactly 0.5 in every fold."""
        auc, std, coef, it = self._paired_probe(shift=1e-4)
        assert (auc, std, coef, it) == (0.5, 0.0, 0.0, 0)

    def test_a_real_paired_shift_is_not_exactly_half(self):
        auc, std, coef, it = self._paired_probe(shift=1e-2)
        assert coef > 0 and it > 0
        assert auc > 0.51

    def test_noisy_chance_does_not_look_like_the_committed_result(self):
        """Genuinely uninformative labels give fold spread, not 0.0000."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold

        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, (400, 64))
        y = rng.permutation(np.r_[np.ones(200), np.zeros(200)])
        aucs = [roc_auc_score(y[te], LogisticRegression(max_iter=5000).fit(x[tr], y[tr])
                              .decision_function(x[te]))
                for tr, te in GroupKFold(5).split(x, y, np.arange(400))]
        assert np.std(aucs) > 0.02


class TestINV18LoudnessCoupling:
    """Does the below-band result survive the pipeline's loudness step? Committed.

    Loudness is normalised on the FULL band (INV-04), so a round trip that removes
    above-band energy gets a slightly larger gain, and that gain shifts every bin
    below LADDER_FMAX. probes.json measures it on 200 LJSpeech files in the
    pipeline order resample -> normalise -> quantise: keys ending ``_norm``, each
    with the mean gain delta it induced.

    This used to rest on a local synthetic dose-response. The committed file
    replaces it (kaiser: 4.62e-05 dB measured against 6.4e-05 dB estimated).
    """

    def test_file_records_both_loudness_variants_in_pipeline_order(self):
        from data.measurements import load_probes

        meta = load_probes()
        assert meta["target_lufs"] == LOUDNESS_TARGET_LUFS
        order = [s.strip() for s in meta["pipeline_order"].split("->")]
        assert order[0] == "resample" and order[-1] == "quantise"
        assert order[1].startswith("loudness normalise")
        probes = meta["probes"]
        raw = {k for k in probes if not k.endswith("_norm")}
        assert {f"{k}_norm" for k in raw} == set(probes) - raw
        for k, v in probes.items():
            assert v["loudness_normalised"] is k.endswith("_norm"), k
            assert ("mean_gain_delta_db" in v) is k.endswith("_norm"), k

    def test_pinned_resampler_survives_loudness_normalisation(self):
        from data.measurements import probe_reading

        kaiser = probe_reading("real_vs_rt_kaiser_norm")
        assert kaiser.loudness_normalised
        assert kaiser.mean_gain_delta_db == pytest.approx(4.62e-05)
        assert kaiser.below_is_degenerate and kaiser.below_at_chance
        assert kaiser.control_auroc >= 0.99

    def test_soxr_gain_delta_is_not_detected_either(self):
        from data.measurements import probe_reading

        soxr = probe_reading("real_vs_rt_soxr_norm")
        assert soxr.mean_gain_delta_db == pytest.approx(0.001534)
        assert soxr.below_is_degenerate

    def test_coupling_is_live_at_the_default_resampler(self):
        """Positive control: the same step DOES add a below-band cue when the gain
        delta is large enough, so the pinned 0.5000 is not a test that cannot fail."""
        from data.measurements import probe_reading

        raw, norm = probe_reading("real_vs_rt_default"), probe_reading("real_vs_rt_default_norm")
        assert norm.mean_gain_delta_db == pytest.approx(0.02986)
        assert not norm.below_at_chance
        assert norm.auroc_below > raw.auroc_below

    # --- TASK 2: a threshold with a margin, not a pass/fail ---------------------
    def test_threshold_is_a_bracket_between_null_and_detection(self):
        from data.measurements import loudness_coupling

        c = loudness_coupling()
        assert (c.highest_null_probe, c.lowest_detected_probe) == (
            "real_vs_rt_soxr_norm", "real_vs_rt_default_norm"
        )
        assert c.highest_null_db == pytest.approx(0.001534)
        assert c.lowest_detected_db == pytest.approx(0.02986)

    def test_pinned_gain_delta_keeps_its_margin_below_the_highest_null(self):
        """A future resampler change cannot silently cross the threshold.

        Two links. probes.json measured exactly VOCOS_RESAMPLER, so changing the
        pin breaks that equality and forces a re-measurement; and the re-measured
        gain delta must then sit LOUDNESS_COUPLING_MIN_MARGIN below the highest
        null. The committed margin is 33x: it clears 10x, not 100x.
        """
        from data.invariants import LOUDNESS_COUPLING_MIN_MARGIN
        from data.measurements import load_probes, loudness_coupling
        from vocoders.registry import get_spec

        pin = get_spec("vocos").resampler
        for k, v in load_probes()["kaiser_params"].items():
            assert getattr(pin, k) == v, f"pin changed ({k}); re-measure probes.json"
        c = loudness_coupling()
        assert c.margin >= LOUDNESS_COUPLING_MIN_MARGIN, f"margin {c.margin:.1f}x"
        assert c.margin == pytest.approx(33.2, abs=0.1)
        assert c.margin_to_detected == pytest.approx(646.3, abs=0.5)

    def test_bracket_is_this_probes_resolution(self):
        """Why the 'threshold' is where it is. A gain delta shifts every
        log-magnitude feature equally; the solver never leaves zero while
        0.25 * shift < tol. Reproduced at the committed deltas, under either log
        convention the file might have used, on synthetic paired features."""
        import math

        probe = TestINV18ProbeControls._paired_probe
        for per_db in (20.0 / math.log(10.0), 20.0):   # natural-log, log10 amplitude
            assert probe(shift=0.001534 / per_db)[:2] == (0.5, 0.0)
            assert probe(shift=0.02986 / per_db)[0] > 0.5


class TestINV17BandLimitResponse:
    """TASK 3: the band-limited primary analysis and the high-band analysis meet
    at LADDER_FMAX.

    Measured DIRECTLY -- impulse through the implementation, and pure tones --
    never energy-weighted. An energy-weighted passband figure ("0.004 dB below
    7 kHz") is what hid a design whose -1 dB point was at 6419 Hz.
    """

    N = 2**17
    _resp: tuple | None = None

    @classmethod
    def _response(cls):
        if cls._resp is None:
            from data.preprocess import band_limit_to_ladder

            imp = np.zeros(cls.N, dtype=np.float32)
            imp[cls.N // 2] = 1.0
            h = band_limit_to_ladder(imp, sr=ARCHIVE_SR).astype(np.float64)
            db = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(h)), 1e-300))
            cls._resp = (np.fft.rfftfreq(cls.N, 1 / ARCHIVE_SR), db)
        return cls._resp

    def _first_below(self, level_db):
        f, db = self._response()
        return float(f[np.nonzero(db < level_db)[0][0]])

    def test_minus_1db_point_is_pinned(self):
        assert self._first_below(-1.0) == pytest.approx(7915.5, abs=2.0)

    def test_minus_20db_point_is_pinned(self):
        assert self._first_below(-20.0) == pytest.approx(7940.3, abs=2.0)

    def test_minus_3db_point(self):
        assert self._first_below(-3.0) == pytest.approx(7922.9, abs=2.0)

    def test_tones_agree_with_the_impulse_measurement(self):
        from data.preprocess import band_limit_to_ladder

        t = np.arange(4 * ARCHIVE_SR) / ARCHIVE_SR
        for f0, want in ((7915.5, -1.0), (7922.9, -3.0), (7940.3, -20.0)):
            x = np.sin(2 * np.pi * f0 * t).astype(np.float32)
            y = band_limit_to_ladder(x, sr=ARCHIVE_SR).astype(np.float64)
            mid = slice(ARCHIVE_SR, 3 * ARCHIVE_SR)
            ref = np.mean(x[mid].astype(np.float64) ** 2)
            got = 10 * np.log10(np.mean(y[mid] ** 2) / ref)
            assert got == pytest.approx(want, abs=0.1), f"{f0} Hz: {got:.3f} dB"

    def test_passband_is_flat_up_to_the_transition(self):
        from data.invariants import BAND_LIMIT_PASSBAND_LOSS_DB, BAND_LIMIT_TRANSITION_HZ

        f, db = self._response()
        passband = f <= LADDER_FMAX - BAND_LIMIT_TRANSITION_HZ
        assert np.abs(db[passband]).max() <= BAND_LIMIT_PASSBAND_LOSS_DB + 1e-3

    def test_nothing_at_or_above_ladder_fmax_survives(self):
        """The transition may narrow but never straddle LADDER_FMAX: attenuated
        above-band content in the primary analysis is Griffin-Lim's cliff again.

        Reads -156 dB: the float32 floor of this impulse measurement (the design
        is -200 dB zero-phase). A transition straddling LADDER_FMAX reads tens
        of dB here, so the margin is not what the check depends on."""
        f, db = self._response()
        assert db[f >= LADDER_FMAX].max() <= -150.0

    def test_the_gap_between_the_analyses_is_the_declared_transition(self):
        from data.invariants import BAND_LIMIT_TRANSITION_HZ

        gap = LADDER_FMAX - self._first_below(-0.1)
        assert 0 < gap <= BAND_LIMIT_TRANSITION_HZ + 1.0

    def test_high_band_analysis_starts_at_the_band_limit_stopband_edge(self):
        import inspect

        from metrics.spectral import high_band_distance

        assert inspect.signature(high_band_distance).parameters["lo"].default == LADDER_FMAX

    def test_order_is_derived_not_chosen(self):
        from scipy.signal import cheb2ord

        from data.invariants import (
            BAND_LIMIT_FILTER_ORDER,
            BAND_LIMIT_PASSBAND_LOSS_DB,
            BAND_LIMIT_STOPBAND_DB,
            BAND_LIMIT_TRANSITION_HZ,
            band_limit_order,
        )
        from data.preprocess import band_limit_sos

        nyq = ARCHIVE_SR / 2
        for cutoff in (LADDER_FMAX, 7600.0, 10_000.0):
            n, _ = cheb2ord((cutoff - BAND_LIMIT_TRANSITION_HZ) / nyq, cutoff / nyq,
                            BAND_LIMIT_PASSBAND_LOSS_DB / 2, BAND_LIMIT_STOPBAND_DB)
            assert band_limit_order(ARCHIVE_SR, cutoff) == n
        assert BAND_LIMIT_FILTER_ORDER == 53
        assert len(band_limit_sos()) == 27

    def test_the_old_order_12_design_fails_these_pins(self):
        """Mutation check: the pins would have caught the 1.6 kHz hole."""
        from scipy.signal import cheby2, sosfiltfilt

        sos = cheby2(12, 100, LADDER_FMAX / (ARCHIVE_SR / 2), btype="low", output="sos")
        imp = np.zeros(self.N)
        imp[self.N // 2] = 1.0
        db = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(sosfiltfilt(sos, imp))), 1e-300))
        f = np.fft.rfftfreq(self.N, 1 / ARCHIVE_SR)
        minus1 = float(f[np.nonzero(db < -1.0)[0][0]])
        assert minus1 == pytest.approx(6418.7, abs=2.0)
        assert minus1 != pytest.approx(7915.5, abs=2.0)
