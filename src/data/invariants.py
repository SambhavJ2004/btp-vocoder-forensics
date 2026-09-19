"""Hard invariants of the controlled-resynthesis design.

Every constant here is a *confound control*, not a tunable. Changing one
silently invalidates every number the study has produced up to that point,
because the measured EER difference between conditions would no longer be
attributable to vocoder identity alone.

The full rationale for each invariant lives in CLAUDE.md. This module is the
machine-readable copy: code imports from here so that no pipeline can quietly
disagree with the documented design.

If a value must change, it changes ONCE, before any audio is generated, and
every condition -- including REAL -- is regenerated from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

# --- INV-01  Two rates, two single-step derivations ---------------------------
# ARCHIVE is the primary artifact: 22.05 kHz mono, forced by LJSpeech's native
# rate. Nyquist is 11.025 kHz, which keeps the high band the study exists to
# measure. Vocoders resample once, native -> archive.
#
# ZEROSHOT is a SEPARATE DERIVED artifact at 16 kHz, produced by one further
# downsample from the archive, because pretrained ASVspoof detectors, PESQ-WB
# and UTMOS all require 16 kHz. It is derived from the archive, NEVER from
# source, and never fed back into anything.
#
# Two derivations, each a single filtering step. Never chain.
ARCHIVE_SR: Final[int] = 22_050
ZEROSHOT_SR: Final[int] = 16_000
SANCTIONED_RATES: Final[tuple[int, ...]] = (ARCHIVE_SR, ZEROSHOT_SR)
CHANNELS: Final[int] = 1
RESAMPLE_METHOD: Final[str] = "soxr_hq"  # librosa res_type; frozen across conditions

# INV-01 / manifest provenance: no record may carry more than one resample step.
MAX_RESAMPLE_STEPS: Final[int] = 1

# Tier names as they appear in the manifest.
ARCHIVE_TIER: Final[str] = "archive"
ZEROSHOT_TIER: Final[str] = "zeroshot"
TIERS: Final[tuple[str, ...]] = (ARCHIVE_TIER, ZEROSHOT_TIER)
TIER_RATES: Final[dict[str, int]] = {ARCHIVE_TIER: ARCHIVE_SR, ZEROSHOT_TIER: ZEROSHOT_SR}

# --- INV-05  Encoding --------------------------------------------------------
# One container, one bit depth, everywhere. No lossy intermediates, ever.
CONTAINER: Final[str] = ".wav"
SUBTYPE: Final[str] = "PCM_16"
DTYPE: Final[str] = "float32"  # in-memory representation before write

# --- INV-03  Silence ---------------------------------------------------------
# Trim boundaries are computed ONCE on the real reference, at ARCHIVE_SR, and
# reused verbatim for every vocoder condition of the same utterance. Running a
# trimmer independently per condition reintroduces the silence-duration
# confound the trim exists to remove.
#
# Frame/hop are sample counts, so their duration follows the archive rate:
# at 22.05 kHz, 512 samples is 23.2 ms (it was 32 ms at 16 kHz). Unchanged by
# the rate migration on purpose -- trim logic is explicitly out of scope.
TRIM_TOP_DB: Final[float] = 30.0
TRIM_FRAME_LENGTH: Final[int] = 512
TRIM_HOP_LENGTH: Final[int] = 128
TRIM_PAD_MS: Final[float] = 20.0  # symmetric context kept around the trim points

# --- INV-04  Loudness --------------------------------------------------------
# Identical normalisation target across all conditions, applied ONCE at the
# archive tier. A conservative target is used so that per-file peak protection
# (which would break "identical") almost never triggers; when it does, the
# utterance is dropped from ALL conditions rather than rescaled in one.
#
# The derived zeroshot tier is NOT re-normalised -- see check_waveform.
LOUDNESS_TARGET_LUFS: Final[float] = -27.0
TRUE_PEAK_CEILING_DBFS: Final[float] = -1.0
LOUDNESS_TOLERANCE_LU: Final[float] = 0.5  # verification tolerance, not a knob

# Downsampling to 16 kHz discards everything above 8 kHz, which lowers measured
# loudness slightly. That drift is recorded, not corrected: re-normalising the
# derived tier would apply a gain proportional to each file's high-band energy,
# which is precisely the artifact under study. See CLAUDE.md INV-01.
DERIVED_LOUDNESS_DRIFT_LIMIT_LU: Final[float] = 2.0

# INV-04, made observable. Every archive manifest row records both loudness
# measurements, the gain applied and the target, so the gain each condition
# received is a recorded fact rather than an estimate. The recorded gain is
# measured from the arrays and must equal target - lufs_in_BAND to within this;
# anything more means the normaliser did something other than one gain, or
# referenced something other than the analysis band.
LOUDNESS_GAIN_RECORD_TOLERANCE_DB: Final[float] = 1e-3

# The smallest uniform level offset the committed linear probe resolves: lbfgs
# stays at zero while 0.25 * shift < tol (1e-4), i.e. ~0.0035 dB for natural-log
# magnitude features (~0.008 dB for log10). Context for reading the loudness
# report, NOT a gate -- a stronger detector could resolve less. See INV-18.
LOUDNESS_PROBE_RESOLUTION_DB: Final[float] = 0.0035

# --- INV-17  Analysis band (NOT a generation-time filter) --------------------
# The audited mel `fmax` of every condition, transcribed from the checkpoint's
# OWN config. Source for each row: docs/mel_configs.md. This is the single
# source of truth -- data.mel builds its MelConfig entries from it, so the table
# and the code cannot drift apart.
#
# `None` in a released config does NOT mean "unbounded": librosa resolves it to
# sr/2, so it is recorded here already resolved, at 22050/2 = 11025.0.
AUDITED_MEL_FMAX: Final[dict[str, float]] = {
    "melgan": 11_025.0,        # Audio2Mel default mel_fmax=None -> librosa sr/2
    "hifigan_v1": 8_000.0,     # jik876/hifi-gan config_v1.json
    "vocos": 12_000.0,         # torchaudio MelSpectrogram default f_max=sr/2 @24k
    "bigvgan_base": 8_000.0,   # nvidia/bigvgan_base_22khz_80band
    "bigvgan_112m": 8_000.0,   # nvidia/bigvgan_v2_22khz_80band_fmax8k_256x
    "bigvgan_v2_22khz_fullband": 11_025.0,  # nvidia/bigvgan_v2_22khz_80band_256x
    # Declared but unavailable -- kept for the audit trail, not in any ladder.
    "specdiff_gan": 8_000.0,   # SpecDiff-GAN configs/config_ljspeech.json
}

# Conditions whose weights are not obtainable. Recorded so the decision to drop
# them is visible in the audit trail rather than being an unexplained absence,
# and so a substitute can be compared against what it replaced.
UNAVAILABLE_CONDITIONS: Final[frozenset[str]] = frozenset({"specdiff_gan"})

# Griffin-Lim has no checkpoint: it is analysis/synthesis with no learned prior,
# so its front-end is a project decision rather than a model property. It is set
# TO the analysis band and therefore does not constrain it.
FMAX_FREE_CONDITIONS: Final[frozenset[str]] = frozenset({"griffin_lim"})

# The primary ladder -- everything generated as a rung and reported as one.
PRIMARY_LADDER: Final[tuple[str, ...]] = (
    "griffin_lim",
    "melgan",
    "hifigan_v1",
    "vocos",          # replaces specdiff_gan, whose weights are unavailable
    "bigvgan_base",
    "bigvgan_112m",
)

# --- INV-18  Per-analysis membership, justified by measurement --------------
# Two analyses run over the ladder, and they have DIFFERENT membership, because
# the confounds that disqualify a condition are band-localised.
#
#   PRIMARY_ANALYSIS    the headline correlation, BAND-LIMITED to LADDER_FMAX.
#   HIGH_BAND_ANALYSIS  [LADDER_FMAX, Nyquist) on the full-band archive.
#
# A condition is excluded from an analysis only when its processing history
# introduces an artifact INSIDE that analysis's band, and every exclusion names
# the committed measurement that shows it. See CLAUDE.md INV-18.
PRIMARY_ANALYSIS: Final[str] = "primary_band_limited"
HIGH_BAND_ANALYSIS: Final[str] = "high_band"
ANALYSES: Final[tuple[str, ...]] = (PRIMARY_ANALYSIS, HIGH_BAND_ANALYSIS)


@dataclass(frozen=True)
class MeasurementRef:
    """A citation of one statistic in a committed measurement artifact.

    Resolved and re-checked against ``data/measurements/`` by
    :func:`data.measurements.verify` -- the value here is a claim about the file,
    not a number to be trusted on its own.
    """

    artifact: str       # file under data/measurements/
    condition: str      # CSV: row selector, in the ARTIFACT's own naming.
                        # probes.json: the probe name, e.g. "real_vs_rt_kaiser"
    metric: str         # CSV: a column, or "colA-colB" for a band difference.
                        # probes.json: a recorded field, e.g. "auroc_below"
    statistic: str      # CSV: mean | median | min | max | auroc_vs:<condition>
                        # probes.json: "recorded" -- the run computed it, the
                        # file stores the result, nothing here recomputes it
    value: float        # what that statistic evaluated to
    rel_tol: float = 1e-4

    def describe(self) -> str:
        return (
            f"{self.artifact}[condition={self.condition}] "
            f"{self.statistic}({self.metric})"
        )


@dataclass(frozen=True)
class ResamplerSpec:
    """A resampler, pinned by every parameter that changes its output.

    Declared per condition, because the one condition that resamples on its way
    to the archive (Vocos) needs a sharper resampler than RESAMPLE_METHOD, and
    the choice is not cosmetic: the torchaudio DEFAULT round trip costs 18% of
    the band above LADDER_FMAX (committed measurement, real_roundtrip vs real).
    A parameter left to a library default is a parameter that changes when the
    library does.
    """

    backend: str                      # "torchaudio"
    resampling_method: str            # e.g. "sinc_interp_kaiser"
    lowpass_filter_width: int
    rolloff: float
    beta: float | None = None
    justification: tuple[MeasurementRef, ...] = ()


@dataclass(frozen=True)
class Exclusion:
    """Why one condition is kept out of one analysis, and the evidence for it."""

    condition: str
    analysis: str
    reason: str
    measurements: tuple[MeasurementRef, ...]

    def render(self) -> str:
        cites = "; ".join(f"{m.describe()} = {m.value:.4g}" for m in self.measurements)
        return f"{self.condition} ({self.analysis}): {self.reason} [evidence: {cites}]"


_BAND_CSV = "band_measurements.csv"
_PROBES = "probes.json"
# Energy in [rt_edge, Nyquist) = [10900, 11025) Hz: the sliver where a resampler's
# transition band leaves its fingerprint. meta.json records rt_edge = 10900.
_EDGE = "hb_8k_nyq-hb_8k_rtedge"

ANALYSIS_EXCLUSIONS: Final[dict[str, dict[str, Exclusion]]] = {
    # Nothing is excluded from the band-limited primary analysis. Both artifacts
    # that disqualify conditions elsewhere -- Griffin-Lim's mel-inversion cliff
    # and Vocos's resampler edge -- live ABOVE LADDER_FMAX, and this analysis
    # removes that band. griffin_lim is back in the headline correlation on
    # purpose, not by oversight.
    PRIMARY_ANALYSIS: {},
    HIGH_BAND_ANALYSIS: {
        "griffin_lim": Exclusion(
            condition="griffin_lim",
            analysis=HIGH_BAND_ANALYSIS,
            reason=(
                "mel-inversion bandwidth cliff. Griffin-Lim inverts the mel to a "
                "linear spectrogram and runs ISTFT, so it cannot emit above its "
                "mel fmax: its high-band fraction sits at the PCM_16 quantisation "
                "floor, four orders of magnitude below real's minimum. Anything "
                "measured for it in this band measures that absence, not a "
                "reconstruction artifact."
            ),
            measurements=(
                MeasurementRef(_BAND_CSV, "griffin_lim", "hb_8k_nyq", "max", 5.492221e-08),
                MeasurementRef(_BAND_CSV, "real", "hb_8k_nyq", "min", 6.383930e-04),
            ),
        ),
        "vocos": Exclusion(
            condition="vocos",
            analysis=HIGH_BAND_ANALYSIS,
            reason=(
                "resampler signature. Vocos is 24 kHz native and is the only "
                "ladder condition resampled to the archive rate. In the band "
                "[rt_edge, Nyquist) it carries ~118x the energy of BigVGAN, is "
                "separable from BigVGAN at AUROC 0.9997, and is INDISTINGUISHABLE "
                "from resampled real audio (AUROC 0.507) -- so what distinguishes "
                "it there is the resampler, not the vocoder. Those sliver figures "
                "describe a Vocos resampled with the torchaudio DEFAULT, before "
                "the pin; under the pinned kaiser resampler the round trip is "
                "still separable above LADDER_FMAX (AUROC 0.9935), and under the "
                "pipeline's own loudness normalisation more so (0.9989 +- 0.0011). "
                "The un-normalised probe understated the exclusion's support, if "
                "anything; it did not overstate it."
            ),
            measurements=(
                MeasurementRef(_BAND_CSV, "vocos", _EDGE, "mean", 8.630273e-07),
                MeasurementRef(_BAND_CSV, "bigvgan", _EDGE, "mean", 7.295493e-09),
                MeasurementRef(_BAND_CSV, "vocos", _EDGE, "auroc_vs:bigvgan", 0.999722),
                MeasurementRef(
                    _BAND_CSV, "vocos", _EDGE, "auroc_vs:real_roundtrip", 0.507222,
                    rel_tol=1e-3,
                ),
                # the imprint under the resampler Vocos actually uses now...
                MeasurementRef(_PROBES, "real_vs_rt_kaiser", "auroc_above", "recorded", 0.9935),
                # ...and under the pipeline order: resample -> normalise -> quantise
                MeasurementRef(
                    _PROBES, "real_vs_rt_kaiser_norm", "auroc_above", "recorded", 0.9989
                ),
            ),
        ),
    },
}


@dataclass(frozen=True)
class RejectedAlternative:
    """A design considered and rejected, kept so it is not proposed again.

    ``evidence`` cites the deciding measurement in data/measurements/.
    ``in_committed_artifacts`` is DERIVED from it rather than set by hand: it
    used to be a hand-written flag, and a flag can say True while the citation
    is missing. A rejection resting on a number nobody can recompute is weaker
    than one that can, and the difference should be visible in code.
    """

    name: str
    reason: str
    measured_value: float
    measurement: str
    evidence: tuple[MeasurementRef, ...] = ()

    @property
    def in_committed_artifacts(self) -> bool:
        return any(m.value == self.measured_value for m in self.evidence)


REJECTED_ALTERNATIVES: Final[dict[str, RejectedAlternative]] = {
    "uniform_round_trip": RejectedAlternative(
        name="uniform round trip",
        reason=(
            "Proposed fix for the Vocos resampler signature: pass EVERY condition "
            "through the same 22050 -> 24000 -> 22050 round trip, so Vocos is no "
            "longer the only one carrying a resampler imprint. Rejected because "
            "resampling is not idempotent. A second round trip is itself "
            "separable from one, so adding a round trip to the other conditions "
            "does not make Vocos path-identical to them; it gives every condition "
            "an imprint and Vocos a different one. Vocos is instead excluded from "
            "the band where its imprint lives (the high-band analysis) and kept "
            "in the band where it does not (the band-limited primary analysis)."
        ),
        # Above the band, where the imprint lives and where a uniform round trip
        # would have had to make Vocos path-identical. The 0.9903 previously
        # recorded here came from a 60-file run and is NOT in any committed
        # artifact; the committed 200-file figure replaces it.
        measured_value=0.9928,
        measurement=(
            "AUROC above LADDER_FMAX, one torchaudio-kaiser round trip vs two, "
            "200 LJSpeech files, PCM_16, GroupKFold(5) by utterance"
        ),
        evidence=(
            MeasurementRef(_PROBES, "kaiser_1x_vs_2x", "auroc_above", "recorded", 0.9928),
            MeasurementRef(_PROBES, "kaiser_1x_vs_2x", "auroc_full", "recorded", 0.9918),
            # Below the band the two are indistinguishable. Consistent with the
            # rejection, not against it: below LADDER_FMAX no round trip was
            # needed in the first place.
            MeasurementRef(_PROBES, "kaiser_1x_vs_2x", "auroc_below", "recorded", 0.5),
            # Under loudness normalisation, the pipeline's own order. Kept as
            # corroboration; measured_value stays at the lower, un-normalised
            # figure.
            MeasurementRef(_PROBES, "kaiser_1x_vs_2x_norm", "auroc_above", "recorded", 0.9941),
        ),
    ),
}

# --- INV-18  Reading a committed paired probe --------------------------------
# data/measurements/probes.json records grouped-CV AUROCs below LADDER_FMAX,
# above it, and full-band. A below-band figure is only interpretable next to its
# above-band CONTROL: the same features, labels, folds and classifier, on the
# band where the round trip is known to leave an imprint. If the control is not
# separable, a below-band 0.5 could be a broken pipeline rather than an absent
# artifact. data.measurements.probe_reading refuses to return one without the
# other.
#
# CONTROL: the above-band probe must reach this. The weakest committed control
# is kaiser 1x vs 2x at 0.9928.
PROBE_CONTROL_MIN_AUROC: Final[float] = 0.95
# AT CHANCE: |auroc_below - 0.5| + 2 * fold_std must not exceed this. The fold
# std is included so a noisy 0.5 does not pass. Validated against a positive
# control in the same band: the torchaudio DEFAULT round trip reads 0.5291 +-
# 0.0024 below the band and must be (and is) classified NOT at chance.
PROBE_CHANCE_MAX_DEVIATION: Final[float] = 0.01

# --- INV-18  Loudness coupling: a threshold with a margin, not a gate --------
# Loudness is normalised on the FULL band (INV-04). A condition that lost
# high-band energy therefore gets a slightly larger gain, and the gain moves
# every bin BELOW LADDER_FMAX too. probes.json measures it on 200 LJSpeech files
# in the pipeline order resample -> normalise -> quantise (keys ending _norm):
#
#     round trip    mean gain delta   AUROC below LADDER_FMAX, raw -> normalised
#     default       0.02986 dB        0.5291 -> 0.5348      detected
#     soxr          0.001534 dB       0.5000 -> 0.5000      not detected
#     kaiser (pin)  4.62e-05 dB       0.5000 -> 0.5000      not detected
#
# Read it as a BRACKET. Onset lies somewhere in (0.001534, 0.02986] dB; three
# points do not locate it more finely than that factor of ~19.
#
# And read it as a property of THIS probe, not of the coupling. A gain delta g
# shifts every log-magnitude feature by the same amount, and lbfgs never leaves
# zero while 0.25 * shift < tol (1e-4): g < ~0.0035 dB for natural-log
# features, ~0.008 dB for log10. That predicts exactly this bracket. The gain
# shift itself does not vanish below 0.0015 dB; it falls below what a linear
# probe on mean spectra resolves. A stronger detector could resolve less.
#
# The operating point is the pinned resampler: 33x below the highest measured
# null (soxr), 646x below the lowest detected delta (default). A future
# resampler must keep this factor below the highest null; the committed margin
# is 33x, which clears 10x but NOT 100x.
LOUDNESS_COUPLING_OPERATING_PROBE: Final[str] = "real_vs_rt_kaiser_norm"
LOUDNESS_COUPLING_MIN_MARGIN: Final[float] = 10.0


def _derive_ladder_fmax() -> float:
    """LADDER_FMAX = min audited fmax over the constraining primary conditions.

    Deliberately computed, never written as a literal. If a checkpoint is
    re-sourced -- swapping MelGAN from `descriptinc` (11025) to a
    ParallelWaveGAN LJSpeech recipe (7600), say -- the ladder band follows the
    audit table instead of silently disagreeing with it.
    """
    constraining = {
        c: f
        for c, f in AUDITED_MEL_FMAX.items()
        if c in PRIMARY_LADDER
        and c not in FMAX_FREE_CONDITIONS
        and c not in UNAVAILABLE_CONDITIONS
    }
    if not constraining:
        raise RuntimeError("INV-17: no audited fmax values for the primary ladder.")
    return float(min(constraining.values()))


LADDER_FMAX: Final[float] = _derive_ladder_fmax()

# --- INV-17  What the band-limit is FOR --------------------------------------
# The archive is FULL-BAND. Band-limiting is an ANALYSIS-time transform applied
# to a comparison set on request, never a generation-time filter.
#
# This reverses the original design, which low-passed at generation on the
# premise that an fmax=8000 mel front-end yields no output above 8000. That
# premise is FALSE. `fmax` constrains the ANALYSIS the vocoder consumes, not the
# synthesis it performs: a time-domain upsampling vocoder emits content across
# the full band regardless of what the mel carried. Measured on 20 LJSpeech
# files, fraction of energy above 8 kHz:
#
#     real           0.01803
#     BigVGAN        0.01480    tracks real per-file
#     Griffin-Lim    0.00000    to 5 d.p. -- NOT exactly zero; see below
#
# The 60-file committed measurement (data/measurements/band_measurements.csv)
# shows Griffin-Lim's high-band fraction at median 5.1e-9, max 5.5e-8: the PCM_16
# quantisation floor, on no signal content. "Exactly zero" was a rounding
# artifact of a 5-decimal table. The conclusion survives -- real's MINIMUM is
# 6.4e-4, 1e4 x Griffin-Lim's maximum -- but the literal claim does not.
#
# Griffin-Lim is the exception, and it is the exception that produced the error:
# it inverts the mel to a linear spectrogram and runs ISTFT, so it genuinely
# cannot exceed fmax. The cliff was measured on Griffin-Lim and generalised by
# config inspection to conditions where it does not hold.
#
# What that cost: everything above LADDER_FMAX in a neural vocoder's output is
# HALLUCINATED -- the mel carried no information there, so the model invented it
# from its prior. That is the most forensically interesting content on the
# ladder, and generation-time filtering destroyed exactly it.
LADDER_FMAX_ROLE: Final[str] = "analysis band, applied on request; archive is full-band"

# Zero-phase Chebyshev Type II, applied identically to every member of a
# comparison set including REAL, at analysis time.
#
# Zero-phase because a causal filter's group delay would put the condition out
# of sample alignment with its reference and trip INV-16.
#
# Chebyshev II rather than Butterworth because an identical filter *attenuates*
# but does not *equalise*: real audio enters with energy above the band and an
# fmax=8000 vocoder enters with none, so whatever the filter leaves behind is
# still a difference between them. Measured on a harmonic-rich 22.05 kHz signal
# (fraction of total energy above the band):
#
#     unfiltered          1.95e-03
#     butterworth o8      2.55e-05      still 4000x the quantisation floor
#     butterworth o24     2.15e-06      still 350x
#     cheby2 o12 rs100    1.78e-16      gone
#
# The target is not "small" but "below the noise floor of the delivered format".
# PCM_16 quantisation at the -27 LUFS delivery level puts ~2e-9 of energy above
# the band on its own, so once the filter is well under that, the residual is
# absent from the delivered file rather than merely faint. Measured end to end
# on the synthetic corpus, real and Griffin-Lim land at 5.84e-09 and 5.71e-09 --
# 2% apart, both quantisation-dominated, against a 1.98e-03 gap before filtering.
#
# --- The transition band: specified by BOTH edges, order derived ------------
# The filter used to be specified by an order (12) and a stopband edge
# (LADDER_FMAX) alone. scipy's cheby2 takes Wn as the frequency where rejection
# FIRST reaches rs, so the passband edge was never chosen: it fell out of order
# and attenuation. At order 12 / 100 dB the prewarped -3 dB point sits a factor
# cosh(acosh(1e5)/12) = 1.56 below the stopband edge, and filtfilt squares the
# magnitude on top. Measured response of that design, zero-phase:
#
#     -0.1 dB 6144 Hz   -1 dB 6419 Hz   -3 dB 6555 Hz   -20 dB 6876 Hz
#
# So the primary analysis was effectively [0, ~6.4 kHz] and ~6.4-8 kHz sat in
# neither analysis -- a 1.6 kHz hole through the fricative region. Nothing
# pinned the passband and the only passband figure ever quoted was energy
# weighted ("0.004 dB below 7 kHz"), which speech's low-frequency energy makes
# small whatever the response does near the edge.
#
# Now: passband edge LADDER_FMAX - BAND_LIMIT_TRANSITION_HZ at a total (zero-
# phase) loss of BAND_LIMIT_PASSBAND_LOSS_DB, stopband edge LADDER_FMAX at
# BAND_LIMIT_STOPBAND_DB per pass, and the order is whatever satisfies both.
# Measured three ways (pure tones, impulse through the implementation, analytic
# sosfreqz), agreeing to 0.1 Hz:
#
#     -0.1 dB 7900.0 Hz   -1 dB 7915.5 Hz   -3 dB 7922.9 Hz   -20 dB 7940.3 Hz
#     -100 dB 7979.1 Hz   full -200 dB stopband from 7998.8 Hz
#
# The transition CANNOT straddle LADDER_FMAX, only narrow. Moving it above the
# band would leak attenuated [8000, 8000+d) content into the primary analysis,
# where Griffin-Lim has none and real does: the bandwidth cliff INV-18 relies on
# the band limit to remove. So [7900, 8000) Hz is still in neither analysis at
# full weight. It is 1.25% of the band and lies inside the falling slope of the
# top 80-band mel filter (peak ~7.7 kHz, zero at 8 kHz).
#
# What the narrower transition costs, measured on this machine:
#     order 12 -> 53 (6 -> 27 SOS sections), max pole radius 0.9948
#     impulse response 99.99%-energy span 2.8 ms -> 24 ms (linear, so identical
#       for every member of a comparison set; smears, does not separate)
#     ~19 -> ~72 ms per 10 s of audio
# Stopband residual on harmonic-rich audio is unchanged: the float32 storage
# floor (~1.8e-16), float64 4.8e-21.
# Rejected: elliptic (order 17 for the same edges, but 0.1 dB passband ripple
# and a 41 ms impulse response); Kaiser FIR once (1415 taps, but only 100 dB
# total stopband, residual 6.8e-16 in float64); a transition wider than 100 Hz
# (fewer taps, bigger hole); narrower (7950: order 75, 37 ms, 108 ms/10 s).
BAND_LIMIT_TRANSITION_HZ: Final[float] = 100.0
BAND_LIMIT_PASSBAND_LOSS_DB: Final[float] = 0.1   # total, zero-phase (0.05 per pass)
BAND_LIMIT_STOPBAND_DB: Final[float] = 100.0      # per pass; filtfilt gives 200


def band_limit_order(sr: int, cutoff_hz: float) -> int:
    """Chebyshev II order meeting both band-limit edges at ``cutoff_hz``.

    Pure arithmetic so the invariant module stays import-light; the test suite
    asserts it agrees with ``scipy.signal.cheb2ord``, which the filter uses.
    """
    import math

    passband = cutoff_hz - BAND_LIMIT_TRANSITION_HZ
    if not 0 < passband < cutoff_hz < sr / 2:
        raise InvariantViolation(
            f"INV-17: band limit [{passband}, {cutoff_hz}] Hz is not inside (0, {sr / 2})."
        )
    wp = math.tan(math.pi * passband / sr)
    ws = math.tan(math.pi * cutoff_hz / sr)
    per_pass = BAND_LIMIT_PASSBAND_LOSS_DB / 2
    ratio = (10 ** (0.1 * BAND_LIMIT_STOPBAND_DB) - 1) / (10 ** (0.1 * per_pass) - 1)
    return math.ceil(math.acosh(math.sqrt(ratio)) / math.acosh(ws / wp))


BAND_LIMIT_FILTER_ORDER: Final[int] = band_limit_order(ARCHIVE_SR, LADDER_FMAX)
BAND_LIMIT_FILTER_SPEC: Final[str] = (
    f"cheby2_zerophase_order{BAND_LIMIT_FILTER_ORDER}"
    f"_pass{LADDER_FMAX - BAND_LIMIT_TRANSITION_HZ:.0f}hz-{BAND_LIMIT_PASSBAND_LOSS_DB}db"
    f"_stop{LADDER_FMAX:.0f}hz-{BAND_LIMIT_STOPBAND_DB:.0f}db"
)

# --- INV-17  Measurement floor (calibrated) ----------------------------------
# out_of_band_energy was calibrated by injecting a tone above the cutoff at
# known amplitudes and checking recovery (tests: TestOutOfBandCalibration).
# Three separate floors sit under any reported figure, and they are NOT the
# same thing:
#
#   1e-22   the measurement itself -- Blackman window + float64 rFFT. Recovery
#           is linear to within 0.1% down to -200 dB. Never the limit.
#   1.8e-16 float32 STORAGE of the pipeline array. band_limit_to_ladder returns
#           float32, and rounding to float32 puts this much energy above the
#           band regardless of how good the filter is. Converting that array
#           back to float64 reproduces the identical figure, which is how we
#           know it is storage and not signal.
#   ~2e-8   PCM_16 at the -27 LUFS delivery level. This is the one that matters
#           physically: it is the floor of the delivered file.
#
# MEASUREMENT_FLOOR is set from the second of these -- the floor of any number
# this codebase can report about a pipeline array. A figure at or below it is
# representation noise and must be reported as such rather than as a value; see
# format_oob(). It is deliberately NOT the gate threshold: delivered files sit
# seven orders of magnitude above it, so gating here would fail everything.
MEASUREMENT_FLOOR: Final[float] = 1e-15

# Measured PCM_16 out-of-band floor at the delivery level, across content types
# (harmonic 5.8e-9, sparse tone 8.6e-9, noise 1.9e-8). Empirical, not chosen.
PCM16_OOB_FLOOR: Final[float] = 2e-8

# Headroom over the delivery floor for file-to-file variation. The gate has to
# sit above PCM16_OOB_FLOOR (nothing delivered reads cleaner) and far below a
# real failure -- an unfiltered condition is ~2e-3, a Butterworth-8 ~2.6e-5.
BAND_LIMIT_FLOOR_MARGIN: Final[float] = 50.0

# Gate tolerance: the fraction of total energy allowed above LADDER_FMAX.
#
# The floor here is not the filter, it is PCM_16. Quantisation adds broadband
# noise, some of which lands above the cutoff; at the -27 LUFS delivery level
# that measures ~2e-9 to ~6e-9 depending on content. A delivered file can never
# read cleaner than that however good the filter is.
#
# Set ~150x above that floor: correctly filtered audio passes with room to
# spare, while a skipped filter (2e-3) or too gentle a one (Butterworth-8,
# 2.6e-5) is caught by 25x or more.
#
# What remains at this level is quantisation noise -- signal-independent and
# identical across conditions by construction, unlike the 2e-3 real-vs-vocoder
# gap this invariant exists to close.
#
# Derived from the measured delivery floor rather than written as a literal, so
# it moves with the format rather than with someone's judgement.
BAND_LIMIT_MAX_STOPBAND_ENERGY: Final[float] = PCM16_OOB_FLOOR * BAND_LIMIT_FLOOR_MARGIN


def _check_analysis(analysis: str) -> None:
    if analysis not in ANALYSES:
        raise InvariantViolation(f"INV-18: unknown analysis {analysis!r}; known {ANALYSES}.")


def exclusion_for(analysis: str, condition: str) -> Exclusion | None:
    """INV-18. The Exclusion keeping ``condition`` out of ``analysis``, or None."""
    _check_analysis(analysis)
    return ANALYSIS_EXCLUSIONS[analysis].get(condition)


def is_excluded(analysis: str, condition: str) -> bool:
    """INV-18. True if ``condition`` is kept out of ``analysis``."""
    return exclusion_for(analysis, condition) is not None


# --- INV-07  Pipeline order --------------------------------------------------
# Fixed and identical for every condition. Trimming after loudness measurement
# changes the measured level; loudness before length alignment measures a
# different span of signal. Order is part of the control.
PIPELINE_ORDER: Final[tuple[str, ...]] = (
    "load_native",         # vocoder output at its own native rate
    "resample_once",       # -> ARCHIVE_SR, mono            (INV-01)
    "align_length",        # -> reference length            (INV-06)
    "apply_ref_trim",      # -> reference trim boundaries   (INV-03)
    "measure_high_band",   # -> recorded, NOT filtered      (INV-17)
    "measure_band_loudness",  # -> the gain's reference           (INV-04)
    "normalise_loudness",  # -> band hits LOUDNESS_TARGET_LUFS  (INV-04)
    "encode",              # -> WAV / PCM_16                (INV-05)
)

# There is no longer a variant pipeline. INV-17 used to low-pass at generation
# and exempt conditions skipped that step, which made INV-07's "no exceptions"
# false; moving the band limit to analysis time removed the exception rather
# than scoping around it. Every condition runs PIPELINE_ORDER exactly.

# The derived tier's only step. Applied to the finished archive file.
DERIVATION_ORDER: Final[tuple[str, ...]] = (
    "load_archive",        # the finished archive artifact
    "downsample_once",     # -> ZEROSHOT_SR                 (INV-01)
    "encode",              # -> WAV / PCM_16                (INV-05)
)

# --- INV-06  Pairing ---------------------------------------------------------
# Every condition contains exactly the same utterance IDs as REAL. Intrusive
# metrics need the paired reference, and an unequal set breaks the claim that
# vocoder identity is the only free variable.
REAL_CONDITION: Final[str] = "real"

# --- INV-08  Determinism -----------------------------------------------------
GLOBAL_SEED: Final[int] = 20260907

# --- INV-09  Splits ----------------------------------------------------------
# Split by utterance ID (and by speaker where the corpus has more than one),
# computed once, shared byte-for-byte across every condition, both tiers and
# both protocols. Content overlap between train and eval inflates matched EER
# separability and would be read as an artifact signal.
SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "dev", "eval")
SPLIT_FRACTIONS: Final[dict[str, float]] = {"train": 0.70, "dev": 0.10, "eval": 0.20}

# --- INV-12  Sanity gate -----------------------------------------------------
# A detector trained on features that carry no vocoder artifact -- silence
# duration alone, or broadband energy alone -- must sit near chance. If it does
# not, a confound has leaked and the condition set is not publishable.
SANITY_EER_FLOOR: Final[float] = 0.40  # 0.5 == chance; below this, investigate

# --- INV-16  Sample alignment ------------------------------------------------
# Reference-derived trim spans (INV-03) assume the vocoder is sample-aligned
# with its input. A fixed vocoder delay would turn a time shift into a measured
# "artifact". Cross-correlate real vs resynthesized over this many files when a
# vocoder is first onboarded; the peak lag must be exactly zero.
ALIGNMENT_PROBE_FILES: Final[int] = 20
ALIGNMENT_MAX_LAG_SAMPLES: Final[int] = 2_048  # search window, not a tolerance
ALIGNMENT_REQUIRED_LAG: Final[int] = 0

# Envelope window for the phase-blind measurement. Load-bearing, not a default:
# a shorter window leaves pitch ripple in the envelope and the correlation peak
# locks onto a pitch period instead of the true offset. Measured on Griffin-Lim
# over 20 utterances, files agreeing on the modal lag: 6/20 at 5 ms, 17/20 at
# 20 ms, 20/20 at 50 ms.
ALIGNMENT_ENVELOPE_MS: Final[float] = 50.0

# The waveform (phase-sensitive) lag is a corroborating check, enforced only
# when it is unambiguous evidence. Two conditions, both required:
#   - mean |correlation| above this floor, i.e. there is a phase relationship at
#     all. Griffin-Lim reconstructs phase from magnitude alone and sits near
#     noise; gating it on waveform lag would block it forever for its defining
#     characteristic rather than a defect.
#   - the lags agree across files. A *fixed* delay is consistent by definition,
#     so scattered waveform lags are measurement noise, not a delay -- and a
#     partial correlation (~0.3) produces exactly that kind of scatter.
# The envelope gate always applies and is the safety net: a genuine fixed delay
# displaces the envelope too.
ALIGNMENT_PHASE_CORR_THRESHOLD: Final[float] = 0.30
ALIGNMENT_MODAL_SHARE_FLOOR: Final[float] = 0.60


class InvariantViolation(RuntimeError):
    """Raised when audio or a manifest row contradicts the controlled design.

    Never catch this to keep a pipeline running. A violation means the affected
    condition must be regenerated, not patched.
    """


@dataclass(frozen=True)
class AudioContract:
    """The state every sanctioned audio file is in by the time it is written."""

    tier: str = ARCHIVE_TIER
    sample_rate: int = ARCHIVE_SR
    channels: int = CHANNELS
    subtype: str = SUBTYPE
    target_lufs: float = LOUDNESS_TARGET_LUFS
    resample_steps: int = 1
    provenance: tuple[str, ...] = field(default=PIPELINE_ORDER)


ARCHIVE_CONTRACT: Final[AudioContract] = AudioContract()
ZEROSHOT_CONTRACT: Final[AudioContract] = AudioContract(
    tier=ZEROSHOT_TIER,
    sample_rate=ZEROSHOT_SR,
    resample_steps=1,
    provenance=DERIVATION_ORDER,
)
CONTRACTS: Final[dict[str, AudioContract]] = {
    ARCHIVE_TIER: ARCHIVE_CONTRACT,
    ZEROSHOT_TIER: ZEROSHOT_CONTRACT,
}


def tier_for_rate(sample_rate: int) -> str:
    """Map a sanctioned rate back to its tier name."""
    for tier, rate in TIER_RATES.items():
        if rate == sample_rate:
            return tier
    raise InvariantViolation(
        f"INV-01: {sample_rate} Hz is not a sanctioned rate. Only "
        f"{ARCHIVE_SR} (archive) and {ZEROSHOT_SR} (derived zero-shot) exist."
    )


def check_waveform(
    wav: np.ndarray,
    sample_rate: int,
    *,
    where: str = "unknown",
    expected_sr: int | None = None,
    measured_lufs: float | None = None,
) -> None:
    """Assert a waveform satisfies INV-01, INV-04 and INV-05 before it is written.

    Called by :func:`data.audio_io.write_audio` on every write. Cheap enough to
    leave on permanently; the cost of a silently mis-specified condition is a
    full re-run of Phase A.

    ``expected_sr`` pins the tier when the caller knows which one it is writing.
    Without it, any sanctioned rate passes -- which is correct for a generic
    reader but too loose for a writer, so writers always pass it.

    ``measured_lufs`` is checked against the target only for the archive tier.
    The derived tier legitimately drifts (downsampling discards the >8 kHz
    band) and is deliberately not re-normalised, so its loudness is recorded in
    the manifest instead of being asserted here.
    """
    if expected_sr is not None:
        if sample_rate != expected_sr:
            raise InvariantViolation(
                f"INV-01 at {where}: sample rate {sample_rate} != expected "
                f"{expected_sr}. Archive is {ARCHIVE_SR}; the zero-shot set is a "
                f"separate derived artifact at {ZEROSHOT_SR}."
            )
    elif sample_rate not in SANCTIONED_RATES:
        raise InvariantViolation(
            f"INV-01 at {where}: sample rate {sample_rate} is not sanctioned. "
            f"Archive {ARCHIVE_SR}, derived zero-shot {ZEROSHOT_SR}. Resample once "
            "from the vocoder's native rate; never chain resamples."
        )
    if wav.ndim != 1:
        raise InvariantViolation(
            f"INV-01 at {where}: expected mono 1-D array, got shape {wav.shape}."
        )
    if wav.size == 0:
        raise InvariantViolation(f"INV-03 at {where}: empty waveform after trimming.")
    if not np.isfinite(wav).all():
        raise InvariantViolation(f"INV-05 at {where}: non-finite samples present.")

    peak_dbfs = 20.0 * np.log10(max(float(np.abs(wav).max()), 1e-12))
    if peak_dbfs > TRUE_PEAK_CEILING_DBFS:
        raise InvariantViolation(
            f"INV-04 at {where}: peak {peak_dbfs:.2f} dBFS exceeds "
            f"{TRUE_PEAK_CEILING_DBFS} dBFS. Do NOT rescale this file alone -- that "
            "breaks identical loudness. Drop the utterance from every condition."
        )
    if measured_lufs is not None:
        drift = abs(measured_lufs - LOUDNESS_TARGET_LUFS)
        if drift > LOUDNESS_TOLERANCE_LU:
            raise InvariantViolation(
                f"INV-04 at {where}: loudness {measured_lufs:.2f} LUFS drifts "
                f"{drift:.2f} LU from target {LOUDNESS_TARGET_LUFS}."
            )


def check_derived_loudness(
    measured_lufs: float, *, parent_lufs: float | None = None, where: str = "unknown"
) -> None:
    """INV-04, derived tier. Drift is expected; a large drift is not.

    Downsampling to 16 kHz costs a fraction of a LU on typical speech. A drift
    of several LU means something other than band-limiting happened -- most
    likely a second gain stage that should not exist.

    Drift is measured against ``parent_lufs``, the archive file this was derived
    from, because that isolates what the downsample actually did. Falling back
    to the global target is safe in production (INV-04 pins the archive there)
    but blurs the two effects together, so pass the parent value when you have
    it -- Phase A always does.
    """
    reference = LOUDNESS_TARGET_LUFS if parent_lufs is None else parent_lufs
    drift = abs(measured_lufs - reference)
    if drift > DERIVED_LOUDNESS_DRIFT_LIMIT_LU:
        raise InvariantViolation(
            f"INV-04 at {where}: derived loudness {measured_lufs:.2f} LUFS drifts "
            f"{drift:.2f} LU from its archive parent ({reference:.2f}), beyond the "
            f"{DERIVED_LOUDNESS_DRIFT_LIMIT_LU} LU that band-limiting alone explains. "
            "Check for an unintended second normalisation."
        )


def out_of_band_energy(wav: np.ndarray, sample_rate: int, cutoff_hz: float) -> float:
    """Fraction of total energy above ``cutoff_hz``.

    Windowed with a **Blackman** window before the transform, which is not
    optional at this dynamic range. A bare rFFT of an off-bin signal leaks
    across the whole spectrum: measured on an off-bin tone, a rectangular window
    floors out at 3.0e-6 -- above the gate -- so a correctly filtered file reads
    as a badly filtered one.

    Blackman specifically, NOT Blackman-Harris, despite the latter's better
    headline number. Blackman-Harris minimises the *peak* sidelobe (-92 dB) but
    its asymptotic rolloff is flat; Blackman's peak sidelobe is worse (-58 dB)
    but it rolls off at -18 dB/octave. The cutoff here sits far from the
    dominant low-frequency content, so asymptotic rolloff is what matters.
    Measured floors on an off-bin tone carrier:

        rectangular       3.0e-06     unusable
        Blackman-Harris   4.2e-14
        Nuttall           1.5e-12
        Blackman          5.9e-25     <- in use

    Calibration (tests: TestOutOfBandCalibration) recovers an injected tone
    linearly to within 0.1% from -40 dB down to -200 dB.

    Used by the INV-17 gate and by the bandwidth sanity probe. Cheap enough to
    run per file during Phase A.
    """
    if wav.size == 0:
        raise InvariantViolation("INV-17: cannot measure the band of an empty signal.")
    x = np.asarray(wav, dtype=np.float64)
    win = np.blackman(len(x)) if len(x) > 1 else np.ones(1)
    spec = np.abs(np.fft.rfft(x * win)) ** 2
    total = float(spec.sum())
    if total <= 0.0:
        return 0.0
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    return float(spec[freqs > cutoff_hz].sum() / total)


def format_oob(value: float) -> str:
    """Render an out-of-band figure, refusing to quote meaningless precision.

    Anything at or below MEASUREMENT_FLOOR is float32 representation noise, not
    a measurement of the signal. Reporting "1.78e-16" invites the reader to
    believe the filter was characterised to that level; it was not, and the true
    residual could be anywhere below. Say so instead.
    """
    if value <= MEASUREMENT_FLOOR:
        return f"below measurement floor (<{MEASUREMENT_FLOOR:.0e})"
    return f"{value:.3e}"


def check_band_limit(
    wav: np.ndarray,
    sample_rate: int,
    *,
    where: str = "unknown",
    cutoff_hz: float = LADDER_FMAX,
) -> float:
    """INV-17 gate on the output of the ANALYSIS-time band-limit.

    Returns the measured out-of-band energy fraction so the caller can record it.

    This gates :func:`data.preprocess.band_limit_comparison_set`, not the
    archive. The archive is full-band by design and would fail this check on
    every file -- that is the point of it being full-band.
    """
    measured = out_of_band_energy(wav, sample_rate, cutoff_hz)
    if measured > BAND_LIMIT_MAX_STOPBAND_ENERGY:
        raise InvariantViolation(
            f"INV-17 at {where}: {format_oob(measured)} of total energy sits above "
            f"{cutoff_hz:.0f} Hz, over the {BAND_LIMIT_MAX_STOPBAND_ENERGY:.0e} "
            f"allowance ({BAND_LIMIT_FLOOR_MARGIN:.0f}x the measured PCM_16 delivery "
            f"floor of {PCM16_OOB_FLOOR:.0e}). Every non-exempt condition, REAL "
            "included, is low-passed to the ladder band; a condition that is not "
            "would be separable from the rest on high-band energy alone."
        )
    return measured


def check_pairing(condition_ids: dict[str, set[str]]) -> None:
    """Assert INV-06: every condition holds exactly the REAL utterance set."""
    if REAL_CONDITION not in condition_ids:
        raise InvariantViolation(
            f"INV-06: no '{REAL_CONDITION}' condition present. The paired reference "
            "is what makes PESQ/MCD available and the comparison controlled."
        )
    reference = condition_ids[REAL_CONDITION]
    for name, ids in condition_ids.items():
        if ids != reference:
            missing = sorted(reference - ids)[:5]
            extra = sorted(ids - reference)[:5]
            raise InvariantViolation(
                f"INV-06: condition '{name}' is not paired with '{REAL_CONDITION}'. "
                f"missing={missing} extra={extra}"
            )
