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

# Conditions that are generated, measured and REPORTED like any other, but are
# refused by the primary correlation. Two different reasons, both recorded so
# the refusal message can say which applies -- a bare exclusion set invites
# someone to delete an entry without knowing what it was protecting.
CORRELATION_EXCLUDED: Final[dict[str, str]] = {
    "griffin_lim": (
        "floor reference. Griffin-Lim inverts the mel to a linear spectrogram "
        "and runs ISTFT, so it genuinely cannot emit above its mel fmax -- "
        "measured high-band fraction 0.00000 against real's 0.01803. Its "
        "detectability is therefore driven by a hard bandwidth zero, not by the "
        "fine reconstruction artifacts the study is about. Sitting at the "
        "low-quality end of the ladder it would anchor a strong positive "
        "Spearman for a reason unrelated to the hypothesis."
    ),
    "bigvgan_v2_22khz_fullband": (
        "paired control. Differs from bigvgan_112m in training mel fmax alone "
        "(11025 vs 8000), which is the variable it exists to isolate; pooling "
        "it into the ladder would put that variable into the headline number."
    ),
}


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
#     Griffin-Lim    0.00000    exactly zero
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
# Passband cost: 0.004 dB below 7 kHz.
BAND_LIMIT_FILTER_ORDER: Final[int] = 12
BAND_LIMIT_STOPBAND_DB: Final[float] = 100.0
BAND_LIMIT_FILTER_SPEC: Final[str] = (
    f"cheby2_zerophase_order{BAND_LIMIT_FILTER_ORDER}_rs{BAND_LIMIT_STOPBAND_DB:.0f}db"
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


def is_correlation_excluded(condition: str) -> bool:
    """INV-17/INV-18. True for conditions reported but kept out of the headline rho."""
    return condition in CORRELATION_EXCLUDED


def exclusion_reason(condition: str) -> str | None:
    """Why a condition is excluded, for the refusal message. None if included."""
    return CORRELATION_EXCLUDED.get(condition)


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
    "normalise_loudness",  # -> LOUDNESS_TARGET_LUFS        (INV-04)
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
