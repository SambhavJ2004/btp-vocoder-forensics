"""Confound enforcement: the fixed post-vocoder pipeline.

This is the single place where INV-01 (two single-step derivations), INV-03
(silence), INV-04 (loudness) and INV-07 (order) are applied. Phase A calls
:func:`process_condition_output` for every vocoder output including the real
reference, then :func:`derive_zeroshot` once per finished archive file.

Two design points that are easy to get wrong:

1. Trim boundaries are derived from the REAL reference and then applied by
   sample index to every condition. If each condition were trimmed
   independently, a vocoder whose padding behaviour differs would end up with
   systematically different silence durations -- exactly the cue ASVspoof
   detectors were shown to key on -- and the trim would have introduced the
   confound it exists to remove.

2. The zero-shot set is derived from the finished ARCHIVE file, never from
   source. Deriving it from source would mean two independent resampling paths
   whose filter imprints differ, and the 16 kHz set would no longer be the same
   audio as the archive set.

3. The archive is FULL-BAND. Band-limiting to LADDER_FMAX (INV-17) is an
   ANALYSIS-time transform -- :func:`band_limit_comparison_set` -- applied to a
   comparison set on request, and applied to REAL on exactly the same terms as
   every vocoded member of that set. It is never applied at generation, because
   information discarded there cannot be recovered, and the content above
   LADDER_FMAX is hallucinated by the vocoder and therefore the most
   forensically interesting content the archive holds.

4. The post-trim ordering (INV-07) has exactly ONE implementation,
   :func:`finalise_trimmed`, behind two thin entry points --
   :func:`finalise_reference` for real, :func:`process_condition_output` for
   vocoder conditions. They differ only in what they do BEFORE the trim, which
   is the only thing that genuinely differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import pyloudnorm as pyln

from .invariants import (
    ARCHIVE_SR,
    BAND_LIMIT_FILTER_ORDER,
    BAND_LIMIT_STOPBAND_DB,
    LADDER_FMAX,
    LOUDNESS_TARGET_LUFS,
    REAL_CONDITION,
    RESAMPLE_METHOD,
    TRIM_FRAME_LENGTH,
    TRIM_HOP_LENGTH,
    TRIM_PAD_MS,
    TRIM_TOP_DB,
    TRUE_PEAK_CEILING_DBFS,
    ZEROSHOT_SR,
    InvariantViolation,
    check_band_limit,
    check_derived_loudness,
    out_of_band_energy,
)


@dataclass(frozen=True)
class TrimSpan:
    """Sample-index trim boundaries derived once from the real reference.

    Indices are in ARCHIVE_SR samples. The derived zero-shot tier is produced by
    downsampling the already-trimmed archive file, so it never needs these
    indices rescaled -- which is one more reason the derivation runs last.

    Serialised into the manifest so a re-run reproduces the same span without
    re-deriving it, and so an auditor can confirm every condition of an
    utterance used identical boundaries.
    """

    utt_id: str
    start: int
    end: int
    ref_length: int

    @property
    def length(self) -> int:
        return self.end - self.start


def resample_once(wav: np.ndarray, sr_in: int, target_sr: int = ARCHIVE_SR) -> np.ndarray:
    """INV-01, step one. Vocoder native rate -> archive rate, single filtering pass.

    For LJSpeech-sourced conditions whose vocoder is already 22.05 kHz this is a
    no-op, which is the point of choosing the corpus rate as the archive rate:
    the most common path applies no resampling filter at all.
    """
    if wav.ndim != 1:
        raise InvariantViolation(f"INV-01: expected mono input, got shape {wav.shape}")
    if sr_in == target_sr:
        return wav.astype(np.float32, copy=False)
    return librosa.resample(
        wav.astype(np.float32, copy=False),
        orig_sr=sr_in,
        target_sr=target_sr,
        res_type=RESAMPLE_METHOD,
    )


def derive_zeroshot(
    wav_archive: np.ndarray,
    *,
    archive_sr: int = ARCHIVE_SR,
    archive_lufs: float | None = None,
    meter: pyln.Meter | None = None,
) -> tuple[np.ndarray, float]:
    """INV-01, step two. The archive artifact -> the derived 16 kHz artifact.

    One downsample, from the finished archive file. Returns the waveform and its
    measured loudness, which is *recorded* rather than corrected: re-normalising
    here would apply a gain proportional to each file's high-band energy, and
    high-band energy is the artifact under study. See CLAUDE.md INV-01/INV-04.

    ``archive_lufs`` is the parent's measured loudness; passing it makes the
    drift check measure the downsample alone rather than the downsample plus any
    residual normalisation error.
    """
    if archive_sr != ARCHIVE_SR:
        raise InvariantViolation(
            f"INV-01: the zero-shot set derives from the {ARCHIVE_SR} Hz archive, not "
            f"from {archive_sr} Hz. Deriving from source would be a second "
            "independent resampling path."
        )
    out = librosa.resample(
        wav_archive.astype(np.float32, copy=False),
        orig_sr=ARCHIVE_SR,
        target_sr=ZEROSHOT_SR,
        res_type=RESAMPLE_METHOD,
    ).astype(np.float32)

    meter = meter or pyln.Meter(ZEROSHOT_SR)
    measured = float(meter.integrated_loudness(out))
    check_derived_loudness(measured, parent_lufs=archive_lufs, where="derive_zeroshot")
    return out, measured


def band_limit_to_ladder(
    wav: np.ndarray, *, sr: int = ARCHIVE_SR, cutoff_hz: float = LADDER_FMAX
) -> np.ndarray:
    """INV-17. Low-pass one signal to the analysis band.

    **Analysis-time only.** Nothing in Phase A calls this; the archive is
    full-band. Reach it through :func:`band_limit_comparison_set`, which applies
    it to every member of a comparison set at once -- including real -- so that
    a caller cannot band-limit half a comparison and produce the same cliff with
    the sign flipped.

    Zero-phase (filtfilt) for two reasons. A causal filter imposes a
    frequency-dependent group delay, which is itself a phase artifact sitting in
    the band the study measures; and it would shift the output relative to the
    reference, which INV-16 exists to forbid. Zero-phase filtering leaves sample
    alignment intact.

    Chebyshev Type II, not Butterworth. The filter has to do more than reduce
    the high band -- it has to leave real audio and an fmax=8000 vocoder
    *indistinguishable* there. Butterworth attenuates far too gently for that:
    even at order 24 it leaves ~2.5e-6 of total energy above the band, orders of
    magnitude above the 16-bit noise floor, so the two remain separable on
    exactly the cue INV-17 exists to remove. Chebyshev II is equiripple in the
    stopband and monotone in the passband, which is the right shape here: 100 dB
    of stopband rejection for 0.004 dB of passband loss.

    ``cutoff_hz`` is the *stopband edge* under this design -- rejection reaches
    the full 100 dB at LADDER_FMAX, with the transition sitting just below it.
    That costs a sliver of the 7-8 kHz region, identically for every condition,
    which is the trade the invariant is making.

    """
    from scipy.signal import cheby2, sosfiltfilt

    nyq = sr / 2.0
    if not 0 < cutoff_hz < nyq:
        raise InvariantViolation(
            f"INV-17: ladder band {cutoff_hz} Hz is not inside (0, {nyq}) at {sr} Hz."
        )
    sos = cheby2(
        BAND_LIMIT_FILTER_ORDER,
        BAND_LIMIT_STOPBAND_DB,
        cutoff_hz / nyq,
        btype="low",
        output="sos",
    )
    return sosfiltfilt(sos, wav).astype(np.float32)


def band_limit_comparison_set(
    wavs: dict[str, np.ndarray],
    *,
    sr: int = ARCHIVE_SR,
    cutoff_hz: float = LADDER_FMAX,
    verify: bool = True,
) -> dict[str, np.ndarray]:
    """INV-17. Band-limit an entire comparison set to the analysis band.

    ``wavs`` maps condition name -> waveform, and MUST include the real
    reference: band-limiting only the vocoded members leaves real holding a high
    band the others lack, which is the original cliff with its sign flipped and
    exactly as learnable. Taking the whole set at once is what makes that
    mistake awkward to write.

    Returns a new mapping; the inputs are not modified. Each output is gated by
    :func:`data.invariants.check_band_limit` unless ``verify`` is off.

    This is the transform the band-limited ablation and any like-for-like
    comparison run through. It replaces the generation-time filter that INV-17
    used to apply -- see the invariant for why that was wrong.
    """
    if REAL_CONDITION not in wavs:
        raise InvariantViolation(
            f"INV-17: the comparison set has no '{REAL_CONDITION}' member. "
            "Band-limiting only the vocoded conditions leaves real with a high "
            "band they lack -- the same separable cliff, sign flipped."
        )
    out: dict[str, np.ndarray] = {}
    for name, wav in wavs.items():
        limited = band_limit_to_ladder(wav, sr=sr, cutoff_hz=cutoff_hz)
        if verify:
            check_band_limit(limited, sr, where=f"{name} (analysis band)", cutoff_hz=cutoff_hz)
        out[name] = limited
    return out


def derive_trim_span(ref_wav: np.ndarray, utt_id: str) -> TrimSpan:
    """INV-03. Compute trim boundaries from the REAL reference only, at ARCHIVE_SR."""
    _, (start, end) = librosa.effects.trim(
        ref_wav,
        top_db=TRIM_TOP_DB,
        frame_length=TRIM_FRAME_LENGTH,
        hop_length=TRIM_HOP_LENGTH,
    )
    pad = int(round(TRIM_PAD_MS * 1e-3 * ARCHIVE_SR))
    start = max(0, int(start) - pad)
    end = min(len(ref_wav), int(end) + pad)
    if end <= start:
        raise InvariantViolation(f"INV-03: degenerate trim span for {utt_id}")
    return TrimSpan(utt_id=utt_id, start=start, end=end, ref_length=len(ref_wav))


def align_length(wav: np.ndarray, ref_length: int, *, sr: int = ARCHIVE_SR) -> np.ndarray:
    """INV-06. Vocoder output length is ``n_frames * hop`` and rarely matches the
    reference exactly. Pad with zeros or truncate at the tail so that reference
    trim indices remain meaningful.

    A mismatch beyond one mel hop means the condition is misaligned at the
    frame level, which would corrupt intrusive metrics -- so it is an error,
    not something to silently absorb. The 50 ms tolerance is expressed against
    ``sr`` so it stays 50 ms of time at either rate.
    """
    delta = len(wav) - ref_length
    if abs(delta) > sr // 20:  # 50 ms
        raise InvariantViolation(
            f"INV-06: length mismatch of {delta} samples ({delta / sr:.3f}s). "
            "Check the mel hop/centre convention for this vocoder before proceeding."
        )
    if delta > 0:
        return wav[:ref_length]
    if delta < 0:
        return np.pad(wav, (0, -delta))
    return wav


def apply_trim_span(wav: np.ndarray, span: TrimSpan) -> np.ndarray:
    """INV-03. Apply the reference's boundaries by index -- never re-detect."""
    if len(wav) != span.ref_length:
        raise InvariantViolation(
            f"INV-03/INV-07: trim span for {span.utt_id} assumes length "
            f"{span.ref_length}, got {len(wav)}. Call align_length first."
        )
    return wav[span.start : span.end]


def normalise_loudness(
    wav: np.ndarray, meter: pyln.Meter | None = None, *, sr: int = ARCHIVE_SR
) -> tuple[np.ndarray, float]:
    """INV-04. Bring the waveform to the fixed LUFS target, at the archive tier.

    Returns the normalised waveform and its measured post-normalisation
    loudness, which :func:`data.audio_io.write_audio` verifies. Peak protection
    is deliberately NOT applied here: silently attenuating one file would break
    the "identical target across conditions" guarantee. The caller drops the
    utterance from every condition instead.
    """
    meter = meter or pyln.Meter(sr)
    loudness_in = meter.integrated_loudness(wav)
    if not np.isfinite(loudness_in):
        raise InvariantViolation("INV-04: loudness undefined (silent or too short).")
    out = pyln.normalize.loudness(wav, loudness_in, LOUDNESS_TARGET_LUFS)
    return out.astype(np.float32), float(meter.integrated_loudness(out))


def would_clip(wav: np.ndarray) -> bool:
    """True when the utterance must be dropped from ALL conditions (INV-04)."""
    peak_dbfs = 20.0 * np.log10(max(float(np.abs(wav).max()), 1e-12))
    return peak_dbfs > TRUE_PEAK_CEILING_DBFS


def finalise_trimmed(
    wav_trimmed: np.ndarray,
    *,
    condition: str,
    meter: pyln.Meter | None = None,
) -> tuple[np.ndarray, float, float | None]:
    """INV-07. The **single** implementation of the post-trim ordering.

    Steps, in this order and no other:
      measure_high_band (INV-17, descriptive) -> normalise_loudness (INV-04)

    Every condition reaches this function, and reaches it by the same route.
    That is the whole point: the real reference and the vocoder conditions have
    different *pre*-trim needs -- real derives the trim span, vocoders receive
    one -- and for a while that difference was expressed by duplicating the
    post-trim steps in two places. Two implementations of an ordering that
    INV-07 exists to keep identical is the invariant's own failure mode. Real is
    the reference every span and every pairing derives from, so a silent
    divergence here would move every condition against real at once.

    So the pre-trim difference stays, in two thin entry points, and the ordering
    lives here once. :func:`process_condition_output` is the vocoder entry
    point; :func:`data.preprocess.finalise_reference` is real's. Equivalence is
    asserted by test, not by prose.

    **No filtering happens here.** The archive is full-band (INV-17); the
    high-band fraction is *measured* and recorded, not removed. Measuring before
    normalisation keeps the number comparable across conditions, since gain
    would otherwise scale it.

    Returns ``(wav, measured_lufs, high_band_fraction)`` -- the fraction of
    energy above LADDER_FMAX, which is evidence rather than a gate. Griffin-Lim
    reads 0.00000 here and real reads ~0.018; that difference is the finding
    this invariant now records instead of erasing.
    """
    wav = wav_trimmed
    high_band = out_of_band_energy(wav, ARCHIVE_SR, LADDER_FMAX)
    wav, measured = normalise_loudness(wav, meter=meter, sr=ARCHIVE_SR)
    return wav, measured, high_band


def finalise_reference(
    wav_archive: np.ndarray,
    span: TrimSpan,
    *,
    condition: str = REAL_CONDITION,
    meter: pyln.Meter | None = None,
) -> tuple[np.ndarray, float, float | None]:
    """INV-07, real's entry point. Trim by the span it just derived, then finalise.

    Real cannot call :func:`process_condition_output` because that function
    consumes a `TrimSpan` and real is what *produces* one -- it has already
    resampled, and its length is the span's `ref_length` by construction, so
    both of those steps would be no-ops at best. What it shares with the vocoder
    path is everything after the trim, and that is exactly what it calls.
    """
    return finalise_trimmed(
        apply_trim_span(wav_archive, span), condition=condition, meter=meter
    )


def process_condition_output(
    wav_native: np.ndarray,
    sr_native: int,
    span: TrimSpan,
    *,
    condition: str,
    meter: pyln.Meter | None = None,
) -> tuple[np.ndarray, float, float | None]:
    """INV-07, the vocoder entry point. Full pipeline on raw generator output.

    Steps, in this order and no other:
      resample_once -> align_length -> apply_trim_span -> band_limit
      -> normalise_loudness

    The last two are :func:`finalise_trimmed`, shared byte-for-byte with real.

    Produces the ARCHIVE artifact. The derived zero-shot artifact comes from
    :func:`derive_zeroshot`, applied to this output after it is written.
    """
    wav = resample_once(wav_native, sr_native, ARCHIVE_SR)
    wav = align_length(wav, span.ref_length, sr=ARCHIVE_SR)
    wav = apply_trim_span(wav, span)
    return finalise_trimmed(wav, condition=condition, meter=meter)


def archive_band_spec() -> tuple[float, str]:
    """What every manifest row records about the archive band (INV-17).

    One value for every condition, because the archive is full-band for all of
    them. A row that says anything else means someone filtered at generation.
    """
    return ARCHIVE_SR / 2.0, "full_band"
