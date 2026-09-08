"""Band-wise spectral analysis — the descriptive half of the mechanism argument.

The prediction under test: because mel compression is heaviest at high
frequencies, the vocoder has least information there and must hallucinate the
most, so log-spectral error should concentrate in the high band. BigVGAN's
anti-aliasing is expected to flatten that profile relative to older vocoders.

This module produces the *description*. The causal claim comes from the
band-limited retraining ablation in :mod:`detectors.bandlimit` -- a band-wise
error profile on its own is a correlation, not a mechanism.

:func:`high_band_distance` is the one to read first. Above LADDER_FMAX the mel
carried nothing, so the vocoder invented the content; whether it invented it
*well* is a different and better question than how much of it there is.

ARCHIVE tier only (INV-01). Band-wise error above 8 kHz is exactly what the
derived 16 kHz set cannot represent, so running this there would report the
downsampler's rolloff as a vocoder property.
"""

from __future__ import annotations

import librosa
import numpy as np

from data.invariants import (
    ARCHIVE_SR,
    LADDER_FMAX,
    InvariantViolation,
    out_of_band_energy,
)

# Bands up to archive Nyquist (11.025 kHz). The boundary at LADDER_FMAX is
# deliberate and load-bearing, but NOT for the reason it used to be. It marks the
# edge above which a vocoder had no mel information: everything emitted there is
# hallucinated from the model's prior. With a full-band archive (INV-17) that
# band holds real content for every time-domain vocoder -- measured 0.01480 for
# BigVGAN against 0.01803 for real -- and exactly zero for Griffin-Lim, which
# cannot exceed fmax by construction.
#
# It also keeps the derived-tier boundary on the same edge, since ZEROSHOT
# Nyquist and LADDER_FMAX coincide at 8 kHz -- if LADDER_FMAX ever drops (a
# 7600 Hz MelGAN re-source, say) these stop coinciding and the table gains a
# band. That is correct: they are different limits that happen to agree today.
DEFAULT_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (0, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 6000),
    (6000, LADDER_FMAX),
    (LADDER_FMAX, ARCHIVE_SR / 2),
)


def log_spectral_distance_by_band(
    reference: np.ndarray,
    degraded: np.ndarray,
    *,
    sr: int = ARCHIVE_SR,
    n_fft: int = 1024,
    hop_length: int = 256,
    bands: tuple[tuple[float, float], ...] = DEFAULT_BANDS_HZ,
) -> dict[str, float]:
    """Per-band log-spectral distance in dB over an aligned pair."""
    if len(reference) != len(degraded):
        raise InvariantViolation("Band analysis needs aligned pairs; see INV-06.")

    def _logspec(x: np.ndarray) -> np.ndarray:
        s = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
        return 20.0 * np.log10(np.maximum(s, 1e-10))

    ref_s, deg_s = _logspec(reference), _logspec(degraded)
    n = min(ref_s.shape[1], deg_s.shape[1])
    diff = ref_s[:, :n] - deg_s[:, :n]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    out: dict[str, float] = {}
    for lo, hi in bands:
        sel = (freqs >= lo) & (freqs < hi)
        if not sel.any():
            continue
        out[f"lsd_{int(lo)}_{int(hi)}"] = float(np.sqrt(np.mean(diff[sel] ** 2)))
    out["lsd_full"] = float(np.sqrt(np.mean(diff**2)))
    return out


def high_band_distance(
    reference: np.ndarray,
    degraded: np.ndarray,
    *,
    sr: int = ARCHIVE_SR,
    lo: float = LADDER_FMAX,
    hi: float | None = None,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> dict[str, float]:
    """How well a condition reconstructs the band its mel never carried.

    Above ``lo`` the vocoder was told nothing, so whatever it emits is invented
    from its prior. Energy alone does not settle whether the invention is any
    good: a model can put the right AMOUNT of energy above 8 kHz and still get
    the STRUCTURE wrong, and wrong structure with right energy is a strong
    detection cue that an energy fraction cannot see. BigVGAN measures 0.01480
    against real's 0.01803 and tracks it per file, which says the amount is
    close and says nothing at all about the content.

    Returns, restricted to [lo, hi):

      ``hb_lsd``          log-spectral distance in dB. The headline number.
      ``hb_corr``         Pearson correlation of the per-bin mean log magnitude.
                          Separates "wrong level, right shape" from "right
                          level, wrong shape" -- LSD alone conflates them.
      ``hb_frac_ref`` / ``hb_frac_deg``
                          energy fraction above ``lo`` for each signal, so the
                          amount and the content are reported side by side.
      ``hb_frac_ratio``   deg/ref. 1.0 means the right amount of energy, which
                          is necessary and nowhere near sufficient.

    A condition with a hard bandwidth zero (Griffin-Lim) yields
    ``hb_frac_deg == 0`` and an ``hb_lsd`` dominated by the spectral floor; read
    those rows as "structurally absent", not as "badly reconstructed".
    """
    if len(reference) != len(degraded):
        raise InvariantViolation("High-band distance needs aligned pairs; see INV-06.")
    hi = sr / 2.0 if hi is None else hi
    if not 0 < lo < hi <= sr / 2.0:
        raise InvariantViolation(
            f"high_band_distance: band [{lo}, {hi}) is not inside (0, {sr / 2.0}]."
        )

    def _spec(x):
        return np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))

    ref_s, deg_s = _spec(reference), _spec(degraded)
    n = min(ref_s.shape[1], deg_s.shape[1])
    ref_s, deg_s = ref_s[:, :n], deg_s[:, :n]

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    sel = (freqs >= lo) & (freqs < hi)
    if not sel.any():
        raise InvariantViolation(
            f"high_band_distance: no FFT bins in [{lo}, {hi}) at n_fft={n_fft}."
        )

    ref_db = 20.0 * np.log10(np.maximum(ref_s[sel], 1e-10))
    deg_db = 20.0 * np.log10(np.maximum(deg_s[sel], 1e-10))
    lsd = float(np.sqrt(np.mean((ref_db - deg_db) ** 2)))

    ref_prof = ref_db.mean(axis=1)
    deg_prof = deg_db.mean(axis=1)
    if ref_prof.std() < 1e-12 or deg_prof.std() < 1e-12:
        corr = float("nan")  # a flat profile: e.g. an exact bandwidth zero
    else:
        corr = float(np.corrcoef(ref_prof, deg_prof)[0, 1])

    frac_ref = out_of_band_energy(reference, sr, lo)
    frac_deg = out_of_band_energy(degraded, sr, lo)
    return {
        "hb_lsd": lsd,
        "hb_corr": corr,
        "hb_frac_ref": frac_ref,
        "hb_frac_deg": frac_deg,
        "hb_frac_ratio": float(frac_deg / frac_ref) if frac_ref > 0 else float("nan"),
    }


def spectral_centroid_shift(
    reference: np.ndarray, degraded: np.ndarray, sr: int = ARCHIVE_SR
) -> float:
    """Mean centroid shift in Hz. Positive means the condition is brighter."""
    ref_c = librosa.feature.spectral_centroid(y=reference, sr=sr).mean()
    deg_c = librosa.feature.spectral_centroid(y=degraded, sr=sr).mean()
    return float(deg_c - ref_c)
