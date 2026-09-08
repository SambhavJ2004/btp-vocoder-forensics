"""Band-wise spectral analysis — the descriptive half of the mechanism argument.

The prediction under test: because mel compression is heaviest at high
frequencies, the vocoder has least information there and must hallucinate the
most, so log-spectral error should concentrate in the high band. BigVGAN's
anti-aliasing is expected to flatten that profile relative to older vocoders.

This module produces the *description*. The causal claim comes from the
band-limited retraining ablation in :mod:`detectors.bandlimit` -- a band-wise
error profile on its own is a correlation, not a mechanism.

ARCHIVE tier only (INV-01). Band-wise error above 8 kHz is exactly what the
derived 16 kHz set cannot represent, so running this there would report the
downsampler's rolloff as a vocoder property.
"""

from __future__ import annotations

import librosa
import numpy as np

from data.invariants import ARCHIVE_SR, LADDER_FMAX, InvariantViolation

# Bands up to archive Nyquist (11.025 kHz). The boundary at LADDER_FMAX is
# deliberate and load-bearing: it separates the band every condition shares from
# the band only INV-17-exempt conditions carry. For a ladder condition the top
# band should read as filter stopband and nothing else; a non-trivial value
# there means the band-limit step did not run.
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


def spectral_centroid_shift(
    reference: np.ndarray, degraded: np.ndarray, sr: int = ARCHIVE_SR
) -> float:
    """Mean centroid shift in Hz. Positive means the condition is brighter."""
    ref_c = librosa.feature.spectral_centroid(y=reference, sr=sr).mean()
    deg_c = librosa.feature.spectral_centroid(y=degraded, sr=sr).mean()
    return float(deg_c - ref_c)
