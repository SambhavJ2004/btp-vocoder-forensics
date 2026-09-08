"""Mel-cepstral distortion — intrusive, paired-reference quality metric.

Available only because the study uses resynthesis rather than TTS: the real
recording is a genuine sample-aligned reference for every condition.

No DTW is applied. Resynthesis is frame-aligned by construction, and the
pipeline enforces identical trim spans (INV-03), lengths (INV-06) and sample
alignment (INV-16), so warping would hide exactly the timing/phase errors the
metric should catch. If a condition needs DTW to score reasonably, that is a bug
in Phase A alignment, not a reason to add DTW.

Runs on the ARCHIVE tier: the derived 16 kHz set has no content above 8 kHz, and
the upper mel bands are where vocoder reconstruction error concentrates.
"""

from __future__ import annotations

import librosa
import numpy as np

from data.invariants import ARCHIVE_SR, InvariantViolation

# 10 / ln(10) * sqrt(2) — the conventional MCD scaling to dB.
_MCD_K = 10.0 / np.log(10.0) * np.sqrt(2.0)


def mel_cepstra(
    wav: np.ndarray,
    *,
    n_mfcc: int = 25,
    n_fft: int = 1024,
    hop_length: int = 256,
    sr: int = ARCHIVE_SR,
) -> np.ndarray:
    """MFCC-style mel cepstra, (n_frames, n_mfcc)."""
    mfcc = librosa.feature.mfcc(
        y=wav, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    return mfcc.T


def mcd(
    reference: np.ndarray,
    degraded: np.ndarray,
    *,
    exclude_c0: bool = True,
    sr: int = ARCHIVE_SR,
) -> float:
    """MCD in dB. Lower is better.

    ``exclude_c0`` drops the energy coefficient, which is standard: c0 tracks
    overall gain, and gain is already pinned by the loudness invariant (INV-04),
    so including it would only add noise.
    """
    if len(reference) != len(degraded):
        raise InvariantViolation(
            f"MCD needs aligned pairs: {len(reference)} vs {len(degraded)} samples. "
            "Phase A guarantees this (INV-06); do not paper over it with DTW."
        )
    ref_c = mel_cepstra(reference, sr=sr)
    deg_c = mel_cepstra(degraded, sr=sr)
    n = min(len(ref_c), len(deg_c))
    start = 1 if exclude_c0 else 0
    diff = ref_c[:n, start:] - deg_c[:n, start:]
    return float(_MCD_K * np.mean(np.sqrt(np.sum(diff**2, axis=1))))
