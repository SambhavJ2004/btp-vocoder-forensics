"""F0 and periodicity error — targets pitch-specific artifacts.

Included because a vocoder can score well on broadband metrics while getting
voicing wrong, and because pitch errors are a mechanism the study can name:
phase is synthesised from nothing, so harmonic-phase coherence is where an
ill-posed inversion shows first.

Three numbers per pair:
  - F0 RMSE in cents, over frames both signals call voiced
  - voicing decision error (fraction of frames that disagree)
  - periodicity RMSE, if a periodicity estimate is available
"""

from __future__ import annotations

import numpy as np

from data.invariants import ARCHIVE_SR, InvariantViolation

F0_FLOOR_HZ = 65.0
F0_CEIL_HZ = 400.0  # LJSpeech is a single female speaker; widen for VCTK
FRAME_PERIOD_MS = 5.0


def extract_f0(wav: np.ndarray, sr: int = ARCHIVE_SR) -> tuple[np.ndarray, np.ndarray]:
    """Return (f0_hz, voiced_flag) on a fixed frame grid, using DIO+StoneMask."""
    import pyworld  # optional dep: pip install '.[quality]'

    x = wav.astype(np.float64)
    f0, t = pyworld.dio(
        x, sr, f0_floor=F0_FLOOR_HZ, f0_ceil=F0_CEIL_HZ, frame_period=FRAME_PERIOD_MS
    )
    f0 = pyworld.stonemask(x, f0, t, sr)
    return f0, f0 > 0


def f0_error(
    reference: np.ndarray, degraded: np.ndarray, sr: int = ARCHIVE_SR
) -> dict[str, float]:
    """F0 RMSE (cents) and voicing decision error over an aligned pair."""
    if len(reference) != len(degraded):
        raise InvariantViolation("F0 comparison needs aligned pairs; see INV-06.")

    f0_ref, v_ref = extract_f0(reference, sr)
    f0_deg, v_deg = extract_f0(degraded, sr)
    n = min(len(f0_ref), len(f0_deg))
    f0_ref, f0_deg = f0_ref[:n], f0_deg[:n]
    v_ref, v_deg = v_ref[:n], v_deg[:n]

    both = v_ref & v_deg
    if both.sum() == 0:
        return {"f0_rmse_cents": float("nan"), "vde": 1.0, "voiced_frames": 0}

    cents = 1200.0 * np.log2(f0_deg[both] / f0_ref[both])
    return {
        "f0_rmse_cents": float(np.sqrt(np.mean(cents**2))),
        "vde": float(np.mean(v_ref != v_deg)),
        "voiced_frames": int(both.sum()),
    }
