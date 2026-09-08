"""Band-limited detection ablation — correlation to mechanism.

The claim under test: vocoder artifacts concentrate in the high band, because
mel compression is heaviest there and the vocoder must hallucinate the most.
BigVGAN's anti-aliasing is expected to suppress exactly that signature.

Protocol: low-pass at a series of cutoffs, RETRAIN the detector on each
band-limited variant, and find where detection collapses. Retraining is not
optional -- evaluating an unmodified detector on filtered audio measures
domain shift, not where the information lives, and would produce a
collapse-looking curve for every condition regardless of mechanism.

The reading: if an older vocoder's EER degrades sharply once content above
~8 kHz is removed while BigVGAN's is unchanged, the artifact is localised to
the high band and the anti-aliasing effect is demonstrated directly.

**This ablation runs on the ARCHIVE tier only** (INV-01). Running the sweep on
the derived 16 kHz set would be measuring the downsampler rather than the
vocoders. :func:`data.manifest.require_primary` enforces this at the manifest
level.

**And it runs inside the ladder band** (INV-17). Every non-exempt condition is
low-passed to LADDER_FMAX before delivery, so cutoffs above it are no-ops that
would duplicate the full-band point under a different label. Use
:func:`cutoffs_for_rate`, which caps the sweep accordingly, rather than
DEFAULT_CUTOFFS directly. The question the sweep can no longer ask -- what
happens above LADDER_FMAX -- is answered by the paired-checkpoint contrast in
:func:`detectors.protocols.paired_bandwidth_contrast` instead.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from data.invariants import ARCHIVE_SR, LADDER_FMAX, InvariantViolation

# Cutoffs in Hz, all strictly below archive Nyquist (11.025 kHz). The 7-9 kHz
# spacing is deliberately fine: that is where the mel front-end's compression
# becomes severe and where BigVGAN's anti-aliasing is expected to show, so it is
# the region the curve has to resolve.
DEFAULT_CUTOFFS: tuple[float, ...] = (
    1000.0,
    2000.0,
    3000.0,
    4000.0,
    5000.0,
    6000.0,
    7000.0,
    8000.0,
    9000.0,
    10000.0,
)
FILTER_ORDER = 8


def nyquist(sr: int = ARCHIVE_SR) -> float:
    return sr / 2.0


def lowpass(wav: np.ndarray, cutoff_hz: float, sr: int = ARCHIVE_SR) -> np.ndarray:
    """Zero-phase Butterworth low-pass.

    Zero-phase (filtfilt) is deliberate: a causal filter imposes its own
    frequency-dependent group delay, which is itself a phase artifact and would
    contaminate the very cue the ablation is trying to isolate. It would also
    put the condition out of sample alignment with its reference, which INV-16
    exists to prevent. The identical filter is applied to real and vocoded audio
    alike.
    """
    nyq = nyquist(sr)
    if not 0 < cutoff_hz < nyq:
        raise InvariantViolation(
            f"cutoff {cutoff_hz} Hz is not inside (0, {nyq}) at {sr} Hz. Above Nyquist "
            "there is nothing to remove -- see the module docstring."
        )
    sos = butter(FILTER_ORDER, cutoff_hz / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, wav).astype(np.float32)


def band_limited_variant(
    wav: np.ndarray, cutoff_hz: float | None, sr: int = ARCHIVE_SR
) -> np.ndarray:
    """``None`` means full band; keeps the sweep loop uniform."""
    return wav if cutoff_hz is None else lowpass(wav, cutoff_hz, sr)


def cutoffs_for_rate(
    sr: int = ARCHIVE_SR, *, band_exempt: bool = False
) -> tuple[float, ...]:
    """The sweep, clipped to what actually exists in the signal.

    A cutoff at or above the signal's upper edge removes nothing, so including
    one would add a duplicate of the full-band point wearing a different label
    and flatten the apparent curve at the top end.

    That upper edge is NOT Nyquist for a ladder condition. INV-17 low-passes
    every non-exempt condition to LADDER_FMAX, so the sweep is capped there --
    a consequence of the ladder band worth stating plainly: **the ablation can
    no longer probe above LADDER_FMAX on ladder conditions, because nothing is
    there to remove.** What the ladder band buys in exchange is that the sweep
    is now measuring the same thing on every condition, which it was not when
    conditions differed in bandwidth.

    The >LADDER_FMAX question is answered instead by the paired-checkpoint
    contrast (`detectors.protocols.paired_bandwidth_contrast`), which varies the
    generator's own band rather than a filter over it. Exempt conditions keep
    their native band, so their sweep runs to Nyquist.
    """
    ceiling = nyquist(sr) if band_exempt else min(LADDER_FMAX, nyquist(sr))
    return tuple(c for c in DEFAULT_CUTOFFS if c < ceiling)


def collapse_point(cutoffs: list[float], eers: list[float], *, threshold: float = 0.25) -> float:
    """Highest cutoff at which EER has degraded past ``threshold``.

    A blunt summary of one curve, for the comparison table. Read the curves
    themselves before quoting this number -- a non-monotonic curve means
    something other than band content is moving.
    """
    order = np.argsort(cutoffs)
    c = np.asarray(cutoffs, dtype=float)[order]
    e = np.asarray(eers, dtype=float)[order]
    bad = np.where(e >= threshold)[0]
    return float(c[bad[-1]]) if bad.size else float("nan")
