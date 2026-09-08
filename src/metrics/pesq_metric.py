"""PESQ (ITU-T P.862) — intrusive quality, wideband mode.

**Runs on the DERIVED zero-shot tier, not the archive.** Wideband PESQ is
*defined* at 16 kHz: the reference implementation accepts 8 kHz or 16 kHz and
nothing else, so it cannot follow the archive rate. This is one of the three
external constraints that force the derived tier to exist at all, alongside
UTMOS and the pretrained ASVspoof detectors.

The consequence must be stated in the thesis rather than discovered by a
reviewer: PESQ is blind to everything above 8 kHz, which is precisely the band
the mechanism argument concerns. It therefore measures a strictly narrower
question than MCD or band-wise LSD, both of which run on the archive. Do not
read a flat PESQ column as evidence that the conditions are equivalent in the
high band -- PESQ cannot see the high band.

PESQ also saturates on high-quality speech: modern vocoders may all cluster near
the ceiling while UTMOS still separates them. Known property, not a bug.
"""

from __future__ import annotations

import numpy as np

from data.invariants import ZEROSHOT_SR, InvariantViolation


def pesq_wb(reference: np.ndarray, degraded: np.ndarray, sr: int = ZEROSHOT_SR) -> float:
    """Wideband PESQ (MOS-LQO, roughly 1.0-4.5). Higher is better."""
    if sr != ZEROSHOT_SR:
        raise InvariantViolation(
            f"INV-01: PESQ-WB is defined only at {ZEROSHOT_SR} Hz; got {sr}. Score it "
            "on the derived zero-shot tier -- never resample inside a metric, which "
            "would be a second filtering pass on the signal."
        )
    if len(reference) != len(degraded):
        raise InvariantViolation("PESQ needs aligned pairs; see INV-06.")

    from pesq import pesq as _pesq  # optional dep: pip install '.[quality]'

    return float(_pesq(sr, reference.astype(np.float64), degraded.astype(np.float64), "wb"))
