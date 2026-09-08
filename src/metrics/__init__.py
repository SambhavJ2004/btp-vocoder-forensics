"""Perceptual quality measurement — the X-axis. Person B.

UTMOS is the primary metric; PESQ, MCD, F0 error and band-wise LSD support it.
Submodules import their heavy optional dependencies lazily so that importing
this package does not require the ``[quality]`` extra.
"""

from .mcd import mcd
from .spectral import (
    high_band_distance,
    log_spectral_distance_by_band,
    spectral_centroid_shift,
)

__all__ = [
    "high_band_distance",
    "log_spectral_distance_by_band",
    "mcd",
    "spectral_centroid_shift",
]
