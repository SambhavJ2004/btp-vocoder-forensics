"""Forensic detectability — the Y-axis. Person C.

Both protocols live in :mod:`detectors.protocols` and every reported number
carries its protocol label (INV-14).
"""

from .base import Detector, DetectorSpec, crop_samples_for_rate, fixed_length_crop
from .eer import bootstrap_eer_ci, compute_eer
from .protocols import DetectionResult, Protocol, score_condition, spearman_headline

__all__ = [
    "DetectionResult",
    "Detector",
    "DetectorSpec",
    "Protocol",
    "bootstrap_eer_ci",
    "compute_eer",
    "crop_samples_for_rate",
    "fixed_length_crop",
    "score_condition",
    "spearman_headline",
]
