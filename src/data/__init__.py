"""Corpus handling, confound enforcement and the dataset manifest. Person A."""

from .alignment import AlignmentReport, assert_aligned, check_vocoder_alignment
from .invariants import (
    ARCHIVE_SR,
    ARCHIVE_TIER,
    BAND_EXEMPT_CONDITIONS,
    CONTRACTS,
    LADDER_FMAX,
    LOUDNESS_TARGET_LUFS,
    REAL_CONDITION,
    SUBTYPE,
    ZEROSHOT_SR,
    ZEROSHOT_TIER,
    InvariantViolation,
    check_band_limit,
    check_pairing,
    check_waveform,
    is_band_exempt,
    tier_for_rate,
)

__all__ = [
    "ARCHIVE_SR",
    "ARCHIVE_TIER",
    "BAND_EXEMPT_CONDITIONS",
    "CONTRACTS",
    "LADDER_FMAX",
    "LOUDNESS_TARGET_LUFS",
    "REAL_CONDITION",
    "SUBTYPE",
    "ZEROSHOT_SR",
    "ZEROSHOT_TIER",
    "AlignmentReport",
    "InvariantViolation",
    "assert_aligned",
    "check_band_limit",
    "check_pairing",
    "check_vocoder_alignment",
    "check_waveform",
    "is_band_exempt",
    "tier_for_rate",
]
