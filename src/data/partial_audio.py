"""Sample-domain operations for a separate, non-citable partial-audio protocol.

These functions do not resynthesize, align, trim, normalize, resample or filter.
All intervals are zero-based and half-open. Crossfades live INSIDE the edit span.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from data.invariants import (
    ARCHIVE_SR,
    GLOBAL_SEED,
    SPLIT_NAMES,
    TRIM_FRAME_LENGTH,
    TRIM_HOP_LENGTH,
    TRIM_TOP_DB,
)

REAL, SYNTHETIC, IGNORE = 0, 1, -1
VOCODERS = ("bigvgan_112m", "griffin_lim", "vocos")
SCHEMA_VERSION = "vocopatch_partial_v1"
REVIEW_SCHEMA_VERSION = "vocopatch_partial_v2"
FADE_METHOD = "complementary_linear_strict_interior"


class UnsuitableUtterance(ValueError):
    """An explicit whole-group sampling exclusion, never an invariant waiver."""


@dataclass(frozen=True)
class PartialAudioConfig:
    seed: int = GLOBAL_SEED
    n_utterances: int = 12
    segment_seconds: float = 1.0
    crossfade_ms: float = 10.0
    boundary_margin_seconds: float = 0.25
    minimum_active_fraction: float = 0.8
    activity_top_db: float = TRIM_TOP_DB
    variants: int = 1
    demo_examples: int = 5
    real_to_real_controls: bool = False
    split_counts: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.split_counts, list):
            object.__setattr__(self, "split_counts", tuple(self.split_counts))

    def validate(self) -> None:
        if not isinstance(self.real_to_real_controls, bool):
            raise ValueError("real_to_real_controls must be an explicit boolean.")
        if self.split_counts is not None:
            if (not isinstance(self.split_counts, tuple)
                    or len(self.split_counts) != len(SPLIT_NAMES)
                    or any(not isinstance(n, int) or isinstance(n, bool) or n < 0
                           for n in self.split_counts)):
                raise ValueError("split_counts must contain integer train/dev/eval counts >= 0.")
            if sum(self.split_counts) != self.n_utterances:
                raise ValueError("Explicit split counts must sum to n_utterances.")
        for name in ("seed", "n_utterances", "variants", "demo_examples"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer.")
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must lie in [0, 2**63).")
        if not 1 <= self.n_utterances <= 12 or self.variants not in (1, 2):
            raise ValueError("Pilot permits 1–12 utterances and 1–2 variants.")
        if not 0 <= self.demo_examples <= 5:
            raise ValueError("demo_examples must lie in [0, 5].")
        for name in ("segment_seconds", "crossfade_ms", "boundary_margin_seconds",
                     "minimum_active_fraction", "activity_top_db"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number.")
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if (self.segment_seconds <= 0 or self.crossfade_ms < 0
                or self.boundary_margin_seconds <= 0 or self.activity_top_db <= 0
                or not 0 < self.minimum_active_fraction <= 1):
            raise ValueError("Invalid duration, margin, fade or activity policy.")
        if self.margin_samples < 1 or self.segment_samples <= 2 * self.fade_samples:
            raise ValueError("Need an interior margin and a nonempty pure-source core.")

    @property
    def schema_version(self) -> str:
        return (REVIEW_SCHEMA_VERSION if self.real_to_real_controls or self.split_counts is not None
                else SCHEMA_VERSION)

    @property
    def segment_samples(self) -> int:
        return round(self.segment_seconds * ARCHIVE_SR)

    @property
    def fade_samples(self) -> int:
        return round(self.crossfade_ms * ARCHIVE_SR / 1000)

    @property
    def margin_samples(self) -> int:
        return round(self.boundary_margin_seconds * ARCHIVE_SR)


DEFAULT_PARTIAL_CONFIG = PartialAudioConfig()


def utterance_rng(seed: int, utt_id: str, variant: int) -> np.random.Generator:
    key = f"{seed}|{utt_id}|{variant}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def activity_candidates(real: np.ndarray, config: PartialAudioConfig) -> tuple[np.ndarray, dict]:
    """RMS activity proxy from real only, not phonetic/word-level forced alignment.

    Full 512-sample frames at hop 128, no padding. Candidate intervals start at
    margin+k*hop, contain >= the recorded fraction of active frame centers, and
    have active first/last frames. Digital silence is never called speech.
    """
    config.validate()
    real = np.asarray(real)
    if real.ndim != 1 or not np.isfinite(real).all():
        raise ValueError("Activity input must be a finite mono waveform.")
    n = len(real)
    length, margin = config.segment_samples, config.margin_samples
    if n < max(TRIM_FRAME_LENGTH, length + 2 * margin):
        raise UnsuitableUtterance("too_short_for_interior_segment")
    starts = np.arange(0, n - TRIM_FRAME_LENGTH + 1, TRIM_HOP_LENGTH)
    cumulative = np.r_[0.0, np.cumsum(real.astype(np.float64) ** 2)]
    rms = np.sqrt(np.maximum(
        (cumulative[starts + TRIM_FRAME_LENGTH] - cumulative[starts]) / TRIM_FRAME_LENGTH, 0,
    ))
    peak = float(rms.max())
    if peak <= 2 / 32768:
        raise UnsuitableUtterance("no_measurable_reference_activity")
    threshold = max(peak * 10 ** (-config.activity_top_db / 20), 2 / 32768)
    active = rms >= threshold
    centers = starts + TRIM_FRAME_LENGTH // 2
    candidates = np.arange(margin, n - margin - length + 1, TRIM_HOP_LENGTH)
    left = np.searchsorted(centers, candidates)
    right = np.searchsorted(centers, candidates + length)
    counts = right - left
    sums = np.r_[0, np.cumsum(active)]
    fractions = (sums[right] - sums[left]) / np.maximum(counts, 1)
    valid = (counts > 0) & (fractions >= config.minimum_active_fraction)
    valid &= active[np.minimum(left, len(active) - 1)] & active[np.maximum(right - 1, 0)]
    candidates = candidates[valid]
    if not len(candidates):
        raise UnsuitableUtterance("no_active_interior_segment")
    return candidates, {
        "method": "real_rms_frame_activity_proxy", "frame_samples": TRIM_FRAME_LENGTH,
        "hop_samples": TRIM_HOP_LENGTH, "rms_threshold": threshold,
        "maximum_rms": peak, "absolute_floor": 2 / 32768,
        "candidate_count": int(len(candidates)),
        "minimum_active_fraction": config.minimum_active_fraction,
        "activity_top_db": config.activity_top_db,
    }


def choose_region(
    real: np.ndarray, config: PartialAudioConfig, utt_id: str, variant: int = 1,
    *, previous: tuple[int, int] | None = None,
) -> tuple[int, int, dict]:
    candidates, activity = activity_candidates(real, config)
    if previous is not None:
        candidates = candidates[(candidates + config.segment_samples <= previous[0])
                                | (candidates >= previous[1])]
        if not len(candidates):
            raise UnsuitableUtterance("no_disjoint_second_variant_region")
    start = int(utterance_rng(config.seed, utt_id, variant).choice(candidates))
    return start, start + config.segment_samples, dict(
        activity, available_candidate_count=int(len(candidates)),
    )


def choose_real_donor(
    real: np.ndarray, config: PartialAudioConfig, utt_id: str, variant: int,
    target: tuple[int, int],
) -> dict:
    """A different active excerpt from the SAME real utterance, never another split.

    This intentionally changes speech content/time position. It is not alignment
    repair. Reject identical cores so this category cannot silently become null.
    """
    candidates, activity = activity_candidates(real, config)
    start, end = target
    candidates = candidates[(candidates + config.segment_samples <= start) | (candidates >= end)]
    fade = config.fade_samples
    target_core = real[start + fade:end - fade]
    for checked, donor_start in enumerate(
        utterance_rng(config.seed, f"real-donor:{utt_id}", variant).permutation(candidates),
        start=1,
    ):
        donor_start = int(donor_start)
        donor_end = donor_start + config.segment_samples
        if not np.array_equal(real[donor_start + fade:donor_end - fade], target_core):
            return {"start_sample": donor_start, "end_sample": donor_end,
                    "activity": dict(activity, disjoint_candidate_count=int(len(candidates)),
                                     candidates_checked=checked)}
    raise UnsuitableUtterance("no_distinct_disjoint_real_donor_region")


def validate_region(n: int, start: int, end: int, fade: int) -> None:
    if any(not isinstance(value, (int, np.integer)) or isinstance(value, bool)
           for value in (n, start, end, fade)):
        raise ValueError("Sample boundaries must be integers.")
    if not 0 < start < end < n or fade < 0 or end - start <= 2 * fade:
        raise ValueError("Invalid interior edit region or empty pure-source core.")


def splice(real: np.ndarray, donor: np.ndarray, start: int, end: int, fade: int,
           *, donor_start: int | None = None) -> np.ndarray:
    """Convex sample-aligned replacement; same-source null edits are exact identities.

    On each fade, donor weights are 1/(F+1),...,F/(F+1), reversed at the exit.
    Both contributions are strictly positive at EVERY annotated transition sample.
    Difference-form interpolation avoids creating rounding changes on null edits.
    """
    real, donor = np.asarray(real), np.asarray(donor)
    if (real.ndim != 1 or real.shape != donor.shape or not np.isfinite(real).all()
            or not np.isfinite(donor).all() or np.iscomplexobj(real) or np.iscomplexobj(donor)):
        raise ValueError("Splicing requires equal-length finite real mono inputs.")
    validate_region(len(real), start, end, fade)
    donor_start = start if donor_start is None else donor_start
    if (not isinstance(donor_start, (int, np.integer)) or isinstance(donor_start, bool)
            or not 0 <= donor_start <= len(donor) - (end - start)):
        raise ValueError("Invalid donor excerpt bounds.")
    weights = np.ones(end - start, dtype=np.float64)
    if fade:
        ramp = np.arange(1, fade + 1, dtype=np.float64) / (fade + 1)
        weights[:fade], weights[-fade:] = ramp, ramp[::-1]
    output = real.astype(np.float32, copy=True)
    reference = real[start:end].astype(np.float64)
    excerpt = donor[donor_start:donor_start + end - start].astype(np.float64)
    output[start:end] = reference + weights * (excerpt - reference)
    return output


def interval_labels(n: int, start: int, end: int, fade: int, *, synthetic: bool) -> list[dict]:
    """Complete run-length labels, including matched ignore buffers on null controls."""
    validate_region(n, start, end, fade)
    intervals = [(0, start, REAL), (start, start + fade, IGNORE),
                 (start + fade, end - fade, SYNTHETIC if synthetic else REAL),
                 (end - fade, end, IGNORE), (end, n, REAL)]
    labels: list[dict] = []
    for left, right, label in intervals:
        if left == right:
            continue
        if labels and labels[-1]["label"] == label:
            labels[-1]["end_sample"] = right
        else:
            labels.append({"start_sample": left, "end_sample": right, "label": label})
    return labels
