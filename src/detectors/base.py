"""Detector adapter interface.

Detectors return a single score per utterance, higher == more bona fide. That is
the only contract; architectures differ freely underneath.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from data.invariants import ARCHIVE_SR, ZEROSHOT_SR

# INV-11. The crop is a CROSS-DETECTOR policy, so it lives here rather than in
# any one adapter. The ASVspoof convention is 64600 SAMPLES, which is 4.0375 s
# only at 16 kHz -- reusing that integer at the archive rate would silently
# shorten the crop to 2.93 s and make the matched protocol see less signal than
# the mismatched one. What carries across rates is the DURATION.
ASVSPOOF_CROP_SAMPLES = 64_600  # zero-shot tier, 16 kHz
ASVSPOOF_CROP_SECONDS = ASVSPOOF_CROP_SAMPLES / ZEROSHOT_SR
ARCHIVE_CROP_SAMPLES = round(ASVSPOOF_CROP_SECONDS * ARCHIVE_SR)  # 89027 @ 22.05 kHz


def crop_samples_for_rate(sr: int) -> int:
    """Samples spanning the ASVspoof crop duration at ``sr`` (INV-11).

    Always ask for the crop by rate. A bare 64600 is correct on one tier only.
    """
    if sr == ZEROSHOT_SR:
        return ASVSPOOF_CROP_SAMPLES
    if sr == ARCHIVE_SR:
        return ARCHIVE_CROP_SAMPLES
    raise ValueError(f"INV-01: {sr} Hz is not a sanctioned tier rate.")


@dataclass(frozen=True)
class DetectorSpec:
    key: str
    display_name: str
    params_m: float
    input_kind: str          # "raw" | "lfcc" | "ssl_features"
    max_seconds: float       # fixed-length crop the model expects
    pretrained: str = ""     # checkpoint URL / Kaggle Dataset path
    notes: str = ""


class Detector(ABC):
    spec: DetectorSpec

    @abstractmethod
    def load(self, checkpoint: str | None = None) -> None: ...

    @abstractmethod
    def score(self, wav: np.ndarray) -> float:
        """Higher == more bona fide."""

    def crop_for(
        self, wav: np.ndarray, sr: int, *, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """INV-11. The sanctioned way for an adapter to reach its input length.

        Every adapter's ``score``/``fit`` calls this rather than cropping for
        itself, so the crop duration and the padding policy are the same across
        detectors, conditions and tiers by construction. Adapters that crop
        independently are the way INV-11 gets broken quietly.
        """
        return fixed_length_crop(wav, crop_samples_for_rate(sr), rng=rng)

    def score_batch(self, wavs: list[np.ndarray]) -> np.ndarray:
        return np.array([self.score(w) for w in wavs], dtype=np.float64)

    def fit(self, *args, **kwargs) -> None:
        """Matched-protocol training. Zero-shot-only detectors may leave this."""
        raise NotImplementedError(f"{self.spec.key} does not support training here.")


def fixed_length_crop(wav: np.ndarray, n_samples: int, *, rng: np.random.Generator | None = None):
    """Repeat-pad or crop to the model's fixed input length.

    Repeat-padding rather than zero-padding is the ASVspoof convention, and it
    matters here for a specific reason: zero-padding would make padded silence
    duration a function of utterance length, handing the detector a cue that has
    nothing to do with vocoder artifacts. The same crop policy is used for every
    condition (INV-11).
    """
    if len(wav) >= n_samples:
        if rng is None:
            return wav[:n_samples]
        start = int(rng.integers(0, len(wav) - n_samples + 1))
        return wav[start : start + n_samples]
    reps = int(np.ceil(n_samples / len(wav)))
    return np.tile(wav, reps)[:n_samples]
