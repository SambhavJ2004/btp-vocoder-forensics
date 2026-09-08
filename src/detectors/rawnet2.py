"""RawNet2 — second detector, for architecture-robustness of the headline claim.

Its role is narrow but important: if the quality-vs-detectability correlation
holds under AASIST but reverses under RawNet2, the finding is about AASIST, not
about vocoders. Two architectures is the minimum needed to say anything about
the signal rather than the model.
"""

from __future__ import annotations

import numpy as np

from .base import Detector, DetectorSpec

SPEC = DetectorSpec(
    key="rawnet2",
    display_name="RawNet2",
    params_m=17.6,
    input_kind="raw",
    max_seconds=4.0375,
    pretrained="TODO: asvspoof.org 2021 LA baseline RawNet2 weights",
    notes="Sinc-conv front-end + residual blocks + GRU.",
)


class RawNet2(Detector):
    spec = SPEC

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    def load(self, checkpoint: str | None = None) -> None:
        raise NotImplementedError(
            "Use the ASVspoof 2021 LA baseline implementation and weights; take the crop "
            "from crop_samples_for_rate(sr) exactly as AASIST does, so the two "
            "detectors see identical inputs on whichever tier is in play (INV-11)."
        )

    def score(self, wav: np.ndarray) -> float:
        raise NotImplementedError("Forward and return the bona fide logit.")
