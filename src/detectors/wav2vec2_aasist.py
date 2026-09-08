"""Frozen wav2vec2 front-end + trainable AASIST back-end.

Strongest detector in the set, and therefore the tightest available estimate of
the matched-protocol upper bound: how much artifact information the signal
carries at all, independent of any single architecture's blind spots.

Compute-gated. Per the plan's risk table, run AASIST alone first and only scale
to this once the quota picture is clear -- the frozen front-end still costs a
forward pass over every utterance every epoch, so cache features to disk rather
than recomputing them.
"""

from __future__ import annotations

import numpy as np

from .base import Detector, DetectorSpec

SPEC = DetectorSpec(
    key="w2v2_aasist",
    display_name="wav2vec2-XLSR + AASIST",
    params_m=317.0,
    input_kind="ssl_features",
    max_seconds=4.0375,
    pretrained="TODO: facebook/wav2vec2-xls-r-300m (frozen) + AASIST head (trained here)",
    notes="Front-end frozen. If it is ever unfrozen, say so -- it changes the claim.",
)


class Wav2Vec2AASIST(Detector):
    spec = SPEC

    def __init__(self, device: str = "cuda", freeze_frontend: bool = True):
        self.device = device
        self.freeze_frontend = freeze_frontend
        self._model = None

    def load(self, checkpoint: str | None = None) -> None:
        raise NotImplementedError(
            "Load XLS-R with requires_grad=False on the front-end, attach the AASIST "
            "back-end to a chosen transformer layer, and record WHICH layer -- layer "
            "choice moves EER enough to be a confound between runs."
        )

    def score(self, wav: np.ndarray) -> float:
        raise NotImplementedError("Forward and return the bona fide logit.")
