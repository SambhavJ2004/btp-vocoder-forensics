"""SpecDiff-GAN — diffusion-flavoured GAN vocoder.

RISK (week-one decision, from the plan's risk table): pretrained weights may not
be publicly available. If they are not obtainable by the week-one deadline,
substitute Vocos or Avocodo and record the substitution in the manifest and the
thesis rather than silently dropping a rung from the ladder.

Its value in the ladder is family diversity: if GAN vocoders cluster in
embedding space (the Phase C family-level hypothesis), a non-pure-GAN condition
is what makes that testable.
"""

from __future__ import annotations

import numpy as np

from data.mel import SPECDIFF_MEL

from .base import Vocoder, VocoderSpec

SPEC = VocoderSpec(
    key="specdiff_gan",
    display_name="SpecDiff-GAN",
    family="diffusion",
    year=2023,
    params_m=None,
    mel=SPECDIFF_MEL,
    checkpoint="BLOCKED: SpecDiff-GAN/SpecDiff-GAN ships configs/ but no pretrained "
               "weights; inference expects a user-supplied --checkpoint_file",
    notes="Substitute Vocos or Avocodo if weights are unavailable; log the swap.",
)


class SpecDiffGAN(Vocoder):
    spec = SPEC

    def load(self) -> None:
        raise NotImplementedError(
            "BLOCKED pending checkpoint availability. See the week-one decision "
            "deadline in the plan's risk table."
        )

    def synthesize(self, mel: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "If a diffusion sampler is involved, fix the step count and the sampling "
            "seed and record both in the manifest (INV-08) -- stochastic sampling "
            "would otherwise make the condition irreproducible."
        )
