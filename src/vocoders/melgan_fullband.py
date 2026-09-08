"""MelGAN delivered at its native band — INV-17 exempt paired control.

Partner to `melgan`: the same checkpoint, the same mel front-end, the same
generator output. The only difference is that this one is not low-passed to
LADDER_FMAX on the way out.

**This is a weaker contrast than the BigVGAN pair, and the difference matters.**
`bigvgan_112m` and `bigvgan_v2_22khz_fullband` are two *separately trained*
checkpoints whose mel front-ends differ in fmax, so their contrast isolates what
a generator trained to produce the high band does differently from one that was
not. The MelGAN pair shares one set of weights, so its contrast isolates only
what the delivery filter removes -- which is the band-limited ablation evaluated
at a single cutoff, not an independent question.

Both are worth having. The BigVGAN pair answers "does training bandwidth change
the artifact?"; this pair answers "is MelGAN's high band carrying detectable
artifact at all?", on a second architecture, so the high-band mechanism argument
does not rest on one model. Do not present them as two instances of the same
comparison.
"""

from __future__ import annotations

import numpy as np

from data.mel import MELGAN_FULLBAND_MEL

from .base import Vocoder, VocoderSpec
from .melgan import SPEC as MELGAN_SPEC

SPEC = VocoderSpec(
    key="melgan_fullband",
    display_name="MelGAN (full band)",
    family="gan",
    year=2019,
    params_m=4.3,
    mel=MELGAN_FULLBAND_MEL,
    checkpoint=MELGAN_SPEC.checkpoint,
    primary_ladder=False,
    notes=(
        "INV-17 exempt, excluded from the primary correlation. Same checkpoint "
        "as `melgan`, delivered without the ladder band. Isolates delivered "
        "bandwidth on a second architecture; shares weights with its partner, "
        "unlike the BigVGAN pair."
    ),
)


class MelGANFullband(Vocoder):
    spec = SPEC

    def load(self) -> None:
        raise NotImplementedError(
            "Identical to MelGAN.load(). Verify the weight file against "
            "checkpoint_sha256 with vocoders.checkpoints.verify_checkpoint before "
            "loading -- this condition and `melgan` must come from the same file, "
            "or the pair stops being a controlled contrast."
        )

    def synthesize(self, mel: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Identical to MelGAN.synthesize(). No post-processing (INV-10); the "
            "difference from `melgan` is applied in data.preprocess, which skips "
            "the ladder band for exempt conditions (INV-17)."
        )
