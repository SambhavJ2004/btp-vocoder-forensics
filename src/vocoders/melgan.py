"""MelGAN -- early, small, fast GAN vocoder. Low-quality end of the neural range.

Audit note (docs/mel_configs.md): the `descriptinc` release leaves
``mel_fmax=None``, which librosa resolves to sr/2 = 11025 Hz. MelGAN is
therefore the *widest*-band condition in the primary ladder, not the narrowest
-- the opposite of what the ladder's age would suggest. INV-17 low-passes it to
LADDER_FMAX like every other condition.

If this condition is ever re-sourced from a ParallelWaveGAN or ESPnet LJSpeech
recipe, fmax there is 7600, and LADDER_FMAX drops to 7600 with it. Re-run the
audit before swapping the source.
"""

from __future__ import annotations

import numpy as np

from data.mel import MELGAN_MEL

from .base import Vocoder, VocoderSpec

SPEC = VocoderSpec(
    key="melgan",
    display_name="MelGAN",
    family="gan",
    year=2019,
    params_m=4.3,
    mel=MELGAN_MEL,
    checkpoint="descriptinc/melgan-neurips",  # torch.hub load_melgan
    notes="No anti-aliasing machinery; expect strong high-band artifacts.",
)


class MelGAN(Vocoder):
    spec = SPEC

    def load(self) -> None:
        raise NotImplementedError(
            "Load the MelGAN generator in its own isolated session (INV-13), record "
            "the upstream commit, and record the weight-file sha256 once downloaded."
        )

    def synthesize(self, mel: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Return the generator output at 22.05 kHz, float32, mono, with NO "
            "post-processing (INV-10). Band-limiting to LADDER_FMAX happens later, "
            "in data.preprocess, applied identically to every condition (INV-17)."
        )
