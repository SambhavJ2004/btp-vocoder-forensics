"""HiFi-GAN v1 — the field's default baseline vocoder.

Ships in the v0 dataset alongside Griffin-Lim so that Persons B and C have two
conditions spanning a wide quality gap to debug their pipelines against.
"""

from __future__ import annotations

import numpy as np

from data.mel import HIFIGAN_V1_MEL

from .base import Vocoder, VocoderSpec

SPEC = VocoderSpec(
    key="hifigan_v1",
    display_name="HiFi-GAN v1",
    family="gan",
    year=2020,
    params_m=13.9,
    mel=HIFIGAN_V1_MEL,
    checkpoint="TODO: jik876/hifi-gan LJ_V1 generator (Google Drive release)",
    notes="Multi-period + multi-scale discriminators. Baseline condition.",
)


class HiFiGAN(Vocoder):
    spec = SPEC

    def load(self) -> None:
        raise NotImplementedError(
            "Clone jik876/hifi-gan, instantiate Generator(h) from config_v1.json, load "
            "the LJ_V1 state dict, call remove_weight_norm(), and set eval(). Pin the "
            "checkpoint sha256 into SPEC before generating any audio."
        )

    def synthesize(self, mel: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Generator forward on the log-mel; return 22.05 kHz float32 mono, no "
            "post-processing (INV-10). Use upstream's mel_spectrogram() rather than "
            "data.mel.compute_mel -- match the training front-end exactly. "
            "Band-limiting to LADDER_FMAX happens later, in data.preprocess (INV-17)."
        )
