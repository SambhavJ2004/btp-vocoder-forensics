"""BigVGAN -- the state-of-the-art top of the ladder, plus the bandwidth pair.

Central to the mechanism argument. BigVGAN's anti-aliased multi-periodicity
(snake activations + low-pass filtering around up/downsampling) exists to
suppress high-frequency fold-back produced by its own nonlinearities. That is
precisely the band where the mel representation is most compressed and where
detectors have the most to work with -- so the band-limited ablation
(:mod:`detectors.bandlimit`) is the experiment that turns this from a story
into evidence.

Three conditions live here, all at the archive rate of 22.05 kHz:

  ``bigvgan_base``     v1, 14M params, fmax 8000. Isolates capacity from design
                       against the 112M model.
  ``bigvgan_112m``     v2, 112M params, fmax 8000. Top of the primary ladder.
  ``bigvgan_v2_22khz_fullband``
                       v2, 112M params, fmax 11025. **Not** in the primary
                       ladder and **exempt from INV-17**. Same architecture,
                       same parameter count, same training recipe as
                       ``bigvgan_112m`` -- differing in mel fmax alone. That
                       makes the two a controlled pair that isolates bandwidth
                       from architecture, which is the one comparison the
                       band-limited ablation cannot make on its own.

All three are 22.05 kHz natively, so none of them takes a resampling filter on
the way to the archive (INV-01).
"""

from __future__ import annotations

import numpy as np

from data.mel import (
    BIGVGAN_BASE_22K_MEL,
    BIGVGAN_V2_22K_FMAX8K_MEL,
    BIGVGAN_V2_22K_FULLBAND_MEL,
)

from .base import Vocoder, VocoderSpec

BASE_SPEC = VocoderSpec(
    key="bigvgan_base",
    display_name="BigVGAN-base",
    family="gan",
    year=2022,
    params_m=14.0,
    mel=BIGVGAN_BASE_22K_MEL,
    checkpoint="nvidia/bigvgan_base_22khz_80band",
    checkpoint_revision="760f0c694aba68b8c200dd52ffd3b4ecafa7028c",
    notes="Same architecture family as the large model; isolates capacity from design.",
)

LARGE_SPEC = VocoderSpec(
    key="bigvgan_112m",
    display_name="BigVGAN v2 (112M, fmax 8k)",
    family="gan",
    year=2024,
    params_m=112.0,
    mel=BIGVGAN_V2_22K_FMAX8K_MEL,
    checkpoint="nvidia/bigvgan_v2_22khz_80band_fmax8k_256x",
    checkpoint_revision="a9d8b711c344f3c49cc1a05d0c8b6741cf733aa8",
    notes="Top of the primary ladder. The interesting cell of the main plot.",
)

FULLBAND_SPEC = VocoderSpec(
    key="bigvgan_v2_22khz_fullband",
    display_name="BigVGAN v2 (112M, full band)",
    family="gan",
    year=2024,
    params_m=112.0,
    mel=BIGVGAN_V2_22K_FULLBAND_MEL,
    checkpoint="nvidia/bigvgan_v2_22khz_80band_256x",
    checkpoint_revision="633ff708ed5b74903e86ff1298cf4a98e921c513",
    primary_ladder=False,
    notes=(
        "INV-17 exempt, excluded from the primary correlation. Paired with "
        "bigvgan_112m: identical architecture and parameter count, differing in "
        "mel fmax alone (11025 vs 8000). Isolates bandwidth from architecture."
    ),
)


class BigVGAN(Vocoder):
    """Wraps all three; pass the spec you want."""

    def __init__(self, spec: VocoderSpec = LARGE_SPEC):
        self.spec = spec
        self._model = None

    def load(self) -> None:
        raise NotImplementedError(
            "Load via bigvgan.BigVGAN.from_pretrained(spec.checkpoint, "
            "revision=spec.checkpoint_revision, use_cuda_kernel=False) -- the custom "
            "CUDA kernel is a compile risk on Kaggle and is not needed for "
            "inference-only work. Pass the pinned revision explicitly; omitting it "
            "silently tracks the branch head (INV-08). Then remove_weight_norm() "
            "and eval()."
        )

    def synthesize(self, mel: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Use upstream's meldataset.get_mel_spectrogram() for analysis so the "
            "front-end matches training exactly, then generator forward. Return the "
            "model's native 22.05 kHz, float32 mono, no post-processing (INV-10)."
        )

    def unload(self) -> None:
        self._model = None
