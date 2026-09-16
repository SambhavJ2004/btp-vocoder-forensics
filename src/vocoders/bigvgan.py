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

import re

import numpy as np

from data.invariants import AUDITED_MEL_FMAX, InvariantViolation
from data.measurements import load_meta
from data.mel import (
    BIGVGAN_BASE_22K_MEL,
    BIGVGAN_V2_22K_FMAX8K_MEL,
    BIGVGAN_V2_22K_FULLBAND_MEL,
    MelConfig,
)

from .base import Vocoder, VocoderSpec

# --- bigvgan_112m: pin and mel config sourced from the measurement run --------
# data/measurements/meta.json was written by the Kaggle run that produced the
# committed band measurements. Reading the pin and the mel config from it gives
# them PROVENANCE: the checkpoint these specs describe is the one that generated
# the measured audio, rather than a literal pasted in afterwards.
#
# Two identifiers, and they are not interchangeable:
#   checkpoint_revision  40-hex git commit of the HF repo. Pins which version of
#                        the repo is fetched. Not in meta.json; audited earlier.
#   checkpoint_sha256    64-hex digest of the weight FILE the run actually loaded.
#                        From meta.json. Verifies the bytes, not the repo state.
# meta.json's key is named `bigvgan_sha256`; it is a file digest, so it fills
# checkpoint_sha256 and leaves checkpoint_revision alone.
_META_REPO = "nvidia/bigvgan_v2_22khz_80band_fmax8k_256x"
_META_MEL_SOURCE = "data/measurements/meta.json[bigvgan_cfg]"


def _mel_from_meta(meta: dict) -> MelConfig:
    cfg = meta["bigvgan_cfg"]
    return MelConfig(
        sample_rate=int(cfg["sr"]),
        n_fft=int(cfg["n_fft"]),
        hop_length=int(cfg["hop"]),
        win_length=int(cfg["win"]),
        n_mels=int(cfg["n_mels"]),
        fmin=float(cfg["fmin"]),
        fmax=float(cfg["fmax"]),
        source=_META_MEL_SOURCE,
    )


def _validated_meta() -> dict:
    """Load meta.json and cross-check it against the config audit.

    Two independent sources now describe bigvgan_112m -- the audit read from the
    HF config (docs/mel_configs.md) and the measurement run's own record. They
    must agree, and a disagreement is loud rather than one silently winning:
    either the run used a different checkpoint than the ladder declares, or the
    audit is wrong, and both invalidate the measurements.
    """
    meta = load_meta()
    if meta.get("bigvgan_repo") != _META_REPO:
        raise InvariantViolation(
            f"INV-08: meta.json measured {meta.get('bigvgan_repo')!r} but bigvgan_112m "
            f"is {_META_REPO!r}. The committed measurements are not of this condition."
        )
    sha = str(meta.get("bigvgan_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise InvariantViolation(
            f"INV-08: meta.json bigvgan_sha256 {sha!r} is not a 64-hex SHA-256 digest."
        )

    measured = _mel_from_meta(meta)
    audited = BIGVGAN_V2_22K_FMAX8K_MEL
    for field in ("sample_rate", "n_fft", "hop_length", "win_length", "n_mels", "fmin", "fmax"):
        if getattr(measured, field) != getattr(audited, field):
            raise InvariantViolation(
                f"INV-02/INV-08: bigvgan_112m {field} is {getattr(measured, field)} in "
                f"meta.json but {getattr(audited, field)} in the config audit. The "
                "measurement run and the audit describe different front-ends."
            )
    if measured.fmax != AUDITED_MEL_FMAX["bigvgan_112m"]:
        raise InvariantViolation(
            "INV-17: meta.json fmax disagrees with AUDITED_MEL_FMAX; LADDER_FMAX "
            "would be derived from a value the measured checkpoint does not have."
        )
    return meta


_META = _validated_meta()

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
    mel=_mel_from_meta(_META),
    checkpoint=_META["bigvgan_repo"],
    checkpoint_revision="a9d8b711c344f3c49cc1a05d0c8b6741cf733aa8",
    checkpoint_sha256=_META["bigvgan_sha256"],
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
