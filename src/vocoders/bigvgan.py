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

import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

from data.invariants import AUDITED_MEL_FMAX, InvariantViolation
from data.measurements import load_meta
from data.mel import (
    BIGVGAN_BASE_22K_MEL,
    BIGVGAN_V2_22K_FMAX8K_MEL,
    BIGVGAN_V2_22K_FULLBAND_MEL,
    MelConfig,
)

from .base import Vocoder, VocoderSpec, resolve_torch_device
from .upstream import check_hparams as _check_hparams
from .upstream import import_pinned

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
    # A literal, unlike bigvgan_112m's, which comes from meta.json because that
    # checkpoint produced the committed band measurements. No measurement run has
    # used this one, so there is no artifact to read it from; recording it here is
    # what turns the revision-only pin into one `verify_checkpoint` can enforce.
    #
    # Computed from `bigvgan_generator.pt` at the pinned revision, and
    # cross-checked against HuggingFace's own LFS oid for that file, which is the
    # same digest -- so a future pin can be recorded from the tree API
    # (`?expand=true`, `lfs.oid`) without downloading 56 MB. Verified equal here
    # rather than assumed, because a wrong digest fails the run, not the pin.
    checkpoint_sha256="e2a93f92971e1b6750d8f9b701121597cd271e473e6c189c8072be7080047735",
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


# --- Loading, without the HF mixin -------------------------------------------
# Upstream's `BigVGAN.from_pretrained` goes through huggingface_hub's
# PyTorchModelHubMixin, whose `_from_pretrained` signature changed under it and
# no longer works against current huggingface_hub. The mixin only ever did three
# things -- fetch config.json, build the model from it, load the generator
# weights -- so those are done here directly.
#
# The model CODE comes from the same pinned revision as the weights: the HF repo
# ships bigvgan.py, meldataset.py and alias_free_activation/ alongside them. So
# one content-addressed revision pins code, config and weights together, and the
# mel front-end is upstream's own get_mel_spectrogram (INV-02), not a copy.

UPSTREAM_CODE_FILES: tuple[str, ...] = (
    "bigvgan.py",
    "activations.py",
    "env.py",
    "meldataset.py",
    "utils.py",
    "alias_free_activation/torch/__init__.py",
    "alias_free_activation/torch/act.py",
    "alias_free_activation/torch/filter.py",
    "alias_free_activation/torch/resample.py",
)
UPSTREAM_CONFIG = "config.json"
UPSTREAM_WEIGHTS = "bigvgan_generator.pt"
# Top-level module names upstream's code imports itself by. They are generic
# ("utils", "env"), so a same-named module imported earlier from anywhere else
# would be picked up silently instead of upstream's.
UPSTREAM_MODULES: tuple[str, ...] = (
    "bigvgan", "activations", "env", "meldataset", "utils", "alias_free_activation",
)

def fetch_pinned_snapshot(
    spec: VocoderSpec, download: Callable[..., str] | None = None
) -> Path:
    """Download code, config and weights from the pinned revision. Verify first.

    Returns the snapshot directory. The weight digest is checked against
    ``spec.checkpoint_sha256`` before anything from the snapshot is imported or
    loaded (INV-08): a swapped checkpoint must fail here, not after audio exists.
    """
    from .checkpoints import hf_token, verify_checkpoint

    if not spec.checkpoint_revision:
        raise InvariantViolation(
            f"INV-08: {spec.key} has no checkpoint_revision; downloading would track "
            "the branch head."
        )
    if not spec.checkpoint_sha256:
        raise InvariantViolation(
            f"INV-08: {spec.key} has no checkpoint_sha256 to verify the weights against."
        )
    if download is None:
        from huggingface_hub import hf_hub_download

        download = hf_hub_download

    token = hf_token()  # optional; None means huggingface_hub's own default
    files = (*UPSTREAM_CODE_FILES, UPSTREAM_CONFIG, UPSTREAM_WEIGHTS)
    paths = {
        f: Path(download(
            repo_id=spec.checkpoint, filename=f, revision=spec.checkpoint_revision, token=token
        ))
        for f in files
    }
    root = paths["bigvgan.py"].parent
    strays = [f for f, p in paths.items() if p != root / f]
    if strays:
        raise InvariantViolation(
            f"INV-08: {spec.key} files {strays} did not land in one snapshot directory; "
            "code, config and weights must come from the same revision."
        )
    verify_checkpoint(paths[UPSTREAM_WEIGHTS], spec.checkpoint_sha256, condition=spec.key)
    return root


UPSTREAM_GUARD_MODULES = UPSTREAM_MODULES
UPSTREAM_WANTED = ("bigvgan", "meldataset", "env")


def check_hparams(h: dict, mel: MelConfig, *, condition: str) -> None:
    """INV-02. The config.json at the pinned revision must match the audit."""
    _check_hparams(h, mel, condition=condition)


def import_upstream(root: Path) -> dict:
    """Import upstream's modules from ``root`` and prove they came from there."""
    return import_pinned(
        Path(root), guard=UPSTREAM_GUARD_MODULES, wanted=UPSTREAM_WANTED,
        why="BigVGAN's code",
    )


class BigVGAN(Vocoder):
    """Wraps all three; pass the spec you want.

    Only ``bigvgan_112m`` carries a weight digest today, so it is the only one
    ``load()`` will accept -- the other two fail closed at the digest check.
    """

    runtime_modules = ("torch", "huggingface_hub", "librosa", "scipy", "matplotlib")
    required_pins = ("checkpoint_revision", "checkpoint_sha256")

    def __init__(self, spec: VocoderSpec = LARGE_SPEC, device: str | None = None):
        self.spec = spec
        self.device = device
        self._model = None
        self._h = None
        self._mel_fn = None

    def load(self) -> None:
        import torch

        root = fetch_pinned_snapshot(self.spec)
        upstream = import_upstream(root)
        h = upstream["env"].AttrDict(
            json.loads((root / UPSTREAM_CONFIG).read_text(encoding="utf-8"))
        )
        check_hparams(h, self.spec.mel, condition=self.spec.key)

        self.device = resolve_torch_device(self.device)
        # use_cuda_kernel=False: the custom CUDA kernel is a compile risk on Kaggle
        # and inference does not need it.
        model = upstream["bigvgan"].BigVGAN(h, use_cuda_kernel=False)
        state = torch.load(root / UPSTREAM_WEIGHTS, map_location="cpu", weights_only=True)
        # Mirrors upstream's own loader: a checkpoint saved without weight norm
        # only loads after removing it; one saved with it loads first and then
        # has it removed for inference.
        try:
            model.load_state_dict(state["generator"])
        except RuntimeError:
            model.remove_weight_norm()
            model.load_state_dict(state["generator"])
        else:
            model.remove_weight_norm()
        self._model = model.eval().to(self.device)
        self._h = h
        self._mel_fn = upstream["meldataset"].get_mel_spectrogram

    def _require_loaded(self):
        if self._model is None:
            raise RuntimeError(f"{self.spec.key}: call load() first.")
        return self._model

    def synthesize(self, mel: np.ndarray) -> np.ndarray:
        import torch

        model = self._require_loaded()
        x = torch.from_numpy(np.ascontiguousarray(mel, dtype=np.float32))
        if x.ndim == 2:
            x = x.unsqueeze(0)
        with torch.inference_mode():
            out = model(x.to(self.device))
        return out.squeeze().float().cpu().numpy().astype(np.float32)

    def resynthesize(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Upstream's own front-end (``meldataset.get_mel_spectrogram``), then forward."""
        import torch

        if sr != self.spec.mel.sample_rate:
            raise ValueError(
                f"{self.spec.key}: expected input at {self.spec.mel.sample_rate} Hz, got {sr}."
            )
        model = self._require_loaded()
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            mel = self._mel_fn(x.to(self.device), self._h)
            out = model(mel)
        return out.squeeze().float().cpu().numpy().astype(np.float32)

    def unload(self) -> None:
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass
