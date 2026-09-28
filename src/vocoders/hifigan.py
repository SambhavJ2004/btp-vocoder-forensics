"""HiFi-GAN v1 -- the field's default baseline vocoder.

The one condition on the ladder whose code and weights come from **two different
hosts**, which is stated here rather than hidden because it is the weakest pin in
the study.

**Code: the official implementation**, ``jik876/hifi-gan`` at a pinned GitHub
commit, fetched as raw files. Not a community reimplementation, and not
``trust_remote_code`` on a mirror: the generator is the thing under measurement,
so it has to be upstream's own.

**Weights: a HuggingFace mirror**, ``jaketae/hifigan-lj-v1`` at a pinned
revision, verified by sha256. Upstream publishes LJ_V1 only through a Google
Drive folder, which cannot be fetched non-interactively (Drive interposes a
confirmation page on large files), so a run that had to download from there could
not be automated at all. The mirror is used as a byte-addressed host and nothing
more -- its own modelling code is never imported.

What makes that mirror acceptable is that it is checkable, and it was checked:

  * its ``config.json`` generator architecture equals upstream's
    ``config_v1.json`` field for field (``upsample_rates [8,8,2,2]``,
    ``upsample_kernel_sizes [16,16,4,4]``, ``upsample_initial_channel 512``,
    ``resblock_kernel_sizes [3,7,11]``, ``resblock_dilation_sizes [[1,3,5]]*3``);
  * its ``state_dict`` keys are upstream's own (``conv_pre``, ``ups.N``,
    ``resblocks.N.convs{1,2}.M``, ``conv_post``), with weight norm still applied,
    so it loads into upstream's ``Generator`` without renaming;
  * the file is 55.8 MB, matching upstream's released ``generator_v1``;
  * and ``checkpoint_sha256`` pins the exact bytes, so a swapped mirror fails at
    :func:`vocoders.checkpoints.verify_checkpoint` before any import (INV-08).

If it is ever mirrored on a host the team controls, only ``WEIGHTS_REPO`` and the
digest change.

**Mel front-end.** Upstream's own ``meldataset.mel_spectrogram`` (INV-02: use the
vocoder's own extractor where it ships one), and :func:`upstream.check_hparams`
requires ``config_v1.json`` to agree with ``docs/mel_configs.md`` at load time.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from data.invariants import InvariantViolation
from data.mel import HIFIGAN_V1_MEL

from .base import Vocoder, VocoderSpec, resolve_torch_device
from .compat import apply_shims, describe
from .upstream import check_hparams, import_pinned

# Official implementation, pinned. Raw GitHub is content-addressed by commit and
# fetches non-interactively, which Drive does not.
CODE_REPO = "jik876/hifi-gan"
CODE_COMMIT = "4769534d45265d52a904b850da5a622601885777"
# utils.py is not optional: models.py does `from utils import init_weights,
# get_padding`, by exactly the generic top-level name the import guard exists
# to protect.
CODE_FILES: tuple[str, ...] = (
    "models.py", "utils.py", "env.py", "meldataset.py", "config_v1.json",
)
RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"

# Weight host. Separate from the code on purpose; see the module docstring.
WEIGHTS_REPO = "jaketae/hifigan-lj-v1"
WEIGHTS_REVISION = "85caaf4ed15cfb83ba79a994a2266aa892645495"
WEIGHTS_FILE = "pytorch_model.bin"
UPSTREAM_CONFIG = "config_v1.json"

# Top-level names upstream's code imports itself by. `models`, `env` and `utils`
# are generic enough that an unrelated module of the same name would be picked up
# silently, which is what the guard in `upstream.import_pinned` refuses.
GUARD_MODULES: tuple[str, ...] = ("models", "env", "meldataset", "utils")
WANTED_MODULES: tuple[str, ...] = ("models", "env", "meldataset")

SPEC = VocoderSpec(
    key="hifigan_v1",
    display_name="HiFi-GAN v1",
    family="gan",
    year=2020,
    params_m=13.9,
    mel=HIFIGAN_V1_MEL,
    checkpoint=f"{WEIGHTS_REPO} ({WEIGHTS_FILE}); code {CODE_REPO}@{CODE_COMMIT[:12]}",
    checkpoint_revision=WEIGHTS_REVISION,
    checkpoint_sha256="6265dc8996049cfe01c4e148d6ad6db1e9a1784b1978bda56fc29d74daddb5a2",
    upstream_commit=CODE_COMMIT,
    notes="Multi-period + multi-scale discriminators. Baseline condition.",
)


def fetch_pinned_snapshot(
    spec: VocoderSpec = SPEC,
    *,
    download: Callable[..., str] | None = None,
    fetch_url: Callable[[str, Path], Path] | None = None,
    root: Path | None = None,
) -> tuple[Path, Path, list[dict]]:
    """Fetch code and weights into one directory. Verify the digest FIRST.

    Returns ``(code_root, weight_path, shim_records)``: two paths because code
    and weights come from two hosts, and the records because upstream's 2020 code
    does not run on current librosa/torch and is rewritten at two call sites
    (:mod:`vocoders.compat`). Mirrors ``vocoders.bigvgan.fetch_pinned_snapshot``: the weight digest
    is checked before anything is imported or loaded, so a swapped checkpoint
    fails here rather than after audio exists (INV-08).
    """
    from .checkpoints import hf_token, verify_checkpoint

    if not spec.checkpoint_revision:
        raise InvariantViolation(
            f"INV-08: {spec.key} has no checkpoint_revision; downloading would track "
            "the branch head of the weight mirror."
        )
    if not spec.checkpoint_sha256:
        raise InvariantViolation(
            f"INV-08: {spec.key} has no checkpoint_sha256. The weights come from a "
            "mirror rather than from upstream, so the digest is the only thing that "
            "says which bytes ran."
        )
    if not spec.upstream_commit:
        raise InvariantViolation(
            f"INV-08: {spec.key} has no upstream_commit; the generator implementation "
            "would track the branch head."
        )

    if download is None:
        from huggingface_hub import hf_hub_download

        def download(**kwargs):
            return hf_hub_download(**kwargs, token=hf_token())

    weights = Path(
        download(
            repo_id=WEIGHTS_REPO, filename=WEIGHTS_FILE, revision=spec.checkpoint_revision
        )
    )
    # Before any code is fetched, let alone imported.
    verify_checkpoint(weights, spec.checkpoint_sha256, condition=spec.key)

    root = Path(root) if root is not None else weights.parent / "hifigan_upstream"
    root.mkdir(parents=True, exist_ok=True)
    if fetch_url is None:
        fetch_url = _download_url
    shims = []
    for name in CODE_FILES:
        target = root / name
        if not target.is_file():
            fetch_url(RAW_URL.format(repo=CODE_REPO, commit=spec.upstream_commit,
                                     path=name), target)
        # After the fetch and after the weight digest, never before: the shim is a
        # recorded modification of upstream's text, not a substitute for pinning it.
        shims.append(apply_shims(target, condition=spec.key))
    return root, weights, [r for r in shims if r.get("shimmed")]


def _download_url(url: str, target: Path) -> Path:
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:  # noqa: S310 -- pinned raw.githubusercontent
        if response.status != 200:
            raise InvariantViolation(f"INV-08: {url} returned HTTP {response.status}")
        target.write_bytes(response.read())
    return target


class HiFiGAN(Vocoder):
    runtime_modules = ("torch", "huggingface_hub", "librosa", "scipy")
    required_pins = ("checkpoint_revision", "checkpoint_sha256", "upstream_commit")
    spec = SPEC

    def __init__(self, device: str | None = None) -> None:
        self.device = device
        self._model = None
        self._h = None
        self._mel_fn = None
        self._shims: list[dict] = []

    def load(self) -> None:
        import torch

        root, weights, shims = fetch_pinned_snapshot(self.spec)
        self._shims = shims
        if shims:
            print(f"[{self.spec.key}] compatibility shims:")
            print(describe(shims))
        upstream = import_pinned(
            root, guard=GUARD_MODULES, wanted=WANTED_MODULES, why="HiFi-GAN's code"
        )
        h = upstream["env"].AttrDict(
            json.loads((root / UPSTREAM_CONFIG).read_text(encoding="utf-8"))
        )
        # INV-02, enforced rather than documented: the audit must equal the config
        # this checkpoint was trained with.
        check_hparams(h, self.spec.mel, condition=self.spec.key)

        self.device = resolve_torch_device(self.device)
        model = upstream["models"].Generator(h)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        # The mirror stores a bare generator state_dict with weight norm applied;
        # upstream's own release wraps it in {"generator": ...}. Accept both, and
        # remove weight norm for inference exactly as upstream's inference.py does.
        model.load_state_dict(state["generator"] if "generator" in state else state)
        model.remove_weight_norm()
        self._model = model.eval().to(self.device)
        self._h = h
        self._mel_fn = upstream["meldataset"].mel_spectrogram

    def _require_loaded(self):
        if self._model is None:
            raise RuntimeError(f"{self.spec.key}: call load() first.")
        return self._model

    def synthesize(self, mel: np.ndarray) -> np.ndarray:
        import torch

        model = self._require_loaded()
        tensor = torch.as_tensor(mel, dtype=torch.float32, device=self.device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        with torch.inference_mode():
            audio = model(tensor)
        # Raw generator output, no post-processing (INV-10). Band-limiting to
        # LADDER_FMAX is an analysis transform and happens later (INV-17).
        return audio.squeeze().detach().cpu().numpy().astype(np.float32)

    def resynthesize(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Real audio -> upstream's own mel -> waveform.

        Overrides the default so the mel matches the TRAINING front-end exactly
        (INV-02), rather than this repository's reimplementation of it.
        """
        import torch

        self._require_loaded()
        if sr != self.spec.mel.sample_rate:
            raise ValueError(
                f"{self.spec.key}: expected input at {self.spec.mel.sample_rate} Hz "
                f"(its training rate), got {sr}."
            )
        h = self._h
        tensor = torch.as_tensor(wav, dtype=torch.float32).unsqueeze(0)
        mel = self._mel_fn(
            tensor, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
            h.fmin, h.fmax, center=False,
        )
        return self.synthesize(mel.squeeze(0).cpu().numpy())

    def unload(self) -> None:
        self._model = None
        self._mel_fn = None
