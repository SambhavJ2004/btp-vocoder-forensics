"""MelGAN -- early, small, fast GAN vocoder. Low-quality end of the neural range.

Audit note (docs/mel_configs.md): the `descriptinc` release leaves
``mel_fmax=None``, which librosa resolves to sr/2 = 11025 Hz. MelGAN is
therefore the *widest*-band condition in the primary ladder, not the narrowest
-- the opposite of what the ladder's age would suggest. INV-17 low-passes it to
LADDER_FMAX like every other condition.

If this condition is ever re-sourced from a ParallelWaveGAN or ESPnet LJSpeech
recipe, fmax there is 7600, and LADDER_FMAX drops to 7600 with it. Re-run the
audit before swapping the source.

**Weights live in the repository, not a release.** ``models/linda_johnson.pt``
is committed to ``descriptinc/melgan-neurips``, so a pinned commit of raw GitHub
gives both code and weights from one content-addressed source -- the cleanest
provenance of any condition here, better than HiFi-GAN's two hosts. "linda
johnson" is LJSpeech: the corpus is one speaker, Linda Johnson, and the
alternative ``multi_speaker.pt`` would be the wrong condition.

**Two front-end differences that are real and recorded, not incidental.**
``Audio2Mel`` takes **log10** of the mel, where HiFi-GAN and BigVGAN take the
natural log, and it pads reflect by ``(n_fft - hop)//2`` with ``center=False``.
Both are part of "vocoder and its own front-end as a unit" (INV-02), which is
why upstream's extractor is used rather than ``data.mel.compute_mel``.

**Its 2019 code does not run on current libraries**, and is shimmed at two call
sites with the differences recorded -- see :mod:`vocoders.compat`.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

import numpy as np

from data.invariants import InvariantViolation
from data.mel import MELGAN_MEL

from .base import Vocoder, VocoderSpec, resolve_torch_device
from .compat import apply_shims, describe
from .upstream import import_pinned

CODE_REPO = "descriptinc/melgan-neurips"
CODE_COMMIT = "6488045bfba1975602288de07a58570c7b4d66ea"
# Only modules.py: it is self-contained (torch, librosa, numpy), whereas
# mel2wav/__init__.py imports interface.py, which pulls in yaml and a Drive-free
# but unnecessary checkpoint loader. Fetched under its own upstream name so the
# bytes stay comparable with upstream's.
CODE_FILES: tuple[str, ...] = ("modules.py",)
WEIGHTS_FILE = "models/linda_johnson.pt"
RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"

# `modules` is about as generic a top-level name as exists, which is exactly what
# the import guard is for.
GUARD_MODULES: tuple[str, ...] = ("modules", "mel2wav")
WANTED_MODULES: tuple[str, ...] = ("modules",)

# Generator(input_size, ngf, n_residual_layers), as upstream's own interface.py
# constructs it for the released checkpoints.
GENERATOR_ARGS = (80, 32, 3)

SPEC = VocoderSpec(
    key="melgan",
    display_name="MelGAN",
    family="gan",
    year=2019,
    params_m=4.3,
    mel=MELGAN_MEL,
    checkpoint=f"{CODE_REPO}@{CODE_COMMIT[:12]}:{WEIGHTS_FILE}",
    checkpoint_sha256="d9f8a9934a162128a276b49122733a315eb434261e9c06162e11e0c5fa7a59e1",
    upstream_commit=CODE_COMMIT,
    notes="No anti-aliasing machinery; expect strong high-band artifacts. "
          "Code and weights from one pinned commit; log10 mel front-end.",
)

# Audio2Mel constructor defaults -> MelConfig field. MelGAN ships no config file,
# so the front-end IS these defaults; this is the only thing there is to verify
# against, and verifying it is what stops the audit drifting from the code.
_AUDIO2MEL_DEFAULTS: tuple[tuple[str, str], ...] = (
    ("sampling_rate", "sample_rate"),
    ("n_fft", "n_fft"),
    ("hop_length", "hop_length"),
    ("win_length", "win_length"),
    ("n_mel_channels", "n_mels"),
    ("mel_fmin", "fmin"),
)


def check_audio2mel_defaults(audio2mel, mel, *, condition: str = "melgan") -> None:
    """INV-02 for a vocoder with no config file: check the code's own defaults.

    ``mel_fmax`` is deliberately absent from the pairs: upstream leaves it
    ``None`` and librosa resolves that to Nyquist, so the audit records 11025 for
    a default of ``None``. That one is checked as "still None", because a release
    that started bounding fmax would change the condition's band and must not
    pass silently.
    """
    params = inspect.signature(audio2mel.__init__).parameters
    for name, field in _AUDIO2MEL_DEFAULTS:
        if name not in params:
            raise InvariantViolation(
                f"INV-02: {condition}: upstream Audio2Mel has no {name!r} parameter; "
                "the pinned commit is not the one this audit describes."
            )
        actual, expected = params[name].default, getattr(mel, field)
        if float(actual) != float(expected):
            raise InvariantViolation(
                f"INV-02: {condition}: upstream Audio2Mel defaults {name}={actual}, "
                f"the audit says {field}={expected}."
            )
    if params["mel_fmax"].default is not None:
        raise InvariantViolation(
            f"INV-02: {condition}: upstream Audio2Mel now defaults "
            f"mel_fmax={params['mel_fmax'].default!r} rather than None. The audit "
            f"records {mel.fmax} on the assumption librosa resolves None to Nyquist; "
            "a bounded fmax changes this condition's band and LADDER_FMAX with it."
        )


def fetch_pinned_snapshot(
    spec: VocoderSpec = SPEC,
    *,
    fetch_url: Callable[[str, Path], Path] | None = None,
    root: Path | None = None,
) -> tuple[Path, Path, list[dict]]:
    """Fetch code and weights from the one pinned commit. Verify the digest first.

    Returns ``(code_root, weight_path, shim_records)``.
    """
    from .checkpoints import verify_checkpoint

    if not spec.upstream_commit:
        raise InvariantViolation(
            f"INV-08: {spec.key} has no upstream_commit; both the code and the weights "
            "come from that commit, so without it nothing is pinned."
        )
    if not spec.checkpoint_sha256:
        raise InvariantViolation(
            f"INV-08: {spec.key} has no checkpoint_sha256 to verify the weights against."
        )
    if fetch_url is None:
        fetch_url = _download_url

    root = Path(root) if root is not None else Path.home() / ".cache" / "btpvf" / "melgan"
    root.mkdir(parents=True, exist_ok=True)

    weights = root / "linda_johnson.pt"
    if not weights.is_file():
        fetch_url(RAW_URL.format(repo=CODE_REPO, commit=spec.upstream_commit,
                                 path=WEIGHTS_FILE), weights)
    verify_checkpoint(weights, spec.checkpoint_sha256, condition=spec.key)

    shims = []
    for name in CODE_FILES:
        target = root / name
        if not target.is_file():
            fetch_url(RAW_URL.format(repo=CODE_REPO, commit=spec.upstream_commit,
                                     path=f"mel2wav/{name}"), target)
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


class MelGAN(Vocoder):
    runtime_modules = ("torch", "librosa", "scipy")
    required_pins = ("checkpoint_sha256", "upstream_commit")
    spec = SPEC

    def __init__(self, device: str | None = None) -> None:
        self.device = device
        self._model = None
        self._mel = None
        self._shims: list[dict] = []

    def load(self) -> None:
        import torch

        root, weights, shims = fetch_pinned_snapshot(self.spec)
        self._shims = shims
        if shims:
            print(f"[{self.spec.key}] compatibility shims:")
            print(describe(shims))
        upstream = import_pinned(
            root, guard=GUARD_MODULES, wanted=WANTED_MODULES, why="MelGAN's code"
        )
        modules = upstream["modules"]
        check_audio2mel_defaults(modules.Audio2Mel, self.spec.mel, condition=self.spec.key)

        self.device = resolve_torch_device(self.device)
        model = modules.Generator(*GENERATOR_ARGS)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self._model = model.eval().to(self.device)
        # Upstream's own extractor, with upstream's own defaults (INV-02).
        self._mel = modules.Audio2Mel().to(self.device)

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
        # Raw generator output (INV-10); the band limit is an analysis transform.
        return audio.squeeze().detach().cpu().numpy().astype(np.float32)

    def resynthesize(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Real audio -> upstream's log10 mel -> waveform."""
        import torch

        self._require_loaded()
        if sr != self.spec.mel.sample_rate:
            raise ValueError(
                f"{self.spec.key}: expected input at {self.spec.mel.sample_rate} Hz "
                f"(its training rate), got {sr}."
            )
        tensor = torch.as_tensor(wav, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            mel = self._mel(tensor.unsqueeze(0).unsqueeze(0))
        return self.synthesize(mel.squeeze(0).cpu().numpy())

    def unload(self) -> None:
        self._model = None
        self._mel = None
