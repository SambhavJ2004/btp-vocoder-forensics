"""Running a vocoder's OWN released code, and proving which code ran.

Two adapters now load upstream implementations rather than reimplementing them
(INV-02 prefers the vocoder's own front-end; reimplementing a generator would
make the condition "our version of HiFi-GAN"). Both therefore face the same two
problems, and this module is the one implementation of both:

**Provenance.** Upstream code imports itself by generic top-level names --
``utils``, ``env``, ``models`` -- so a same-named module imported earlier from
anywhere else is picked up silently and the condition runs somebody else's code.
:func:`import_pinned` refuses that, and refuses an import that resolved anywhere
but the pinned snapshot. Both sides of the comparison are resolved paths, because
the HuggingFace cache keeps content in ``blobs/`` and makes
``snapshots/<revision>/`` a tree of symlinks into it.

**Config agreement.** The mel numbers in this repository are an audit
(``docs/mel_configs.md``); the numbers the checkpoint was trained with live in
the checkpoint's own config. :func:`check_hparams` requires them to be equal at
load time, so INV-02's "copy the numbers from the upstream release, never guess
them" is enforced rather than documented.

Extracted from ``vocoders.bigvgan`` when ``vocoders.hifigan`` became the second
caller. The alternative -- a second copy -- is how the silence floor came to be
computed two different ways under one name.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from data.invariants import InvariantViolation
from data.mel import MelConfig

# Upstream config key -> MelConfig field. jik876's `config_v1.json` and NVIDIA's
# BigVGAN `config.json` happen to use the same names, because BigVGAN's training
# code descends from HiFi-GAN's.
HPARAM_FIELDS: tuple[tuple[str, str], ...] = (
    ("sampling_rate", "sample_rate"),
    ("n_fft", "n_fft"),
    ("hop_size", "hop_length"),
    ("win_size", "win_length"),
    ("num_mels", "n_mels"),
    ("fmin", "fmin"),
    ("fmax", "fmax"),
)


def check_hparams(h: dict, mel: MelConfig, *, condition: str) -> None:
    """INV-02. The config at the pinned revision must match the audit."""
    for key, field in HPARAM_FIELDS:
        if key not in h:
            raise InvariantViolation(f"INV-02: {condition} config has no {key!r}.")
        value = h[key]
        expected = getattr(mel, field)
        if value is None and field == "fmax":
            continue  # unbounded fmax is resolved to Nyquist by the MelConfig
        if float(value) != float(expected):
            raise InvariantViolation(
                f"INV-02: {condition} config says {key}={value}, the audit says "
                f"{field}={expected}. The mel front-end is recorded per condition and "
                "cannot be guessed; reconcile docs/mel_configs.md with the checkpoint."
            )


def module_origin(module) -> list[Path]:
    """Where Python actually loaded ``module`` from, with every symlink followed."""
    file = getattr(module, "__file__", None)
    if file:
        return [Path(file).resolve()]
    return [Path(p).resolve() for p in getattr(module, "__path__", [])]


def expected_origin(root: Path, name: str, module) -> Path:
    """Where ``name`` lives in the snapshot, with every symlink followed.

    A module with a file is ``<name>.py`` or ``<name>/__init__.py``; a namespace
    package (BigVGAN's ``alias_free_activation`` has no ``__init__``) is its
    directory.
    """
    if getattr(module, "__file__", None):
        for entry in (root / f"{name}.py", root / name / "__init__.py"):
            if os.path.lexists(entry):
                return entry.resolve()
        return (root / f"{name}.py").resolve()  # absent: matches no real file
    return (root / name).resolve()


def from_snapshot(module, root: Path, name: str) -> bool:
    """True only if every origin of ``module`` IS the snapshot's own entry.

    Both sides are resolved. The first version of this check resolved the
    module's path but compared it against the snapshot DIRECTORY, whose files are
    symlinks -- so every correctly loaded module failed it, which is what stopped
    the first real Kaggle run. Comparing the resolved module file with the
    resolved snapshot ENTRY is at least as strict: an equality on what Python
    loaded, not a prefix test on a directory.
    """
    expected = expected_origin(root, name, module)
    origins = module_origin(module)
    return bool(origins) and all(origin == expected for origin in origins)


def import_pinned(
    root: Path, *, guard: tuple[str, ...], wanted: tuple[str, ...], why: str
) -> dict:
    """Import ``wanted`` from ``root`` and prove they came from there.

    ``guard`` is every top-level name upstream's code might import itself by,
    checked for a pre-existing import before ``root`` goes on ``sys.path``;
    ``wanted`` is what the caller needs handed back. ``why`` names the code, for
    the refusal message.
    """
    root = Path(root)
    for name in guard:
        existing = sys.modules.get(name)
        if existing is not None and not from_snapshot(existing, root, name):
            raise InvariantViolation(
                f"INV-13: a module named {name!r} is already imported from "
                f"{[str(o) for o in module_origin(existing)]}, which is not the pinned "
                f"snapshot's {name!r} ({expected_origin(root, name, existing)}). "
                f"{why} imports itself by that top-level name and would silently get "
                "the wrong module. Run the condition in a fresh process."
            )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    modules = {n: importlib.import_module(n) for n in wanted}
    for name, module in modules.items():
        if not from_snapshot(module, root, name):
            raise InvariantViolation(
                f"INV-13: imported {name!r} from "
                f"{[str(o) for o in module_origin(module)]}, not from the pinned "
                f"snapshot's {name!r} ({expected_origin(root, name, module)})."
            )
    return modules
