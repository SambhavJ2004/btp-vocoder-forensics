"""Run configuration. Paths and compute knobs only.

Nothing that affects the scientific comparison belongs here. Confound controls
live in :mod:`data.invariants` precisely so they cannot be overridden by a YAML
file that a tired person edits at 2am to make a run go through.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Paths:
    data_root: Path = Path(os.environ.get("BTPVF_DATA_ROOT", "./data"))
    corpus_root: Path = Path("./data/LJSpeech-1.1")
    audio_out: Path = Path("./data/resynth")
    manifest: Path = Path("./manifests/dataset.csv")
    results: Path = Path("./results")
    checkpoints: Path = Path("./checkpoints")

    def __post_init__(self) -> None:
        for f in ("data_root", "corpus_root", "audio_out", "manifest", "results", "checkpoints"):
            setattr(self, f, Path(getattr(self, f)))


@dataclass
class RunConfig:
    """Per-run knobs. ``dataset_version`` distinguishes the throwaway v0 dataset
    (200 utterances, Griffin-Lim + HiFi-GAN, shipped in week two so B and C can
    build in parallel) from the full six-condition set. Swapping between them is
    a path change, nothing more -- keep it that way.
    """

    dataset_version: str = "v0"
    conditions: tuple[str, ...] = ()
    n_utterances: int | None = 200
    device: str = "cuda"
    batch_size: int = 8
    num_workers: int = 2
    paths: Paths = field(default_factory=Paths)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        paths = Paths(**raw.pop("paths", {}))
        if "conditions" in raw:
            raw["conditions"] = tuple(raw["conditions"])
        return cls(paths=paths, **raw)
