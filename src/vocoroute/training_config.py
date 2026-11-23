"""Versioned, strict configuration for the base-generator protocol."""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from . import PROTOTYPE_VERSION
from .spectral import FRONTEND_V1

CONFIG_SCHEMA = "vocoroute-base-training-v1"
TRAIN_ROLE, DEV_ROLE, EVAL_ROLE = "train", "dev", "eval"
ROLES = (TRAIN_ROLE, DEV_ROLE, EVAL_ROLE)
STFT_RESOLUTIONS = ((256, 64), (512, 128), (1024, 256))
SPECTRAL_FLOOR = 1e-4


def canonical_json(value: object) -> bytes:
    """Encode metadata reproducibly, with strict finite JSON numbers."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def parse_json(raw: bytes) -> dict:
    def unique(pairs):
        obj = dict(pairs)
        if len(obj) != len(pairs):
            raise ValueError("duplicate JSON field")
        return obj

    obj = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(obj, dict):
        raise ValueError("expected a JSON object")
    return obj


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def finite_number(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def fixed_contract() -> dict:
    return {
        "schema": CONFIG_SCHEMA,
        "model": {"architecture": PROTOTYPE_VERSION, "component": "spectral_backbone"},
        "frontend": {
            **asdict(FRONTEND_V1),
            "win_length": FRONTEND_V1.n_fft,
            "window": "periodic_hann",
            "center": True,
            "pad_mode": "constant_zero",
            "normalized": False,
            "onesided": True,
            "mel_scale": "slaney",
            "mel_norm": "slaney",
            "mel_power": 2,
            "mel_log": "natural",
            "f_min": 0,
            "f_max": FRONTEND_V1.sample_rate / 2,
        },
        "roles": {"fit": TRAIN_ROLE, "select": DEV_ROLE, "held_out": EVAL_ROLE},
        "batch_size": 1,
        "device": "cpu",
        "dtype": "float32",
        "stft_loss": {
            "resolutions": [list(pair) for pair in STFT_RESOLUTIONS],
            "magnitude_floor": SPECTRAL_FLOOR,
        },
    }


@dataclass(frozen=True)
class LossWeights:
    waveform: float = 1.0
    magnitude: float = 1.0
    log_magnitude: float = 1.0
    phase_enabled: bool = False
    phase_time: float = 0.05
    phase_frequency: float = 0.05

    def __post_init__(self) -> None:
        active = [self.waveform, self.magnitude, self.log_magnitude]
        phase = [self.phase_time, self.phase_frequency]
        if type(self.phase_enabled) is not bool or any(
            not finite_number(weight) or weight < 0 for weight in active + phase
        ):
            raise ValueError("expected finite nonnegative weights and boolean phase_enabled")
        if not any(active + (phase if self.phase_enabled else [])):
            raise ValueError("at least one enabled loss weight must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260907
    epochs: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    scheduler_gamma: float = 0.98
    loss: LossWeights = LossWeights()
    manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not finite_number(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not finite_number(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not finite_number(self.scheduler_gamma) or not 0 < self.scheduler_gamma <= 1:
            raise ValueError("scheduler gamma must be in (0, 1]")
        if not isinstance(self.loss, LossWeights):
            raise ValueError("loss must be a LossWeights instance")
        if self.manifest_sha256 is not None and not is_sha256(self.manifest_sha256):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest or null")

    def document(self) -> dict:
        return {
            **fixed_contract(),
            "seed": self.seed,
            "epochs": self.epochs,
            "manifest_sha256": self.manifest_sha256,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "foreach": False,
                "fused": False,
            },
            "scheduler": {
                "name": "ExponentialLR",
                "interval": "epoch",
                "gamma": self.scheduler_gamma,
            },
            "loss": asdict(self.loss),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.document())).hexdigest()

    @classmethod
    def from_document(cls, document: dict) -> "TrainingConfig":
        try:
            cfg = cls(
                seed=document["seed"],
                epochs=document["epochs"],
                manifest_sha256=document["manifest_sha256"],
                learning_rate=document["optimizer"]["learning_rate"],
                weight_decay=document["optimizer"]["weight_decay"],
                scheduler_gamma=document["scheduler"]["gamma"],
                loss=LossWeights(**document["loss"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid training configuration fields") from exc
        if canonical_json(cfg.document()) != canonical_json(document):
            raise ValueError("training configuration disagrees with the v1 schema or contract")
        return cfg


def load_training_config(path: str | Path) -> TrainingConfig:
    return TrainingConfig.from_document(parse_json(Path(path).read_bytes()))
