"""Weights-only checkpoints and provenance for the base training protocol."""

import hashlib
import io
import pickle
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from .model import SpectralBackbone
from .training_config import (
    BUDGET_CONFIG_SCHEMA,
    TrainingConfig,
    canonical_json,
    finite_number,
    is_sha256,
    parse_json,
)
from .training_data import GeneratorManifest, sha256_file
from .training_device import accelerator_record, resolve_device

RUN_SCHEMA = "vocoroute-base-run-v1"
CHECKPOINT_SCHEMA = "vocoroute-base-checkpoint-v1"
WEIGHTS_FILE = "base.pt"
RECEIPT_FILE = "checkpoint.json"


def initialize_backbone(seed: int) -> SpectralBackbone:
    """Seed initialization without consuming the caller's CPU RNG stream."""
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        return SpectralBackbone().cpu().float()


def run_provenance(config: TrainingConfig, manifest: GeneratorManifest) -> dict:
    if config.manifest_sha256 != manifest.sha256:
        raise ValueError("run config and approved manifest identity disagree")
    source_root = Path(__file__).resolve().parent.parent
    source_paths = list((source_root / "vocoroute").glob("*.py"))
    source_paths.extend(source_root / "data" / name for name in ("audio_io.py", "invariants.py"))
    try:
        soundfile = version("soundfile")
    except PackageNotFoundError:
        soundfile = None
    record = {
        "schema": RUN_SCHEMA,
        "evidence": "development",
        "config": config.document(),
        "config_sha256": config.sha256,
        "dataset_id": manifest.dataset_id,
        "manifest_sha256": manifest.sha256,
        "source_sha256": {
            path.relative_to(source_root).as_posix(): sha256_file(path)
            for path in sorted(source_paths)
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torchaudio": version("torchaudio"),
            "soundfile": soundfile,
            "device": config.device,
            "threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }
    if config.schema == BUDGET_CONFIG_SCHEMA:
        record["runtime"]["accelerator"] = accelerator_record(resolve_device(config.device))
    return record


def _validate_receipt(receipt: dict, config: TrainingConfig) -> None:
    if set(receipt) != {"schema", "run", "history", "best_epoch", "weights_sha256"} or (
        receipt["schema"] != CHECKPOINT_SCHEMA or not is_sha256(receipt["weights_sha256"])
    ):
        raise ValueError("invalid base checkpoint receipt schema")
    run = receipt["run"]
    if not isinstance(run, dict) or set(run) != {
        "schema",
        "evidence",
        "config",
        "config_sha256",
        "dataset_id",
        "manifest_sha256",
        "source_sha256",
        "runtime",
    }:
        raise ValueError("invalid run provenance schema")
    if (
        run["schema"] != RUN_SCHEMA
        or run["evidence"] != "development"
        or not isinstance(run["dataset_id"], str)
        or not run["dataset_id"].strip()
        or not is_sha256(run["manifest_sha256"])
        or run["manifest_sha256"] != config.manifest_sha256
        or run["config_sha256"] != config.sha256
        or canonical_json(run["config"]) != canonical_json(config.document())
    ):
        raise ValueError("checkpoint run configuration or manifest identity disagrees")
    source = run["source_sha256"]
    if (
        not isinstance(source, dict)
        or not {"vocoroute/model.py", "vocoroute/spectral.py", "data/audio_io.py"} <= source.keys()
        or not all(is_sha256(digest) for digest in source.values())
    ):
        raise ValueError("invalid checkpoint code provenance")
    runtime = run["runtime"]
    extra_keys = {"accelerator"} if config.schema == BUDGET_CONFIG_SCHEMA else set()
    if (
        not isinstance(runtime, dict)
        or set(runtime)
        != extra_keys
        | {
            "python",
            "platform",
            "torch",
            "torchaudio",
            "soundfile",
            "device",
            "threads",
            "deterministic_algorithms",
        }
        or runtime["device"] != config.device
        or type(runtime["threads"]) is not int
        or runtime["threads"] < 1
        or type(runtime["deterministic_algorithms"]) is not bool
        or any(
            not isinstance(runtime[key], str) or not runtime[key]
            for key in (
                "python",
                "platform",
                "torch",
                "torchaudio",
            )
        )
        or (runtime["soundfile"] is not None and not isinstance(runtime["soundfile"], str))
    ):
        raise ValueError("invalid checkpoint runtime provenance")
    if config.device == "cuda":
        accelerator = runtime.get("accelerator")
        if not isinstance(accelerator, dict) or (
            not isinstance(accelerator.get("name"), str)
            or not isinstance(accelerator.get("cuda_build"), str)
            or accelerator.get("matmul_tf32") is not False
            or accelerator.get("cudnn_tf32") is not False
        ):
            raise ValueError("invalid checkpoint CUDA provenance")
    elif runtime.get("accelerator") is not None:
        raise ValueError("CPU checkpoint cannot declare a CUDA accelerator")
    history = receipt["history"]
    if not isinstance(history, list) or len(history) != config.epochs:
        raise ValueError("checkpoint history must contain the full configured run")
    for epoch, row in enumerate(history, 1):
        if (
            not isinstance(row, dict)
            or set(row) != {"epoch", "train_total", "dev_total"}
            or type(row["epoch"]) is not int
            or row["epoch"] != epoch
            or any(
                not finite_number(row[key]) or row[key] < 0 for key in ("train_total", "dev_total")
            )
        ):
            raise ValueError("invalid checkpoint epoch metrics")
    best = min(history, key=lambda row: row["dev_total"])["epoch"]
    if type(receipt["best_epoch"]) is not int or receipt["best_epoch"] != best:
        raise ValueError("checkpoint epoch must be the first minimum dev total")


def _validate_weights(weights: object, model: SpectralBackbone) -> None:
    expected = model.state_dict()
    if not isinstance(weights, dict) or weights.keys() != expected.keys():
        raise ValueError("checkpoint tensor keys do not match the spectral backbone")
    for name, original in expected.items():
        tensor = weights[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.layout != torch.strided
            or tensor.device.type != "cpu"
            or tensor.dtype != original.dtype
            or tensor.shape != original.shape
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"invalid checkpoint tensor: {name}")


def save_base_checkpoint(
    directory: Path,
    config: TrainingConfig,
    run: dict,
    history: list[dict],
    best_epoch: int,
    weights: dict[str, torch.Tensor],
) -> str:
    """Write to a fresh directory and publish the receipt last. Return its SHA-256."""
    _validate_weights(weights, initialize_backbone(config.seed))
    buffer = io.BytesIO()
    torch.save(weights, buffer)
    payload = buffer.getvalue()
    receipt = {
        "schema": CHECKPOINT_SCHEMA,
        "run": run,
        "history": history,
        "best_epoch": best_epoch,
        "weights_sha256": hashlib.sha256(payload).hexdigest(),
    }
    _validate_receipt(receipt, config)
    metadata = canonical_json(receipt) + b"\n"
    directory.mkdir(exist_ok=False)
    (directory / WEIGHTS_FILE).write_bytes(payload)
    (directory / RECEIPT_FILE).write_bytes(metadata)
    return hashlib.sha256(metadata).hexdigest()


def load_base_checkpoint(
    directory: Path, config: TrainingConfig, *, receipt_sha256: str
) -> tuple[SpectralBackbone, dict]:
    """Verify both content pins, provenance schema, and weight compatibility."""
    raw = (directory / RECEIPT_FILE).read_bytes()
    if not is_sha256(receipt_sha256) or hashlib.sha256(raw).hexdigest() != receipt_sha256:
        raise ValueError("checkpoint receipt does not match its external SHA-256 pin")
    receipt = parse_json(raw)
    _validate_receipt(receipt, config)
    code = receipt["run"]["source_sha256"]
    for filename in ("model.py", "spectral.py"):
        if sha256_file(Path(__file__).with_name(filename)) != code[f"vocoroute/{filename}"]:
            raise ValueError(f"checkpoint backbone or frontend source differs: {filename}")
    payload = (directory / WEIGHTS_FILE).read_bytes()
    if hashlib.sha256(payload).hexdigest() != receipt["weights_sha256"]:
        raise ValueError("checkpoint weights SHA-256 mismatch")
    model = initialize_backbone(config.seed)
    try:
        weights = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError("checkpoint payload was rejected by the weights-only loader") from exc
    _validate_weights(weights, model)
    model.load_state_dict(weights, strict=True)
    return model.eval(), receipt
