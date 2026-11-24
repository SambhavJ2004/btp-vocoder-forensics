"""Pinned training-state checkpoints, separate from the inference bundle."""

import hashlib
import io
import math
import pickle
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .training_checkpoint import _validate_weights
from .training_config import TrainingConfig, canonical_json, finite_number, is_sha256, parse_json

RESUME_SCHEMA = "vocoroute-base-resume-v1"


@dataclass
class Progress:
    updates: int = 0
    history: list[dict] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    dev_losses: list[float] = field(default_factory=list)

    def cursor(self) -> dict:
        return {
            "updates": self.updates,
            "completed_epochs": len(self.history),
            "next_train_row": len(self.train_losses),
            "next_dev_row": len(self.dev_losses),
        }


def _loss(value: object) -> bool:
    return finite_number(value) and value >= 0


def _validate_progress(
    progress: Progress, config: TrainingConfig, train_rows: int, dev_rows: int
) -> None:
    histories = (progress.history, progress.train_losses, progress.dev_losses)
    if any(not isinstance(series, list) for series in histories):
        raise ValueError("invalid resume loss history")
    for epoch, row in enumerate(progress.history, 1):
        if (
            not isinstance(row, dict)
            or set(row) != {"epoch", "train_total", "dev_total"}
            or type(row["epoch"]) is not int
            or row["epoch"] != epoch
            or not _loss(row["train_total"])
            or not _loss(row["dev_total"])
        ):
            raise ValueError("invalid resume epoch history")
    completed = len(progress.history)
    train = len(progress.train_losses)
    dev = len(progress.dev_losses)
    if (
        type(progress.updates) is not int
        or progress.updates != completed * train_rows + train
        or progress.updates > config.max_updates
        or completed > config.epochs
        or train > train_rows
        or dev > dev_rows
        or (dev and train != train_rows)
        or (completed == config.epochs and (train or dev))
        or any(not _loss(value) for value in progress.train_losses + progress.dev_losses)
    ):
        raise ValueError("resume cursor is inconsistent with the frozen order or budget")


def _validate_optimizer(state, optimizer, scheduler_state, scheduler, model, progress) -> None:
    expected = optimizer.state_dict()
    if not isinstance(state, dict) or state.keys() != expected.keys():
        raise ValueError("invalid resume optimizer schema")
    if not isinstance(state["param_groups"], list) or len(state["param_groups"]) != 1:
        raise ValueError("invalid resume optimizer parameter groups")
    group = state["param_groups"][0]
    original = expected["param_groups"][0]
    if (
        not isinstance(group, dict)
        or group.keys() != original.keys()
        or any(group[key] != original[key] for key in original if key != "lr")
    ):
        raise ValueError("resume optimizer options or parameter order disagree")
    current = scheduler.state_dict()
    epochs = len(progress.history)
    learning_rate = original["initial_lr"] * scheduler.gamma**epochs
    if (
        not isinstance(scheduler_state, dict)
        or scheduler_state.keys() != current.keys()
        or any(scheduler_state.get(key) != current[key] for key in ("gamma", "base_lrs"))
        or scheduler_state.get("last_epoch") != epochs
        or scheduler_state.get("_step_count") != epochs + 1
        or scheduler_state.get("_last_lr") != [group["lr"]]
        or not finite_number(group["lr"])
        or not math.isclose(group["lr"], learning_rate, rel_tol=1e-12)
    ):
        raise ValueError("resume scheduler state disagrees with the epoch or learning rate")
    params = dict(zip(original["params"], model.parameters(), strict=True))
    moments = state["state"]
    if not isinstance(moments, dict) or set(moments) != (
        set(params) if progress.updates else set()
    ):
        raise ValueError("resume optimizer state does not cover the expected parameters")
    for key, moment in moments.items():
        if not isinstance(moment, dict) or set(moment) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("invalid resume optimizer moments")
        step = moment["step"]
        if not isinstance(step, torch.Tensor) or step.ndim != 0 or step.item() != progress.updates:
            raise ValueError("resume optimizer update count disagrees")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = moment[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.layout != torch.strided
                or tensor.shape != params[key].shape
                or tensor.dtype != params[key].dtype
                or not torch.isfinite(tensor).all()
                or (name == "exp_avg_sq" and bool((tensor < 0).any()))
            ):
                raise ValueError("invalid resume optimizer moment tensor")


def _rng_state(device: torch.device) -> dict:
    return {
        "python": random.getstate(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng(state: dict, device: torch.device) -> None:
    if not isinstance(state, dict) or set(state) != {"python", "cpu", "cuda"}:
        raise ValueError("invalid resume RNG schema")
    current = _rng_state(device)
    for kind in ("cpu", "cuda"):
        tensor = state[kind]
        expected = current[kind]
        if expected is None:
            if tensor is not None:
                raise ValueError("unexpected CUDA RNG in CPU checkpoint")
        elif (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.uint8
            or tensor.shape != expected.shape
        ):
            raise ValueError(f"invalid resume {kind} RNG tensor")
    random.Random().setstate(state["python"])
    torch.Generator().set_state(state["cpu"])
    if device.type == "cuda":
        torch.Generator(device).set_state(state["cuda"])
        torch.cuda.set_rng_state(state["cuda"], device)
    torch.set_rng_state(state["cpu"])
    random.setstate(state["python"])


def write_resume(
    path: Path,
    *,
    run: dict,
    progress: Progress,
    model,
    optimizer,
    scheduler,
    best_weights,
    device: torch.device,
) -> str:
    """Write an immutable checkpoint and publish its receipt last."""
    buffer = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "progress": asdict(progress),
            "best_weights": best_weights,
            "rng": _rng_state(device),
        },
        buffer,
    )
    payload = buffer.getvalue()
    receipt = (
        canonical_json(
            {
                "schema": RESUME_SCHEMA,
                "run": run,
                "cursor": progress.cursor(),
                "state_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        + b"\n"
    )
    path.mkdir(exist_ok=False)
    (path / "training.pt").write_bytes(payload)
    (path / "resume.json").write_bytes(receipt)
    return hashlib.sha256(receipt).hexdigest()


def read_resume(
    path: Path,
    *,
    receipt_sha256: str,
    run: dict,
    config: TrainingConfig,
    train_rows: int,
    dev_rows: int,
    model,
    optimizer,
    scheduler,
    device: torch.device,
) -> tuple[Progress, dict | None]:
    raw = (path / "resume.json").read_bytes()
    if not is_sha256(receipt_sha256) or hashlib.sha256(raw).hexdigest() != receipt_sha256:
        raise ValueError("resume receipt SHA-256 mismatch")
    receipt = parse_json(raw)
    if (
        set(receipt) != {"schema", "run", "cursor", "state_sha256"}
        or receipt["schema"] != RESUME_SCHEMA
        or not is_sha256(receipt["state_sha256"])
    ):
        raise ValueError("invalid resume receipt schema")
    if canonical_json(receipt["run"]) != canonical_json(run):
        raise ValueError("resume config, manifest, code or runtime identity disagrees")
    payload = (path / "training.pt").read_bytes()
    if hashlib.sha256(payload).hexdigest() != receipt["state_sha256"]:
        raise ValueError("resume training state SHA-256 mismatch")
    try:
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError("resume payload was rejected by the weights-only loader") from exc
    if not isinstance(state, dict) or set(state) != {
        "model",
        "optimizer",
        "scheduler",
        "progress",
        "best_weights",
        "rng",
    }:
        raise ValueError("invalid resume training state")
    try:
        progress = Progress(**state["progress"])
    except TypeError as exc:
        raise ValueError("invalid resume progress fields") from exc
    _validate_progress(progress, config, train_rows, dev_rows)
    if canonical_json(progress.cursor()) != canonical_json(receipt["cursor"]):
        raise ValueError("resume receipt cursor disagrees with the saved state")
    _validate_weights(state["model"], model)
    best = state["best_weights"]
    if progress.history:
        _validate_weights(best, model)
    elif best is not None:
        raise ValueError("resume state selected weights before completing a dev epoch")
    _validate_optimizer(
        state["optimizer"], optimizer, state["scheduler"], scheduler, model, progress
    )
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    # PyTorch's scheduler guard tracks step ordering on the live optimizer.
    # A checkpoint taken during the first dev epoch may resume before another update.
    if progress.updates:
        optimizer._opt_called = True
    _restore_rng(state["rng"], device)
    return progress, best
