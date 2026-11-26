"""Matched, independently initialized train-only phase-loss ablation."""

import csv
import hashlib
import random
import shutil
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import monotonic

import torch
from torch import Tensor

from .losses import ReconstructionLoss
from .overfit import (
    N2B1_CORPUS,
    PARAMETERS,
    CorpusPin,
    _checkpoint,
    _record_waveform,
    _runtime,
    _step,
    diagnostic_config,
    load_utterance,
)
from .phase_diagnostic import phase_evaluation_loss, reconstruction_metrics
from .train_base import BaseTrainingModel, _graceful_stop, base_optimizer
from .training_config import TrainingConfig, canonical_json
from .training_data import TrainingExample, prepare_example, sha256_file
from .training_device import training_runtime
from .vctk_prepare import write_json

SNAPSHOT_STEPS = (0, 100, 250, 500, 1000, 2000, 4000)
WORKER_SECONDS = 660
SESSION_SECONDS = 780
PLANNED_GPU_SECONDS = 900
MIN_PAIR_MARGIN_SECONDS = 5.0
PHASE_WEIGHT = 0.05


def _config(phase_enabled: bool) -> TrainingConfig:
    return TrainingConfig(
        loss=replace(
            TrainingConfig().loss,
            phase_enabled=phase_enabled,
            phase_time=PHASE_WEIGHT,
            phase_frequency=PHASE_WEIGHT,
        )
    )


def phase_protocol() -> dict:
    common = diagnostic_config(steps=SNAPSHOT_STEPS, wall_seconds=WORKER_SECONDS)
    return {
        **common,
        "schema": "vocoroute-n2b1p-phase-ablation-v1",
        "loss": {
            arm: asdict(_config(enabled).loss) for arm, enabled in (("A", False), ("B", True))
        },
        "armed_training": "lockstep matched pairs with independent optimizers",
        "admission_min_pair_margin_seconds": MIN_PAIR_MARGIN_SECONDS,
        "session_timeout_seconds": SESSION_SECONDS,
        "planned_gpu_seconds": PLANNED_GPU_SECONDS,
        "listening_seed": common["seed"],
    }


def _flat(model: BaseTrainingModel) -> Tensor:
    return torch.cat([p.detach().cpu().reshape(-1) for p in model.backbone.parameters()])


@dataclass
class _Arm:
    path: Path
    model: BaseTrainingModel
    optimizer: torch.optim.AdamW
    initial: Tensor
    snapshots: list[dict] = field(default_factory=list)
    train_seconds: float = 0.0
    peak_allocated: int = 0
    peak_reserved: int = 0
    peak_incremental: int = 0

    def update(self, example: TrainingExample) -> dict:
        device = example.target.device
        if device.type == "cuda":
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        began = monotonic()
        values, grad = _step(self.model.train(), example, self.optimizer)
        elapsed = monotonic() - began
        self.train_seconds += elapsed
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            self.peak_allocated = max(self.peak_allocated, peak)
            self.peak_incremental = max(self.peak_incremental, peak - baseline)
            self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved(device))
        return {
            "optimization_total": values["total"],
            "common_reconstruction": values["waveform"]
            + values["magnitude"]
            + values["log_magnitude"],
            "waveform_mae": values["waveform"],
            "linear_magnitude_mae": values["magnitude"],
            "log_magnitude_mae": values["log_magnitude"],
            "optimization_phase_time": values["phase_time"],
            "optimization_phase_frequency": values["phase_frequency"],
            "gradient_l2": grad,
            "seconds": elapsed,
        }

    @torch.no_grad()
    def snapshot(self, example: TrainingExample, evaluation: ReconstructionLoss, step: int) -> None:
        waveform = self.model.eval().synthesize(example)
        if waveform.shape != example.target.shape or waveform.dtype != torch.float32:
            raise ValueError("ablation snapshot violated waveform geometry or FP32")
        stem = f"step-{step:04d}"
        stats = _record_waveform(self.path / f"{stem}.wav", waveform)
        if stats["nonfinite_samples"]:
            raise ValueError(f"nonfinite {self.path.name} snapshot at step {step}")
        values = reconstruction_metrics(self.model, evaluation, waveform, example.target)
        row = {
            "step": step,
            **stats,
            **values,
            "optimization_total": float(self.model.loss(waveform, example.target).total),
        }
        write_json(self.path / f"{stem}.json", row)
        self.snapshots.append(row)

    def finish(self, step: int) -> dict:
        return {
            "snapshots": self.snapshots,
            "checkpoint": _checkpoint(self.path, self.model, self.initial, step),
            "update_seconds": self.train_seconds,
            "update_peak_gpu_memory": {
                "process_allocated_bytes": self.peak_allocated,
                "process_reserved_bytes": self.peak_reserved,
                "incremental_allocated_bytes": self.peak_incremental,
            },
        }


def _listen(output: Path, arms: dict[str, _Arm], step: int, seed: int) -> None:
    candidates = [("reference", output / "reference.wav")] + [
        (arm, obj.path / obj.snapshots[-1]["wav"]) for arm, obj in arms.items()
    ]
    random.Random(seed).shuffle(candidates)
    blind = output / "blinded"
    blind.mkdir()
    key = {}
    for i, (condition, source) in enumerate(candidates, start=1):
        name = f"clip-{i:02d}.wav"
        shutil.copyfile(source, blind / name)
        key[name] = {
            "condition": condition,
            "matched_step": step if condition != "reference" else None,
            "sha256": sha256_file(blind / name),
        }
    write_json(output / "listening-key.json", key)


def _baseline_comparison(runtime: dict, original: dict, snapshots: list[dict]) -> dict:
    keys = runtime.keys() | original["runtime"].keys()
    differences = sorted(key for key in keys if runtime.get(key) != original["runtime"].get(key))
    comparison = {"runtime_matches": not differences, "runtime_differences": differences}
    at_1000 = next((row for row in snapshots if row["step"] == 1000), None)
    if at_1000 is not None:
        comparison["run_a_step_1000"] = {
            "raw_sha256_matches": at_1000["raw_sha256"] == original["step_1000_raw_sha256"],
            "common_reconstruction_delta": (
                at_1000["common_reconstruction"] - original["step_1000_common_reconstruction"]
            ),
        }
    return comparison


def run_phase_ablation(
    inputs: Path,
    mount: Path,
    output: Path,
    *,
    original: dict,
    code: dict,
    pin: CorpusPin = N2B1_CORPUS,
    device: str = "cuda",
    steps: tuple[int, ...] = SNAPSHOT_STEPS,
    deadline_seconds: float = WORKER_SECONDS,
    started_at: float | None = None,
) -> dict:
    """Fit a pair and admit each pair atomically until a common stopping step."""
    start = monotonic() if started_at is None else started_at
    if (
        output.resolve().is_relative_to(inputs.resolve())
        or len(steps) < 2
        or steps[0] != 0
        or tuple(sorted(set(steps))) != steps
        or not 0 < deadline_seconds <= WORKER_SECONDS
        or (
            device == "cuda"
            and (
                pin != N2B1_CORPUS or steps != SNAPSHOT_STEPS or deadline_seconds != WORKER_SECONDS
            )
        )
    ):
        raise ValueError("unexpected ablation inputs, schedule, deadline or output directory")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "vocoroute-n2b1p-phase-ablation-receipt-v1",
        "status": "running",
        "code": code,
        "protocol": phase_protocol(),
        "corpus": asdict(pin),
        "baseline": original,
        "matched_updates": 0,
        "executed_snapshot_steps": list(steps),
    }
    receipt["protocol_sha256"] = hashlib.sha256(canonical_json(receipt["protocol"])).hexdigest()
    try:
        with (
            training_runtime(device) as target,
            torch.autocast(device_type=target.type, enabled=False),
        ):
            receipt["runtime"] = _runtime(target)
            waveform, ref_path, frozen_row = load_utterance(inputs, mount, pin)
            receipt["frozen_row"] = asdict(frozen_row)
            shutil.copyfile(ref_path, output / "reference.wav")
            if sha256_file(output / "reference.wav") != pin.prepared_sha256:
                raise ValueError("reference WAV changed after validation")
            _fit_pair(output, waveform.to(target), steps, start + deadline_seconds, receipt)
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt["worker_seconds"] = monotonic() - start
        receipt["artifacts_sha256"] = {
            p.relative_to(output).as_posix(): sha256_file(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        }
        write_json(output / "ablation-receipt.json", receipt)
    return receipt


def _fit_pair(
    output: Path, waveform: Tensor, steps: tuple[int, ...], deadline: float, receipt: dict
):
    arms = {}
    for label, enabled in (("A", False), ("B", True)):
        path = output / label
        path.mkdir()
        config = _config(enabled)
        model = BaseTrainingModel(config).to(waveform.device)
        arms[label] = _Arm(path, model, base_optimizer(model, config), _flat(model))
    initial = arms["A"].initial
    if initial.numel() != PARAMETERS or not torch.equal(initial, arms["B"].initial):
        raise ValueError("arms did not start from identical pinned-backbone parameters")
    original, runtime = receipt["baseline"], receipt["runtime"]
    initial_sha = hashlib.sha256(initial.numpy().astype("<f4", copy=False).tobytes()).hexdigest()
    receipt["initial_fp32_sha256"] = initial_sha
    receipt["baseline_initial_hash_matches"] = initial_sha == original["initial_fp32_sha256"]
    if runtime == original["runtime"] and not receipt["baseline_initial_hash_matches"]:
        raise ValueError("original runtime matched but the original initial weights did not")
    evaluation = phase_evaluation_loss().to(waveform.device)
    example = prepare_example(arms["A"].model.frontend, waveform)
    for arm in arms.values():
        arm.snapshot(example, evaluation, 0)
    if arms["A"].snapshots[0]["raw_sha256"] != arms["B"].snapshots[0]["raw_sha256"]:
        raise ValueError("step-zero matched-arm waveforms differ")
    columns = [
        "step",
        "optimization_total",
        "common_reconstruction",
        "waveform_mae",
        "linear_magnitude_mae",
        "log_magnitude_mae",
        "optimization_phase_time",
        "optimization_phase_frequency",
        "gradient_l2",
        "seconds",
    ]
    reason = "max_updates"
    longest_pair = 0.0
    began = monotonic()
    with ExitStack() as stack:
        streams = {
            key: stack.enter_context(
                (arm.path / "train-loss.csv").open("x", newline="", encoding="utf-8")
            )
            for key, arm in arms.items()
        }
        writers = {
            key: csv.DictWriter(stream, fieldnames=columns) for key, stream in streams.items()
        }
        for writer in writers.values():
            writer.writeheader()
        stopping = stack.enter_context(_graceful_stop())
        for step in range(1, steps[-1] + 1):
            wait = max(MIN_PAIR_MARGIN_SECONDS, 2 * longest_pair)
            if stopping() or monotonic() + wait >= deadline:
                reason = "signal" if stopping() else "wall_time"
                break
            pair_start = monotonic()
            pair = {key: arm.update(example) for key, arm in arms.items()}
            longest_pair = max(longest_pair, monotonic() - pair_start)
            for key, values in pair.items():
                writers[key].writerow({"step": step, **values})
                streams[key].flush()
            receipt["matched_updates"] = step
            if step in steps:
                for arm in arms.values():
                    arm.snapshot(example, evaluation, step)
    matched = receipt["matched_updates"]
    receipt["fit_and_snapshot_seconds"] = monotonic() - began
    for arm in arms.values():
        if arm.snapshots[-1]["step"] != matched:
            arm.snapshot(example, evaluation, matched)
    receipt["arms"] = {key: arm.finish(matched) for key, arm in arms.items()}
    receipt["baseline_comparison"] = _baseline_comparison(runtime, original, arms["A"].snapshots)
    _listen(output, arms, matched, receipt["protocol"]["listening_seed"])
    receipt.update(
        status="completed" if reason == "max_updates" else "budget_limited", stop_reason=reason
    )
