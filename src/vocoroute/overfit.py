"""Train-only, one-utterance diagnostic for the existing spectral backbone."""

import csv
import hashlib
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch import Tensor

from data.audio_io import write_generator_pcm16

from .spectral import FRONTEND_V1
from .train_base import BaseTrainingModel, _graceful_stop, base_optimizer
from .training_config import SPECTRAL_FLOOR, STFT_RESOLUTIONS, TrainingConfig, canonical_json
from .training_data import GeneratorRow, TrainingAudio, prepare_example, sha256_file
from .training_device import accelerator_record, training_runtime
from .vctk_prepare import load_overfit_manifest, read_pinned_json, write_json

SNAPSHOT_STEPS = (0, 25, 100, 250, 500, 1000)
MAX_UPDATES = 1000
WORKER_TIME_LIMIT = 840
GPU_SESSION_LIMIT = 1080
PLANNED_GPU_SECONDS = 1200
SAMPLE_RATE = FRONTEND_V1.sample_rate
PARAMETERS = 566_144
PCM16_MAX = 32767 / 32768


@dataclass(frozen=True)
class CorpusPin:
    notebook: str
    notebook_id: int
    freeze_sha256: str
    selection_sha256: str
    pilot_sha256: str
    utterance: str
    native_sha256: str
    prepared_sha256: str
    samples: int

    def file_names(self) -> tuple[str, ...]:
        return (
            "freeze.json",
            "manifest.json",
            "parent-manifest.json",
            f"{self.utterance}_mic2.flac",
            f"{self.utterance}.wav",
        )


N2B1_CORPUS = CorpusPin(
    notebook="sambhavjainsj/vocoroute-vctk092-n2b0-train-dev/1",
    notebook_id=147338652,
    freeze_sha256="39ae136dbf240762a8c2abd38e39f8c56f9d246cd191c6f7dae014400808ec6b",
    selection_sha256="20601c92e22cb203c38dc1ed85d5827c8814e9dfdf46cc7310aab978056f9f5c",
    pilot_sha256="5ed731b8c47dcd0331f14ab1b5d877200fb6f7e09500fbb7507f94d998167d2f",
    utterance="p229_128",
    native_sha256="dee0b6e5e34c2682e9dad351edbe6988b231aa9c423f228c8b428b81cc1117d7",
    prepared_sha256="5648c74a7014cbc335417687b32d819209a2e0ee77b85b90687ce66b771d10f8",
    samples=76_523,
)


def load_utterance(inputs: Path, mount: Path, pin: CorpusPin) -> tuple[Tensor, Path, GeneratorRow]:
    """Allow only the five selected files across the attached input tree."""
    inputs = inputs.resolve(strict=True)
    mount = mount.resolve(strict=True)
    if not inputs.is_dir() or not mount.is_dir() or not mount.is_relative_to(inputs):
        raise ValueError("invalid diagnostic data mount")
    expected_names = set(pin.file_names())
    files = {}
    for path in inputs.rglob("*"):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ValueError(f"non-regular diagnostic input path: {path}")
        if path.is_file():
            if (
                not path.is_relative_to(mount)
                or path.name not in expected_names
                or path.name in files
            ):
                raise ValueError(f"unexpected, duplicate or extra mounted diagnostic input: {path}")
            files[path.name] = path
    if files.keys() != expected_names:
        raise ValueError(f"missing diagnostic input files: {sorted(expected_names - files.keys())}")

    freeze = read_pinned_json(files["freeze.json"], pin.freeze_sha256)
    if (
        freeze.get("schema") != "vocoroute-vctk092-freeze-v1"
        or freeze.get("protocol", {}).get("microphone") != "mic2"
        or freeze.get("manifests", {}).get("overfit") != pin.selection_sha256
        or freeze.get("manifests", {}).get("pilot") != pin.pilot_sha256
        or files["parent-manifest.json"].parent != files["manifest.json"].parent
    ):
        raise ValueError("corpus freeze does not match the diagnostic and pilot identities")
    diagnostic = load_overfit_manifest(files["manifest.json"].parent, pin.selection_sha256)
    if diagnostic.sha256 != pin.pilot_sha256 or len(diagnostic.rows) != 8:
        raise ValueError("expected the frozen eight-utterance TRAIN selection")
    row = next((row for row in diagnostic.rows if row.utterance_id == pin.utterance), None)
    if row is None or (row.role, row.native_sha256, row.prepared_sha256, row.prepared_samples) != (
        "train",
        pin.native_sha256,
        pin.prepared_sha256,
        pin.samples,
    ):
        raise ValueError("fixed TRAIN utterance identities do not match")

    names = (Path(row.native_path).name, Path(row.prepared_path).name)
    if names != pin.file_names()[-2:]:
        raise ValueError("the frozen recording paths do not identify the approved mic2 files")
    native, prepared = (files[name] for name in names)
    mounted_row = replace(
        row,
        native_path=native.relative_to(mount).as_posix(),
        prepared_path=prepared.relative_to(mount).as_posix(),
    )
    audio = TrainingAudio(replace(diagnostic, rows=(mounted_row,)), mount)
    audio.validate_headers()
    return audio.waveform(mounted_row), prepared, row


def diagnostic_config(
    *,
    device: str = "cuda",
    steps: tuple[int, ...] = SNAPSHOT_STEPS,
    wall_seconds: int = WORKER_TIME_LIMIT,
) -> dict:
    base = TrainingConfig().document()
    return {
        "schema": "vocoroute-single-utterance-v1",
        "role": "train_only",
        "model": base["model"],
        "frontend": base["frontend"],
        "stft_loss": base["stft_loss"],
        "loss": base["loss"],
        "optimizer": base["optimizer"],
        "scheduler": "constant_for_train_only_diagnostic",
        "seed": base["seed"],
        "batch_size": 1,
        "dtype": "float32",
        "device": device,
        "snapshot_steps": list(steps),
        "max_updates": steps[-1],
        "worker_training_deadline_seconds": wall_seconds,
    }


def _terms(terms) -> dict[str, float]:
    values = torch.stack(terms).detach().cpu().tolist()
    if not np.isfinite(values).all():
        raise ValueError("nonfinite reconstruction loss component")
    return dict(zip(terms._fields, values, strict=True))


def spectral_convergence(model: BaseTrainingModel, predicted: Tensor, target: Tensor) -> float:
    values = []
    for fft, hop in STFT_RESOLUTIONS:
        magnitude = torch.stft(
            torch.cat((predicted, target)),
            n_fft=fft,
            hop_length=hop,
            window=getattr(model.loss, f"hann_{fft}").to(predicted),
            center=True,
            pad_mode="constant",
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()
        pred_mag, ref_mag = magnitude.chunk(2)
        values.append(
            torch.linalg.vector_norm(pred_mag - ref_mag)
            / torch.linalg.vector_norm(ref_mag).clamp_min(SPECTRAL_FLOOR)
        )
    return float(torch.stack(values).mean())


def _record_waveform(path: Path, waveform: Tensor) -> dict:
    raw = waveform.detach().cpu().numpy().reshape(-1)
    raw_path = path.with_suffix(".npy")
    np.save(raw_path, raw, allow_pickle=False)
    finite = np.isfinite(raw)
    clips = (raw[finite] < -1.0) | (raw[finite] > PCM16_MAX)
    stats = {
        "samples": raw.size,
        "sample_rate": SAMPLE_RATE,
        "raw": raw_path.name,
        "raw_sha256": sha256_file(raw_path),
        "wav": None,
        "wav_sha256": None,
        "nonfinite_samples": int((~finite).sum()),
        "clipped_samples_for_pcm16": int(clips.sum()),
        "rms": None,
        "peak_abs": None,
        "std": None,
    }
    if finite.all():
        raw64 = raw.astype(np.float64)
        stats.update(
            rms=float(np.sqrt(np.mean(raw64**2))),
            peak_abs=float(np.max(np.abs(raw64))),
            std=float(raw64.std()),
        )
        with path.open("xb") as stream:
            write_generator_pcm16(stream, np.clip(raw, -1.0, PCM16_MAX), SAMPLE_RATE)
        stats.update(wav=path.name, wav_sha256=sha256_file(path))
    return stats


@torch.no_grad()
def _snapshot(model: BaseTrainingModel, example, step: int, output: Path) -> dict:
    predicted = model.synthesize(example)
    if predicted.shape != example.target.shape or predicted.dtype != torch.float32:
        raise ValueError("diagnostic synthesis changed the pinned shape or FP32 contract")
    name = f"step-{step:04d}"
    row = {"step": step, **_record_waveform(output / f"{name}.wav", predicted)}
    if not row["nonfinite_samples"]:
        try:
            row.update(_terms(model.loss(predicted, example.target)))
            row.update(
                waveform_mae=row["waveform"],
                log_magnitude_stft_error=row["log_magnitude"],
                spectral_convergence=spectral_convergence(model, predicted, example.target),
            )
            if not np.isfinite(row["spectral_convergence"]):
                raise ValueError("nonfinite spectral convergence")
        except ValueError as exc:
            row.pop("spectral_convergence", None)
            row["error"] = str(exc)
    else:
        row["error"] = "nonfinite waveform samples"
    write_json(output / f"{name}.json", row)
    if "error" in row:
        raise ValueError(f"{name}: {row['error']}")
    return row


def _runtime(device: torch.device) -> dict:
    visible = []
    if device.type == "cuda":
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            visible.append(
                {
                    "index": index,
                    "name": prop.name,
                    "memory_bytes": prop.total_memory,
                    "capability": [prop.major, prop.minor],
                }
            )
        if (
            device.index != 0
            or len(visible) != 2
            or any(gpu["name"] != "Tesla T4" or gpu["capability"] != [7, 5] for gpu in visible)
        ):
            raise RuntimeError(f"expected T4 x2, using only cuda:0: {visible}")
    smi = shutil.which("nvidia-smi") if visible else None
    driver = None
    if smi:
        query = subprocess.run(
            [smi, "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if query.returncode == 0:
            driver = query.stdout.splitlines()
    return {
        "device": str(device),
        "accelerator": accelerator_record(device),
        "visible_gpus": visible,
        "driver": driver,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torchaudio": version("torchaudio"),
        "soundfile": version("soundfile"),
        "numpy": np.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _step(model: BaseTrainingModel, example, optimizer) -> tuple[dict, float]:
    optimizer.zero_grad(set_to_none=True)
    prediction = model.synthesize(example)
    if not torch.isfinite(prediction).all():
        raise ValueError("nonfinite training forward waveform")
    terms = model.loss(prediction, example.target)
    values = _terms(terms)
    terms.total.backward()
    gradients = [param.grad for param in model.backbone.parameters()]
    if any(grad is None for grad in gradients):
        raise ValueError("missing backbone gradient")
    joined = torch.cat([grad.detach().reshape(-1) for grad in gradients])
    norm = float(torch.linalg.vector_norm(joined))
    if not np.isfinite(norm):
        raise ValueError("nonfinite backbone gradient or gradient norm")
    optimizer.step()
    if not torch.stack([p.detach().isfinite().all() for p in model.backbone.parameters()]).all():
        raise ValueError("nonfinite backbone parameter after AdamW")
    return values, norm


def _checkpoint(output: Path, model: BaseTrainingModel, initial: Tensor, updates: int) -> dict:
    weights = {name: value.detach().cpu() for name, value in model.backbone.state_dict().items()}
    with (output / "backbone.pt").open("xb") as stream:
        torch.save(weights, stream)
    final = torch.cat([p.detach().cpu().reshape(-1) for p in model.backbone.parameters()])
    delta = final - initial
    return {
        "file": "backbone.pt",
        "schema": "spectral-backbone-weights-only",
        "step": updates,
        "sha256": sha256_file(output / "backbone.pt"),
        "initial_flat_fp32_sha256": hashlib.sha256(
            initial.numpy().astype("<f4", copy=False).tobytes()
        ).hexdigest(),
        "final_flat_fp32_sha256": hashlib.sha256(
            final.numpy().astype("<f4", copy=False).tobytes()
        ).hexdigest(),
        "changed_parameters": int(torch.count_nonzero(delta)),
        "delta_l2": float(torch.linalg.vector_norm(delta)),
        "delta_max_abs": float(delta.abs().max()),
    }


def _fit(
    model: BaseTrainingModel,
    base: TrainingConfig,
    waveform: Tensor,
    output: Path,
    steps: tuple[int, ...],
    deadline: float,
    receipt: dict,
) -> None:
    device = model.frontend.window.device
    with torch.no_grad():
        example = prepare_example(model.frontend, waveform.to(device=device, dtype=torch.float32))
    initial = torch.cat([p.detach().cpu().reshape(-1) for p in model.backbone.parameters()])
    if initial.numel() != PARAMETERS:
        raise ValueError("diagnostic requires the existing 566144-parameter spectral backbone")
    optimizer = base_optimizer(model, base)
    receipt["parameter_count"] = initial.numel()
    receipt["snapshots"].append(_snapshot(model.eval(), example, 0, output))
    receipt["step_zero_elapsed_seconds"] = monotonic() - receipt["worker_started_monotonic"]
    columns = [
        "step",
        "total",
        "waveform",
        "magnitude",
        "log_magnitude",
        "phase_time",
        "phase_frequency",
        "gradient_l2",
        "seconds",
    ]
    stop_reason = "max_updates"
    update_started = monotonic()
    with (
        (output / "train-loss.csv").open("x", encoding="utf-8", newline="") as stream,
        _graceful_stop() as stopping,
    ):
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for step in range(1, steps[-1] + 1):
            if stopping() or monotonic() >= deadline:
                stop_reason = "signal" if stopping() else "wall_time"
                break
            began = monotonic()
            values, grad = _step(model.train(), example, optimizer)
            writer.writerow(
                {"step": step, **values, "gradient_l2": grad, "seconds": monotonic() - began}
            )
            stream.flush()
            receipt["updates"] = step
            receipt["finite_gradient_updates"] = step
            if step in steps:
                receipt["snapshots"].append(_snapshot(model.eval(), example, step, output))
    receipt["fit_and_snapshot_seconds"] = monotonic() - update_started
    updates = receipt["updates"]
    if updates != receipt["snapshots"][-1]["step"]:
        receipt["snapshots"].append(_snapshot(model.eval(), example, updates, output))
    receipt["checkpoint"] = _checkpoint(output, model, initial, updates)
    first, last = receipt["snapshots"][0], receipt["snapshots"][-1]
    receipt["comparisons"] = {
        "train_total_decreased": last["total"] < first["total"],
        "waveform_mae_decreased": last["waveform_mae"] < first["waveform_mae"],
        "nontrivial_final_waveform": last["std"] > 1 / 32768,
        "parameters_changed": receipt["checkpoint"]["changed_parameters"] > 0,
    }
    receipt.update(
        status="completed" if stop_reason == "max_updates" else "budget_limited",
        stop_reason=stop_reason,
    )
    if device.type == "cuda":
        receipt["gpu_memory"] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }


def run_overfit(
    inputs: Path,
    mount: Path,
    output: Path,
    *,
    pin: CorpusPin = N2B1_CORPUS,
    device: str = "cuda",
    steps: tuple[int, ...] = SNAPSHOT_STEPS,
    wall_seconds: int = WORKER_TIME_LIMIT,
    worker_started_at: float | None = None,
    code: dict | None = None,
) -> dict:
    """Run a bounded diagnostic using only an explicitly pinned TRAIN utterance."""
    start = monotonic() if worker_started_at is None else worker_started_at
    if (
        not steps
        or steps[0] != 0
        or len(steps) < 2
        or any(type(step) is not int for step in steps)
        or steps != tuple(sorted(set(steps)))
        or steps[-1] > MAX_UPDATES
        or type(wall_seconds) is not int
        or not 0 < wall_seconds <= WORKER_TIME_LIMIT
        or output.resolve().is_relative_to(inputs.resolve())
    ):
        raise ValueError("invalid single-utterance snapshot schedule, wall limit or output path")
    config = diagnostic_config(device=device, steps=steps, wall_seconds=wall_seconds)
    receipt = {
        "schema": "vocoroute-single-utterance-receipt-v1",
        "status": "running",
        "corpus": asdict(pin),
        "code": code,
        "config": config,
        "config_sha256": hashlib.sha256(canonical_json(config)).hexdigest(),
        "updates": 0,
        "finite_gradient_updates": 0,
        "snapshots": [],
        "worker_started_monotonic": start,
    }
    output.mkdir(exist_ok=False)
    try:
        with (
            training_runtime(device) as target,
            torch.autocast(device_type=target.type, enabled=False),
        ):
            receipt["runtime"] = _runtime(target)
            if target.type == "cuda":
                torch.cuda.reset_peak_memory_stats(target)
            waveform, reference, row = load_utterance(inputs, mount, pin)
            receipt["frozen_row"] = asdict(row)
            shutil.copyfile(reference, output / "reference.wav")
            if sha256_file(output / "reference.wav") != pin.prepared_sha256:
                raise ValueError("reference bytes changed after audio validation")
            receipt["reference"] = {
                "file": "reference.wav",
                "sha256": pin.prepared_sha256,
                "samples": pin.samples,
                "sample_rate": SAMPLE_RATE,
            }
            base = TrainingConfig()
            model = BaseTrainingModel(base).to(target)
            _fit(model, base, waveform, output, steps, start + wall_seconds, receipt)
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        del receipt["worker_started_monotonic"]
        receipt["elapsed_seconds"] = monotonic() - start
        receipt["artifacts_sha256"] = {p.name: sha256_file(p) for p in sorted(output.iterdir())}
        write_json(output / "diagnostic-receipt.json", receipt)
    return receipt
