"""Balanced eight-utterance phase-aware diagnostic on frozen pilot TRAIN audio."""

import csv
import hashlib
import random
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean, median
from time import monotonic

import numpy as np
import torch

from data.audio_io import probe

from .overfit import (
    N2B1_CORPUS,
    PARAMETERS,
    _checkpoint,
    _record_waveform,
    _runtime,
    _step,
    diagnostic_config,
)
from .phase_ablation import _config, _flat
from .phase_diagnostic import reconstruction_metrics
from .spectral import FRONTEND_V1, frames_for_length
from .train_base import BaseTrainingModel, _graceful_stop, base_optimizer
from .training_config import TRAIN_ROLE, canonical_json
from .training_data import GeneratorManifest, TrainingAudio, prepare_example, sha256_file
from .training_device import training_runtime
from .vctk_prepare import (
    FREEZE_SCHEMA,
    NATIVE_RATE,
    RESAMPLE_PROFILE,
    load_overfit_manifest,
    read_pinned_json,
    write_json,
)

ROUND_SIZE = 8
SNAPSHOT_STEPS = (0, 800, 4000, 8000, 16000)
WORKER_SECONDS = 960
SESSION_SECONDS = 1080
PLANNED_GPU_SECONDS = 1200
ROUND_MARGIN_SECONDS = 5.0
RELOAD_ATOL, RELOAD_RTOL = 2e-6, 1e-5
ORDER_DOMAIN = "vocoroute-n2b2-round-v1"
METADATA_PATHS = {
    "freeze.json": "freeze.json",
    "manifest.json": "overfit/manifest.json",
    "parent-manifest.json": "overfit/parent-manifest.json",
    "resampling-receipt.json": "overfit/source/resampling-receipt.json",
    "license_text.txt": "overfit/source/license_text.txt",
}
SUMMARY_METRICS = (
    "common_reconstruction",
    "optimization_total",
    "waveform_mae",
    "linear_magnitude_mae",
    "log_magnitude_mae",
    "spectral_convergence",
    "circular_phase_error",
    "phase_time",
    "phase_frequency",
    "rms",
    "peak_abs",
)
TRACE_COLUMNS = (
    "step",
    "round",
    "utterance_id",
    "speaker_id",
    "samples",
    "common_reconstruction",
    "total",
    "waveform",
    "magnitude",
    "log_magnitude",
    "phase_time",
    "phase_frequency",
    "gradient_l2",
    "seconds",
)


@dataclass(frozen=True)
class CorpusPin:
    notebook: str = N2B1_CORPUS.notebook
    notebook_id: int = N2B1_CORPUS.notebook_id
    freeze_sha256: str = N2B1_CORPUS.freeze_sha256
    selection_sha256: str = N2B1_CORPUS.selection_sha256
    pilot_sha256: str = N2B1_CORPUS.pilot_sha256
    resampling_sha256: str = "06c316bf2323589e670628ff15b766a7992f72bc90b2cdbfef521c9009f61d6f"
    license_sha256: str = "b34e17103bfb246f2549fc82a279e6ba28834e0cb42f76a92efc14b72e3a3723"


N2B2_CORPUS = CorpusPin()


def _metadata_pins(pin: CorpusPin) -> dict[str, str]:
    return {
        "freeze.json": pin.freeze_sha256,
        "manifest.json": pin.selection_sha256,
        "parent-manifest.json": pin.pilot_sha256,
        "resampling-receipt.json": pin.resampling_sha256,
        "license_text.txt": pin.license_sha256,
    }


def _read_selection(files: dict[str, Path], pin: CorpusPin) -> GeneratorManifest:
    freeze = read_pinned_json(files["freeze.json"], pin.freeze_sha256)
    resampling = read_pinned_json(files["resampling-receipt.json"], pin.resampling_sha256)
    if (
        freeze.get("schema") != FREEZE_SCHEMA
        or freeze.get("protocol", {}).get("microphone") != "mic2"
        or freeze.get("manifests", {}).get("overfit") != pin.selection_sha256
        or freeze.get("manifests", {}).get("pilot") != pin.pilot_sha256
        or freeze.get("resampling_receipt_sha256") != pin.resampling_sha256
        or resampling.get("schema") != RESAMPLE_PROFILE
        or sha256_file(files["license_text.txt"]) != pin.license_sha256
        or files["parent-manifest.json"].parent != files["manifest.json"].parent
    ):
        raise ValueError("freeze, resampling, parent manifest or license provenance differs")
    selection = load_overfit_manifest(files["manifest.json"].parent, pin.selection_sha256)
    rows = selection.rows
    if (
        selection.sha256 != pin.pilot_sha256
        or len(rows) != ROUND_SIZE
        or len({row.speaker_id.casefold() for row in rows}) != ROUND_SIZE
        or len({row.source_group.casefold() for row in rows}) != ROUND_SIZE
    ):
        raise ValueError("expected eight distinct frozen pilot TRAIN speakers and sources")
    for row in rows:
        uid = row.utterance_id
        if (
            row.role != TRAIN_ROLE
            or row.source_group != f"vctk-0.92/{uid}"
            or not re.fullmatch(r"p[0-9]{3}_[0-9]{3}", uid)
            or uid.split("_", 1)[0] != row.speaker_id
            or row.native_rate != NATIVE_RATE
            or row.resample_steps != 1
            or row.resampling_profile != RESAMPLE_PROFILE
            or row.resampling_receipt_sha256 != pin.resampling_sha256
            or row.native_path != f"native/train/{row.speaker_id}/{uid}_mic2.flac"
            or row.prepared_path != f"prepared/train/{row.speaker_id}/{uid}.wav"
        ):
            raise ValueError(f"unexpected frozen TRAIN recording provenance: {uid}")
    return selection


def _audio_pins(selection: GeneratorManifest) -> dict[str, str]:
    return {
        relative: digest
        for row in selection.rows
        for relative, digest in (
            (row.native_path, row.native_sha256),
            (row.prepared_path, row.prepared_sha256),
        )
    }


def source_inventory(source: Path, pin: CorpusPin = N2B2_CORPUS) -> dict:
    """Build the exact saved-output allowlist using only recovered N2B.0 metadata."""
    selection = _read_selection({name: source / path for name, path in METADATA_PATHS.items()}, pin)
    return _describe_inventory(selection, pin)


def _describe_inventory(selection: GeneratorManifest, pin: CorpusPin) -> dict:
    paths = {METADATA_PATHS[name]: digest for name, digest in _metadata_pins(pin).items()}
    paths.update({f"overfit/{path}": digest for path, digest in _audio_pins(selection).items()})
    return {
        "schema": "vocoroute-n2b2-input-inventory-v1",
        "notebook": pin.notebook,
        "notebook_id": pin.notebook_id,
        "files": {
            path: {"source_path": f"n2b0-train-dev/{path}", "sha256": digest}
            for path, digest in paths.items()
        },
        "recordings": [
            {
                "utterance_id": r.utterance_id,
                "speaker_id": r.speaker_id,
                "native_samples": r.native_samples,
                "prepared_samples": r.prepared_samples,
            }
            for r in selection.rows
        ],
    }


def load_diagnostic(inputs: Path, mount: Path, pin: CorpusPin = N2B2_CORPUS, *, inventory=None):
    """Refuse any file outside the exact selection before opening audio."""
    if mount.is_symlink():
        raise ValueError("diagnostic mount must be a regular directory")
    inputs, mount = inputs.resolve(strict=True), mount.resolve(strict=True)
    if not mount.is_dir() or not mount.is_relative_to(inputs):
        raise ValueError("invalid diagnostic input mount")
    files = {}
    for path in inputs.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError(f"non-regular diagnostic input: {path}")
        if path.is_file():
            if not path.is_relative_to(mount) or path.name in files:
                raise ValueError(f"unexpected or duplicate diagnostic input: {path}")
            files[path.name] = path
    if _metadata_pins(pin).keys() - files.keys():
        raise ValueError("missing frozen diagnostic metadata")
    selection = _read_selection(files, pin)
    if inventory is not None and inventory != _describe_inventory(selection, pin):
        raise ValueError("mounted frozen selection differs from the RECORD plan inventory")
    expected = _metadata_pins(pin).keys() | {Path(path).name for path in _audio_pins(selection)}
    if len(expected) != 21 or files.keys() != expected:
        raise ValueError("expected only the 16 approved audio files and five provenance files")
    mounted_rows = tuple(
        replace(
            row,
            native_path=files[Path(row.native_path).name].relative_to(mount).as_posix(),
            prepared_path=files[Path(row.prepared_path).name].relative_to(mount).as_posix(),
        )
        for row in selection.rows
    )
    audio = TrainingAudio(replace(selection, rows=mounted_rows), mount)
    audio.validate_headers()
    for row in mounted_rows:
        for path, magic in ((row.native_path, b"fLaC"), (row.prepared_path, b"RIFF")):
            with (mount / path).open("rb") as stream:
                if stream.read(4) != magic:
                    raise ValueError(f"unexpected recording encoding: {path}")
                stream.seek(0)
                if probe(stream)["subtype"] != "PCM_16":
                    raise ValueError(f"unexpected recording bit depth: {path}")
    return audio, mounted_rows


def round_order(ids: tuple[str, ...], seed: int, index: int) -> tuple[str, ...]:
    """A stateless permutation using a zero-based round index."""

    def rank(uid):
        return hashlib.sha256(f"{ORDER_DOMAIN}:{seed}:{index}:{uid}".encode()).digest(), uid

    return tuple(sorted(ids, key=rank))


def protocol() -> dict:
    base = diagnostic_config(steps=SNAPSHOT_STEPS, wall_seconds=WORKER_SECONDS)
    return {
        **base,
        "schema": "vocoroute-n2b2-eight-overfit-v1",
        "loss": _config(True).document()["loss"],
        "initialization": "fresh_seeded",
        "parameters": PARAMETERS,
        "round_size": ROUND_SIZE,
        "order_domain": ORDER_DOMAIN,
        "order": "ascending SHA-256 rank and utterance ID, zero-based rounds",
        "round_margin_seconds": ROUND_MARGIN_SECONDS,
        "round_timing_multiplier": 2,
        "session_timeout_seconds": SESSION_SECONDS,
        "planned_gpu_seconds": PLANNED_GPU_SECONDS,
        "reload_tolerance": {"atol": RELOAD_ATOL, "rtol": RELOAD_RTOL},
    }


def _summarize(rows: list[dict]) -> dict:
    low, high = (
        sort(rows, key=lambda row: (row["common_reconstruction"], row["utterance_id"]))
        for sort in (min, max)
    )
    return {
        "macro_mean": {field: fmean(row[field] for row in rows) for field in SUMMARY_METRICS},
        "median": {field: median(row[field] for row in rows) for field in SUMMARY_METRICS},
        "best_by_common_loss": low,
        "worst_by_common_loss": high,
    }


@torch.no_grad()
def _snapshot(model, examples, rows, exposure, step, output):
    directory = output / "snapshots" / f"step-{step:05d}"
    directory.mkdir(parents=True)
    was_training = model.training
    model.eval()
    try:
        metrics = []
        for row in rows:
            uid = row.utterance_id
            example = examples[uid]
            predicted = model.synthesize(example)
            if predicted.shape != example.target.shape or predicted.dtype != torch.float32:
                raise ValueError(f"unexpected diagnostic waveform shape or dtype: {uid}")
            stats = _record_waveform(directory / f"{uid}.wav", predicted)
            if stats["nonfinite_samples"]:
                raise ValueError(f"nonfinite snapshot: {uid}")
            measures = reconstruction_metrics(model, model.loss, predicted, example.target)
            objective = float(model.loss(predicted, example.target).total)
            metrics.append(
                {
                    "utterance_id": uid,
                    "speaker_id": row.speaker_id,
                    "reference": f"reference/{uid}.wav",
                    "utterance_updates": exposure[uid],
                    **stats,
                    **measures,
                    "optimization_total": objective,
                }
            )
        snapshot = {"step": step, "summary": _summarize(metrics), "recordings": metrics}
        write_json(directory / "snapshot.json", snapshot)
        return snapshot
    finally:
        model.train(was_training)


@torch.no_grad()
def _reload(config, output, checkpoint, examples):
    model = BaseTrainingModel(config).to(next(iter(examples.values())).target.device).eval()
    file = output / checkpoint["file"]
    if sha256_file(file) != checkpoint["sha256"]:
        raise ValueError("checkpoint file hash changed before reload")
    model.backbone.load_state_dict(
        torch.load(file, map_location="cpu", weights_only=True), strict=True
    )
    flat = _flat(model).numpy().astype("<f4", copy=False).tobytes()
    if hashlib.sha256(flat).hexdigest() != checkpoint["final_flat_fp32_sha256"]:
        raise ValueError("checkpoint weight hash changed across reload")
    last = output / "snapshots" / f"step-{checkpoint['step']:05d}"
    report = {}
    for uid, example in examples.items():
        actual = model.synthesize(example).cpu().numpy().reshape(-1)
        expected = np.load(last / f"{uid}.npy", allow_pickle=False)
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, atol=RELOAD_ATOL, rtol=RELOAD_RTOL
        ):
            raise ValueError(f"checkpoint waveform replay failed: {uid}")
        report[uid] = {
            "max_abs": float(np.max(np.abs(actual - expected))),
            "mae": float(np.mean(np.abs(actual - expected))),
        }
    return report


def _listening_set(output, ids, step, seed):
    rng = random.Random(seed)
    order = list(ids)
    rng.shuffle(order)
    blind = output / "blinded"
    blind.mkdir()
    last = output / "snapshots" / f"step-{step:05d}"
    key = {}
    for pair, uid in enumerate(order, 1):
        variants = [
            ("reference", output / "reference" / f"{uid}.wav"),
            ("reconstruction", last / f"{uid}.wav"),
        ]
        rng.shuffle(variants)
        for label, (condition, source) in zip("ab", variants, strict=True):
            name = f"pair-{pair:02d}-{label}.wav"
            shutil.copyfile(source, blind / name)
            key[name] = {
                "utterance_id": uid,
                "condition": condition,
                "step": step if condition == "reconstruction" else None,
                "sha256": sha256_file(blind / name),
            }
    write_json(output / "listening-key.json", key)


def run_diagnostic(
    inputs,
    mount,
    output,
    *,
    code,
    pin=N2B2_CORPUS,
    device="cuda",
    steps=SNAPSHOT_STEPS,
    deadline_seconds=WORKER_SECONDS,
    started_at=None,
):
    start = monotonic() if started_at is None else started_at
    if (
        len(steps) < 2
        or steps[0] != 0
        or steps != tuple(sorted(set(steps)))
        or steps[-1] > SNAPSHOT_STEPS[-1]
        or any(type(n) is not int or n % ROUND_SIZE for n in steps)
        or not 0 < deadline_seconds <= WORKER_SECONDS
        or output.resolve().is_relative_to(inputs.resolve())
        or (
            device == "cuda"
            and (
                pin != N2B2_CORPUS or steps != SNAPSHOT_STEPS or deadline_seconds != WORKER_SECONDS
            )
        )
    ):
        raise ValueError("invalid diagnostic scope, balanced schedule or budget")
    output.mkdir(parents=True, exist_ok=False)
    settings = protocol()
    receipt = {
        "schema": "vocoroute-n2b2-overfit-receipt-v1",
        "status": "running",
        "code": code,
        "protocol": settings,
        "protocol_sha256": hashlib.sha256(canonical_json(settings)).hexdigest(),
        "updates": 0,
        "actual_device_request": device,
        "scheduled_steps": list(steps),
        "actual_deadline_seconds": deadline_seconds,
    }
    try:
        with (
            training_runtime(device) as target,
            torch.autocast(device_type=target.type, enabled=False),
        ):
            receipt["runtime"] = _runtime(target)
            if target.type == "cuda":
                torch.cuda.reset_peak_memory_stats(target)
            audio, rows = load_diagnostic(inputs, mount, pin, inventory=code.get("input_inventory"))
            _fit(output, audio, rows, target, steps, start + deadline_seconds, receipt)
            if "expected_full_order_sha256" in code and receipt["updates"] == SNAPSHOT_STEPS[-1]:
                if receipt["actual_order_sha256"] != code["expected_full_order_sha256"]:
                    raise ValueError("completed training order did not match the frozen schedule")
            if target.type == "cuda":
                receipt["gpu_memory"] = {
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(target),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(target),
                }
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt["worker_seconds"] = monotonic() - start
        receipt["artifacts_sha256"] = {
            path.relative_to(output).as_posix(): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        }
        write_json(output / "diagnostic-receipt.json", receipt)
    return receipt


def _fit(output, audio, rows, device, steps, deadline, receipt):
    config = _config(True)
    model = BaseTrainingModel(config).to(device)
    initial = _flat(model)
    if initial.numel() != PARAMETERS:
        raise ValueError("unexpected backbone parameter count")
    receipt["parameters"] = initial.numel()
    optimizer = base_optimizer(model, config)
    examples, exposures, by_speaker = {}, {}, {}
    references = output / "reference"
    references.mkdir()
    with torch.no_grad():
        for row in rows:
            uid = row.utterance_id
            waveform = audio.waveform(row).to(device)
            example = prepare_example(model.frontend, waveform)
            if (
                example.length != row.prepared_samples
                or not torch.equal(example.target[0], waveform)
                or example.mel.shape != (1, FRONTEND_V1.n_mels, frames_for_length(example.length))
                or not torch.isfinite(example.mel).all()
            ):
                raise ValueError(f"prepared waveform and mel alignment failed: {uid}")
            examples[uid] = example
            exposures[uid] = by_speaker[row.speaker_id] = 0
            reference = references / f"{uid}.wav"
            shutil.copyfile(audio.root / row.prepared_path, reference)
            if sha256_file(reference) != row.prepared_sha256:
                raise ValueError(f"reference WAV bytes changed: {uid}")
    ids = tuple(examples)
    speakers = {row.utterance_id: row.speaker_id for row in rows}
    receipt.update(
        utterance_exposure=exposures,
        speaker_exposure=by_speaker,
        training_ids=list(ids),
        snapshots=[],
    )
    longest, update_seconds, gradients = 0.0, 0.0, []
    order_digest = hashlib.sha256()
    reason = "max_updates"
    began = monotonic()
    with (
        (output / "train-loss.csv").open("x", newline="", encoding="utf-8") as stream,
        _graceful_stop() as stopping,
    ):
        writer = csv.DictWriter(stream, fieldnames=TRACE_COLUMNS)
        writer.writeheader()
        receipt["snapshots"].append(_snapshot(model, examples, rows, exposures, 0, output))
        for index in range(steps[-1] // ROUND_SIZE):
            if stopping() or monotonic() + max(ROUND_MARGIN_SECONDS, 2 * longest) >= deadline:
                reason = "signal" if stopping() else "wall_time"
                break
            round_start = monotonic()
            for uid in round_order(ids, config.seed, index):
                before = monotonic()
                losses, gradient = _step(model.train(), examples[uid], optimizer)
                elapsed = monotonic() - before
                receipt["updates"] += 1
                step, speaker = receipt["updates"], speakers[uid]
                exposures[uid] += 1
                by_speaker[speaker] += 1
                order_digest.update((uid + "\n").encode())
                update_seconds += elapsed
                gradients.append(gradient)
                writer.writerow(
                    {
                        "step": step,
                        "round": index + 1,
                        "utterance_id": uid,
                        "speaker_id": speaker,
                        "samples": examples[uid].length,
                        "common_reconstruction": sum(
                            losses[key] for key in ("waveform", "magnitude", "log_magnitude")
                        ),
                        **losses,
                        "gradient_l2": gradient,
                        "seconds": elapsed,
                    }
                )
            stream.flush()
            longest = max(longest, monotonic() - round_start)
            if step in steps:
                receipt["snapshots"].append(
                    _snapshot(model, examples, rows, exposures, step, output)
                )
    step = receipt["updates"]
    rounds = step // ROUND_SIZE
    if (
        step % ROUND_SIZE
        or set(exposures.values()) != {rounds}
        or set(by_speaker.values()) != {rounds}
    ):
        raise RuntimeError("stopping exposed an imbalanced training round")
    if receipt["snapshots"][-1]["step"] != step:
        receipt["snapshots"].append(_snapshot(model, examples, rows, exposures, step, output))
    receipt["fit_and_snapshot_seconds"] = monotonic() - began
    checkpoint = _checkpoint(output, model, initial, step)
    receipt.update(
        status="completed" if reason == "max_updates" else "budget_limited",
        stop_reason=reason,
        completed_rounds=rounds,
        actual_order_sha256=order_digest.hexdigest(),
        checkpoint=checkpoint,
        update_seconds=update_seconds,
        updates_per_second=step / update_seconds if step else None,
        gradient_l2={"min": min(gradients), "median": median(gradients), "max": max(gradients)}
        if gradients
        else None,
        reload=_reload(config, output, checkpoint, examples),
    )
    _listening_set(output, ids, step, config.seed)
