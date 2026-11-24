"""FP32 base-model training entry point with a read-only dry-run mode."""

import argparse
import json
import math
import random
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import Tensor, nn

from .losses import ReconstructionLoss, ReconstructionTerms
from .spectral import SpectralFrontend
from .training_checkpoint import (
    initialize_backbone,
    load_base_checkpoint,
    run_provenance,
    save_base_checkpoint,
)
from .training_config import (
    BUDGET_CONFIG_SCHEMA,
    DEV_ROLE,
    ROLES,
    TRAIN_ROLE,
    TrainingConfig,
    load_training_config,
)
from .training_data import GeneratorManifest, TrainingAudio, load_manifest, prepare_example
from .training_device import accelerator_record, training_runtime
from .training_resume import Progress, read_resume, write_resume


class BaseTrainingModel(nn.Module):
    """Run mel conditioning, base synthesis and waveform-domain objectives."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.frontend = SpectralFrontend()
        self.backbone = initialize_backbone(config.seed)
        self.loss = ReconstructionLoss(config.loss)

    def forward(self, waveform: Tensor) -> ReconstructionTerms:
        device = self.frontend.window.device
        example = prepare_example(self.frontend, waveform.to(device=device, dtype=torch.float32))
        spectrum = self.backbone(example.mel)
        synthesized = self.frontend.synthesize(spectrum, example.length)
        return self.loss(synthesized, example.target)


def _training_parts(config: TrainingConfig, device: torch.device):
    model = BaseTrainingModel(config).to(device)
    spec = config.document()["optimizer"]
    optimizer = torch.optim.AdamW(
        model.backbone.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=tuple(spec["betas"]),
        eps=spec["eps"],
        foreach=spec["foreach"],
        fused=spec["fused"],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.scheduler_gamma)
    return model, optimizer, scheduler


def _train_example(model, optimizer, waveform: Tensor, uid: str) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = model(waveform).total
    if not torch.isfinite(loss):
        raise ValueError(f"nonfinite training loss for {uid}")
    loss.backward()
    if any(
        param.grad is None or not torch.isfinite(param.grad).all()
        for param in model.backbone.parameters()
    ):
        raise ValueError(f"missing or nonfinite gradient for {uid}")
    optimizer.step()
    return loss.item()


def _run_training(
    config: TrainingConfig,
    manifest: GeneratorManifest,
    audio: TrainingAudio,
    output: Path,
) -> dict:
    """Retain the v1 CPU inference-only output and directory contract."""
    audio.validate_headers()
    model, optimizer, scheduler = _training_parts(config, torch.device("cpu"))
    provenance = run_provenance(config, manifest)
    history = []
    best_loss = math.inf
    best_epoch = 0
    best_weights = {}
    for epoch in range(1, config.epochs + 1):
        training = manifest.examples(TRAIN_ROLE, seed=config.seed, epoch=epoch)
        model.train()
        train_totals = [
            _train_example(model, optimizer, audio.waveform(row), row.utterance_id)
            for row in training
        ]

        model.eval()
        with torch.no_grad():
            dev_totals = [
                model(audio.waveform(row)).total.item() for row in manifest.examples(DEV_ROLE)
            ]
        if not all(map(math.isfinite, dev_totals)):
            raise ValueError("nonfinite dev loss")
        dev_loss = math.fsum(dev_totals) / len(dev_totals)
        history.append(
            {
                "epoch": epoch,
                "train_total": math.fsum(train_totals) / len(train_totals),
                "dev_total": dev_loss,
            }
        )
        if dev_loss < best_loss:
            best_loss, best_epoch = dev_loss, epoch
            best_weights = {
                key: tensor.detach().clone() for key, tensor in model.backbone.state_dict().items()
            }
        scheduler.step()

    digest = save_base_checkpoint(output, config, provenance, history, best_epoch, best_weights)
    return {"best_epoch": best_epoch, "checkpoint_receipt_sha256": digest, "output": str(output)}


@contextmanager
def _graceful_stop():
    requested = False

    def receive(_signum, _frame):
        nonlocal requested
        requested = True

    previous = {
        signum: signal.signal(signum, receive) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield lambda: requested
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _cuda_preflight(config: TrainingConfig, device: torch.device) -> None:
    if device.type == "cuda":
        model = BaseTrainingModel(config).to(device)
        waveform = 0.1 * torch.sin(torch.arange(1025, dtype=torch.float32) * 0.11)
        loss = model(waveform).total
        loss.backward()
        if not torch.isfinite(loss) or any(
            param.grad is None or not torch.isfinite(param.grad).all()
            for param in model.backbone.parameters()
        ):
            raise RuntimeError(
                "CUDA FP32 STFT/iSTFT preflight produced nonfinite loss or gradients"
            )
        torch.cuda.synchronize(device)


def _write_summary(output: Path, record: dict) -> None:
    temp = output / "run.json.partial"
    temp.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temp.replace(output / "run.json")


def _run_budgeted(
    config: TrainingConfig,
    manifest: GeneratorManifest,
    audio: TrainingAudio,
    output: Path,
    device: torch.device,
    *,
    resume: tuple[Path, str] | None,
    attempt_updates: int | None,
) -> dict:
    start = time.monotonic()
    audio.validate_headers()
    model, optimizer, scheduler = _training_parts(config, device)
    run = run_provenance(config, manifest)
    train_rows = len(manifest.examples(TRAIN_ROLE))
    dev = manifest.examples(DEV_ROLE)
    progress, best = Progress(), None
    if resume is None:
        random.seed(config.seed)
        torch.random.default_generator.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(config.seed)
    else:
        progress, best = read_resume(
            resume[0],
            receipt_sha256=resume[1],
            run=run,
            config=config,
            train_rows=train_rows,
            dev_rows=len(dev),
            model=model.backbone,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        if len(progress.history) == config.epochs or (
            progress.updates == config.max_updates and len(progress.train_losses) < train_rows
        ):
            raise ValueError(
                "the pinned training plan has already exhausted its epoch or update cap"
            )

    initial_updates = progress.updates
    snapshots = 0
    best_loss = min((row["dev_total"] for row in progress.history), default=math.inf)
    output.mkdir(exist_ok=False)
    record = {
        "schema": "vocoroute-base-attempt-v1",
        "status": "running",
        "reason": None,
        "run": run,
        "convergence_claimed": False,
        "attempt_start_updates": initial_updates,
        "resume_input_sha256": resume[1] if resume else None,
        "latest_resume": None,
    }

    def report() -> None:
        record.update(
            progress.cursor(), history=progress.history, elapsed_seconds=time.monotonic() - start
        )
        _write_summary(output, record)

    def checkpoint() -> None:
        nonlocal snapshots
        snapshots += 1
        tag = f"resume-{snapshots:06d}-{progress.updates:09d}"
        digest = write_resume(
            output / tag,
            run=run,
            progress=progress,
            model=model.backbone,
            optimizer=optimizer,
            scheduler=scheduler,
            best_weights=best,
            device=device,
        )
        record["latest_resume"] = {"directory": tag, "receipt_sha256": digest}
        report()

    report()
    try:
        reason = "max_epochs"
        with _graceful_stop() as stopping:
            while len(progress.history) < config.epochs:
                epoch = len(progress.history) + 1
                train = manifest.examples(TRAIN_ROLE, seed=config.seed, epoch=epoch)
                for row in train[len(progress.train_losses) :]:
                    reason = _budget_stop(
                        config,
                        progress.updates,
                        initial_updates,
                        attempt_updates,
                        start,
                        stopping,
                        fitting=True,
                    )
                    if reason:
                        break
                    model.train()
                    value = _train_example(model, optimizer, audio.waveform(row), row.utterance_id)
                    progress.train_losses.append(value)
                    progress.updates += 1
                    if progress.updates % config.checkpoint_every_updates == 0:
                        checkpoint()
                else:
                    model.eval()
                    with torch.no_grad():
                        for row in dev[len(progress.dev_losses) :]:
                            reason = _budget_stop(
                                config,
                                progress.updates,
                                initial_updates,
                                attempt_updates,
                                start,
                                stopping,
                                fitting=False,
                            )
                            if reason:
                                break
                            value = model(audio.waveform(row)).total.item()
                            if not math.isfinite(value):
                                raise ValueError(f"nonfinite dev loss for {row.utterance_id}")
                            progress.dev_losses.append(value)
                    if len(progress.dev_losses) == len(dev):
                        dev_loss = math.fsum(progress.dev_losses) / len(dev)
                        progress.history.append(
                            {
                                "epoch": epoch,
                                "train_total": math.fsum(progress.train_losses) / len(train),
                                "dev_total": dev_loss,
                            }
                        )
                        if dev_loss < best_loss:
                            best_loss = dev_loss
                            best = {
                                key: tensor.detach().cpu().clone()
                                for key, tensor in model.backbone.state_dict().items()
                            }
                        scheduler.step()
                        progress.train_losses.clear()
                        progress.dev_losses.clear()
                        reason = "max_epochs"
                        continue
                break

        checkpoint()
        if len(progress.history) == config.epochs:
            best_epoch = min(progress.history, key=lambda row: row["dev_total"])["epoch"]
            digest = save_base_checkpoint(
                output / "inference", config, run, progress.history, best_epoch, best
            )
            record.update(
                status="completed",
                reason="max_epochs",
                best_epoch=best_epoch,
                checkpoint_receipt_sha256=digest,
            )
        else:
            record.update(
                status="interrupted" if reason == "signal" else "budget_limited", reason=reason
            )
        report()
    except Exception as exc:
        record.update(status="failed", reason=type(exc).__name__, error=str(exc))
        report()
        raise
    return record


def _budget_stop(config, updates, initial, attempt, start, stopping, *, fitting):
    if stopping():
        return "signal"
    if time.monotonic() - start >= config.wall_time_seconds:
        return "wall_time"
    if fitting:
        if updates >= config.max_updates:
            return "max_updates"
        if attempt is not None and updates - initial >= attempt:
            return "attempt_updates"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-default-config", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--validate-checkpoint", type=Path, metavar="DIRECTORY")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--receipt-sha256")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-receipt-sha256")
    parser.add_argument("--attempt-updates", type=int)
    args = parser.parse_args(argv)

    try:
        if args.print_default_config:
            result = _default_config(args.device).document()
        elif args.validate_checkpoint is not None:
            if args.config is None or args.receipt_sha256 is None:
                raise ValueError("checkpoint validation requires --config and --receipt-sha256")
            config = load_training_config(args.config)
            _, result = load_base_checkpoint(
                args.validate_checkpoint, config, receipt_sha256=args.receipt_sha256
            )
        else:
            config = (
                load_training_config(args.config) if args.config else _default_config(args.device)
            )
            result = _prepare_run(args, config)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _default_config(device: str | None) -> TrainingConfig:
    if device is None:
        return TrainingConfig()
    return TrainingConfig(
        schema=BUDGET_CONFIG_SCHEMA,
        device=device,
        max_updates=100,
        wall_time_seconds=600,
        checkpoint_every_updates=25,
    )


def _prepare_run(args: argparse.Namespace, config: TrainingConfig) -> dict:
    if args.device is not None and args.device != config.device:
        raise ValueError("--device must match the pinned training configuration")
    if (args.resume is None) != (args.resume_receipt_sha256 is None):
        raise ValueError("resume requires both --resume and --resume-receipt-sha256")
    if args.attempt_updates is not None and args.attempt_updates < 1:
        raise ValueError("--attempt-updates must be positive")
    if (args.resume is not None or args.attempt_updates is not None) and (
        not args.train or config.schema != BUDGET_CONFIG_SCHEMA
    ):
        raise ValueError("resume and per-attempt budget require --train with a v2 config")
    if args.train and (args.config is None or args.out_dir is None):
        raise ValueError("training requires explicit --config and a fresh --out-dir")
    if args.dry_run and args.out_dir is not None:
        raise ValueError("dry-run has no output directory")
    with training_runtime(config.device) as device:
        _cuda_preflight(config, device)
        return _ready_run(args, config, device)


def _ready_run(args: argparse.Namespace, config: TrainingConfig, device: torch.device) -> dict:
    requested = args.manifest is not None or args.data_root is not None or args.train
    if requested:
        if args.manifest is None or args.data_root is None or config.manifest_sha256 is None:
            raise ValueError("real data needs --manifest, --data-root, and an approved config pin")
        manifest = load_manifest(args.manifest, config.manifest_sha256)
        audio = TrainingAudio(manifest, args.data_root)
        if args.train:
            output = args.out_dir.resolve()
            if output.exists() or not output.parent.is_dir() or output.is_relative_to(audio.root):
                raise ValueError("use a fresh output with an existing parent outside the data root")
            if config.schema == BUDGET_CONFIG_SCHEMA:
                resume = (args.resume, args.resume_receipt_sha256) if args.resume else None
                rng = random.getstate()
                try:
                    with torch.random.fork_rng(
                        devices=[device.index] if device.type == "cuda" else []
                    ):
                        return _run_budgeted(
                            config,
                            manifest,
                            audio,
                            output,
                            device,
                            resume=resume,
                            attempt_updates=args.attempt_updates,
                        )
                finally:
                    random.setstate(rng)
            return _run_training(config, manifest, audio, output)
        data = {
            "dataset_id": manifest.dataset_id,
            "manifest_sha256": manifest.sha256,
            "role_counts": {role: sum(row.role == role for row in manifest.rows) for role in ROLES},
            "train_dev_file_hashes_verified": True,
        }
    else:
        if config.manifest_sha256 is not None:
            raise ValueError("a pinned data configuration requires --manifest and --data-root")
        data = None

    model = initialize_backbone(config.seed)
    return {
        "mode": "dry-run",
        "device": config.device,
        "accelerator": accelerator_record(device),
        "config_sha256": config.sha256,
        "base_parameters": sum(param.numel() for param in model.parameters()),
        "data": data,
    }


if __name__ == "__main__":
    raise SystemExit(main())
