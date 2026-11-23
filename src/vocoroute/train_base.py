"""CPU base-model training entry point with an explicitly read-only dry-run mode."""

import argparse
import json
import math
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
from .training_config import DEV_ROLE, ROLES, TRAIN_ROLE, TrainingConfig, load_training_config
from .training_data import GeneratorManifest, TrainingAudio, load_manifest, prepare_example


class BaseTrainingModel(nn.Module):
    """Run mel conditioning, base synthesis and waveform-domain objectives."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.frontend = SpectralFrontend()
        self.backbone = initialize_backbone(config.seed)
        self.loss = ReconstructionLoss(config.loss)

    def forward(self, waveform: Tensor) -> ReconstructionTerms:
        example = prepare_example(self.frontend, waveform)
        spectrum = self.backbone(example.mel)
        synthesized = self.frontend.synthesize(spectrum, example.length)
        return self.loss(synthesized, example.target)


def _run_training(
    config: TrainingConfig,
    manifest: GeneratorManifest,
    audio: TrainingAudio,
    output: Path,
) -> dict:
    audio.validate_headers()
    model = BaseTrainingModel(config)
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
    provenance = run_provenance(config, manifest)
    history = []
    best_loss = math.inf
    best_epoch = 0
    best_weights = {}
    for epoch in range(1, config.epochs + 1):
        training = manifest.examples(TRAIN_ROLE, seed=config.seed, epoch=epoch)
        model.train()
        train_totals = []
        for row in training:
            optimizer.zero_grad(set_to_none=True)
            loss = model(audio.waveform(row)).total
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite training loss for {row.utterance_id}")
            loss.backward()
            if any(
                param.grad is None or not torch.isfinite(param.grad).all()
                for param in model.backbone.parameters()
            ):
                raise ValueError(f"missing or nonfinite gradient for {row.utterance_id}")
            optimizer.step()
            train_totals.append(loss.item())

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
    args = parser.parse_args(argv)

    try:
        if args.print_default_config:
            result = TrainingConfig().document()
        elif args.validate_checkpoint is not None:
            if args.config is None or args.receipt_sha256 is None:
                raise ValueError("checkpoint validation requires --config and --receipt-sha256")
            config = load_training_config(args.config)
            _, result = load_base_checkpoint(
                args.validate_checkpoint, config, receipt_sha256=args.receipt_sha256
            )
        else:
            config = load_training_config(args.config) if args.config else TrainingConfig()
            result = _prepare_run(args, config)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _prepare_run(args: argparse.Namespace, config: TrainingConfig) -> dict:
    if args.train and (args.config is None or args.out_dir is None):
        raise ValueError("training requires explicit --config and a fresh --out-dir")
    if args.dry_run and args.out_dir is not None:
        raise ValueError("dry-run has no output directory")
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
            deterministic = torch.are_deterministic_algorithms_enabled()
            warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
            torch.use_deterministic_algorithms(True)
            try:
                return _run_training(config, manifest, audio, output)
            finally:
                torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
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
        "device": "cpu",
        "config_sha256": config.sha256,
        "base_parameters": sum(param.numel() for param in model.parameters()),
        "data": data,
    }


if __name__ == "__main__":
    raise SystemExit(main())
