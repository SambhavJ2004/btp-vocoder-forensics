"""Fixed development evaluation and listening artifacts for the base pilot."""

import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import torch

from .training_config import DEV_ROLE, TRAIN_ROLE, canonical_json
from .training_data import GeneratorManifest, prepare_example, sha256_file

PILOT_SCHEMA = "vocoroute-base-pilot-v1"
PILOT_MANIFEST = "5ed731b8c47dcd0331f14ab1b5d877200fb6f7e09500fbb7507f94d998167d2f"
PILOT_SOURCE = "sambhavjainsj/vocoroute-vctk092-n2b0-train-dev/1"
PILOT_NOTEBOOK_ID = 147338652
PILOT_DATA_REF = "sambhavjainsj/vocoroute-vctk092-n2b0-pilot"
PILOT_METADATA = {
    "manifest.json": PILOT_MANIFEST,
    "dataset-receipt.json": "8bfe9a935b7c6e11dea8aa565308639aaf3f60520eb0ccda33fb8a52f0f249ca",
    "freeze.json": "39ae136dbf240762a8c2abd38e39f8c56f9d246cd191c6f7dae014400808ec6b",
    "source/resampling-receipt.json": (
        "06c316bf2323589e670628ff15b766a7992f72bc90b2cdbfef521c9009f61d6f"
    ),
    "source/license_text.txt": "b34e17103bfb246f2549fc82a279e6ba28834e0cb42f76a92efc14b72e3a3723",
}
CHECKPOINT_RULE = "lowest per-utterance mean weighted dev objective; exact ties keep earliest"
PANEL_EPOCHS = (1, 5, 10, 20)


def _rank(seed: int, label: str, value: str) -> bytes:
    return hashlib.sha256(f"{PILOT_SCHEMA}:{seed}:{label}:{value}".encode()).digest()


def panel_ids(manifest: GeneratorManifest, seed: int, *, per_speaker: int = 3) -> list[str]:
    rows = manifest.examples(DEV_ROLE)
    panel = []
    for speaker in sorted({row.speaker_id for row in rows}):
        ids = [row.utterance_id for row in rows if row.speaker_id == speaker]
        if len(ids) < per_speaker:
            raise ValueError("insufficient frozen dev recordings for the listening panel")
        panel.extend(sorted(ids, key=lambda uid: (_rank(seed, "panel", uid), uid))[:per_speaker])
    return panel


def blind_order(ids: list[str], seed: int, stage: str) -> list[tuple[str, bool]]:
    if len(ids) % 2 or len(set(ids)) != len(ids):
        raise ValueError("the balanced panel requires an even number of unique utterances")
    order = sorted(ids, key=lambda uid: (_rank(seed, f"order:{stage}", uid), uid))
    sides = sorted(ids, key=lambda uid: (_rank(seed, f"side:{stage}", uid), uid))
    reference_first = set(sides[: len(sides) // 2])
    return [(uid, uid in reference_first) for uid in order]


def protocol_digest(document: dict) -> str:
    return hashlib.sha256(canonical_json(document)).hexdigest()


def pilot_config():
    from .training_config import BUDGET_CONFIG_SCHEMA, LossWeights, TrainingConfig

    return TrainingConfig(
        schema=BUDGET_CONFIG_SCHEMA,
        device="cuda",
        manifest_sha256=PILOT_MANIFEST,
        epochs=20,
        max_updates=20000,
        scheduler_gamma=1.0,
        wall_time_seconds=1500,
        checkpoint_every_updates=1000,
        loss=LossWeights(phase_enabled=True),
    )


def pilot_protocol(manifest: GeneratorManifest, config) -> dict:
    if config != pilot_config() or manifest.sha256 != PILOT_MANIFEST:
        raise ValueError("pilot requires the exact frozen manifest and training configuration")
    for role, utterances, speakers in ((TRAIN_ROLE, 1000, 20), (DEV_ROLE, 200, 4)):
        rows = manifest.examples(role)
        if len(rows) != utterances or len({row.speaker_id for row in rows}) != speakers:
            raise ValueError(f"frozen pilot {role} allocation disagrees")
    return {
        "schema": PILOT_SCHEMA,
        "training_config_sha256": config.sha256,
        "source_notebook": PILOT_SOURCE,
        "source_notebook_id": PILOT_NOTEBOOK_ID,
        "attachment_metadata": PILOT_METADATA,
        "best_checkpoint_rule": CHECKPOINT_RULE,
        "panel_utterance_ids": panel_ids(manifest, config.seed),
        "panel_epochs": list(PANEL_EPOCHS),
        "panel_policy": "SHA-256 rank by speaker; stage-specific order, six reference-first",
        "accelerator": {"machine_shape": "NvidiaTeslaT4", "visible_t4s": 2, "training": "cuda:0"},
        "admission_seconds": 1500,
        "kaggle_timeout_seconds": 1680,
        "planned_gpu_budget_seconds": 1800,
    }


def attachment_pins(manifest: GeneratorManifest) -> dict[str, str]:
    files = dict(PILOT_METADATA)
    for role in (TRAIN_ROLE, DEV_ROLE):
        for row in manifest.examples(role):
            if row.resample_steps and (
                row.resampling_receipt_sha256 != files["source/resampling-receipt.json"]
            ):
                raise ValueError("pilot audio disagrees with the frozen resampling receipt")
            files[row.native_path] = row.native_sha256
            files[row.prepared_path] = row.prepared_sha256
    return files


def prepare_attachment(source: Path, output: Path) -> dict:
    """Project only pinned files from the saved N2B.0 output, on the same host."""
    from .training_data import load_manifest

    pilot = source / "pilot"
    manifest = load_manifest(pilot / "manifest.json", PILOT_MANIFEST)
    pins = attachment_pins(manifest)
    output = output.resolve()
    if output.exists() or output.is_relative_to(source.resolve()):
        raise ValueError("pilot projection requires a fresh output outside the source")
    source_files = {
        name: (source if name == "freeze.json" else pilot) / name for name in pins
    }
    for name, path in source_files.items():
        if (
            not path.resolve(strict=True).is_relative_to(source.resolve())
            or sha256_file(path) != pins[name]
        ):
            raise ValueError(f"saved pilot output SHA-256 mismatch or unsafe path: {name}")
    output.mkdir()
    for name, source_file in source_files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
        if sha256_file(target) != pins[name]:
            raise ValueError(f"pilot projection copy SHA-256 mismatch: {name}")
    validate_attachment(output, manifest)
    return {
        "files": len(pins),
        "bytes": sum((output / name).stat().st_size for name in pins),
        "inventory_sha256": protocol_digest(pins),
    }


def validate_attachment(root: Path, manifest: GeneratorManifest) -> None:
    expected = attachment_pins(manifest)
    files = set()
    for path in root.rglob("*"):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("pilot attachment contains an unsafe path")
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    if files != expected.keys():
        raise ValueError("pilot attachment inventory disagrees with the exact train/dev projection")
    for name, digest in PILOT_METADATA.items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"pilot metadata SHA-256 mismatch: {name}")


def loss_metrics(terms) -> dict[str, float]:
    waveform, linear, log, time, frequency, total = (
        torch.stack((*terms[1:], terms.total)).detach().cpu().tolist()
    )
    if not all(map(math.isfinite, (waveform, linear, log, time, frequency, total))):
        raise ValueError("nonfinite pilot objective component")
    return {
        "weighted_objective": total,
        "common_reconstruction": waveform + linear + log,
        "waveform_mae": waveform,
        "linear_magnitude_mae": linear,
        "log_magnitude_mae": log,
        "phase_time": time,
        "phase_frequency": frequency,
    }


def macro_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {
        key: math.fsum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }


def exposure_counts(manifest: GeneratorManifest, config, progress) -> dict:
    count = len(progress.history)
    order = manifest.examples(TRAIN_ROLE, seed=config.seed, epoch=count + 1)
    exposures = Counter({row.utterance_id: count for row in order})
    exposures.update(row.utterance_id for row in order[:len(progress.train_losses)])
    return dict(sorted(exposures.items()))


class PilotDiagnostics:
    """Read-only evaluation hooks around the existing training and resume loop."""

    def __init__(self, document, manifest, config, audio, output):
        self.output = output
        self.audio = audio
        self.seed = config.seed
        self.panel = document["panel_utterance_ids"]
        self.epochs = document["panel_epochs"]
        self.dev = {row.utterance_id: row for row in manifest.examples(DEV_ROLE)}

    @torch.no_grad()
    def evaluate(self, model, row, waveform):
        from .overfit import PCM16_MAX, spectral_convergence
        from .phase_diagnostic import circular_phase_error

        example = prepare_example(model.frontend, waveform.to(model.frontend.window.device))
        predicted = model.synthesize(example)
        terms = model.loss(predicted, example.target)
        values = loss_metrics(terms)
        phase, _ = circular_phase_error(
            model.frontend.analyze(predicted), model.frontend.analyze(example.target)
        )
        finite = torch.isfinite(predicted)
        values.update(
            spectral_convergence=spectral_convergence(model, predicted, example.target),
            circular_phase_error=float(phase),
            rms=float(predicted.square().mean().sqrt()),
            peak_abs=float(predicted.abs().max()),
            clipped_samples_for_pcm16=int(((predicted < -1) | (predicted > PCM16_MAX)).sum()),
            nonfinite_samples=int((~finite).sum()),
        )
        if not all(map(math.isfinite, values.values())):
            raise ValueError(f"nonfinite pilot dev metrics: {row.utterance_id}")
        return {"utterance_id": row.utterance_id, "speaker_id": row.speaker_id, "metrics": values}

    @staticmethod
    def finish_epoch(epoch, state):
        by_speaker = {}
        for row in state["dev"]:
            by_speaker.setdefault(row["speaker_id"], []).append(row["metrics"])
        dev_rows = [row["metrics"] for row in state["dev"]]
        state["epochs"].append(
            {
                "epoch": epoch,
                "train_macro": macro_metrics(state["train"]),
                "dev_macro": macro_metrics(dev_rows),
                "dev_sample_count_totals": {
                    key: sum(row[key] for row in dev_rows)
                    for key in ("clipped_samples_for_pcm16", "nonfinite_samples")
                },
                "dev_by_speaker": {
                    speaker: {
                        "utterances": len(metrics),
                        **macro_metrics(metrics),
                        "clipped_samples_total": sum(
                            row["clipped_samples_for_pcm16"] for row in metrics
                        ),
                        "nonfinite_samples_total": sum(row["nonfinite_samples"] for row in metrics),
                    }
                    for speaker, metrics in sorted(by_speaker.items())
                },
                "dev_utterances": state["dev"],
            }
        )
        state.update(train=[], dev=[])

    @torch.no_grad()
    def render(self, model, stage):
        from .overfit import _record_waveform

        parent = self.output / "listening"
        reference = parent / "reference"
        reference.mkdir(parents=True, exist_ok=True)
        directory = parent / stage
        directory.mkdir()
        key = {}
        for index, (uid, reference_first) in enumerate(
            blind_order(self.panel, self.seed, stage), 1
        ):
            row = self.dev[uid]
            source = reference / f"{uid}.wav"
            if not source.exists():
                shutil.copyfile(self.audio.root / row.prepared_path, source)
                if sha256_file(source) != row.prepared_sha256:
                    raise ValueError(f"pilot panel reference SHA-256 mismatch: {uid}")
            waveform = self.audio.waveform(row).to(model.frontend.window.device)
            example = prepare_example(model.frontend, waveform)
            predicted = model.synthesize(example)
            for side in ("a", "b"):
                wav = directory / f"pair-{index:02d}-{side}.wav"
                is_reference = (side == "a") == reference_first
                stats = None
                if is_reference:
                    shutil.copyfile(source, wav)
                else:
                    unblinded = parent / "raw" / stage
                    unblinded.mkdir(parents=True, exist_ok=True)
                    generated = unblinded / f"{uid}.wav"
                    stats = _record_waveform(generated, predicted)
                    if stats["wav"] is None:
                        raise ValueError(f"nonfinite listening panel reconstruction: {uid}")
                    generated.replace(wav)
                    stats.update(raw=generated.with_suffix(".npy").relative_to(parent).as_posix(),
                                 wav=wav.relative_to(parent).as_posix())
                key[wav.name] = {
                    "utterance_id": uid,
                    "speaker_id": row.speaker_id,
                    "condition": "reference" if is_reference else "reconstruction",
                    "sha256": sha256_file(wav),
                    "stats": stats,
                }
        (parent / f"{stage}-key.json").write_bytes(canonical_json(key) + b"\n")

    def save_metrics(self, progress, manifest, config) -> str:
        state = progress.pilot
        path = self.output / "pilot-metrics.json"
        document = {
            "schema": PILOT_SCHEMA,
            "epochs": state["epochs"],
            "partial_train_macro": macro_metrics(state["train"]),
            "partial_dev_utterances": state["dev"],
            "train_utterance_exposures": exposure_counts(manifest, config, progress),
            "listening_keys_sha256": {
                path.relative_to(self.output).as_posix(): sha256_file(path)
                for path in sorted((self.output / "listening").glob("*-key.json"))
            },
        }
        path.write_text(json.dumps(document, sort_keys=True, allow_nan=False), encoding="utf-8")
        return sha256_file(path)
