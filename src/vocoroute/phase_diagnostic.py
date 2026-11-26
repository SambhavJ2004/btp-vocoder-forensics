"""Local phase-oracle controls for the pinned N2B.1 reconstruction."""

import argparse
import hashlib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from data.audio_io import read_native

from .losses import ReconstructionLoss
from .overfit import N2B1_CORPUS, _record_waveform, spectral_convergence
from .spectral import SpectralFrontend
from .train_base import BaseTrainingModel
from .training_config import SPECTRAL_FLOOR, TrainingConfig, is_sha256, parse_json
from .training_data import prepare_example, sha256_file
from .training_device import training_runtime
from .vctk_prepare import read_pinned_json, write_json

BASELINE_RECEIPT_SHA256 = "588ae733675b5f6deb5b1a6a6f7474d4bdd70b60edf338e17e463aa9152a70a8"
GRIFFIN_LIM_ITERATIONS = 64
GRIFFIN_LIM_SEED = 20260907
ORACLE_ATOL = 2e-6
ORACLE_RTOL = 1e-5


def baseline_record(receipt: Path) -> dict:
    original = read_pinned_json(receipt, BASELINE_RECEIPT_SHA256)
    return {
        "receipt_sha256": BASELINE_RECEIPT_SHA256,
        "runtime": original["runtime"],
        "initial_fp32_sha256": original["checkpoint"]["initial_flat_fp32_sha256"],
        "step_1000_raw_sha256": original["snapshots"][-1]["raw_sha256"],
        "step_1000_common_reconstruction": original["snapshots"][-1]["total"],
        "notebook": {"id": 147395155, "version": 1},
        "commit": original["code"]["commit"],
    }


def _unit_phase(spectrum: Tensor) -> Tensor:
    magnitude = spectrum.abs()
    unit = torch.where(
        magnitude > 0,
        spectrum / magnitude.clamp_min(torch.finfo(magnitude.dtype).tiny),
        1.0,
    )
    # In a one-sided real FFT both signed endpoints are constrained to the real axis.
    first, last = unit[:, :1].real, unit[:, -1:].real
    first = torch.where(first < 0, -1.0, 1.0).to(unit.dtype)
    last = torch.where(last < 0, -1.0, 1.0).to(unit.dtype)
    return torch.cat((first, unit[:, 1:-1], last), dim=1)


def magnitude_with_phase(magnitude: Tensor, donor: Tensor) -> Tensor:
    return magnitude * _unit_phase(donor)


def griffin_lim(frontend: SpectralFrontend, magnitude: Tensor, length: int) -> Tensor:
    """Return the final imposed spectrum, before the last consistency projection."""
    rng = torch.Generator(device="cpu").manual_seed(GRIFFIN_LIM_SEED)
    angle = torch.rand(magnitude.shape, generator=rng, dtype=magnitude.dtype) * (2 * torch.pi)
    imposed = magnitude_with_phase(magnitude, torch.polar(torch.ones_like(angle), angle))
    for _ in range(GRIFFIN_LIM_ITERATIONS):
        consistent = frontend.analyze(frontend.synthesize(imposed, length))
        imposed = magnitude_with_phase(magnitude, consistent)
    return imposed


def _relative_spectrum_error(candidate: Tensor, original: Tensor) -> Tensor:
    return torch.linalg.vector_norm(candidate - original) / torch.linalg.vector_norm(
        original
    ).clamp_min(SPECTRAL_FLOOR)


def circular_phase_error(predicted: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Average 1-cos(delta) by reference energy, with cost 1 for undefined prediction."""
    pred_inner, target_inner = predicted[:, 1:-1], target[:, 1:-1]
    pred_magnitude, target_magnitude = pred_inner.abs(), target_inner.abs()
    weight = torch.where(target_magnitude >= SPECTRAL_FLOOR, target_magnitude.square(), 0.0)
    defined = pred_magnitude >= SPECTRAL_FLOOR
    target_unit = target_inner / target_magnitude.clamp_min(SPECTRAL_FLOOR)
    pred_unit = pred_inner / pred_magnitude.clamp_min(SPECTRAL_FLOOR)
    cosine = (pred_unit * target_unit.conj()).real.clamp(-1, 1)
    distance = torch.where(defined, 1 - cosine, 1.0)
    denominator = weight.sum().clamp_min(SPECTRAL_FLOOR**2)
    return (weight * distance).sum() / denominator, (weight * defined).sum() / denominator


def phase_evaluation_loss() -> ReconstructionLoss:
    return ReconstructionLoss(replace(TrainingConfig().loss, phase_enabled=True))


def reconstruction_metrics(
    model: BaseTrainingModel, loss: ReconstructionLoss, predicted: Tensor, target: Tensor
) -> dict:
    terms = loss(predicted, target)
    analyzed = model.frontend.analyze(predicted)
    phase, coverage = circular_phase_error(analyzed, model.frontend.analyze(target))
    reprojection = model.frontend.analyze(model.frontend.synthesize(analyzed, predicted.shape[-1]))
    values = {
        "waveform_mae": float(terms.waveform),
        "linear_magnitude_mae": float(terms.magnitude),
        "log_magnitude_mae": float(terms.log_magnitude),
        "common_reconstruction": float(terms.waveform + terms.magnitude + terms.log_magnitude),
        "spectral_convergence": spectral_convergence(model, predicted, target),
        "circular_phase_error": float(phase),
        "defined_phase_reference_energy_fraction": float(coverage),
        "phase_time": float(terms.phase_time),
        "phase_frequency": float(terms.phase_frequency),
        "reanalyzed_stft_consistency": float(_relative_spectrum_error(reprojection, analyzed)),
    }
    if not np.isfinite(list(values.values())).all():
        raise ValueError("nonfinite reconstruction metric")
    return values


def _pinned_file(source: Path, name: str, digest: str) -> Path:
    if Path(name).name != name or not is_sha256(digest):
        raise ValueError("invalid pinned artifact path or SHA-256")
    path = source / name
    if sha256_file(path) != digest:
        raise ValueError(f"original N2B.1 artifact SHA-256 mismatch: {name}")
    return path


def baseline_waveform(output: Path, baseline: Path) -> dict:
    """Compare future Run A step 1000 against the pinned original raw waveform."""
    record = baseline_record(baseline / "diagnostic-receipt.json")
    receipt = parse_json((output / "ablation-receipt.json").read_bytes())
    filename = "A/step-1000.npy"
    if (
        receipt["schema"] != "vocoroute-n2b1p-phase-ablation-receipt-v1"
        or receipt["status"] not in ("completed", "budget_limited")
        or receipt["baseline"] != record
        or receipt["matched_updates"] < 1000
        or sha256_file(output / filename) != receipt["artifacts_sha256"][filename]
    ):
        raise ValueError("could not verify the Run A artifact against its ablation receipt")
    original = np.load(
        _pinned_file(baseline, "step-1000.npy", record["step_1000_raw_sha256"]),
        allow_pickle=False,
    )
    candidate = np.load(output / filename, allow_pickle=False)
    if (
        original.shape != candidate.shape
        or candidate.shape != (N2B1_CORPUS.samples,)
        or candidate.dtype != np.float32
        or not np.isfinite(candidate).all()
    ):
        raise ValueError(
            "Run A's baseline comparison requires a finite pinned-length FP32 waveform"
        )
    error = np.abs(candidate - original)
    return {
        "runtime_matches": receipt["runtime"] == record["runtime"],
        "initial_weights_match": receipt["initial_fp32_sha256"] == record["initial_fp32_sha256"],
        "max_abs": float(error.max()),
        "mae": float(error.mean(dtype=np.float64)),
        "exact_array_match": bool(np.array_equal(candidate, original)),
        "allclose_atol": 2e-4,
        "allclose_rtol": 1e-3,
        "within_tolerance": bool(np.allclose(candidate, original, atol=2e-4, rtol=1e-3)),
    }


def _gradient_probe(model: BaseTrainingModel, evaluation: ReconstructionLoss, example) -> dict:
    predicted = model.synthesize(example)
    terms = evaluation(predicted, example.target)
    common = terms.waveform + terms.magnitude + terms.log_magnitude
    phase = evaluation.weights.phase_time * terms.phase_time
    phase = phase + evaluation.weights.phase_frequency * terms.phase_frequency
    parameters = tuple(model.backbone.parameters())
    values = {}
    for label, term, retain in (
        ("common", common, True),
        ("time", terms.phase_time, True),
        ("frequency", terms.phase_frequency, True),
        ("weighted_phase", phase, False),
    ):
        grads = torch.autograd.grad(term, parameters, retain_graph=retain)
        norm = torch.linalg.vector_norm(torch.cat([grad.reshape(-1) for grad in grads]))
        if not norm.isfinite():
            raise ValueError(f"nonfinite backbone gradient for {label}")
        values[label] = {"loss": float(term.detach()), "backbone_gradient_l2": float(norm)}
    return values


def run_oracles(source: Path, output: Path) -> dict:
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("diagnostic outputs must be outside the original artifacts")
    path = source / "diagnostic-receipt.json"
    original = read_pinned_json(path, BASELINE_RECEIPT_SHA256)
    checkpoint = original["checkpoint"]
    pins = original["artifacts_sha256"]
    for name in ("reference.wav", "step-1000.npy", "backbone.pt"):
        _pinned_file(source, name, pins[name])
    reference, rate = read_native(source / "reference.wav")
    saved = np.load(source / "step-1000.npy", allow_pickle=False)
    length = N2B1_CORPUS.samples
    if (
        rate != original["config"]["frontend"]["sample_rate"]
        or reference.shape != saved.shape
        or saved.shape != (length,)
        or saved.dtype != np.float32
        or not np.isfinite(saved).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("baseline reference/reconstruction geometry or finite check failed")
    model = BaseTrainingModel(TrainingConfig()).eval()
    weights = torch.load(source / "backbone.pt", map_location="cpu", weights_only=True)
    model.backbone.load_state_dict(weights, strict=True)
    flat = torch.cat([weight.detach().reshape(-1) for weight in model.backbone.parameters()])
    tensor_sha = hashlib.sha256(flat.numpy().astype("<f4", copy=False).tobytes()).hexdigest()
    if not flat.isfinite().all() or tensor_sha != checkpoint["final_flat_fp32_sha256"]:
        raise ValueError("baseline checkpoint tensor identity or finite check failed")
    frontend = model.frontend
    evaluation = phase_evaluation_loss()
    target, predicted = (torch.as_tensor(x, dtype=torch.float32)[None] for x in (reference, saved))
    example = prepare_example(frontend, target[0])
    output.mkdir(parents=True, exist_ok=False)
    with training_runtime("cpu"), torch.no_grad():
        replay = model.synthesize(example)
        torch.testing.assert_close(replay, predicted, atol=2e-4, rtol=1e-3)
        ref_stft, pred_stft = (frontend.analyze(x) for x in (target, predicted))
        spectra = {
            "reference-magnitude_reference-phase": magnitude_with_phase(ref_stft.abs(), ref_stft),
            "predicted-magnitude_predicted-phase": magnitude_with_phase(pred_stft.abs(), pred_stft),
            "predicted-magnitude_reference-phase": magnitude_with_phase(pred_stft.abs(), ref_stft),
            "reference-magnitude_predicted-phase": magnitude_with_phase(ref_stft.abs(), pred_stft),
            "reference-magnitude_griffin-lim": griffin_lim(frontend, ref_stft.abs(), length),
            "predicted-magnitude_griffin-lim": griffin_lim(frontend, pred_stft.abs(), length),
        }
        rows = {}
        for index, (name, imposed) in enumerate(spectra.items()):
            reconstructed = frontend.synthesize(imposed, length)
            stats = _record_waveform(output / f"{name}.wav", reconstructed)
            if stats["nonfinite_samples"]:
                raise ValueError(f"nonfinite diagnostic waveform: {name}")
            consistency = _relative_spectrum_error(frontend.analyze(reconstructed), imposed)
            row = {
                **reconstruction_metrics(model, evaluation, reconstructed, target),
                **stats,
                "imposed_stft_consistency": float(consistency),
            }
            if index < 2:
                waveform = (target, predicted)[index]
                torch.testing.assert_close(
                    reconstructed, waveform, atol=ORACLE_ATOL, rtol=ORACLE_RTOL
                )
                row["identity_max_abs"] = float((reconstructed - waveform).abs().max())
            rows[name] = row

    with training_runtime("cpu"):
        gradients = _gradient_probe(model, evaluation, example)
    report = {
        "schema": "vocoroute-n2b1-phase-oracles-v1",
        "frontend": original["config"]["frontend"],
        "length": length,
        "runtime": {"torch": str(torch.__version__), "device": "cpu", "dtype": "float32"},
        "inputs_sha256": {k: pins[k] for k in ("reference.wav", "step-1000.npy", "backbone.pt")},
        "original_receipt_sha256": BASELINE_RECEIPT_SHA256,
        "checkpoint_final_fp32_sha256": tensor_sha,
        "checkpoint_replay_max_abs": float((replay - predicted).abs().max()),
        "identity_tolerance": {"atol": ORACLE_ATOL, "rtol": ORACLE_RTOL},
        "griffin_lim": {
            "seed": GRIFFIN_LIM_SEED,
            "iterations": GRIFFIN_LIM_ITERATIONS,
            "momentum": 0,
            "same_random_initial_phase": True,
        },
        "metric_loss_weights": asdict(evaluation.weights),
        "final_checkpoint_gradient_probe": gradients,
        "conditions": rows,
    }
    write_json(output / "oracle-receipt.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="retrieved N2B.1 notebook output")
    parser.add_argument("output", type=Path, help="fresh local diagnostics folder")
    parser.add_argument("--compare-baseline", action="store_true", help="audit retrieved Run A")
    args = parser.parse_args()
    if args.compare_baseline:
        print(baseline_waveform(args.output, args.source))
    else:
        torch.set_num_threads(1)
        run_oracles(args.source, args.output)


if __name__ == "__main__":
    main()
