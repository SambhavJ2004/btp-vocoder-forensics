"""Synthetic phase safety, matched fitting, and RECORD staging checks."""

import csv
import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from test_vocoroute_n2b1 import mounted as mounted  # noqa: E402

from experiments import vocoroute_phase_kaggle as kaggle  # noqa: E402
from vocoroute import phase_ablation, phase_diagnostic  # noqa: E402
from vocoroute.losses import ReconstructionLoss  # noqa: E402
from vocoroute.overfit import diagnostic_config, run_overfit  # noqa: E402
from vocoroute.phase_diagnostic import (  # noqa: E402
    BASELINE_RECEIPT_SHA256,
    circular_phase_error,
    griffin_lim,
    magnitude_with_phase,
)
from vocoroute.spectral import SpectralFrontend  # noqa: E402
from vocoroute.training_config import LossWeights  # noqa: E402
from vocoroute.training_data import sha256_file  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _read_trace(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def _baseline():
    return {
        "receipt_sha256": BASELINE_RECEIPT_SHA256,
        "runtime": {"device": "test-placeholder"},
        "initial_fp32_sha256": "b" * 64,
        "step_1000_raw_sha256": "c" * 64,
        "step_1000_common_reconstruction": 0.72,
    }


def test_oracle_reuses_native_frontend_and_canonicalizes_real_endpoints():
    frontend = SpectralFrontend()
    n = 1139
    reference = torch.sin(torch.arange(n) * 0.123)[None] * 0.25
    other = torch.cos(torch.arange(n) * 0.68)[None] * 0.1
    ref, pred = (frontend.analyze(wav) for wav in (reference, other))
    for stft, wav in ((ref, reference), (pred, other)):
        rebuilt = frontend.synthesize(magnitude_with_phase(stft.abs(), stft), n)
        torch.testing.assert_close(rebuilt, wav, atol=2e-6, rtol=1e-5)
    mixed = magnitude_with_phase(pred.abs(), ref)
    torch.testing.assert_close(mixed.abs(), pred.abs())
    for endpoint in (0, -1):
        assert torch.count_nonzero(mixed[:, endpoint].imag) == 0
        nonzero = mixed[:, endpoint].abs() > 0
        assert torch.equal(mixed[:, endpoint].real[nonzero] < 0, ref[:, endpoint].real[nonzero] < 0)
    gl = griffin_lim(frontend, ref.abs(), n)
    assert torch.equal(gl, griffin_lim(frontend, ref.abs(), n))
    torch.testing.assert_close(gl.abs(), ref.abs())
    assert frontend.synthesize(gl, n).isfinite().all()


def test_circular_error_checks_distinct_phase_and_undefined_convention():
    frontend = SpectralFrontend()
    original = frontend.analyze(torch.sin(torch.arange(1500) * 0.3)[None])
    for predicted, expected, coverage in [
        (original, 0, 1),
        (-original, 2, 1),
        (torch.zeros_like(original), 1, 0),
    ]:
        measured, fraction = circular_phase_error(predicted, original)
        assert float(measured) == pytest.approx(expected, abs=1e-6)
        assert float(fraction) == pytest.approx(coverage)


@pytest.mark.parametrize("length", [1, 1171])
@pytest.mark.parametrize("scale", [0.0, 1e-10, 0.1])
def test_existing_phase_loss_backward_is_finite_at_and_below_floor(length, scale):
    series = torch.arange(length)
    target = torch.sin(series * 0.125)[None] * scale
    predicted = (torch.cos(series * 0.13)[None] * scale).requires_grad_()
    criterion = ReconstructionLoss(LossWeights(phase_enabled=True))
    losses = criterion(predicted, target)
    phase = losses.phase_time + losses.phase_frequency
    gradient = torch.autograd.grad(phase, predicted)[0]
    assert phase.isfinite() and gradient.isfinite().all()
    if scale <= 1e-10 or length == 1:
        assert phase == 0 and gradient.count_nonzero() == 0
    else:
        assert phase > 0 and gradient.norm() > 0


@pytest.mark.parametrize("scale", [0.0, 1e-10])
def test_weak_prediction_against_active_reference_has_finite_gradient(scale):
    time = torch.arange(1171, dtype=torch.float32)
    target = 0.1 * torch.sin(time[None] * 0.125)
    predicted = (scale * torch.cos(time[None] * 0.13)).requires_grad_()
    terms = ReconstructionLoss(LossWeights(phase_enabled=True))(predicted, target)
    phase = terms.phase_time + terms.phase_frequency
    gradient = torch.autograd.grad(phase, predicted)[0]
    assert phase.isfinite() and phase > 0 and gradient.isfinite().all()
    if scale == 0:
        assert gradient.count_nonzero() == 0


def test_verified_baseline_comparison_measures_difference_and_refuses_tampering(
    tmp_path,
    monkeypatch,
):
    baseline, run = tmp_path / "baseline", tmp_path / "run"
    baseline.mkdir()
    (run / "A").mkdir(parents=True)
    original = np.linspace(-0.5, 0.5, 9, dtype=np.float32)
    original_path, candidate_path = baseline / "step-1000.npy", run / "A/step-1000.npy"
    np.save(original_path, original, allow_pickle=False)
    changed = original.copy()
    changed[4] = 1e-5
    np.save(candidate_path, changed, allow_pickle=False)
    record = _baseline() | {"step_1000_raw_sha256": sha256_file(original_path)}
    monkeypatch.setattr(phase_diagnostic, "baseline_record", lambda _: record)
    monkeypatch.setattr(
        phase_diagnostic, "N2B1_CORPUS", replace(phase_diagnostic.N2B1_CORPUS, samples=9)
    )
    receipt = {
        "schema": "vocoroute-n2b1p-phase-ablation-receipt-v1",
        "status": "completed",
        "baseline": record,
        "matched_updates": 4000,
        "artifacts_sha256": {"A/step-1000.npy": sha256_file(candidate_path)},
        "runtime": record["runtime"],
        "initial_fp32_sha256": record["initial_fp32_sha256"],
    }
    (run / "ablation-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    result = phase_diagnostic.baseline_waveform(run, baseline)
    assert result["runtime_matches"] and result["initial_weights_match"]
    assert not result["exact_array_match"] and result["within_tolerance"]
    assert result["max_abs"] == pytest.approx(1e-5) and result["mae"] == pytest.approx(1e-5 / 9)
    np.save(candidate_path, original, allow_pickle=False)
    with pytest.raises(ValueError, match="could not verify the Run A artifact"):
        phase_diagnostic.baseline_waveform(run, baseline)


def test_matched_runner_preserves_original_arm_and_reports_shared_metrics(mounted, tmp_path):
    inputs, mount, _, pin = mounted
    schedule = (0, 1, 3)
    legacy = tmp_path / "legacy"
    run_overfit(inputs, mount, legacy, pin=pin, device="cpu", steps=schedule)
    output = tmp_path / "ablation"
    report = phase_ablation.run_phase_ablation(
        inputs,
        mount,
        output,
        pin=pin,
        device="cpu",
        steps=schedule,
        original=_baseline(),
        code={},
    )
    assert report["matched_updates"] == 3 and report["status"] == "completed"
    a, b = (report["arms"][arm]["snapshots"] for arm in ("A", "B"))
    assert a[0]["raw_sha256"] == b[0]["raw_sha256"]
    for old, no_phase, phase in zip(schedule, a, b, strict=True):
        assert old == no_phase["step"] == phase["step"]
        assert no_phase["raw_sha256"] == sha256_file(legacy / no_phase["raw"])
        assert no_phase["phase_time"] > 0 and phase["phase_time"] > 0
        assert no_phase["optimization_total"] == pytest.approx(no_phase["common_reconstruction"])
        assert phase["optimization_total"] == pytest.approx(
            phase["common_reconstruction"] + 0.05 * (phase["phase_time"] + phase["phase_frequency"])
        )
    assert a[-1]["raw_sha256"] != b[-1]["raw_sha256"]
    for key in "AB":
        trace = _read_trace(output / key / "train-loss.csv")
        assert len(trace) == 3 and all(row["gradient_l2"] > 0 for row in trace)
        assert all(np.isfinite(list(row.values())).all() for row in trace)
        check = report["arms"][key]["checkpoint"]
        assert check["step"] == 3 and check["changed_parameters"] > 0
        assert check["sha256"] == sha256_file(output / key / "backbone.pt")
    assert _read_trace(output / "A" / "train-loss.csv")[0]["optimization_phase_time"] == 0
    assert _read_trace(output / "B" / "train-loss.csv")[0]["optimization_phase_time"] > 0
    key = json.loads((output / "listening-key.json").read_bytes())
    assert {row["condition"] for row in key.values()} == {"A", "B", "reference"}
    for name, row in key.items():
        assert sha256_file(output / "blinded" / name) == row["sha256"]
    assert json.loads((output / "ablation-receipt.json").read_bytes()) == report


def test_deadline_crossing_finishes_admitted_pair_and_keeps_matched_last_step(
    mounted,
    tmp_path,
    monkeypatch,
):
    inputs, mount, _, pin = mounted
    output = tmp_path / "limited"
    now = [100.0]
    operations = []
    actual_update = phase_ablation._Arm.update

    def cross_deadline(arm, example):
        operations.append(arm.path.name)
        values = actual_update(arm, example)
        now[0] = 111.0
        return values

    monkeypatch.setattr(phase_ablation, "monotonic", lambda: now[0])
    monkeypatch.setattr(phase_ablation._Arm, "update", cross_deadline)
    report = phase_ablation.run_phase_ablation(
        inputs,
        mount,
        output,
        pin=pin,
        device="cpu",
        steps=(0, 3),
        original=_baseline(),
        code={},
        deadline_seconds=10,
        started_at=100,
    )
    assert operations == ["A", "B"]
    assert report["matched_updates"] == 1
    assert report["status"] == "budget_limited" and report["stop_reason"] == "wall_time"
    for arm in report["arms"].values():
        assert arm["checkpoint"]["step"] == 1
        assert [row["step"] for row in arm["snapshots"]] == [0, 1]


def test_record_stager_pins_schedule_budget_and_fails_closed_on_missing_mount(
    tmp_path, monkeypatch
):
    commit = "b" * 40
    monkeypatch.setattr(kaggle, "_git", lambda *_: SimpleNamespace(stdout=commit))
    monkeypatch.setattr("vocoroute.phase_diagnostic.baseline_record", lambda _: _baseline())

    def archive(mode, sha, paths):
        assert (mode, sha) == ("record", commit)
        return {name: (kaggle.REPO_ROOT / name).read_bytes() for name in paths}

    monkeypatch.setattr(kaggle, "_archive_bytes", archive)
    stage = tmp_path / "stage"
    staged = kaggle.stage_phase_kernel(commit, tmp_path / "pinned.json", stage)
    metadata = json.loads((stage / "kernel/kernel-metadata.json").read_bytes())
    plan = staged["plan"]
    protocol = plan["protocol"]
    old = diagnostic_config()
    assert protocol["loss"]["A"] == old["loss"]
    assert protocol["loss"]["B"] == old["loss"] | {"phase_enabled": True}
    assert protocol["optimizer"] == old["optimizer"] and protocol["seed"] == 20260907
    assert protocol["snapshot_steps"] == [0, 100, 250, 500, 1000, 2000, 4000]
    assert protocol["worker_training_deadline_seconds"] == 660
    assert protocol["session_timeout_seconds"] == 780 and protocol["planned_gpu_seconds"] == 900
    assert metadata["dataset_sources"] == [kaggle.DATASET["ref"]]
    assert metadata["is_private"] == metadata["enable_gpu"] == "true"
    assert metadata["enable_internet"] == metadata["enable_tpu"] == "false"
    assert metadata["machine_shape"] == "NvidiaTeslaT4" and metadata["kernel_sources"] == []
    script = stage / "kernel/phase_ablation.py"
    assert sha256_file(script) == staged["script_sha256"]
    (tmp_path / "input").mkdir()
    (tmp_path / "working").mkdir()
    probe = tmp_path / "probe.py"
    probe.write_text(
        script.read_text(encoding="utf-8")
        .replace(repr("/kaggle/input"), repr(str(tmp_path / "input")))
        .replace(repr("/kaggle/working"), repr(str(tmp_path / "working"))),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(probe)],
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    failure = json.loads(
        (tmp_path / "working" / kaggle.OUTPUT_ROOT / "ablation-receipt.json").read_bytes()
    )
    assert result.returncode != 0
    assert failure["status"] == "failed" and failure["matched_updates"] == 0
    assert "expected exactly one private single-utterance" in failure["error"]
