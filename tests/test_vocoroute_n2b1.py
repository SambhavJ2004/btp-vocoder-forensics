"""Synthetic proof of the single-utterance loader, metrics and training mechanics."""

import csv
import json
import subprocess
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from experiments import vocoroute_overfit_kaggle as kaggle  # noqa: E402
from experiments.kaggle_harness import REPO_ROOT  # noqa: E402
from vocoroute import overfit  # noqa: E402
from vocoroute.overfit import (  # noqa: E402
    N2B1_CORPUS,
    PCM16_MAX,
    _record_waveform,
    diagnostic_config,
    load_utterance,
    run_overfit,
    spectral_convergence,
)
from vocoroute.train_base import BaseTrainingModel  # noqa: E402
from vocoroute.training_config import TrainingConfig  # noqa: E402
from vocoroute.training_data import GeneratorRow, prepare_example, sha256_file  # noqa: E402
from vocoroute.vctk_prepare import write_json  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def mounted(tmp_path, request):
    inputs = tmp_path / "input"
    mount = inputs / kaggle.DATA_SLUG
    metadata = mount / "n2b0-train-dev" / "overfit"
    metadata.mkdir(parents=True)
    native = metadata / "p229_128_mic2.flac"
    wav = metadata / "p229_128.wav"
    n = getattr(request, "param", 1100)
    prepared_length = (n * 147 + 319) // 320
    soundfile.write(native, 0.11 * np.sin(np.arange(n) / 15), 48_000, subtype="PCM_16")
    soundfile.write(wav, 0.11 * np.sin(np.arange(prepared_length) / 7), 22_050, subtype="PCM_16")
    row = GeneratorRow(
        utterance_id="p229_128",
        speaker_id="p229",
        source_group="fixture/p229_128",
        role="train",
        native_path="native/train/p229/p229_128_mic2.flac",
        native_sha256=sha256_file(native),
        native_rate=48_000,
        native_samples=n,
        prepared_path="prepared/train/p229/p229_128.wav",
        prepared_sha256=sha256_file(wav),
        prepared_samples=prepared_length,
        resample_steps=1,
        resampling_profile="synthetic-test",
        resampling_receipt_sha256="f" * 64,
    )
    # These rows represent parent split metadata. Only the fixed TRAIN files exist.
    others = [
        replace(
            row,
            utterance_id=f"item{i}",
            speaker_id=f"sp{i}",
            source_group=f"group{i}",
            role={8: "dev", 9: "eval"}.get(i, "train"),
            native_path=f"native/{i}.flac",
            prepared_path=f"prepared/{i}.wav",
            native_sha256=f"{i:064x}",
            prepared_sha256=f"{100 + i:064x}",
        )
        for i in range(1, 10)
    ]
    pilot_sha = write_json(
        metadata / "parent-manifest.json",
        {
            "schema": "vocoroute-generator-manifest-v1",
            "source_kind": "native_generator",
            "target_rate": 22_050,
            "dataset_id": "synthetic-test-pilot",
            "rows": [asdict(r) for r in [row, *others]],
        },
    )
    diag_sha = write_json(
        metadata / "manifest.json",
        {
            "schema": "vocoroute-train-only-selection-v1",
            "parent_manifest_sha256": pilot_sha,
            "train_utterance_ids": [
                row.utterance_id,
                *(r.utterance_id for r in others if r.role == "train"),
            ],
        },
    )
    freeze_sha = write_json(
        metadata.parent / "freeze.json",
        {
            "schema": "vocoroute-vctk092-freeze-v1",
            "protocol": {"microphone": "mic2"},
            "manifests": {"overfit": diag_sha, "pilot": pilot_sha},
        },
    )
    pin = replace(
        N2B1_CORPUS,
        freeze_sha256=freeze_sha,
        selection_sha256=diag_sha,
        pilot_sha256=pilot_sha,
        native_sha256=row.native_sha256,
        prepared_sha256=row.prepared_sha256,
        samples=prepared_length,
    )
    return inputs, mount, wav, pin


def test_loader_validates_frozen_train_projection_and_rejects_extra_inputs(mounted):
    inputs, mount, wav, pin = mounted
    waveform, path, row = load_utterance(inputs, mount, pin)
    assert waveform.shape == (pin.samples,) and path == wav
    assert row.role == "train" and row.utterance_id == pin.utterance
    unexpected = inputs / "eval-index.json"
    unexpected.touch()
    with pytest.raises(ValueError, match="extra mounted diagnostic input"):
        load_utterance(inputs, mount, pin)
    unexpected.unlink()
    wav.write_bytes(wav.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="recording SHA-256 mismatch"):
        load_utterance(inputs, mount, pin)


def test_five_file_connector_may_flatten_the_bundle(mounted):
    inputs, mount, _, pin = mounted
    for file in list(mount.rglob("*")):
        if file.is_file():
            file.rename(mount / file.name)
    waveform, path, _ = load_utterance(inputs, mount, pin)
    assert path == mount / "p229_128.wav" and waveform.numel() == pin.samples


def test_rejects_freeze_tampering_before_audio(mounted, monkeypatch):
    inputs, mount, _, pin = mounted

    def forbidden(*_):
        pytest.fail("unexpected audio read")

    monkeypatch.setattr("vocoroute.overfit.TrainingAudio", forbidden)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_utterance(inputs, mount, replace(pin, freeze_sha256="a" * 64))


def test_snapshot_preserves_raw_and_counts_pcm16_export_clipping(tmp_path):
    path = tmp_path / "clipping.wav"
    raw = torch.tensor([[-1.2, -1.0, 0.25, PCM16_MAX, 1.3]])
    stats = _record_waveform(path, raw)
    assert stats["clipped_samples_for_pcm16"] == 2 and stats["nonfinite_samples"] == 0
    np.testing.assert_array_equal(np.load(path.with_suffix(".npy")), raw.numpy()[0])
    wav, rate = soundfile.read(path)
    np.testing.assert_array_equal(wav, [-1.0, -1.0, 0.25, PCM16_MAX, PCM16_MAX])
    assert rate == 22_050 and stats["peak_abs"] == pytest.approx(1.3)
    nonfinite = _record_waveform(tmp_path / "nonfinite.wav", torch.tensor([[float("nan"), 1.1]]))
    assert nonfinite["nonfinite_samples"] == nonfinite["clipped_samples_for_pcm16"] == 1
    assert nonfinite["wav"] is None and not (tmp_path / "nonfinite.wav").exists()


def test_tiny_train_only_run_preserves_metrics_and_checkpoint(mounted, tmp_path):
    inputs, mount, wav, pin = mounted
    output = tmp_path / "diagnostic"
    result = run_overfit(inputs, mount, output, pin=pin, device="cpu", steps=(0, 1, 3))
    assert result["status"] == "completed" and result["updates"] == 3
    assert result["finite_gradient_updates"] == 3 and result["parameter_count"] == 566_144
    assert result["config"]["role"] == "train_only"
    assert result["checkpoint"]["changed_parameters"] > 0
    assert sha256_file(output / "backbone.pt") == result["checkpoint"]["sha256"]
    assert (output / "reference.wav").read_bytes() == wav.read_bytes()
    assert [row["step"] for row in result["snapshots"]] == [0, 1, 3]
    for snapshot in result["snapshots"]:
        assert snapshot["samples"] == pin.samples and snapshot["nonfinite_samples"] == 0
        assert snapshot["waveform_mae"] == snapshot["waveform"]
        assert snapshot["log_magnitude_stft_error"] == snapshot["log_magnitude"]
        assert snapshot["phase_time"] == snapshot["phase_frequency"] == 0
        assert np.isfinite(snapshot["spectral_convergence"])
        assert snapshot["raw_sha256"] == sha256_file(output / snapshot["raw"])
    with (output / "train-loss.csv").open(newline="", encoding="utf-8") as csv_file:
        history = list(csv.DictReader(csv_file))
    assert len(history) == 3 and all(float(r["gradient_l2"]) > 0 for r in history)
    model = BaseTrainingModel(TrainingConfig())
    model.backbone.load_state_dict(torch.load(output / "backbone.pt", weights_only=True))
    waveform, _, _ = load_utterance(inputs, mount, pin)
    example = prepare_example(model.frontend, waveform)
    with torch.no_grad():
        generated = model.synthesize(example).numpy()[0]
        assert spectral_convergence(model, example.target, example.target) == 0
    np.testing.assert_allclose(generated, np.load(output / "step-0003.npy"), rtol=1e-6, atol=1e-7)
    assert json.loads((output / "diagnostic-receipt.json").read_bytes()) == result


@pytest.mark.parametrize("mounted", [166_580], indirect=True)
def test_full_target_length_synthetic_forward_backward(mounted, tmp_path):
    inputs, mount, _, pin = mounted
    assert pin.samples == N2B1_CORPUS.samples
    report = run_overfit(
        inputs, mount, tmp_path / "full-length", pin=pin, device="cpu", steps=(0, 1)
    )
    assert report["status"] == "completed" and report["finite_gradient_updates"] == 1
    assert report["checkpoint"]["changed_parameters"] > 0
    assert [row["samples"] for row in report["snapshots"]] == [pin.samples, pin.samples]


def test_budget_stops_before_first_optimizer_step(mounted, tmp_path, monkeypatch):
    inputs, mount, _, pin = mounted

    def forbidden(*_):
        pytest.fail("update admitted after the deadline")

    monkeypatch.setattr("vocoroute.overfit._step", forbidden)
    monkeypatch.setattr("vocoroute.overfit.monotonic", lambda: 12.0)
    receipt = run_overfit(
        inputs,
        mount,
        tmp_path / "limited",
        pin=pin,
        device="cpu",
        steps=(0, 1),
        wall_seconds=1,
        worker_started_at=10.0,
    )
    assert receipt["status"] == "budget_limited" and receipt["stop_reason"] == "wall_time"
    assert receipt["updates"] == receipt["checkpoint"]["changed_parameters"] == 0
    assert len(receipt["snapshots"]) == 1


def test_budget_preserves_last_step_between_milestones(mounted, tmp_path, monkeypatch):
    inputs, mount, _, pin = mounted
    now = [10.0]
    update = overfit._step

    def cross_deadline(*args):
        result = update(*args)
        now[0] = 12.0
        return result

    monkeypatch.setattr(overfit, "monotonic", lambda: now[0])
    monkeypatch.setattr(overfit, "_step", cross_deadline)
    receipt = run_overfit(
        inputs,
        mount,
        tmp_path / "limited",
        pin=pin,
        device="cpu",
        steps=(0, 3),
        wall_seconds=1,
        worker_started_at=10.0,
    )
    assert receipt["status"] == "budget_limited" and receipt["updates"] == 1
    assert [row["step"] for row in receipt["snapshots"]] == [0, 1]
    assert receipt["checkpoint"]["step"] == 1


@pytest.mark.parametrize("names", [("Tesla T4",), ("Tesla T4", "NVIDIA L4")])
def test_runtime_rejects_unapproved_cuda_shape(names, monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(names))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(
            name=names[index], total_memory=16_000_000_000, major=7, minor=5
        ),
    )
    with pytest.raises(RuntimeError, match="expected T4 x2"):
        overfit._runtime(torch.device("cuda:0"))


def test_record_stage_only_attaches_private_diagnostic_dataset(tmp_path, monkeypatch):
    commit = "b" * 40
    monkeypatch.setattr(kaggle, "_git", lambda *_: SimpleNamespace(stdout=commit))

    def archived(mode, sha, requested):
        assert mode == "record" and sha == commit
        return {name: (REPO_ROOT / name).read_bytes() for name in requested}

    monkeypatch.setattr(kaggle, "_archive_bytes", archived)
    stage = tmp_path / "record"
    result = kaggle.stage_overfit_kernel(commit, stage)
    metadata = json.loads((stage / "kernel/kernel-metadata.json").read_bytes())
    assert metadata["dataset_sources"] == [f"{kaggle.OWNER}/{kaggle.DATA_SLUG}"]
    assert metadata["is_private"] == metadata["enable_gpu"] == "true"
    assert metadata["enable_internet"] == metadata["enable_tpu"] == "false"
    assert metadata["machine_shape"] == "NvidiaTeslaT4" and metadata["kernel_sources"] == []
    assert result["plan"]["config"] == diagnostic_config()
    assert result["plan"]["session_timeout_seconds"] == 1080
    assert result["plan"]["config"]["worker_training_deadline_seconds"] == 840
    assert result["plan"]["corpus"]["notebook"].endswith("/1")
    script = stage / "kernel/overfit.py"
    assert sha256_file(script) == result["script_sha256"]
    compile(script.read_bytes(), str(script), "exec")
    empty = tmp_path / "empty-input"
    working = tmp_path / "working"
    empty.mkdir()
    working.mkdir()
    probe = tmp_path / "probe.py"
    source = script.read_text(encoding="utf-8")
    probe.write_text(
        source.replace(repr("/kaggle/input"), repr(str(empty))).replace(
            repr("/kaggle/working"), repr(str(working))
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(probe)],
        capture_output=True,
        text=True,
        timeout=40,
        check=True,
    )
    assert "expected exactly one private single-utterance Kaggle mount" in result.stdout
    failure = json.loads((working / "vocoroute-n2b1/diagnostic-receipt.json").read_bytes())
    assert failure["status"] == "failed" and failure["updates"] == 0
