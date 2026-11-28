"""Synthetic CPU resume, budget and offline Kaggle staging checks."""

import hashlib
import json
import math
import random
import signal
import struct
import subprocess
import sys
import wave
import zipfile
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from experiments import vocoroute_kaggle as kaggle  # noqa: E402
from experiments.kaggle_harness import SNAPSHOT_ROOT_IN_ZIP, Snapshot, _tree_sha256  # noqa: E402
from vocoroute import train_base  # noqa: E402
from vocoroute.training_checkpoint import load_base_checkpoint  # noqa: E402
from vocoroute.training_config import (  # noqa: E402
    BUDGET_CONFIG_SCHEMA,
    TrainingConfig,
    canonical_json,
)
from vocoroute.training_data import MANIFEST_SCHEMA  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def put_json(path, value):
    path.write_bytes(canonical_json(value))
    return path


def pcm(index):
    samples = [int(3500 * math.sin(i * (0.12 + index * 0.015))) for i in range(1025 + index)]
    stream = BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setparams((1, 2, 22_050, len(samples), "NONE", "not compressed"))
        writer.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return stream.getvalue(), len(samples)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    rows = []
    for index, role in enumerate(("train", "train", "train", "dev", "dev", "eval")):
        payload, length = pcm(index)
        name = f"example-{index}.wav"
        if role != "eval":
            (root / name).write_bytes(payload)
        rows.append(
            {
                "utterance_id": str(index),
                "speaker_id": role,
                "source_group": str(index),
                "role": role,
                "native_path": name,
                "native_sha256": digest(payload),
                "native_rate": 22_050,
                "native_samples": length,
                "prepared_path": name,
                "prepared_sha256": digest(payload),
                "prepared_samples": length,
                "resample_steps": 0,
                "resampling_profile": "identity",
                "resampling_receipt_sha256": None,
            }
        )
    doc = {
        "schema": MANIFEST_SCHEMA,
        "dataset_id": "synthetic-n2k",
        "rows": rows,
        "source_kind": "native_generator",
        "target_rate": 22_050,
    }
    manifest = put_json(root / "manifest.json", doc)
    cfg = TrainingConfig(
        schema=BUDGET_CONFIG_SCHEMA,
        epochs=2,
        max_updates=6,
        wall_time_seconds=120,
        checkpoint_every_updates=2,
        manifest_sha256=digest(manifest.read_bytes()),
    )
    return root, manifest, cfg


def args(corpus, output, *extra):
    root, manifest, config = corpus
    conf = put_json(output.with_suffix(".json"), config.document())
    return [
        "--train",
        "--device",
        "cpu",
        "--manifest",
        str(manifest),
        "--data-root",
        str(root),
        "--config",
        str(conf),
        "--out-dir",
        str(output),
        *extra,
    ]


def run(corpus, path, *extra):
    assert train_base.main(args(corpus, path, *extra)) == 0
    return json.loads((path / "run.json").read_bytes())


def resume(path, report):
    latest = report["latest_resume"]
    return [
        "--resume",
        str(path / latest["directory"]),
        "--resume-receipt-sha256",
        latest["receipt_sha256"],
    ]


def state(path, report):
    return torch.load(
        path / report["latest_resume"]["directory"] / "training.pt",
        map_location="cpu",
        weights_only=True,
    )


def same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    else:
        assert left == right


def test_cuda_fails_closed_and_v1_config_is_still_identical(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="2"):
        train_base.main(["--dry-run", "--device", "cuda"])
    assert "CUDA was requested" in capsys.readouterr().err
    assert TrainingConfig().sha256 == (
        "c3bc3c704e81eca632eca39a0e70fdc217e0516ee759f4c3919355b7ce60ee9d"
    )
    assert train_base.main(["--dry-run", "--device", "cpu"]) == 0
    assert json.loads(capsys.readouterr().out)["device"] == "cpu"


def test_interrupted_resume_matches_uninterrupted_including_rng(corpus, tmp_path, monkeypatch):
    original = train_base._train_example
    draws = []

    def observe(model, optimizer, waveform, uid):
        draws.append((uid, random.random(), torch.rand(1).item()))
        return original(model, optimizer, waveform, uid)

    monkeypatch.setattr(train_base, "_train_example", observe)
    continuous = tmp_path / "continuous"
    expected = run(corpus, continuous)
    sequence = draws.copy()
    draws.clear()
    partial = tmp_path / "partial"
    paused = run(corpus, partial, "--attempt-updates", "4")
    assert paused["status"] == "budget_limited" and paused["reason"] == "attempt_updates"
    assert paused["updates"] == 4 and paused["next_train_row"] == 1
    assert len(list(partial.glob("resume-*"))) >= 2
    assert not (partial / "inference").exists() and not paused["convergence_claimed"]
    resumed = tmp_path / "resumed"
    actual = run(corpus, resumed, *resume(partial, paused))
    assert actual["status"] == "completed" and actual["updates"] == 6
    assert actual["history"] == expected["history"] and draws == sequence
    same(state(continuous, expected), state(resumed, actual))
    _, receipt = load_base_checkpoint(
        resumed / "inference", corpus[2], receipt_sha256=actual["checkpoint_receipt_sha256"]
    )
    assert receipt["best_epoch"] == expected["best_epoch"]


def test_update_cap_is_total_and_does_not_publish_full_inference(corpus, tmp_path, capsys):
    limited = (*corpus[:2], replace(corpus[2], max_updates=1))
    path = tmp_path / "limited"
    result = run(limited, path)
    assert (result["status"], result["reason"], result["updates"]) == (
        "budget_limited",
        "max_updates",
        1,
    )
    assert not (path / "inference").exists()
    with pytest.raises(SystemExit, match="2"):
        train_base.main(args(limited, tmp_path / "blocked", *resume(path, result)))
    assert "already exhausted" in capsys.readouterr().err


def test_wall_budget_resumes_inside_dev_without_replaying_train(corpus, tmp_path, monkeypatch):
    now = 0.0
    original = train_base.BaseTrainingModel.forward
    stop_once = True

    def forward(self, waveform):
        nonlocal now, stop_once
        output = original(self, waveform)
        if not self.training and stop_once:
            now += 6
            stop_once = False
        return output

    monkeypatch.setattr(train_base, "time", SimpleNamespace(monotonic=lambda: now))
    monkeypatch.setattr(train_base.BaseTrainingModel, "forward", forward)
    corpus = (*corpus[:2], replace(corpus[2], wall_time_seconds=5))
    path = tmp_path / "wall"
    result = run(corpus, path)
    assert result["reason"] == "wall_time" and result["updates"] == 3
    assert result["next_train_row"] == 3 and result["next_dev_row"] == 1
    actual = run(corpus, tmp_path / "continued", *resume(path, result))
    assert actual["attempt_start_updates"] == 3 and actual["status"] == "completed"


def test_resume_refuses_changed_config_manifest_and_payload(corpus, tmp_path, capsys):
    partial = tmp_path / "partial"
    paused = run(corpus, partial, "--attempt-updates", "1")
    extra = resume(partial, paused)
    doc = json.loads(corpus[1].read_bytes())
    doc["dataset_id"] = "different-manifest"
    manifest = put_json(corpus[0] / "new-manifest.json", doc)
    for index, changed in enumerate(
        (
            (*corpus[:2], replace(corpus[2], seed=41)),
            (
                corpus[0],
                manifest,
                replace(corpus[2], manifest_sha256=digest(manifest.read_bytes())),
            ),
        )
    ):
        with pytest.raises(SystemExit, match="2"):
            train_base.main(args(changed, tmp_path / f"changed-{index}", *extra))
        assert "resume config, manifest, code or runtime identity disagrees" in (
            capsys.readouterr().err
        )
    payload = Path(extra[1]) / "training.pt"
    payload.write_bytes(payload.read_bytes() + b"changed")
    with pytest.raises(SystemExit, match="2"):
        train_base.main(args(corpus, tmp_path / "tampered", *extra))
    assert "training state SHA-256 mismatch" in capsys.readouterr().err


def test_signal_saves_safe_boundary_and_failure_keeps_last_pin(corpus, tmp_path, monkeypatch):
    corpus = (*corpus[:2], replace(corpus[2], checkpoint_every_updates=1))
    train = train_base._train_example

    def ask_to_stop(*args):
        value = train(*args)
        signal.raise_signal(signal.SIGTERM)
        return value

    monkeypatch.setattr(train_base, "_train_example", ask_to_stop)
    interrupted = tmp_path / "interrupted"
    result = run(corpus, interrupted)
    assert (result["status"], result["reason"], result["updates"]) == ("interrupted", "signal", 1)
    assert state(interrupted, result)["progress"]["updates"] == 1

    def fail_before_update(*_args):
        raise ValueError("injected training error")

    monkeypatch.setattr(train_base, "_train_example", fail_before_update)
    output = tmp_path / "failure"
    with pytest.raises(SystemExit, match="2"):
        train_base.main(args(corpus, output, *resume(interrupted, result)))
    report = json.loads((output / "run.json").read_bytes())
    assert (
        report["status"] == "failed"
        and report["resume_input_sha256"] == (result["latest_resume"]["receipt_sha256"])
    )
    assert state(interrupted, result)["progress"]["updates"] == 1


def spec_document():
    return {
        "schema": kaggle.SPEC_SCHEMA,
        "title": "VocoRoute N2K Test",
        "resume": None,
        "code": {"ref": "owner/vocoroute-code", "version": 4, "mount": "mounted/code"},
        "data": {
            "ref": "owner/vocoroute-train",
            "version": 7,
            "mount": "mounted/train",
            "receipt_sha256": "1" * 64,
        },
        "runtime": dict.fromkeys(kaggle._RUNTIME_KEYS, "test-pin"),
    }


def test_local_stage_checks_mount_before_import(corpus, tmp_path, monkeypatch):
    files = {
        "pyproject.toml": b"# fixture",
        "src/experiments/__init__.py": b"",
        "src/vocoroute/training_resume.py": b"",
        "src/experiments/vocoroute_kaggle.py": (
            b"def run_kernel_plan(*args):\n    return {'status': 'stub-verified'}\n"
        ),
    }

    def snapshot(path, *, mode, sha):
        assert mode == "record" and sha == "c9a62e2"
        path.mkdir()
        record = Snapshot(mode, sha + "0" * 33, _tree_sha256(files), 4, 100, "test")
        with zipfile.ZipFile(path / record.archive, "w") as archive:
            for name, payload in files.items():
                archive.writestr(f"{SNAPSHOT_ROOT_IN_ZIP}/{name}", payload)
        put_json(path / "btpvf_snapshot.json", record.as_record())
        return record

    monkeypatch.setattr(kaggle, "stage_snapshot", snapshot)
    spec = put_json(tmp_path / "spec.json", spec_document())
    conf = put_json(tmp_path / "config.json", replace(corpus[2], device="cuda").document())
    staged = tmp_path / "plan"
    plan = kaggle.stage_vocoroute_plan("c9a62e2", spec, conf, staged)
    meta = json.loads((staged / "kernel" / "kernel-metadata.json").read_bytes())
    assert meta["is_private"] == "true" and meta["enable_internet"] == "false"
    assert meta["dataset_sources"] == ["owner/vocoroute-code", "owner/vocoroute-train"]
    assert plan["spec"]["data"]["version"] == 7
    inputs = tmp_path / "inputs"
    (inputs / "mounted").mkdir(parents=True)
    (staged / "code").rename(inputs / "mounted" / "code")
    source = (staged / "kernel" / "train.py").read_text(encoding="utf-8")
    local = source.replace('Path("/kaggle/input")', f"Path({str(inputs)!r})")
    for label, expected in (("valid", "stub-verified"), ("tampered", "failed")):
        output = tmp_path / label
        job = tmp_path / f"{label}.py"
        job.write_text(
            local.replace('Path("/kaggle/working/vocoroute-run")', f"Path({str(output)!r})"),
            encoding="utf-8",
        )
        subprocess.run([sys.executable, "-I", "-B", str(job)], check=True, capture_output=True)
        receipt = json.loads((output / "kernel-receipt.json").read_bytes())
        assert receipt["status"] == expected
        if label == "valid":
            archive = inputs / "mounted" / "code" / plan["snapshot"]["archive"]
            archive.write_bytes(archive.read_bytes() + b"tamper")
    assert (tmp_path / "tampered" / "failure.log").is_file()


def test_mounted_dataset_pin_and_no_eval_audio(corpus, tmp_path):
    root, manifest, config = corpus
    spec = spec_document()
    spec["data"]["mount"] = root.name
    receipt = {
        "schema": kaggle.DATASET_SCHEMA,
        "ref": spec["data"]["ref"],
        "version": 7,
        "manifest_sha256": config.manifest_sha256,
        "audio_roles": ["train", "dev"],
    }
    path = put_json(root / "dataset-receipt.json", receipt)
    spec["data"]["receipt_sha256"] = digest(path.read_bytes())
    plan = {"spec": spec, "config": config.document()}
    assert kaggle._dataset_manifest(plan, tmp_path) == (root.resolve(), manifest.resolve())
    put_json(path, {**receipt, "version": 8})
    with pytest.raises(ValueError, match="receipt SHA-256 mismatch"):
        kaggle._dataset_manifest(plan, tmp_path)
    put_json(path, receipt)
    (root / "example-5.wav").write_bytes(b"must not be opened")
    with pytest.raises(ValueError, match="held-out evaluation audio"):
        kaggle._dataset_manifest(plan, tmp_path)


def test_pilot_wall_stop_during_dev_and_resume(corpus, tmp_path, monkeypatch):
    from vocoroute import pilot_diagnostics as pilot
    from vocoroute.training_config import LossWeights
    from vocoroute.training_data import TrainingAudio, load_manifest

    root, manifest_path, config = corpus
    config = replace(config, loss=LossWeights(phase_enabled=True), scheduler_gamma=1.0)
    manifest = load_manifest(manifest_path, config.manifest_sha256)
    audio = TrainingAudio(manifest, root)
    panel = pilot.panel_ids(manifest, config.seed, per_speaker=2)
    protocol = {"panel_utterance_ids": panel, "panel_epochs": [1, 2]}
    clock = SimpleNamespace(now=0.0, enabled=False, rows=0)
    monkeypatch.setattr(train_base, "time", SimpleNamespace(monotonic=lambda: clock.now))
    evaluate = pilot.PilotDiagnostics.evaluate

    def evaluate_checked(self, model, row, waveform):
        assert row.role == "dev" and not model.training and not torch.is_grad_enabled()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        result = evaluate(self, model, row, waveform)
        same(before, model.state_dict())
        same(rng, torch.get_rng_state())
        if clock.enabled:
            clock.rows += 1
            if clock.rows == 3:
                clock.now = config.wall_time_seconds
        return result

    monkeypatch.setattr(pilot.PilotDiagnostics, "evaluate", evaluate_checked)

    def run_pilot(label, resume_from=None, document=protocol):
        return train_base._run_budgeted(
            config, manifest, audio, tmp_path / label, torch.device("cpu"),
            pilot_protocol=document, resume=resume_from, attempt_updates=None,
        )

    full = run_pilot("pilot-full")
    clock.enabled = True
    partial = run_pilot("pilot-partial")
    assert (partial["status"], partial["reason"]) == ("budget_limited", "wall_time")
    assert (partial["completed_epochs"], partial["updates"], partial["next_dev_row"]) == (1, 6, 1)
    assert not partial["convergence_claimed"]
    partial_root = tmp_path / "pilot-partial"
    _, best = load_base_checkpoint(
        partial_root / "inference", config, receipt_sha256=partial["checkpoint_receipt_sha256"]
    )
    assert best["best_epoch"] == 1 and len(best["history"]) == 1
    assert best["run"]["pilot_protocol_sha256"] == pilot.protocol_digest(protocol)
    current = partial["latest_resume"]
    resume_from = (partial_root / current["directory"], current["receipt_sha256"])

    clock.enabled, clock.now = False, 0.0
    with pytest.raises(ValueError, match="identity disagrees"):
        run_pilot("pilot-bad-resume", resume_from, {**protocol, "panel_epochs": [2]})
    recovered = run_pilot("pilot-resumed", resume_from)
    assert (recovered["status"], recovered["updates"]) == ("completed", 6)
    same(state(tmp_path / "pilot-full", full), state(tmp_path / "pilot-resumed", recovered))
    merged = tmp_path / "pilot-resumed" / "pilot-metrics.json"
    metrics = json.loads(merged.read_bytes())
    full_metrics = json.loads((tmp_path / "pilot-full" / merged.name).read_bytes())
    attempt_keys = metrics.pop("listening_keys_sha256")
    full_keys = full_metrics.pop("listening_keys_sha256")
    assert metrics == full_metrics
    assert attempt_keys == {key: full_keys[key] for key in attempt_keys}
    assert metrics["train_utterance_exposures"] == dict.fromkeys(("0", "1", "2"), 2)
    assert metrics["epochs"][0]["dev_by_speaker"]["dev"]["utterances"] == 2
    assert metrics["epochs"][0]["dev_macro"]["circular_phase_error"] > 0

    listening = partial_root / "listening"
    assert sorted(path.name for path in listening.iterdir()) == [
        "best", "best-key.json", "epoch-01", "epoch-01-key.json",
        "initialization", "initialization-key.json", "raw", "reference",
    ]
    key = json.loads((listening / "initialization-key.json").read_bytes())
    assert len(list((listening / "initialization").iterdir())) == 4
    assert sum(name.endswith("-a.wav") and row["condition"] == "reference"
               for name, row in key.items()) == 1
    for name, row in key.items():
        assert digest((listening / "initialization" / name).read_bytes()) == row["sha256"]
        if row["condition"] == "reference":
            reference = root / f"example-{row['utterance_id']}.wav"
            assert row["sha256"] == digest(reference.read_bytes())


def test_pilot_attachment_inventory_and_record_staging(corpus, tmp_path, monkeypatch):
    from vocoroute import pilot_diagnostics as pilot
    from vocoroute.training_data import load_manifest

    root, manifest_file, config = corpus
    manifest = load_manifest(manifest_file, config.manifest_sha256)
    receipt = b"fixture metadata"
    monkeypatch.setattr(pilot, "PILOT_METADATA", {"dataset-receipt.json": digest(receipt)})
    (root / "dataset-receipt.json").write_bytes(receipt)
    manifest_file.unlink()
    pilot.validate_attachment(root, manifest)
    (root / "forbidden.wav").write_bytes(b"extra")
    with pytest.raises(ValueError, match="exact train/dev projection"):
        pilot.validate_attachment(root, manifest)
    (root / "forbidden.wav").unlink()
    (root / "dataset-receipt.json").write_bytes(b"wrong metadata")
    with pytest.raises(ValueError, match="metadata SHA-256 mismatch"):
        pilot.validate_attachment(root, manifest)

    spec = spec_document()
    spec["data"].update(ref=pilot.PILOT_DATA_REF, version=1, receipt_sha256=digest(receipt))
    cuda = replace(config, device="cuda")
    protocol = {"schema": pilot.PILOT_SCHEMA, "training_config_sha256": cuda.sha256}
    spec["pilot"] = {"protocol": protocol, "sha256": pilot.protocol_digest(protocol)}
    conf_path = put_json(tmp_path / "config.json", cuda.document())
    spec_path = put_json(tmp_path / "spec.json", spec)

    def snapshot(path, *, mode, sha):
        assert mode == "record" and sha == "1234567"
        path.mkdir()
        files = {
            "src/experiments/vocoroute_kaggle.py": b"",
            "src/vocoroute/training_resume.py": b"",
            "src/vocoroute/pilot_diagnostics.py": b"",
        }
        record = Snapshot(mode, sha + "0" * 33, _tree_sha256(files), len(files), 0, "test")
        with zipfile.ZipFile(path / record.archive, "w") as archive:
            for name, payload in files.items():
                archive.writestr(f"{SNAPSHOT_ROOT_IN_ZIP}/{name}", payload)
        put_json(path / "btpvf_snapshot.json", record.as_record())
        return record

    monkeypatch.setattr(kaggle, "stage_snapshot", snapshot)
    staged = tmp_path / "staged"
    plan = kaggle.stage_vocoroute_plan("1234567", spec_path, conf_path, staged)
    metadata = json.loads((staged / "kernel/kernel-metadata.json").read_bytes())
    assert metadata["machine_shape"] == "NvidiaTeslaT4" and metadata["is_private"] == "true"
    assert metadata["enable_internet"] == "false"
    assert metadata["dataset_sources"] == [spec["code"]["ref"], spec["data"]["ref"]]
    assert plan["spec"]["pilot"] == spec["pilot"]
    inputs = tmp_path / "input"
    (inputs / spec["data"]["mount"]).mkdir(parents=True)
    (staged / "code").rename(inputs / spec["code"]["mount"])
    kaggle._verify_pilot_mounts(spec, inputs, plan)
    extra = inputs / "raw"
    extra.mkdir()
    with pytest.raises(ValueError, match="unapproved pilot input mount"):
        kaggle._verify_pilot_mounts(spec, inputs, plan)
    extra.rmdir()
    (inputs / spec["code"]["mount"] / "some-audio.wav").write_bytes(b"extra")
    with pytest.raises(ValueError, match="alongside the pilot code snapshot"):
        kaggle._verify_pilot_mounts(spec, inputs, plan)


def test_pilot_attachment_projection_copies_only_verified_files(corpus, tmp_path, monkeypatch):
    from vocoroute import pilot_diagnostics as pilot

    root, manifest_file, config = corpus
    source = tmp_path / "saved-n2b0"
    source.mkdir()
    root.rename(source / "pilot")
    freeze = source / "freeze.json"
    freeze.write_bytes(b"frozen receipt")
    monkeypatch.setattr(pilot, "PILOT_MANIFEST", config.manifest_sha256)
    monkeypatch.setattr(pilot, "PILOT_METADATA", {
        manifest_file.name: config.manifest_sha256,
        freeze.name: digest(freeze.read_bytes()),
    })
    forbidden = source / "pilot" / "example-5.wav"
    forbidden.write_bytes(b"held-out placeholder")
    output = tmp_path / "projection"
    receipt = pilot.prepare_attachment(source, output)
    assert receipt["files"] == 7
    assert not (output / forbidden.name).exists()
    assert forbidden.read_bytes() == b"held-out placeholder"
    assert receipt["bytes"] == sum(path.stat().st_size for path in output.iterdir())
    freeze.write_bytes(b"tampered")
    refused = tmp_path / "refused"
    with pytest.raises(ValueError, match="saved pilot output SHA-256 mismatch"):
        pilot.prepare_attachment(source, refused)
    assert not refused.exists()
