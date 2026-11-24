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
