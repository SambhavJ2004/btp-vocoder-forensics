"""Synthetic checks for the frozen eight-utterance diagnostic and RECORD stage."""

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from experiments import vocoroute_overfit_eight_kaggle as kaggle  # noqa: E402
from vocoroute import overfit_eight as eight  # noqa: E402
from vocoroute.phase_ablation import _config  # noqa: E402
from vocoroute.train_base import BaseTrainingModel  # noqa: E402
from vocoroute.training_data import GeneratorRow, prepare_example, sha256_file  # noqa: E402
from vocoroute.vctk_prepare import RESAMPLE_PROFILE, write_json  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _pin_selection(mount, ids, pilot_sha, resample_sha, license_sha):
    selection = write_json(
        mount / "overfit/manifest.json",
        {
            "schema": "vocoroute-train-only-selection-v1",
            "parent_manifest_sha256": pilot_sha,
            "train_utterance_ids": list(ids),
        },
    )
    freeze = write_json(
        mount / "freeze.json",
        {
            "schema": "vocoroute-vctk092-freeze-v1",
            "protocol": {"microphone": "mic2"},
            "manifests": {"overfit": selection, "pilot": pilot_sha},
            "resampling_receipt_sha256": resample_sha,
        },
    )
    return replace(
        eight.N2B2_CORPUS,
        freeze_sha256=freeze,
        selection_sha256=selection,
        pilot_sha256=pilot_sha,
        resampling_sha256=resample_sha,
        license_sha256=license_sha,
    )


@pytest.fixture
def corpus(tmp_path):
    inputs = tmp_path / "input"
    mount = inputs / kaggle.DATASET_SLUG
    source = mount / "overfit/source"
    source.mkdir(parents=True)
    license_path = source / "license_text.txt"
    license_path.write_text("synthetic fixture attribution", encoding="utf-8")
    resample = write_json(source / "resampling-receipt.json", {"schema": RESAMPLE_PROFILE})
    rows = []
    for i in range(10):
        speaker = f"p{200 + i}"
        uid = f"{speaker}_001"
        role = "train" if i < 8 else ("dev" if i == 8 else "eval")
        native = f"native/{role}/{speaker}/{uid}_mic2.flac"
        prepared = f"prepared/{role}/{speaker}/{uid}.wav"
        length = 2048 + i * 23
        samples = (length * 147 + 319) // 320
        if role == "train":
            for relative, size, rate in ((native, length, 48000), (prepared, samples, 22050)):
                path = mount / "overfit" / relative
                path.parent.mkdir(parents=True)
                wave = 0.1 * np.sin(np.arange(size) * (400 + i * 15) * 2 * np.pi / rate)
                soundfile.write(path, wave, rate, subtype="PCM_16")
            native_sha, prepared_sha = (
                sha256_file(mount / "overfit" / r) for r in (native, prepared)
            )
        else:
            native_sha, prepared_sha = (f"{value:064x}" for value in (100 + i, 200 + i))
        rows.append(
            GeneratorRow(
                uid,
                speaker,
                f"vctk-0.92/{uid}",
                role,
                native,
                native_sha,
                48000,
                length,
                prepared,
                prepared_sha,
                samples,
                1,
                RESAMPLE_PROFILE,
                resample,
            )
        )
    parent = write_json(
        mount / "overfit/parent-manifest.json",
        {
            "schema": "vocoroute-generator-manifest-v1",
            "source_kind": "native_generator",
            "dataset_id": "synthetic-eight-fixture",
            "target_rate": 22050,
            "rows": [asdict(row) for row in rows],
        },
    )
    pin = _pin_selection(
        mount, [row.utterance_id for row in rows[:8]], parent, resample, sha256_file(license_path)
    )
    return inputs, mount, pin, rows


def test_inventory_and_flattened_mount_have_the_same_exact_recordings(corpus):
    inputs, mount, pin, source_rows = corpus
    inventory = eight.source_inventory(mount, pin)
    assert len(inventory["files"]) == 21
    assert sum(name.endswith(".wav") for name in inventory["files"]) == 8
    for path in tuple(mount.rglob("*")):
        if path.is_file() and path.parent != mount:
            path.rename(mount / path.name)
    loader, rows = eight.load_diagnostic(inputs, mount, pin, inventory=inventory)
    assert [row.utterance_id for row in rows] == [row.utterance_id for row in source_rows[:8]]
    for row in rows:
        assert loader.waveform(row).shape == (row.prepared_samples,)


@pytest.mark.parametrize("path", ["extra.wav", "router/other.flac", "eval/other.wav"])
def test_refuses_extra_audio_before_the_audio_loader(corpus, monkeypatch, path):
    inputs, mount, pin, _ = corpus
    extra = mount / path
    extra.parent.mkdir(exist_ok=True)
    extra.touch()
    monkeypatch.setattr(eight, "TrainingAudio", lambda *_: pytest.fail("opened audio"))
    with pytest.raises(ValueError, match="16 approved audio files"):
        eight.load_diagnostic(inputs, mount, pin)


@pytest.mark.parametrize("kind", ["native_path", "prepared_path"])
def test_refuses_changed_recording_bytes(corpus, kind):
    inputs, mount, pin, rows = corpus
    path = mount / "overfit" / getattr(rows[1], kind)
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="recording SHA-256 mismatch"):
        eight.load_diagnostic(inputs, mount, pin)


def test_refuses_dev_in_selection_and_modified_record_inventory(corpus):
    inputs, mount, pin, rows = corpus
    draft = eight.source_inventory(mount, pin)
    draft["files"].pop("freeze.json")
    with pytest.raises(ValueError, match="differs from the RECORD plan inventory"):
        eight.load_diagnostic(inputs, mount, pin, inventory=draft)
    new_pin = _pin_selection(
        mount,
        [row.utterance_id for row in rows[1:9]],
        pin.pilot_sha256,
        pin.resampling_sha256,
        pin.license_sha256,
    )
    with pytest.raises(ValueError, match="unique subset of parent TRAIN"):
        eight.load_diagnostic(inputs, mount, new_pin)


def test_full_seeded_schedule_has_exact_balanced_exposure(corpus):
    _, _, _, rows = corpus
    ids = tuple(row.utterance_id for row in rows[:8])
    seed = eight.protocol()["seed"]
    observed = Counter()
    for index in range(2000):
        order = eight.round_order(ids, seed, index)
        assert set(order) == set(ids)
        assert order == eight.round_order(ids[::-1], seed, index)
        observed.update(order)
    assert observed == {uid: 2000 for uid in ids}
    assert {eight.round_order(ids, seed, i) for i in range(3)} != {
        eight.round_order(ids, seed + 1, i) for i in range(3)
    }


def test_phase_terms_backpropagate_through_synthesis(corpus):
    inputs, mount, pin, _ = corpus
    loader, rows = eight.load_diagnostic(inputs, mount, pin)
    model = BaseTrainingModel(_config(True))
    example = prepare_example(model.frontend, loader.waveform(rows[0]))
    losses = model.loss(model.synthesize(example), example.target)
    assert losses.phase_time > 0 and losses.phase_frequency > 0
    for loss in (losses.phase_time, losses.phase_frequency):
        grads = torch.autograd.grad(loss, tuple(model.backbone.parameters()), retain_graph=True)
        assert all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) > 0 for g in grads)


def test_snapshot_schedule_preserves_updates_and_checkpoint_reload(corpus, tmp_path):
    inputs, mount, pin, rows = corpus
    paths = [tmp_path / "diagnostic", tmp_path / "without-middle-snapshot"]
    options = {"pin": pin, "device": "cpu", "code": {}}
    detailed, sparse = [
        eight.run_diagnostic(inputs, mount, path, steps=steps, **options)
        for path, steps in zip(paths, ((0, 8, 16), (0, 16)), strict=True)
    ]
    assert detailed["status"] == "completed" and detailed["completed_rounds"] == 2
    for label in ("utterance_exposure", "speaker_exposure"):
        assert len(detailed[label]) == 8 and set(detailed[label].values()) == {2}
    assert (
        detailed["checkpoint"]["final_flat_fp32_sha256"]
        == sparse["checkpoint"]["final_flat_fp32_sha256"]
    )
    assert detailed["actual_order_sha256"] == sparse["actual_order_sha256"]
    assert detailed["checkpoint"]["changed_parameters"] > 0
    uid_set = {row.utterance_id for row in rows[:8]}
    assert set(detailed["reload"]) == uid_set
    assert all(row["max_abs"] <= 2e-6 for row in detailed["reload"].values())
    assert [snapshot["step"] for snapshot in detailed["snapshots"]] == [0, 8, 16]
    for snapshot in detailed["snapshots"]:
        recordings = snapshot["recordings"]
        assert {row["utterance_id"] for row in recordings} == uid_set
        assert {row["utterance_updates"] for row in recordings} == {snapshot["step"] // 8}
        for record, source in zip(recordings, rows[:8], strict=True):
            assert record["samples"] == source.prepared_samples and record["nonfinite_samples"] == 0
            assert record["optimization_total"] >= record["common_reconstruction"]
            raw_path = paths[0] / "snapshots" / f"step-{snapshot['step']:05d}" / record["raw"]
            assert np.load(raw_path, allow_pickle=False).shape == (source.prepared_samples,)
    with (paths[0] / "train-loss.csv").open(newline="", encoding="utf-8") as stream:
        trace = list(csv.DictReader(stream))
    assert len(trace) == 16 and all(float(row["gradient_l2"]) > 0 for row in trace)
    assert all(float(row["phase_time"]) > 0 for row in trace)
    expected_order = [
        uid
        for index in range(2)
        for uid in eight.round_order(tuple(detailed["training_ids"]), 20260907, index)
    ]
    assert [row["utterance_id"] for row in trace] == expected_order
    key = json.loads((paths[0] / "listening-key.json").read_bytes())
    assert len(key) == 16
    assert all(
        Counter(row["condition"] for row in key.values() if row["utterance_id"] == uid)
        == {"reference": 1, "reconstruction": 1}
        for uid in uid_set
    )


def test_budget_stop_preserves_completed_round_and_all_outputs(corpus, tmp_path, monkeypatch):
    inputs, mount, pin, _ = corpus
    clock = [100.0]
    step = eight._step

    def cross_deadline(*args):
        result = step(*args)
        clock[0] = 111.0
        return result

    monkeypatch.setattr(eight, "monotonic", lambda: clock[0])
    monkeypatch.setattr(eight, "_step", cross_deadline)
    output = tmp_path / "limited"
    receipt = eight.run_diagnostic(
        inputs,
        mount,
        output,
        pin=pin,
        device="cpu",
        code={},
        steps=(0, 16),
        deadline_seconds=10,
        started_at=100,
    )
    assert receipt["status"] == "budget_limited" and receipt["stop_reason"] == "wall_time"
    assert receipt["updates"] == 8 and set(receipt["utterance_exposure"].values()) == {1}
    assert receipt["checkpoint"]["step"] == 8
    assert [snapshot["step"] for snapshot in receipt["snapshots"]] == [0, 8]
    assert len(receipt["snapshots"][-1]["recordings"]) == len(receipt["reload"]) == 8
    assert sha256_file(output / "backbone.pt") == receipt["checkpoint"]["sha256"]


def test_record_stage_compiles_and_fails_closed_in_an_empty_mount(corpus, tmp_path, monkeypatch):
    _, source, pin, _ = corpus
    inventory = eight.source_inventory(source, pin)
    monkeypatch.setattr(eight, "source_inventory", lambda _: inventory)
    commit = "b" * 40
    monkeypatch.setattr(kaggle, "_git", lambda *_: SimpleNamespace(stdout=commit))

    def archive(mode, sha, paths):
        assert (mode, sha) == ("record", commit)
        return {name: (kaggle.REPO_ROOT / name).read_bytes() for name in paths}

    monkeypatch.setattr(kaggle, "_archive_bytes", archive)
    stage = tmp_path / "stage"
    staged = kaggle.stage_eight_kernel(commit, source, stage)
    plan = staged["plan"]
    assert plan["expected_full_order_sha256"] == kaggle._order_digest(
        tuple(r["utterance_id"] for r in inventory["recordings"]), eight.protocol()
    )
    metadata = json.loads((stage / "kernel/kernel-metadata.json").read_bytes())
    assert metadata["is_private"] == metadata["enable_gpu"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaT4" and metadata["enable_internet"] == "false"
    assert (
        metadata["dataset_sources"] == [kaggle.DATASET["ref"]] and metadata["kernel_sources"] == []
    )
    script = stage / "kernel/overfit_eight.py"
    assert sha256_file(script) == staged["script_sha256"]
    assert (
        hashlib.sha256((stage / "kernel/kernel-metadata.json").read_bytes()).hexdigest()
        == staged["metadata_sha256"]
    )
    empty, working, local = (tmp_path / name for name in ("empty", "working", "local.py"))
    empty.mkdir()
    working.mkdir()
    test_source = script.read_text(encoding="utf-8").replace(
        repr("/kaggle/input"), repr(str(empty))
    )
    local.write_text(
        test_source.replace(repr("/kaggle/working"), repr(str(working))), encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(local)],
        check=False,
        timeout=40,
        capture_output=True,
        text=True,
    )
    failure = json.loads((working / kaggle.OUTPUT_ROOT / "diagnostic-receipt.json").read_bytes())
    assert result.returncode != 0 and failure["status"] == "failed" and failure["updates"] == 0
    assert "expected one private eight-utterance Kaggle mount" in failure["error"]
