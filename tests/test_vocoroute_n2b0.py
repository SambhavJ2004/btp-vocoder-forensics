"""CPU fixtures for safe VCTK indexing and reproducible corpus preparation."""

import hashlib
import io
import json
import stat
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import numpy as np
import pytest
import soundfile

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from experiments import vocoroute_corpus_kaggle as kaggle_prep  # noqa: E402
from experiments.vocoroute_kaggle import _dataset_manifest  # noqa: E402
from vocoroute import vctk_inventory, vctk_prepare  # noqa: E402
from vocoroute.training_data import TrainingAudio, load_manifest  # noqa: E402
from vocoroute.vctk_inventory import (  # noqa: E402
    SplitProtocol,
    allocate_speakers,
    inspect_archive,
    ranked_ids,
    sha256,
)
from vocoroute.vctk_prepare import (  # noqa: E402
    ARCHIVE_NAME,
    AcquisitionPin,
    InvalidRecording,
    load_overfit,
    prepare_held_out,
    prepare_recording,
    prepare_train_dev,
    verify_acquisition,
    write_json,
)

SMALL_PROTOCOL = SplitProtocol(
    base_speakers=3,
    dev_speakers=2,
    eval_speakers=1,
    router_fit_speakers=1,
    router_diag_speakers=1,
    pilot_train_speakers=2,
    pilot_dev_speakers=1,
    expanded_per_speaker=3,
    eval_per_speaker=2,
    overfit_utterances=2,
    pilot_per_speaker=2,
)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _flac(samples, rate=48_000):
    stream = io.BytesIO()
    soundfile.write(stream, samples, rate, format="FLAC", subtype="PCM_16")
    return stream.getvalue()


def _members():
    entries = {}
    speakers = ["# VCTK speaker table", "ID AGE GENDER ACCENTS REGION COMMENTS"]
    for sid in range(200, 210):
        gender = ("M", "F")[sid % 2]
        accent = ("English", "Welsh", "Scottish")[sid % 3]
        speakers.append(f"{sid} 24 {gender} {accent} Place")
        for index in range(4):
            uid = f"p{sid}_{index:03}"
            phase = np.arange(560 + index + sid) * (sid + index * 7) * np.pi / 24_000
            audio = _flac((np.sin(phase) * (sid / 1_000)).astype(np.float32))
            path = f"wav48_silence_trimmed/p{sid}/{uid}"
            entries[f"{path}_mic2.flac"] = audio
            if index == 0:
                entries[f"{path}_mic1.flac"] = audio
            text = ["A Shared Sentence.", " a SHARED sentence! ", uid, "Text four."][index]
            entries[f"txt/p{sid}/{uid}.txt"] = text.encode()
    entries["speaker-info.txt"] = ("\n".join(speakers) + "\n").encode()
    entries["txt/p200/p200_009.txt"] = b"missing both mics"
    entries["txt/p200/p200_010.txt"] = b"only mic one"
    entries["wav48_silence_trimmed/p200/p200_010_mic1.flac"] = b"not selected"
    entries["wav48_silence_trimmed/p201/p201_011_mic2.flac"] = b"missing text"
    entries["txt/p201/p201_012.txt"] = b"..."
    entries["wav48_silence_trimmed/p201/p201_012_mic2.flac"] = b"bad text counterpart"
    return entries


def _archive(path, entries, *, wrapper="VCTK-Corpus-0.92/", reverse=False):
    with ZipFile(path, "w") as zipped:
        for name in sorted(entries, reverse=reverse):
            zipped.writestr(wrapper + name, entries[name])


def _acquisition(root, entries):
    root.mkdir(parents=True, exist_ok=True)
    path = root / ARCHIVE_NAME
    _archive(path, entries)
    source = {"README.txt": b"synthetic fixture README", "license_text.txt": b"fixture only"}
    for name, raw in source.items():
        (root / name).write_bytes(raw)
    archive = path.read_bytes()
    info = {
        "name": ARCHIVE_NAME,
        "bytes": len(archive),
        "sha256": sha256(archive),
        "md5": hashlib.md5(archive, usedforsecurity=False).hexdigest(),
    }
    receipt = write_json(
        root / "acquisition-receipt.json",
        {
            "schema": "vocoroute-vctk-acquisition-v1",
            "status": "verified",
            "doi": "10.7488/ds/2645",
            "license": "CC BY 4.0",
            "archive": info,
            "metadata": {name: {"sha256": sha256(raw)} for name, raw in source.items()},
        },
    )
    return path, AcquisitionPin(receipt, info["bytes"], info["md5"], info["sha256"])


def test_mount_checks_archive_bytes_and_metadata(tmp_path):
    archive, pin = _acquisition(tmp_path, _members())
    source = verify_acquisition(archive, pin)
    assert set(source) == {"README.txt", "license_text.txt", "acquisition-receipt.json"}
    data = archive.read_bytes()
    license_path = tmp_path / "license_text.txt"
    license_path.write_bytes(b"unlicensed")
    with pytest.raises(ValueError, match="metadata SHA-256"):
        verify_acquisition(archive, pin)
    license_path.write_bytes(source[license_path.name])
    archive.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match="archive checksum"):
        verify_acquisition(archive, pin)
    with pytest.raises(ValueError, match="receipt.json"):
        verify_acquisition(archive, replace(pin, receipt_sha256="a" * 64))


@pytest.mark.parametrize(
    "bad",
    [
        "../evil",
        "/evil",
        "C:/evil",
        "win\\evil",
        "x/./y",
        "symlink",
        "SPEAKER-INFO.TXT",
        "speaker-info.txt/child",
    ],
)
def test_unsafe_archive_inventory_is_rejected(tmp_path, bad):
    path = tmp_path / "bad.zip"
    _archive(path, _members(), wrapper="")
    member = ZipInfo(bad)
    member.filename = bad  # Put a literal backslash on disk even on Windows.
    if bad == "symlink":
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(path, "a") as zipped:
        zipped.writestr(member, b"invalid")
    with ZipFile(path) as zipped, pytest.raises(ValueError, match="ZIP"):
        inspect_archive(zipped, "mic2")


def test_inventory_and_allocation_use_availability_and_metadata(tmp_path):
    entries = _members()
    left, right = tmp_path / "a.zip", tmp_path / "b.zip"
    _archive(left, entries)
    _archive(right, entries, wrapper="", reverse=True)
    with ZipFile(left) as a, ZipFile(right) as b:
        inv = inspect_archive(a, "mic2")
        allocation = allocate_speakers(inv, SMALL_PROTOCOL)
        assert allocation == allocate_speakers(inspect_archive(b, "mic2"), SMALL_PROTOCOL)
    assert inv.report["selected_mic_without_transcript"] == ["p201_011"]
    assert inv.report["transcript_without_recording"] == ["p200_009"]
    assert inv.report["transcript_without_selected_mic"] == ["p200_009", "p200_010"]
    assert inv.report["issues"]["invalid_transcript"] == ["p201_012"]
    assert inv.report["paired_microphones"] == 10
    assert all(item.audio.filename.endswith("_mic2.flac") for item in inv.utterances.values())
    groups = [
        allocation[name]
        for name in ("base_expert_fit", "dev", "eval", "router_fit", "router_diagnostics")
    ]
    assert len({sid for group in groups for sid in group}) == sum(map(len, groups)) == 8
    assert set(allocation["pilot_train"]) <= set(allocation["base_expert_fit"])
    with pytest.raises(ValueError, match="need 95"):
        allocate_speakers(inv, SplitProtocol())


def test_prepare_preserves_gain_geometry_and_native_bytes(tmp_path):
    path = tmp_path / "headroom.zip"
    entries = _members()
    waveform = (0.94 * np.sin(np.arange(2_401) / 38.0)).astype(np.float32)
    native_name = "wav48_silence_trimmed/p200/p200_002_mic2.flac"
    entries[native_name] = _flac(waveform)
    _archive(path, entries)
    with ZipFile(path) as zipped:
        inv = inspect_archive(zipped, "mic2")
        record = prepare_recording(
            zipped, "p200_002", inv.utterances["p200_002"], "train", "a" * 64
        )
    result, rate = soundfile.read(io.BytesIO(record.prepared))
    assert rate == 22_050
    assert result.shape == ((waveform.size * 147 + 319) // 320,)
    assert 0.93 < np.max(np.abs(result)) < 0.95
    assert record.native == entries[native_name]
    assert record.row.native_sha256 == sha256(record.native)
    assert record.row.prepared_sha256 == sha256(record.prepared)
    assert record.row.source_group == "vctk-0.92/p200_002"


@pytest.mark.parametrize(
    "value,rate,reason",
    [
        (0.0, 48_000, "digital silence"),
        (-1.0, 48_000, "full-scale"),
        (0.1, 24_000, "48 kHz"),
    ],
)
def test_bad_recording_fails_eligibility(tmp_path, value, rate, reason):
    path = tmp_path / "bad-audio.zip"
    entries = _members()
    entries["wav48_silence_trimmed/p200/p200_002_mic2.flac"] = _flac(np.full(832, value), rate)
    _archive(path, entries)
    with ZipFile(path) as zipped:
        inv = inspect_archive(zipped, "mic2")
        with pytest.raises(InvalidRecording, match=reason):
            prepare_recording(zipped, "p200_002", inv.utterances["p200_002"], "train", "a" * 64)


def test_frozen_train_and_eval_against_real_loader_contract(tmp_path, monkeypatch):
    inputs, working, eval_working = (tmp_path / name for name in ("input", "work", "hold"))
    working.mkdir()
    eval_working.mkdir()
    archive, pin = _acquisition(inputs / "vctk-0.92", _members())
    monkeypatch.setattr(vctk_inventory, "SplitProtocol", lambda: SMALL_PROTOCOL)
    monkeypatch.setattr(
        vctk_prepare, "verify_acquisition", lambda path: verify_acquisition(path, pin)
    )
    monkeypatch.setattr(kaggle_prep, "MAX_AUDIO_BYTES", {"train-dev": 10**6, "held-out": 10**6})
    monkeypatch.setattr(kaggle_prep, "STORAGE_MARGIN_BYTES", 0)
    initial = kaggle_prep.stage_corpus_kernel(tmp_path / "stage")
    receipt = kaggle_prep.run_corpus_kernel(initial["plan"], inputs, working)
    assert receipt["status"] == "verified", receipt
    assert receipt["runtime"]["cpu_only"]
    result = receipt["release"]
    release_path = working / receipt["output"]
    manifests = {}
    for release in ("expanded", "pilot"):
        verify = {
            "spec": {
                "data": {
                    "ref": f"sambhavjainsj/vocoroute-vctk092-n2b0-{release}",
                    "version": 1,
                    "mount": release,
                    "receipt_sha256": result["dataset_receipts"][release],
                }
            },
            "config": {"manifest_sha256": result["manifests"][release]},
        }
        dataset, path = _dataset_manifest(verify, release_path)
        manifest = manifests[release] = load_manifest(path, result["manifests"][release])
        assert all(
            not (dataset / row.prepared_path).exists()
            for row in manifest.rows
            if row.role == "eval"
        )
    assert set(manifests["pilot"].rows) < set(manifests["expanded"].rows)
    diag, rows = load_overfit(release_path / "overfit", result["manifests"]["overfit"])
    assert len(rows) == SMALL_PROTOCOL.overfit_utterances and all(
        row.role == "train" for row in rows
    )
    assert not (diag.root / "native" / "dev").exists()
    assert set(rows) < set(manifests["pilot"].examples("train"))
    assert result["audit"]["expanded"]["cross_role"]["train/eval"]["speaker_overlap"] == 0
    assert (
        result["audit"]["expanded"]["cross_role"]["train/dev"][
            "distinct_shared_normalized_transcripts"
        ]
        > 0
    )
    held_plan = kaggle_prep.stage_corpus_kernel(
        tmp_path / "eval-stage",
        freeze_source="sambhavjainsj/vocoroute-vctk092-n2b0-train-dev/1",
        freeze_sha256=result["freeze_sha256"],
    )["plan"]
    release_path.rename(inputs / release_path.name)
    eval_receipt = kaggle_prep.run_corpus_kernel(held_plan, inputs, eval_working)
    assert eval_receipt["status"] == "verified", eval_receipt
    evaluation = eval_working / eval_receipt["output"]
    for row in manifests["expanded"].rows:
        native, prepared = evaluation / row.native_path, evaluation / row.prepared_path
        if row.role == "eval":
            assert sha256(native.read_bytes()) == row.native_sha256
            assert sha256(prepared.read_bytes()) == row.prepared_sha256
        else:
            assert not native.exists() and not prepared.exists()
    with pytest.raises(FileNotFoundError):
        TrainingAudio(manifests["expanded"], evaluation)
    with ZipFile(archive) as zipped:
        inv = inspect_archive(zipped, "mic2")
        with pytest.raises(ValueError, match="freeze.json"):
            prepare_held_out(
                zipped,
                inv,
                SMALL_PROTOCOL,
                inputs / release_path.name,
                "f" * 64,
                tmp_path / "unapproved",
                verify_acquisition(archive, pin),
                max_audio_bytes=10**6,
            )


def test_cross_role_duplicate_fails_and_exclusions_are_reported(tmp_path):
    entries = _members()
    archive, pin = _acquisition(tmp_path / "acq", entries)
    with ZipFile(archive) as zipped:
        roles = allocate_speakers(inspect_archive(zipped, "mic2"), SMALL_PROTOCOL)
    train_sid, dev_sid = roles["base_expert_fit"][0], roles["dev"][0]
    chosen = ranked_ids(
        SMALL_PROTOCOL.seed,
        f"utterance:{train_sid}",
        [f"{train_sid}_{index:03}" for index in range(4)],
    )
    # First train row is invalid, the second should collide with the first dev row.
    dev_id = ranked_ids(
        SMALL_PROTOCOL.seed, f"utterance:{dev_sid}", [f"{dev_sid}_{index:03}" for index in range(4)]
    )[0]
    entries[f"wav48_silence_trimmed/{train_sid}/{chosen[0]}_mic2.flac"] = _flac(np.zeros(500))
    entries[f"wav48_silence_trimmed/{train_sid}/{chosen[1]}_mic2.flac"] = entries[
        f"wav48_silence_trimmed/{dev_sid}/{dev_id}_mic2.flac"
    ]
    archive, pin = _acquisition(tmp_path / "dual", entries)
    with ZipFile(archive) as zipped, pytest.raises(ValueError, match="duplicate audio detected"):
        prepare_train_dev(
            zipped,
            inspect_archive(zipped, "mic2"),
            SMALL_PROTOCOL,
            tmp_path / "release",
            tmp_path / "reports",
            verify_acquisition(archive, pin),
            max_audio_bytes=10**6,
        )
    rejection = json.loads((tmp_path / "reports" / "audio-rejections.jsonl").read_text())
    assert rejection == {
        "utterance": chosen[0],
        "role": "train",
        "reason": "native digital silence",
    }


def test_staging_is_deterministic_and_private(tmp_path):
    directories = [tmp_path / "one", tmp_path / "two"]
    left, right = (kaggle_prep.stage_corpus_kernel(directory) for directory in directories)
    assert left == right
    script = (directories[0] / "prepare.py").read_bytes()
    meta = json.loads((directories[0] / "kernel-metadata.json").read_bytes())
    assert left["staged_script_sha256"] == sha256(script)
    compile(script, "prepare.py", "exec")
    assert meta["is_private"] == "true"
    assert meta["enable_gpu"] == meta["enable_tpu"] == meta["enable_internet"] == "false"
    assert meta["kernel_sources"] == ["sambhavjainsj/vocoroute-vctk092-acquisition"]
    assert left["plan"]["source"]["version"] == 2
    assert meta["dataset_sources"] == []
    assert left["plan"]["mode"] == "iteration"
    assert all(Path(name).parts[0] == "src" for name in left["plan"]["source_files"])
    with pytest.raises(ValueError, match="exact train/dev notebook version"):
        kaggle_prep.stage_corpus_kernel(
            tmp_path / "bad", freeze_source="owner/unpinned", freeze_sha256="a" * 64
        )


@pytest.mark.parametrize(
    "marker", ["vctk-0.92/VCTK-Corpus-0.92.zip", "eval-index.json", "native/eval/p201/a.flac"]
)
def test_training_preflight_refuses_additional_eval_input(tmp_path, marker):
    attached = tmp_path / "separate-input" / marker
    attached.parent.mkdir(parents=True)
    attached.write_bytes(b"must not be opened")
    with pytest.raises(ValueError, match="held-out evaluation audio is mounted"):
        _dataset_manifest({}, tmp_path)
