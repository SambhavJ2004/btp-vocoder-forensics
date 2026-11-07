"""Deterministic CPU intake contracts; all audio is labeled synthetic fixture data."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from vctk_fixture import make_fixture, native_fixture, put_source

from data import source_ingestion as intake
from data.corpus import Utterance, assign_splits
from data.preprocess import resample_once
from experiments.cli import main


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return make_fixture(tmp_path_factory.mktemp("native-vctk") / "source", invalid=True)


@pytest.fixture(scope="module")
def admitted(source, tmp_path_factory):
    out = tmp_path_factory.mktemp("vctk-intake") / "cohort"
    result = intake.ingest_vctk(
        source, out, policy=intake.IntakePolicy(6, 2), source_kind="fixture"
    )
    return source, out, result


def records(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def seal(root):
    intake.write_json(
        root / "checksums.json",
        {
            p.name: {"bytes": p.stat().st_size, "sha256": intake.digest(p)}
            for p in root.iterdir()
            if p.is_file() and p.name != "checksums.json"
        },
    )


def test_valid_fixture_cohort_counts_speaker_disjointness_and_actual_exclusions(admitted):
    source, out, result = admitted
    assert result["status"] == "passed" and result["sources"] == 12 and result["speakers"] == 6
    assert result["source_kind"] == "fixture" and not result["scientific_admission_complete"]
    assert result["source_counts_by_split"] == {"train": 8, "dev": 2, "eval": 2}
    assert result["speaker_counts_by_split"] == {"train": 4, "dev": 1, "eval": 1}
    assert result["rejected_candidates"] == 7
    selection = intake.read_json(out / "selection.json")
    assert selection["candidates"] == 25
    assert selection["eligible_utterances"] == 18
    assert selection["candidates"] == (
        selection["eligible_utterances"] + result["rejected_candidates"]
    )
    rows = records(out / "source_manifest.csv")
    speakers = {}
    hashes = {}
    for r in rows:
        assert r["source_kind"] == "fixture" and r["citable"] == "False"
        assert r["native_sample_rate"] == "48000" and r["microphone"] == "mic2"
        speakers.setdefault(r["speaker_id"], set()).add(r["split"])
        hashes.setdefault(r["decoded_audio_sha256"], set()).add(r["split"])
        assert intake.digest(source / r["source_path"]) == r["source_sha256"]
        assert len(r["archive_preview_sha256"]) == 64
        assert 4.1 <= float(r["archive_trim_seconds"]) <= 8
    assert all(len(s) == 1 for s in [*speakers.values(), *hashes.values()])
    assert len({r["source_id"] for r in rows}) == len(rows)
    exclusions = intake.read_json(out / "exclusions.json")
    assert {r["reason"] for r in exclusions} == {
        "silent_source",
        "trimmed_duration_outside_cohort_bounds",
        "unreadable_audio",
        "missing_audio",
        "unsupported_native_format",
        "duplicate_source_or_content",
    }
    collision = [r for r in exclusions if r["reason"] == "duplicate_source_or_content"]
    assert {r["candidate_id"] for r in collision} == {"p206_905", "p201_906"}
    assert not list(out.rglob("*.wav")) and not list(out.rglob("*.flac"))
    assert intake.verify_ingestion(out) == result


def test_repeat_intake_is_deterministic_and_native_files_are_unchanged(admitted, tmp_path):
    source, out, _ = admitted
    before = intake.inventory(source)
    again = tmp_path / "repeat"
    intake.ingest_vctk(source, again, policy=intake.IntakePolicy(6, 2), source_kind="fixture")
    for name in (
        "source_manifest.csv",
        "selection.json",
        "speakers.json",
        "exclusions.json",
        "source_inventory.json",
        "resampling.json",
        "validation.json",
    ):
        assert (again / name).read_bytes() == (out / name).read_bytes()
    assert before == intake.inventory(source)


def test_corpus_bridge_preserves_recorded_groups_and_split_map(admitted):
    _, out, _ = admitted
    corpus = intake.IngestedVCTK(out)
    values = corpus.utterances()
    assert len(values) == 12 and len({u.speaker for u in values}) == 6
    assert corpus.is_multi_speaker and corpus.native_sample_rate == 48000
    assert {u.utt_id for u in values} == set(corpus.frozen_splits)
    assert assign_splits(values, speaker_disjoint=True) == corpus.frozen_splits
    returned = corpus.frozen_splits
    returned.clear()
    assert len(corpus.frozen_splits) == 12


@pytest.mark.parametrize(
    "policy",
    [
        intake.IntakePolicy(5, 2),
        intake.IntakePolicy(True, 2),
        intake.IntakePolicy(6, 0),
        intake.IntakePolicy(6, True),
        intake.IntakePolicy(6, 2, 4),
    ],
)
def test_invalid_or_insufficient_split_topology_is_explicit(policy):
    with pytest.raises(ValueError, match="frozen seed"):
        policy.validate()


def test_fifty_speakers_reuse_existing_seeded_speaker_split_policy():
    values = [
        Utterance(f"vctk_{s}_{u}", Path(f"{s}_{u}.flac"), f"speaker_{s:03d}")
        for s in range(50)
        for u in range(10)
    ]
    mapping = assign_splits(values, speaker_disjoint=True)
    assert mapping == assign_splits(list(reversed(values)), speaker_disjoint=True)
    assert {s: sum(v == s for v in mapping.values()) for s in ("train", "dev", "eval")} == {
        "train": 350,
        "dev": 50,
        "eval": 100,
    }
    assert intake.ranked(["a", "b", "c"], "speaker") == intake.ranked(["c", "a", "b"], "speaker")


def test_actual_shortfall_writes_failed_receipt_without_fabricating_coverage(source, tmp_path):
    out = tmp_path / "shortfall"
    with pytest.raises(ValueError, match="Insufficient qualified speakers"):
        intake.ingest_vctk(source, out, policy=intake.IntakePolicy(7, 2), source_kind="fixture")
    assert intake.read_json(out / "validation.json")["status"] == "failed"
    assert intake.read_json(out / "selection.json")["available_qualified_speakers"] == 6
    assert len(intake.read_json(out / "selection.json")["decisions"]) == 18
    assert not (out / "source_manifest.csv").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "extra",
        "nested",
        "removed",
        "modified",
        "split_resealed",
        "hash_resealed",
        "claims_resealed",
        "selection_resealed",
        "plan_resealed",
    ],
)
def test_manifest_and_package_integrity_rejects_tampering(admitted, tmp_path, damage):
    _, original, _ = admitted
    root = tmp_path / "copy"
    shutil.copytree(original, root)
    if damage == "extra":
        (root / "unexpected.txt").write_text("unrelated", encoding="utf-8")
    elif damage == "nested":
        (root / "unrelated").mkdir()
        (root / "unrelated/hidden.txt").write_text("unrelated", encoding="utf-8")
    elif damage == "removed":
        (root / "speakers.json").unlink()
    elif damage == "modified":
        (root / "source_manifest.csv").write_text("corrupt", encoding="utf-8")
    elif damage in ("split_resealed", "hash_resealed"):
        rows = records(root / "source_manifest.csv")
        key = "split" if damage == "split_resealed" else "source_sha256"
        rows[0][key] = "eval" if rows[0][key] != "eval" else "train"
        (root / "source_manifest.csv").write_text(
            intake.csv_text(rows), encoding="utf-8", newline="\n"
        )
    else:
        name = {"claims_resealed": "validation.json", "plan_resealed": "plan.json"}.get(
            damage, "selection.json"
        )
        d = intake.read_json(root / name)
        if damage == "claims_resealed":
            d["citable"] = True
        elif damage == "plan_resealed":
            d["source_stage"] = "scientifically_validated_archive"
        else:
            d["selected_speakers"].reverse()
        intake.write_json(root / name, d)
    if damage.endswith("resealed") or damage == "extra":
        seal(root)
    with pytest.raises(ValueError):
        intake.verify_ingestion(root)


def test_changed_native_inventory_is_rejected_without_repair(admitted, tmp_path):
    source, out, _ = admitted
    copy = tmp_path / "native"
    shutil.copytree(source, copy)
    wav = next(copy.glob("wav48_silence_trimmed/*/*_001_mic2.flac"))
    wav.write_bytes(wav.read_bytes()[:-15])
    with pytest.raises(ValueError, match="inventory changed"):
        intake.verify_ingestion(out, source_root=copy)


def test_relocated_explicit_source_root_preserves_identity(admitted, tmp_path):
    source, out, result = admitted
    copy = tmp_path / "native"
    shutil.copytree(source, copy)
    assert intake.verify_ingestion(out, source_root=copy) == result


@pytest.mark.parametrize(
    "value", ["../escape", "/absolute", "C:drive", "a\\b", "a/../b", "./a", ""]
)
def test_unsafe_paths_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="relative"):
        intake.safe_path(tmp_path, value)


def test_unsafe_source_link_is_never_followed(source, tmp_path):
    root = tmp_path / "source"
    shutil.copytree(source, root)
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    link = root / "README.txt"
    link.unlink()
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"OS does not allow source symlink creation: {exc}")
    with pytest.raises(ValueError, match="Unsafe link"):
        intake.inventory(root)


def test_hardlinked_native_file_is_rejected_without_symlink_privilege(source, tmp_path):
    root = tmp_path / "source"
    shutil.copytree(source, root)
    target = tmp_path / "outside.txt"
    target.write_text("fixture only", encoding="utf-8")
    link = root / "README.txt"
    link.unlink()
    try:
        link.hardlink_to(target)
    except OSError as exc:
        pytest.skip(f"OS/filesystem does not allow hardlink creation: {exc}")
    with pytest.raises(ValueError, match="Unsafe link"):
        intake.inventory(root)


def test_original_evidence_changed_is_rejected(admitted, tmp_path):
    source, out, _ = admitted
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for name in ("README.txt", "license_text.txt"):
        shutil.copyfile(source / name, evidence / name)
    with (evidence / "license_text.txt").open("a", encoding="utf-8") as stream:
        stream.write("\nChanged fixture evidence.\n")
    with pytest.raises(ValueError, match="evidence byte identity"):
        intake.verify_ingestion(out, evidence_dir=evidence)


def test_fixture_cannot_be_reported_as_lawful_real_ingestion(source, tmp_path):
    with pytest.raises(ValueError, match="cannot be declared"):
        intake.ingest_vctk(
            source,
            tmp_path / "false-real",
            policy=intake.IntakePolicy(6, 2),
            source_kind="local_dataset",
        )


def test_native_duplicate_text_all_members_excluded(source, tmp_path):
    root = tmp_path / "source"
    shutil.copytree(source, root)
    put_source(root, "p201_990", native_fixture(8001), text="Shared   PROMPT")
    put_source(root, "p202_990", native_fixture(8002), text=" shared prompt\n")
    out = tmp_path / "cohort"
    intake.ingest_vctk(root, out, policy=intake.IntakePolicy(6, 2), source_kind="fixture")
    exclusions = intake.read_json(out / "exclusions.json")
    for nid in ("p201_990", "p202_990"):
        r = next(e for e in exclusions if e["candidate_id"] == nid)
        assert r["reason"] == "duplicate_source_or_content"
        assert r["observed"]["collisions"][0]["field"] == "normalized_text_sha256"


@pytest.mark.parametrize("mode", ["stereo", "pcm24", "wrong_rate", "endpoints", "empty_text"])
def test_native_format_and_input_exclusions_have_measurement_receipts(source, tmp_path, mode):
    root = tmp_path / "source"
    shutil.copytree(source, root)
    wav = native_fixture(10101)
    if mode == "stereo":
        wav = np.column_stack((wav, wav))
    if mode == "endpoints":
        wav[4000] = -1
    put_source(
        root,
        "p201_990",
        wav,
        rate=44100 if mode == "wrong_rate" else 48000,
        subtype="PCM_24" if mode == "pcm24" else "PCM_16",
        text="" if mode == "empty_text" else None,
    )
    out = tmp_path / "cohort"
    intake.ingest_vctk(root, out, policy=intake.IntakePolicy(6, 2), source_kind="fixture")
    e = next(
        e for e in intake.read_json(out / "exclusions.json") if e["candidate_id"] == "p201_990"
    )
    expected = (
        "native_pcm_endpoint_samples"
        if mode == "endpoints"
        else ("empty_transcript" if mode == "empty_text" else "unsupported_native_format")
    )
    assert e["reason"] == expected
    if mode in ("stereo", "pcm24", "wrong_rate"):
        assert set(e["observed"]) >= {"sample_rate", "channels", "subtype", "frames"}


def test_nonfinite_decoder_results_cannot_enter_manifest(source, monkeypatch):
    inventory = intake.inventory(source)
    speakers = intake.speaker_records(source)
    monkeypatch.setattr(intake, "read_native", lambda _: (np.full(220800, np.nan), 48000))
    with pytest.raises(intake.SourceRejection, match="invalid_decoded"):
        intake.inspect_source(source, "p201_001", speakers, inventory)


def test_output_reuse_overlap_and_unexpected_source_files_are_not_repaired(
    source, admitted, tmp_path
):
    with pytest.raises(FileExistsError):
        intake.ingest_vctk(source, admitted[1], source_kind="fixture")
    with pytest.raises(ValueError, match="separate"):
        intake.ingest_vctk(source, source / "must-not-write", source_kind="fixture")
    root = tmp_path / "native"
    shutil.copytree(source, root)
    (root / ".env").write_text("synthetic forbidden fixture filename", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected source file"):
        intake.inventory(root)


@pytest.mark.parametrize("samples", [1, 2, 3, 47999, 48000, 48001])
def test_resampling_exact_ceil_length_determinism_and_unchanged_legacy(samples):
    x = np.linspace(-0.1, 0.1, samples, dtype=np.float32)
    out = intake.resample_source(x)
    assert len(out) == -(-samples * 22050 // 48000)
    np.testing.assert_array_equal(out, intake.resample_source(x))
    np.testing.assert_array_equal(out, resample_once(x, 48000))
    np.testing.assert_array_equal(x, np.linspace(-0.1, 0.1, samples, dtype=np.float32))


@pytest.mark.parametrize("x", [[], [[0, 1]], [np.nan], [np.inf], [1j], ["text"], [True]])
def test_resampling_invalid_input_is_explicit(x):
    with pytest.raises(ValueError, match="finite nonempty"):
        intake.resample_source(x)


def test_resampling_rate_and_environment_pin_are_explicit(monkeypatch):
    with pytest.raises(ValueError):
        intake.resample_source([0, 0.1], 44100)
    version = intake.version
    monkeypatch.setattr(
        intake, "version", lambda name: "changed" if name == "soxr" else version(name)
    )
    with pytest.raises(ValueError, match="Unsupported intake environment"):
        intake.resample_source([0, 0.1])


def test_resampler_reference_vectors_and_analytic_band_behavior():
    reference = intake.read_json(Path(__file__).parent / "fixtures/vctk_resampling_v1.json")
    assert reference["profile"] == intake.resampling_record()
    for v in reference["vectors"]:
        actual = intake.resample_source(np.asarray(v["input"], dtype=np.float32))
        np.testing.assert_allclose(
            actual, np.asarray(v["expected"], dtype=np.float32), rtol=1e-6, atol=1e-6
        )
    t = np.arange(48000 * 2) / 48000
    passed = intake.resample_source((0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32))[
        2000:-2000
    ]
    stopped = intake.resample_source((0.1 * np.sin(2 * np.pi * 12000 * t)).astype(np.float32))[
        2000:-2000
    ]
    assert 0.069 < np.sqrt(np.mean(passed**2)) < 0.072
    assert np.sqrt(np.mean(stopped**2)) < 1e-5


def test_cli_ingestion_verification_is_readonly_and_reports_fixture(admitted, capsys):
    assert main(["ingest", "verify", "--ingestion-dir", str(admitted[1])]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["source_kind"] == "fixture" and result["models_trained"] == 0
    assert result["audio_outputs_generated"] == 0
