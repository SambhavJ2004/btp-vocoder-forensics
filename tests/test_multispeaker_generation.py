"""Fixture-only integration: mock results never imply neural/VCTK success."""

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from multispeaker_fixture import make_generation_fixture
from vctk_fixture import native_fixture, put_source

from data import source_ingestion
from data.audio_io import read_native
from data.invariants import InvariantViolation
from data.source_ingestion import (
    IntakePolicy,
    array_digest,
    digest,
    ingest_vctk,
    inventory,
    read_json,
    write_json,
)
from experiments import multispeaker_generation as multi
from experiments.cli import main
from vocoders.griffin_lim import GriffinLim


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    base = tmp_path_factory.mktemp("multispeaker")
    source = make_generation_fixture(base / "source")
    intake = base / "intake"
    ingest_vctk(source, intake, policy=IntakePolicy(8, 4), source_kind="fixture")
    plan = base / "plan"
    multi.create_plan(intake, plan, source_kind="fixture", condition="fixture_mock")
    return source, intake, plan


@pytest.fixture(scope="module")
def compatible(cohort, tmp_path_factory):
    out = tmp_path_factory.mktemp("compatibility") / "artifact"
    summary = multi.generate(cohort[2], out, stage="compatibility")
    return out, summary


def test_fixture_end_to_end_uses_exact_training_groups_and_shared_gates(cohort, compatible):
    out, summary = compatible
    assert summary["status"] == "passed"
    assert summary["source_kind"] == "fixture" and summary["real_vctk_sources"] == 0
    assert summary["sources"] == 20 and summary["speakers"] == 6
    assert summary["outputs"] == 80 and summary["alignment_files"] == 20
    assert summary["generator_kind"] == "fixture_mock" and not summary["citable"]
    assert not summary["scientific_admission_complete"]
    frame = pd.read_csv(out / "manifest.csv")
    assert set(frame["split"]) == {"train"}
    assert set(frame["source_kind"]) == {"fixture"}
    assert set(frame["reference_semantics"]) == {"synthetic_fixture_input_not_genuine"}
    assert set(frame["condition"]) == {"real", "fixture_mock"}
    assert all(Path(p).name.startswith("fixture__") for p in frame["path"])
    plan = read_json(cohort[2] / "plan.json")
    assert plan["generator_spec"]["family"] == "fixture_mock"
    assert not plan["generator_spec"]["primary_ladder"]
    assert set(frame["utt_id"]) == {r["source_id"] for r in plan["sources"]}
    assert all(plan["frozen_splits"][uid] == "train" for uid in frame["utt_id"])
    assert sorted(frame[frame["tier"] == "archive"].groupby("speaker")["utt_id"].nunique()) == [
        3,
        3,
        3,
        3,
        4,
        4,
    ]
    provenance = read_json(out / "provenance.json")
    assert not provenance["generator"]["neural_inference"]
    assert not provenance["cached_checkpoint_assets"]
    assert multi.validate(out) == summary


def test_generation_requires_valid_compatibility_and_repeats_exact_wavs(
    cohort, compatible, tmp_path
):
    original, _ = compatible
    before = inventory(cohort[0])
    out = tmp_path / "limited"
    summary = multi.generate(cohort[2], out, stage="generation", compatibility_dir=original)
    assert summary["stage"] == "generation"
    for path in original.rglob("*.wav"):
        assert digest(path) == digest(out / path.relative_to(original))
    assert inventory(cohort[0]) == before
    assert not summary["citable"] and summary["real_vctk_sources"] == 0
    with pytest.raises(ValueError, match="separately validated"):
        multi.generate(cohort[2], tmp_path / "missing", stage="generation")
    with pytest.raises(FileExistsError):
        multi.generate(cohort[2], out, stage="generation", compatibility_dir=original)


def test_generation_never_resplits_or_feeds_metadata_or_heldout_sources(cohort, compatible):
    plan = read_json(cohort[2] / "plan.json")
    frozen = plan["frozen_splits"]
    source_ids = [r["source_id"] for r in plan["sources"]]
    assert len(frozen) == 32 and len(source_ids) == 20
    assert all(frozen[uid] == "train" for uid in source_ids)
    counts = {s: sum(v == s for v in frozen.values()) for s in ("train", "dev", "eval")}
    assert counts == {"train": 24, "dev": 4, "eval": 4}
    assert read_json(compatible[0] / "plan/plan.json") == plan


@pytest.mark.parametrize("damage", ["speaker", "split", "hash", "rate", "microphone", "resampler"])
def test_resealed_invalid_intake_identity_is_rejected(cohort, tmp_path, damage):
    root = tmp_path / "intake"
    shutil.copytree(cohort[1], root)
    if damage in ("speaker", "split", "hash", "rate"):
        frame = pd.read_csv(root / "source_manifest.csv")
        column, value = {
            "speaker": ("speaker_id", "unknown_speaker"),
            "split": ("split", "INVALID_SPLIT"),
            "hash": ("source_sha256", "0" * 64),
            "rate": ("native_sample_rate", 44100),
        }[damage]
        frame.loc[0, column] = value
        frame.to_csv(root / "source_manifest.csv", index=False)
    else:
        name = "plan.json" if damage == "microphone" else "resampling.json"
        doc = read_json(root / name)
        doc["microphone" if damage == "microphone" else "res_type"] = "invalid"
        write_json(root / name, doc)
    multi._seal(root)
    with pytest.raises(ValueError):
        multi.create_plan(
            root, tmp_path / "must-not-exist", source_kind="fixture", condition="fixture_mock"
        )
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("condition", ["bigvgan_112m", "vocos", "unrecognized"])
def test_fixture_neural_requests_fail_before_generator_load(
    cohort, tmp_path, condition, monkeypatch
):
    monkeypatch.setattr(multi, "get_vocoder", lambda _: pytest.fail("No model may be loaded"))
    with pytest.raises(ValueError):
        multi.create_plan(cohort[1], tmp_path / "bad", source_kind="fixture", condition=condition)


def test_fixture_cannot_be_reclassified_as_real_even_with_receipt(cohort, tmp_path, monkeypatch):
    monkeypatch.setattr(multi, "verify_ingestion", lambda *a, **kw: pytest.fail("No real access"))
    fake_review = tmp_path / "attestation.json"
    write_json(fake_review, {"local_processing_authorized": True})
    with pytest.raises(ValueError, match="Source-kind"):
        multi.create_plan(
            cohort[1],
            tmp_path / "bad",
            source_kind="local_dataset",
            condition="griffin_lim",
            acquisition_receipt=fake_review,
        )


def test_wrong_native_source_rate_is_rejected_without_silent_resampling(cohort, tmp_path):
    root, intake = tmp_path / "source", tmp_path / "intake"
    shutil.copytree(cohort[0], root)
    shutil.copytree(cohort[1], intake)
    put_source(root, "p201_001", native_fixture(1, seconds=5, rate=44100), rate=44100)
    with pytest.raises(ValueError, match="inventory changed"):
        multi.create_plan(
            intake,
            tmp_path / "bad",
            source_kind="fixture",
            condition="fixture_mock",
            source_root=root,
        )


@pytest.mark.parametrize("change", ["split", "speaker", "source", "flags", "count", "condition"])
def test_resealed_generation_plan_cannot_authorize_different_sources(cohort, tmp_path, change):
    plan = tmp_path / "plan"
    shutil.copytree(cohort[2], plan)
    doc = read_json(plan / "plan.json")
    if change == "split":
        doc["sources"][0]["split"] = "eval"
    elif change == "speaker":
        doc["sources"][0]["speaker_id"] = "impostor"
    elif change == "source":
        doc["sources"][0]["source_path"] = "../escape.flac"
    elif change == "flags":
        doc["citable"] = True
    elif change == "count":
        doc["sources"].pop()
    else:
        doc["condition"] = "vocos"
    write_json(plan / "plan.json", doc)
    multi._seal(plan)
    with pytest.raises(ValueError):
        multi.load_plan(plan)


@pytest.mark.parametrize("mode", ["wrong_rate", "stereo", "nan", "empty", "excessive_length"])
def test_invalid_mock_native_outputs_fail_and_preserve_failed_receipt(cohort, tmp_path, mode):
    class Bad(multi.FixtureIdentity):
        def resynthesize(self, x, sr):
            if mode == "wrong_rate":
                return multi.NativeOutput(x, 16000)
            if mode == "stereo":
                return multi.NativeOutput(np.column_stack((x, x)), sr)
            if mode == "nan":
                return multi.NativeOutput(np.full_like(x, np.nan), sr)
            if mode == "empty":
                return multi.NativeOutput(np.zeros(0, np.float32), sr)
            return multi.NativeOutput(np.pad(x, (0, 4000)), sr)

    out = tmp_path / "failed"
    with pytest.raises((ValueError, InvariantViolation)):
        multi.generate(cohort[2], out, stage="compatibility", generator=Bad())
    assert read_json(out / "validation.json")["status"] == "failed"
    assert list(out.rglob("fixture__*.wav"))  # Reference retained; no cleanup or false acceptance.
    with pytest.raises((ValueError, FileNotFoundError)):
        multi.validate(out)


def test_actual_delayed_mock_fails_existing_twenty_file_alignment_gate(cohort, tmp_path):
    class Delayed(multi.FixtureIdentity):
        def resynthesize(self, x, sr):
            return multi.NativeOutput(np.roll(x, 100), sr)

    out = tmp_path / "delayed"
    with pytest.raises(InvariantViolation, match="not sample-aligned"):
        multi.generate(cohort[2], out, stage="compatibility", generator=Delayed())
    assert read_json(out / "validation.json")["status"] == "failed"
    assert read_json(out / "alignment.json")["pre_encoding"]["n_files"] == 20


@pytest.mark.parametrize(
    "damage",
    [
        "root_extra",
        "nested_extra_resealed",
        "missing",
        "wav_modified",
        "wav_resealed",
        "container_resealed",
        "speaker_resealed",
        "split_resealed",
        "path_resealed",
        "length_resealed",
        "alignment_resealed",
        "kind_resealed",
    ],
)
def test_strict_output_schema_hashes_and_semantics_reject_tampering(compatible, tmp_path, damage):
    root = tmp_path / "artifact"
    shutil.copytree(compatible[0], root)
    wav = next(root.glob("archive/fixture_mock/*.wav"))
    if damage == "root_extra":
        (root / "extra.txt").write_text("fixture", encoding="utf-8")
    elif damage == "nested_extra_resealed":
        (root / "archive/unrelated").mkdir()
        (root / "archive/unrelated/extra.txt").write_text("fixture", encoding="utf-8")
    elif damage == "missing":
        wav.unlink()
    elif damage in ("wav_modified", "wav_resealed"):
        data = bytearray(wav.read_bytes())
        data[-20] ^= 1
        wav.write_bytes(data)
    elif damage == "container_resealed":
        samples, rate = read_native(wav)
        # Test-only malformed packaging: same decoded samples, wrong container.
        sf.write(wav, samples, rate, format="FLAC", subtype="PCM_16")
        provenance = read_json(root / "provenance.json")
        provenance["file_identities"][wav.relative_to(root).as_posix()] = {
            "sha256": digest(wav),
            "decoded_float32_sha256": array_digest(samples),
        }
        write_json(root / "provenance.json", provenance)
    elif damage in ("speaker_resealed", "split_resealed", "path_resealed"):
        frame = pd.read_csv(root / "manifest.csv")
        key, value = {
            "speaker_resealed": ("speaker", "impostor"),
            "split_resealed": ("split", "eval"),
            "path_resealed": ("path", "../outside.wav"),
        }[damage]
        frame.loc[0, key] = value
        frame.to_csv(root / "manifest.csv", index=False)
    else:
        name = {
            "length_resealed": "lengths.json",
            "alignment_resealed": "alignment.json",
            "kind_resealed": "provenance.json",
        }[damage]
        doc = read_json(root / name)
        if damage == "length_resealed":
            doc["records"][0]["generator_input_samples"] += 1
        elif damage == "alignment_resealed":
            doc["pre_encoding"]["n_files"] = 19
        else:
            doc["generator"]["kind"] = "neural_vocoder"
        write_json(root / name, doc)
    if damage.endswith("resealed"):
        multi._seal(root)
    with pytest.raises((ValueError, InvariantViolation, AssertionError)):
        multi.validate(root)


def test_relative_manifests_survive_relocation_with_original_intake(cohort, compatible, tmp_path):
    out = tmp_path / "relocated"
    shutil.copytree(compatible[0], out)
    assert multi.validate(out) == compatible[1]


def test_derived_reconstruction_uses_the_archive_parent_loudness(compatible, monkeypatch):
    original = multi.derive_zeroshot
    parents = []

    def observed(waveform, **kwargs):
        assert kwargs.get("archive_lufs") is not None, (
            "Do not substitute the target for parent loudness"
        )
        parents.append(kwargs["archive_lufs"])
        return original(waveform, **kwargs)

    monkeypatch.setattr(multi, "derive_zeroshot", observed)
    assert multi.validate(compatible[0])["status"] == "passed"
    frame = pd.read_csv(compatible[0] / "manifest.csv")
    assert len(parents) == 40
    np.testing.assert_allclose(
        sorted(parents), sorted(frame[frame["tier"] == "archive"]["lufs_out"]), atol=1e-12, rtol=0
    )


def test_cli_verification_reports_fixture_and_legacy_root_gate_is_unchanged(
    compatible, capsys, cohort, tmp_path
):
    assert main(["multispeaker", "validate", "--directory", str(compatible[0])]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["real_vctk_sources"] == 0 and report["generator_kind"] == "fixture_mock"
    from experiments.phase_a_resynthesis import preflight

    with pytest.raises(InvariantViolation, match="not an LJSpeech root"):
        preflight(cohort[0], ["griffin_lim"], tmp_path / "legacy", 20)


def test_real_griffin_lim_is_deterministic_on_one_training_fixture_only(cohort):
    plan, root = multi.load_plan(cohort[2])
    source = plan["sources"][0]
    wav, _ = read_native(root / source["source_path"])
    spec = multi._spec("griffin_lim")
    x = multi._input(wav, spec)
    vocoder = GriffinLim()  # Actual 60-iteration DSP; no neural model or weights.
    a = multi._native_output(vocoder.resynthesize(x, spec.mel.sample_rate), spec.mel.sample_rate)
    b = multi._native_output(vocoder.resynthesize(x, spec.mel.sample_rate), spec.mel.sample_rate)
    np.testing.assert_array_equal(a, b)
    assert abs(len(a) - len(x)) <= 1102 and np.isfinite(a).all()
    assert asdict(vocoder.spec)["checkpoint"] == ""
    # One-file determinism is NOT the twenty-file compatibility gate.


def test_neural_cache_lookup_requires_offline_before_import_or_download(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(ValueError, match="HF_HUB_OFFLINE"):
        multi._cached_models("bigvgan_112m")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    fake = SimpleNamespace(
        constants=SimpleNamespace(HF_HUB_OFFLINE=True), try_to_load_from_cache=lambda *a, **kw: None
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    with pytest.raises(ValueError, match="Missing pinned local"):
        multi._cached_models("vocos")


@pytest.mark.parametrize("condition", ["fixture_mock", "griffin_lim"])
def test_mock_or_griffin_lim_receipt_cannot_claim_neural_evidence(condition):
    plan = {"condition": condition}
    with pytest.raises(ValueError, match="Generator label"):
        multi._assert_generator_label(plan, {"kind": "neural_vocoder", "neural_inference": True})
    multi._assert_generator_label(plan, {"kind": condition, "neural_inference": False})


def test_generator_specification_must_match_plan_before_load(monkeypatch):
    monkeypatch.setattr(multi, "get_vocoder", lambda _: multi.FixtureIdentity())
    plan = {
        "condition": "griffin_lim",
        "source_kind": "fixture",
        "generator_spec": asdict(multi._spec("griffin_lim")),
    }
    with pytest.raises(ValueError, match="specification"):
        multi._prepare_generator(plan, None, "cpu")


def test_scoped_revalidation_reads_only_chosen_training_waveforms(cohort, monkeypatch):
    plan = read_json(cohort[2] / "plan.json")
    paths = []
    original = source_ingestion.read_native

    def observed(path):
        paths.append(Path(path).relative_to(cohort[0]).as_posix())
        return original(path)

    monkeypatch.setattr(source_ingestion, "read_native", observed)
    multi._verify_selected_sources(cohort[1], cohort[0], plan["sources"])
    assert paths == [r["source_path"] for r in plan["sources"]]
    heldout = dict(plan["sources"][0], split="eval")
    with pytest.raises(ValueError, match="held-out"):
        multi._verify_selected_sources(cohort[1], cohort[0], [heldout])
    assert len(paths) == 20


@pytest.mark.parametrize("relative", ["../escape", "C:/escape", "/escape", "a/../b", "a\\b"])
def test_unsafe_output_relative_paths_are_rejected(tmp_path, relative):
    with pytest.raises(ValueError):
        multi.safe_path(tmp_path, relative)
