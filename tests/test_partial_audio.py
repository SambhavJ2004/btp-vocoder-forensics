"""Deterministic CPU tests for edit provenance and exact decoded sample contracts."""

from __future__ import annotations

import json
import shutil
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.audio_io import write_audio
from data.invariants import ARCHIVE_SR, InvariantViolation
from data.manifest import ManifestRow, write_manifest
from data.partial_audio import (
    IGNORE,
    REAL,
    SYNTHETIC,
    VOCODERS,
    PartialAudioConfig,
    UnsuitableUtterance,
    activity_candidates,
    choose_region,
    interval_labels,
    splice,
    validate_region,
)
from experiments.partial_audio import (
    FLAGS,
    build_partial_dataset,
    package_path,
    plan_partial_dataset,
    read_input,
    validate_partial_dataset,
    write_json,
)
from vocoders.checkpoints import sha256_file


@pytest.mark.parametrize("fade", [0, 1, 2, 5])
def test_splice_matches_independent_complementary_weights(fade):
    real = np.linspace(-0.25, 0.25, 40, dtype=np.float32)
    donor = np.linspace(0.4, -0.4, 40, dtype=np.float32)
    first, second = real.copy(), donor.copy()
    start, end = 7, 32
    output = splice(real, donor, start, end, fade)
    expected = real.astype(float)
    for sample in range(start, end):
        weight = 1.0
        if fade and sample < start + fade:
            weight = (sample - start + 1) / (fade + 1)
        elif fade and sample >= end - fade:
            weight = (end - sample) / (fade + 1)
        expected[sample] = (1 - weight) * float(real[sample]) + weight * float(donor[sample])
    np.testing.assert_allclose(output, expected, rtol=0, atol=2e-8)
    np.testing.assert_array_equal(output[:start], real[:start])
    np.testing.assert_array_equal(output[end:], real[end:])
    np.testing.assert_array_equal(output[start + fade:end - fade], donor[start + fade:end - fade])
    np.testing.assert_array_equal(real, first)
    np.testing.assert_array_equal(donor, second)
    assert output.dtype == np.float32


@pytest.mark.parametrize("fade", [0, 1, 3, 10])
def test_null_edits_are_bit_exact(fade):
    real = np.random.default_rng(3).uniform(-0.2, 0.2, 80).astype(np.float32)
    np.testing.assert_array_equal(splice(real, real.copy(), 10, 60, fade), real)


@pytest.mark.parametrize("synthetic", [False, True])
@pytest.mark.parametrize("fade", [0, 1, 3])
def test_label_coverage_and_transition_samples(synthetic, fade):
    labels = interval_labels(40, 8, 30, fade, synthetic=synthetic)
    dense = np.full(40, 99)
    previous = 0
    for item in labels:
        assert item["start_sample"] == previous < item["end_sample"]
        dense[item["start_sample"]:item["end_sample"]] = item["label"]
        previous = item["end_sample"]
    assert previous == 40 and np.count_nonzero(dense == IGNORE) == 2 * fade
    assert np.all(dense[:8] == REAL) and np.all(dense[30:] == REAL)
    assert np.all(dense[8 + fade:30 - fade] == (SYNTHETIC if synthetic else REAL))
    if fade:
        assert np.all(dense[8:8 + fade] == IGNORE)
        assert np.all(dense[30 - fade:30] == IGNORE)


@pytest.mark.parametrize("start,end,fade", [(0, 8, 1), (2, 40, 1), (8, 8, 0),
                                           (9, 8, 0), (3, 8, -1), (3, 8, 3),
                                           (True, 8, 1), (2.5, 8, 1)])
def test_invalid_sample_regions_rejected(start, end, fade):
    with pytest.raises(ValueError):
        validate_region(40, start, end, fade)


@pytest.mark.parametrize("kind", ["length", "stereo", "nan", "inf", "complex"])
def test_invalid_splice_arrays(kind):
    real, donor = np.zeros(40), np.ones(40)
    if kind == "length":
        donor = donor[:-1]
    elif kind == "stereo":
        real = real.reshape(20, 2)
    elif kind == "nan":
        real[0] = np.nan
    elif kind == "inf":
        donor[0] = np.inf
    else:
        donor = donor.astype(complex)
    with pytest.raises(ValueError):
        splice(real, donor, 5, 20, 2)


@pytest.mark.parametrize("field,value", [
    ("seed", -1), ("seed", True), ("seed", 2**63), ("n_utterances", 0),
    ("n_utterances", 13), ("n_utterances", 1.5), ("variants", 3),
    ("demo_examples", 6), ("segment_seconds", 0), ("segment_seconds", float("nan")),
    ("segment_seconds", "1"), ("crossfade_ms", -1), ("crossfade_ms", 500),
    ("crossfade_ms", float("inf")), ("boundary_margin_seconds", 0),
    ("minimum_active_fraction", 0), ("minimum_active_fraction", 1.1),
    ("activity_top_db", -1), ("boundary_margin_seconds", 1e-12),
])
def test_invalid_config_rejected(field, value):
    with pytest.raises(ValueError):
        replace(PartialAudioConfig(), **{field: value}).validate()


def test_activity_policy_determinism_and_disjoint_variant():
    config = PartialAudioConfig(segment_seconds=0.3, boundary_margin_seconds=0.2)
    t = np.arange(2 * ARCHIVE_SR) / ARCHIVE_SR
    reference = (0.08 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    reference[:int(0.4 * ARCHIVE_SR)] = 0
    reference[int(1.7 * ARCHIVE_SR):] = 0
    candidates, activity = activity_candidates(reference, config)
    first = choose_region(reference, config, "u1")
    assert first == choose_region(reference, config, "u1")
    assert first[0] in candidates
    assert first[0] >= config.margin_samples
    assert first[1] <= len(reference) - config.margin_samples
    assert first[0] >= int(0.38 * ARCHIVE_SR)
    assert first[1] <= int(1.72 * ARCHIVE_SR)
    second = choose_region(reference, config, "u1", 2, previous=first[:2])
    assert second[1] <= first[0] or second[0] >= first[1]
    assert activity["method"] == "real_rms_frame_activity_proxy"
    assert activity["minimum_active_fraction"] == 0.8


@pytest.mark.parametrize("reference", [np.zeros(ARCHIVE_SR), np.ones(30),
                                       np.full(ARCHIVE_SR, 1 / 32768)])
def test_unsuitable_reference_activity(reference):
    with pytest.raises(UnsuitableUtterance):
        choose_region(reference, PartialAudioConfig(), "u")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    root = tmp_path_factory.mktemp("partial-source") / "source"
    rows = []
    n = round(1.6 * ARCHIVE_SR)
    t = np.arange(n) / ARCHIVE_SR
    for i, split in enumerate(("train", "dev", "eval")):
        for j, condition in enumerate(("real", *VOCODERS)):
            wav = (0.07 * np.sin(2 * np.pi * (180 + 91 * j + i) * t)).astype(np.float32)
            path = root / "archive" / condition / f"u{i}.wav"
            write_audio(path, wav, ARCHIVE_SR)
            # Synthetic test-only declarations, not measured scientific gate receipts.
            rows.append(ManifestRow(
                utt_id=f"u{i}", condition=condition, path=str(path), speaker="fixture",
                split=split, duration_s=n / ARCHIVE_SR, lufs_out=-27.0,
                trim_start=0, trim_end=n, trim_ref_length=n, lufs_in=-27.0,
                lufs_in_band=-27.0, lufs_out_band=-27.0, loudness_gain_db=0.0,
                loudness_target_lufs=-27.0, alignment_checked=True,
                alignment_peak_lag=0, high_band_fraction=0.0,
            ))
    manifest = write_manifest(rows, root / "manifest.csv")
    write_json(root / "manifest.provenance.json", {"mode": "legacy", "sha": "unknown",
                                                  "citable": False})
    return manifest


@pytest.fixture(scope="module")
def package(source, tmp_path_factory):
    root = tmp_path_factory.mktemp("partial-output") / "pilot"
    config = PartialAudioConfig(n_utterances=3, segment_seconds=0.3,
                                boundary_margin_seconds=0.1)
    before = {path.relative_to(source.parent).as_posix(): sha256_file(path)
              for path in source.parent.rglob("*") if path.is_file()}
    summary = build_partial_dataset(source, root, config)
    after = {path.relative_to(source.parent).as_posix(): sha256_file(path)
             for path in source.parent.rglob("*") if path.is_file()}
    assert before == after
    return source, root, config, summary


def test_package_counts_labels_sources_and_splits(package):
    source, root, config, summary = package
    assert summary == validate_partial_dataset(root, source)
    assert summary["utterances_used"] == 3
    assert summary["mixed_files"] == 9 and summary["real_only_controls"] == 3
    assert summary["total_generated_files"] == 12
    assert summary["total_duration_seconds"] == pytest.approx(19.2)
    assert summary["utterance_groups_by_split"] == {"train": 1, "dev": 1, "eval": 1}
    assert summary["source_files_verified_unchanged"] == 12
    document = json.loads((root / "manifest.json").read_text())
    sources = json.loads((root / "sources.json").read_text())["sources"]
    groups = {}
    for row in document["examples"]:
        groups.setdefault(row["group_id"], []).append(row)
        assert row["development_only"] is True and row["citable"] is False
        assert row["validation_status"] == "passed"
        output = read_input(root / row["output_filename"])
        real = read_input(source.parent / sources[row["source_real_id"]]["path"])
        donor = read_input(source.parent / sources[row["source_donor_id"]]["path"])
        start, end, fade = row["start_sample"], row["end_sample"], config.fade_samples
        np.testing.assert_array_equal(output[:start], real[:start])
        np.testing.assert_array_equal(output[end:], real[end:])
        np.testing.assert_array_equal(
            output[start + fade:end - fade], donor[start + fade:end - fade],
        )
        assert row["generated_sha256"] == sha256_file(root / row["output_filename"])
        assert row["start_seconds"] == start / ARCHIVE_SR
        assert row["ignore_sample_count"] == 2 * fade
        assert Path(row["output_filename"]).stem == row["example_id"]
        if row["same_source_null_edit"]:
            assert row["changed_sample_count"] == 0 and row["source_synthetic_id"] is None
            np.testing.assert_array_equal(output, real)
    for values in groups.values():
        assert len({(r["start_sample"], r["end_sample"]) for r in values}) == 1
        controls = [r for r in values if r["same_source_null_edit"]]
        assert len(controls) == 1
        assert all(r["matched_control_id"] == controls[0]["example_id"] for r in values)
    assert all(item["upstream_vocoder_commit_recorded"] is None for item in sources.values())
    assert not list(root.parent.glob("pilot.staging"))


def test_preflight_writes_nothing_and_ignores_manifest_row_order(source, tmp_path, monkeypatch):
    import experiments.partial_audio as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight may not write audio")

    monkeypatch.setattr(module, "write_audio", forbidden)
    config = PartialAudioConfig(n_utterances=3, segment_seconds=0.3)
    first = plan_partial_dataset(source, config)
    copy = tmp_path / "source"
    shutil.copytree(source.parent, copy)
    frame = pd.read_csv(copy / "manifest.csv").iloc[::-1]
    frame.to_csv(copy / "manifest.csv", index=False)
    second = plan_partial_dataset(copy / "manifest.csv", config)
    assert [item["utt_id"] for item in first["selected"]] == [
        item["utt_id"] for item in second["selected"]
    ]
    assert [item["first_region"] for item in first["selected"]] == [
        item["first_region"] for item in second["selected"]
    ]
    assert not first["excluded"]


def test_reproducible_package_and_relocated_validation(package, tmp_path):
    source, root, config, _ = package
    repeated = tmp_path / "repeat"
    build_partial_dataset(source, repeated, config)
    assert (root / "artifacts.sha256.json").read_bytes() == (
        repeated / "artifacts.sha256.json").read_bytes()
    relocated_source = tmp_path / "relocated-source"
    shutil.copytree(source.parent, relocated_source)
    result = validate_partial_dataset(repeated, relocated_source / "manifest.csv")
    assert result["status"] == "passed"


def test_second_variant_is_disjoint_and_grouped(source, tmp_path):
    config = PartialAudioConfig(n_utterances=1, segment_seconds=0.3, variants=2, demo_examples=0)
    root = tmp_path / "two-variants"
    summary = build_partial_dataset(source, root, config)
    assert summary["mixed_files"] == 6 and summary["real_only_controls"] == 2
    rows = json.loads((root / "manifest.json").read_text())["examples"]
    first, second = rows[0], rows[4]
    assert (first["end_sample"] <= second["start_sample"]
            or second["end_sample"] <= first["start_sample"])
    assert len({row["group_id"] for row in rows}) == 1


def test_unavailable_second_variant_recorded(source, tmp_path):
    config = PartialAudioConfig(n_utterances=1, variants=2, demo_examples=0)
    summary = build_partial_dataset(source, tmp_path / "no-second", config)
    assert summary["mixed_files"] == 3 and summary["excluded_variants"] == 1
    assert summary["variant_exclusions"][0]["reason"] == "no_disjoint_second_variant_region"


def test_first_variant_failure_never_attempts_second_or_publishes(source, tmp_path, monkeypatch):
    import experiments.partial_audio as module

    seen = []
    original = module.validate_written_example

    def validate(root, row, real, donor):
        seen.append(row["variant"])
        if row["source_condition"] == "vocos":
            raise ValueError("deliberate validation failure")
        return original(root, row, real, donor)

    monkeypatch.setattr(module, "validate_written_example", validate)
    target = tmp_path / "failed"
    with pytest.raises(ValueError, match="deliberate"):
        build_partial_dataset(source, target, PartialAudioConfig(n_utterances=1, variants=2))
    assert set(seen) == {1} and not target.exists()
    assert json.loads((tmp_path / "failed.staging/failure.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("kind", ["missing", "rate", "stereo", "subtype", "container",
                                  "length", "silence"])
def test_unsuitable_utterance_excluded_as_a_whole(source, tmp_path, kind):
    copy = tmp_path / "source"
    shutil.copytree(source.parent, copy)
    path = copy / "archive/vocos/u0.wav"
    if kind == "missing":
        path.unlink()
    elif kind in ("rate", "stereo", "subtype"):
        # Invalid-header fixtures intentionally bypass the sanctioned writer.
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(2 if kind == "stereo" else 1)
            stream.setsampwidth(4 if kind == "subtype" else 2)
            stream.setframerate(16000 if kind == "rate" else ARCHIVE_SR)
            stream.writeframes(b"\0" * 10000)
    elif kind == "container":
        path.write_bytes(b"fLaC" + b"\0" * 100)
    elif kind == "length":
        write_audio(path, read_input(path)[:-100], ARCHIVE_SR)
    else:
        reference = copy / "archive/real/u0.wav"
        write_audio(reference, np.zeros_like(read_input(reference)), ARCHIVE_SR)
    result = plan_partial_dataset(copy / "manifest.csv", PartialAudioConfig(n_utterances=3))
    assert len(result["selected"]) == 2
    assert len(result["excluded"]) == 1 and result["excluded"][0]["utt_id"] == "u0"
    assert all(item["utt_id"] != "u0" for item in result["selected"])


def test_existing_invariant_failure_is_not_converted_to_exclusion(source, tmp_path, monkeypatch):
    import experiments.partial_audio as module

    def refuse(*args, **kwargs):
        raise InvariantViolation("deliberate source gate refusal")

    monkeypatch.setattr(module, "read_processed", refuse)
    with pytest.raises(InvariantViolation, match="gate refusal"):
        build_partial_dataset(source, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist.staging").exists()


@pytest.mark.parametrize("damage", ["split", "timestamp", "labels", "transition", "hash",
                                    "donor", "seed", "status", "citable", "development"])
def test_manifest_semantic_corruption_rejected(package, tmp_path, damage):
    source, original, _, _ = package
    root = tmp_path / "copy"
    shutil.copytree(original, root)
    path = root / "manifest.json"
    document = json.loads(path.read_text())
    row = document["examples"][1]
    changes = {"split": ("split", "other"), "timestamp": ("start_seconds", 99),
               "labels": ("interval_labels", []), "transition": ("transition_ignore_ranges", []),
               "hash": ("generated_sha256", "0" * 64), "donor": ("source_donor_id", "wrong"),
               "seed": ("seed", 3), "status": ("validation_status", "failed"),
               "citable": ("citable", True), "development": ("development_only", False)}
    key, value = changes[damage]
    row[key] = value
    write_json(path, document)
    with pytest.raises(ValueError):
        validate_partial_dataset(root, source)


@pytest.mark.parametrize("where", ["outside", "core", "transition"])
def test_changed_output_samples_rejected(package, tmp_path, where):
    source, original, _, _ = package
    root = tmp_path / "copy"
    shutil.copytree(original, root)
    row = json.loads((root / "manifest.json").read_text())["examples"][1]
    path = root / row["output_filename"]
    output = read_input(path)
    index = {"outside": 0, "core": row["start_sample"] + 400,
             "transition": row["start_sample"] + 2}[where]
    output[index] += 1 / 32768
    write_audio(path, output, ARCHIVE_SR)
    with pytest.raises(ValueError):
        validate_partial_dataset(root, source)


@pytest.mark.parametrize("damage", ["labels", "csv", "timeline", "demo_audio", "receipt",
                                    "extra_audio", "variant_exclusions", "validation_summary"])
def test_sidecar_preview_and_inventory_corruption_rejected(package, tmp_path, damage):
    source, original, _, _ = package
    root = tmp_path / "copy"
    shutil.copytree(original, root)
    if damage == "labels":
        path = root / "labels.json"
        document = json.loads(path.read_text())
        document["indexing"] = "inclusive"
        write_json(path, document)
    elif damage in ("csv", "timeline", "receipt"):
        path = root / {"csv": "manifest.csv", "timeline": "demo/timeline.svg",
                       "receipt": "artifacts.sha256.json"}[damage]
        path.write_text("corrupt", encoding="utf-8")
    elif damage == "demo_audio":
        next((root / "demo/audio").glob("*.wav")).write_bytes(b"bad wave")
    elif damage == "extra_audio":
        shutil.copyfile(next((root / "audio").glob("*.wav")), root / "audio/extra.wav")
    else:
        path = root / {"variant_exclusions": "variant_exclusions.json",
                       "validation_summary": "validation.json"}[damage]
        write_json(path, {"wrong": True, **FLAGS})
    with pytest.raises((ValueError, OSError)):
        validate_partial_dataset(root, source)


@pytest.mark.parametrize("damage", ["audio", "manifest", "provenance"])
def test_changed_original_source_identity_rejected(package, tmp_path, damage):
    source, root, _, _ = package
    copied = tmp_path / "source"
    shutil.copytree(source.parent, copied)
    if damage == "audio":
        path = copied / "archive/vocos/u0.wav"
        wav = read_input(path)
        wav[0] += 1 / 32768
        write_audio(path, wav, ARCHIVE_SR)
    else:
        names = {"manifest": "manifest.csv", "provenance": "manifest.provenance.json"}
        path = copied / names[damage]
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        validate_partial_dataset(root, copied / "manifest.csv")


def test_path_containment_and_existing_outputs(source, package, tmp_path):
    with pytest.raises(ValueError):
        package_path(tmp_path, "../escape.wav")
    with pytest.raises(ValueError):
        build_partial_dataset(source, source.parent / "new")
    with pytest.raises(ValueError):
        build_partial_dataset(source, source.parent.parent)
    _, existing, config, _ = package
    before = sha256_file(existing / "artifacts.sha256.json")
    with pytest.raises(FileExistsError):
        build_partial_dataset(source, existing, config)
    assert sha256_file(existing / "artifacts.sha256.json") == before


def test_cli_build_and_read_only_validation(source, tmp_path, capsys):
    from experiments.cli import main

    root = tmp_path / "cli"
    assert main(["partial", "build", "--manifest", str(source), "--out-dir", str(root),
                 "--n-utterances", "1", "--segment-seconds", "0.3"]) == 0
    assert json.loads(capsys.readouterr().out)["total_generated_files"] == 4
    before = {p.relative_to(root).as_posix(): sha256_file(p)
              for p in root.rglob("*") if p.is_file()}
    assert main(["partial", "validate", "--dataset-dir", str(root),
                 "--source-manifest", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["citable"] is False
    after = {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()}
    assert before == after


def test_zero_crossfade_package_and_rounding(source, tmp_path):
    config = PartialAudioConfig(n_utterances=1, crossfade_ms=0, demo_examples=0)
    root = tmp_path / "hard-splice"
    build_partial_dataset(source, root, config)
    rows = json.loads((root / "manifest.json").read_text())["examples"]
    assert all(row["ignore_sample_count"] == 0 for row in rows)
    assert all(row["transition_ignore_ranges"] == [] for row in rows)
    assert rows[0]["interval_labels"] == [{"start_sample": 0, "end_sample": rows[0]["num_samples"],
                                           "label": REAL}]
    assert PartialAudioConfig().fade_samples == 220  # nearest, ties-to-even at 220.5
    assert PartialAudioConfig().margin_samples == 5512


@pytest.mark.parametrize("damage", ["split", "missing_pair", "duplicate", "speaker"])
def test_bad_source_table_is_fatal_not_a_sampling_exclusion(source, tmp_path, damage):
    copy = tmp_path / "source"
    shutil.copytree(source.parent, copy)
    frame = pd.read_csv(copy / "manifest.csv")
    if damage == "split":
        frame.loc[0, "split"] = "other"
    elif damage == "missing_pair":
        frame = frame.iloc[1:]
    elif damage == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    else:
        frame.loc[0, "speaker"] = "other"
    frame.to_csv(copy / "manifest.csv", index=False)
    with pytest.raises((ValueError, InvariantViolation)):
        build_partial_dataset(copy / "manifest.csv", tmp_path / "no-output")
    assert not (tmp_path / "no-output").exists()


def test_actual_pair_length_mismatch_is_not_repaired(source, tmp_path):
    copy = tmp_path / "source"
    shutil.copytree(source.parent, copy)
    path = copy / "archive/vocos/u0.wav"
    shortened = read_input(path)[:-100]
    write_audio(path, shortened, ARCHIVE_SR)
    frame = pd.read_csv(copy / "manifest.csv")
    frame.loc[(frame.utt_id == "u0") & (frame.condition == "vocos"), "duration_s"] = (
        len(shortened) / ARCHIVE_SR
    )
    frame.to_csv(copy / "manifest.csv", index=False)
    plan = plan_partial_dataset(copy / "manifest.csv", PartialAudioConfig(n_utterances=3))
    assert plan["excluded"][0]["reason"] == "paired_sample_count_mismatch"
    assert len(read_input(path)) == len(shortened)


def test_no_eligible_source_does_not_generate_audio(source, tmp_path):
    copy = tmp_path / "source"
    shutil.copytree(source.parent, copy)
    for path in (copy / "archive/real").glob("*.wav"):
        write_audio(path, np.zeros_like(read_input(path)), ARCHIVE_SR)
    with pytest.raises(ValueError, match="No eligible"):
        build_partial_dataset(copy / "manifest.csv", tmp_path / "empty")
    assert not (tmp_path / "empty").exists() and not (tmp_path / "empty.staging").exists()
