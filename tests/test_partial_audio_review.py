"""Scientific review regressions and explicit real-origin editing controls."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_partial_audio import package as package
from test_partial_audio import source as source

from data.audio_io import read_processed, write_audio
from data.invariants import ARCHIVE_SR
from data.partial_audio import (
    REVIEW_SCHEMA_VERSION,
    SCHEMA_VERSION,
    PartialAudioConfig,
    UnsuitableUtterance,
    choose_real_donor,
    choose_region,
    splice,
)
from experiments.partial_audio import (
    FLAGS,
    artifact_inventory,
    build_partial_dataset,
    config_record,
    plan_partial_dataset,
    validate_partial_dataset,
    write_json,
)
from vocoders.checkpoints import sha256_file


def reseal(root):
    """Allow semantic tests to reach checks beyond ordinary accidental hash changes."""
    write_json(root / "artifacts.sha256.json", {"files": artifact_inventory(root), **FLAGS})


@pytest.fixture(scope="module")
def reviewed(source, tmp_path_factory):
    root = tmp_path_factory.mktemp("reviewed-output") / "pilot"
    config = PartialAudioConfig(n_utterances=3, segment_seconds=0.3,
                                boundary_margin_seconds=0.1, split_counts=(1, 1, 1),
                                real_to_real_controls=True, demo_examples=3)
    before = {p.relative_to(source.parent).as_posix(): sha256_file(p)
              for p in source.parent.rglob("*") if p.is_file()}
    summary = build_partial_dataset(source, root, config)
    assert before == {p.relative_to(source.parent).as_posix(): sha256_file(p)
                      for p in source.parent.rglob("*") if p.is_file()}
    return source, root, config, summary


def test_distinct_donor_is_same_utterance_active_and_disjoint():
    t = np.arange(2 * ARCHIVE_SR) / ARCHIVE_SR
    real = (0.08 * np.sin(2 * np.pi * 137 * t)).astype(np.float32)
    config = PartialAudioConfig(segment_seconds=0.4, real_to_real_controls=True)
    start, end, _ = choose_region(real, config, "fixture")
    donor = choose_real_donor(real, config, "fixture", 1, (start, end))
    assert donor == choose_real_donor(real, config, "fixture", 1, (start, end))
    assert donor["end_sample"] <= start or donor["start_sample"] >= end
    assert donor["end_sample"] - donor["start_sample"] == end - start
    assert donor["activity"]["method"] == "real_rms_frame_activity_proxy"
    assert not np.array_equal(real[start:end], real[donor["start_sample"]:donor["end_sample"]])


def test_identical_real_excerpts_are_not_relabelled_as_non_null_controls():
    real = np.full(2 * ARCHIVE_SR, 0.1, dtype=np.float32)
    config = PartialAudioConfig(segment_seconds=0.4, real_to_real_controls=True)
    start, end, _ = choose_region(real, config, "constant")
    with pytest.raises(UnsuitableUtterance, match="no_distinct_disjoint"):
        choose_real_donor(real, config, "constant", 1, (start, end))


@pytest.mark.parametrize("fade", [0, 1, 3])
def test_relocated_splice_uses_declared_donor_indices(fade):
    real = np.arange(60, dtype=np.float32) / 1000
    out = splice(real, real, 35, 50, fade, donor_start=5)
    expected = real.copy()
    for offset in range(15):
        weight = (offset + 1) / (fade + 1) if offset < fade else (
            (15 - offset) / (fade + 1) if offset >= 15 - fade else 1.0
        )
        expected[35 + offset] = ((1 - weight) * float(real[35 + offset])
                                 + weight * float(real[5 + offset]))
    np.testing.assert_allclose(out, expected, rtol=0, atol=5e-9)
    np.testing.assert_array_equal(out[:35], real[:35])
    np.testing.assert_array_equal(out[50:], real[50:])
    np.testing.assert_array_equal(out[35 + fade:50 - fade], real[5 + fade:20 - fade])


@pytest.mark.parametrize("origin", [-1, 59, 0.5, True])
def test_bad_real_donor_bounds(origin):
    with pytest.raises(ValueError, match="donor excerpt"):
        splice(np.zeros(60), np.zeros(60), 35, 50, 3, donor_start=origin)


@pytest.mark.parametrize("values", [(1, 1), (1, 1, 1, 1), (1, -1, 3),
                                   (True, 1, 1), (1.0, 1, 1), (0, 0, 0)])
def test_invalid_explicit_split_counts(values):
    with pytest.raises(ValueError):
        PartialAudioConfig(n_utterances=3, split_counts=values).validate()


@pytest.mark.parametrize("flag", [0, 1, "true", None])
def test_control_flag_requires_boolean(flag):
    with pytest.raises(ValueError):
        PartialAudioConfig(real_to_real_controls=flag).validate()


def test_config_versions_and_json_roundtrip():
    old = PartialAudioConfig()
    assert old.schema_version == SCHEMA_VERSION
    assert "split_counts" not in config_record(old)
    assert "real_to_real_controls" not in config_record(old)
    new = replace(old, real_to_real_controls=True, split_counts=(4, 4, 4))
    assert new.schema_version == REVIEW_SCHEMA_VERSION
    assert PartialAudioConfig(**json.loads(json.dumps(config_record(new)))) == new


def test_reviewed_counts_and_original_split_inheritance(reviewed):
    source, root, _, summary = reviewed
    assert validate_partial_dataset(root, source) == summary
    assert summary["total_generated_files"] == 15
    assert summary["mixed_files"] == 9
    assert summary["null_controls"] == summary["real_to_real_controls"] == 3
    assert summary["real_only_controls"] == 6
    assert summary["split_coverage_complete"] is True
    assert summary["requested_utterances_by_split"] == {"train": 1, "dev": 1, "eval": 1}
    for split, counts in summary["files_by_class_and_split"].items():
        assert split in ("train", "dev", "eval")
        assert set(counts.values()) == {1}
    original = pd.read_csv(source).groupby("utt_id").split.first().to_dict()
    rows = json.loads((root / "manifest.json").read_text())["examples"]
    assert all(row["split"] == original[row["group_id"]] for row in rows)
    assert len({row["example_id"] for row in rows}) == 15


def test_real_to_real_audio_and_label_semantics(reviewed):
    source, root, _, _ = reviewed
    rows = json.loads((root / "manifest.json").read_text())["examples"]
    catalog = json.loads((root / "sources.json").read_text())["sources"]
    by_id = {row["example_id"]: row for row in rows}
    for row in rows:
        control = by_id[row["matched_real_to_real_control_id"]]
        assert control["example_class"] == "real_to_real_control"
        assert control["source_real_id"] == control["source_donor_id"]
        if row["example_class"] != "real_to_real_control":
            continue
        assert row["source_condition"] == row["source_donor_condition"] == "real"
        assert not row["has_synthetic"] and not row["same_source_null_edit"]
        assert row["source_synthetic_id"] is None and not row["pure_synthetic_ranges"]
        assert row["intentional_excerpt_relocation"] is True
        assert row["content_policy"] == "different_real_region_not_content_matched"
        assert all(label["label"] in (-1, 0) for label in row["interval_labels"])
        assert sum(label["end_sample"] - label["start_sample"] for label in row["interval_labels"]
                   if label["label"] == -1) == row["ignore_sample_count"]
        s, e, d, f = (row[key] for key in ("start_sample", "end_sample", "donor_start_sample",
                                           "crossfade_samples_per_boundary"))
        real = read_processed(source.parent / catalog[row["source_real_id"]]["path"], ARCHIVE_SR)
        out = read_processed(root / row["output_filename"], ARCHIVE_SR)
        np.testing.assert_array_equal(out[:s], real[:s])
        np.testing.assert_array_equal(out[e:], real[e:])
        np.testing.assert_array_equal(out[s + f:e - f], real[d + f:d + e - s - f])
        assert row["changed_core_sample_count"] > 0


def test_v2_deterministic_regeneration_and_relocation(reviewed, tmp_path):
    source, original, config, _ = reviewed
    root = tmp_path / "repeated"
    build_partial_dataset(source, root, config)
    assert (root / "artifacts.sha256.json").read_bytes() == (
        original / "artifacts.sha256.json").read_bytes()
    moved_source = tmp_path / "source"
    shutil.copytree(source.parent, moved_source)
    assert validate_partial_dataset(root, moved_source / "manifest.csv")["status"] == "passed"


def test_every_reviewed_variant_inherits_the_same_frozen_source_group(source, tmp_path):
    root = tmp_path / "variants"
    config = PartialAudioConfig(n_utterances=1, split_counts=(0, 1, 0),
                                real_to_real_controls=True, variants=2, segment_seconds=0.3,
                                boundary_margin_seconds=0.1, demo_examples=0)
    summary = build_partial_dataset(source, root, config)
    assert summary["total_generated_files"] == 10
    rows = json.loads((root / "manifest.json").read_text())["examples"]
    assert {row["variant"] for row in rows} == {1, 2}
    assert {row["split"] for row in rows} == {"dev"}
    assert len({row["group_id"] for row in rows}) == 1
    assert all(row["source_real_id"] == row["source_donor_id"] for row in rows
               if row["example_class"] == "real_to_real_control")


def test_real_control_failure_blocks_second_variant(source, tmp_path, monkeypatch):
    import experiments.partial_audio as module

    original = module.validate_written_example
    seen = []

    def validate(root, row, real, donor):
        seen.append(row["variant"])
        if row.get("example_class") == "real_to_real_control":
            raise ValueError("deliberate real control failure")
        return original(root, row, real, donor)

    monkeypatch.setattr(module, "validate_written_example", validate)
    with pytest.raises(ValueError, match="real control failure"):
        build_partial_dataset(source, tmp_path / "bad-real-control", PartialAudioConfig(
            n_utterances=1, variants=2, real_to_real_controls=True, segment_seconds=0.3,
        ))
    assert seen == [1] * 5
    assert not (tmp_path / "bad-real-control").exists()


def test_insufficient_split_reports_shortfall_without_backfill(source, tmp_path):
    root = tmp_path / "shortfall"
    config = PartialAudioConfig(n_utterances=4, split_counts=(1, 2, 1), real_to_real_controls=True,
                                segment_seconds=0.3, boundary_margin_seconds=0.1, demo_examples=0)
    summary = build_partial_dataset(source, root, config)
    assert summary["utterance_groups_by_split"] == {"train": 1, "dev": 1, "eval": 1}
    assert summary["split_shortfalls"] == {"train": 0, "dev": 1, "eval": 0}
    assert not summary["split_coverage_complete"]
    assert summary["status"] == "passed"  # Integrity and requested coverage are distinct.


def test_explicit_selection_never_draws_from_an_unrequested_split(source):
    config = PartialAudioConfig(n_utterances=1, split_counts=(0, 1, 0))
    result = plan_partial_dataset(source, config)
    assert len(result["selected"]) == 1
    assert result["selected"][0]["sources"]["real"]["split"] == "dev"
    assert result["shortfalls_by_split"] == {"train": 0, "dev": 0, "eval": 0}


def test_no_distinct_donor_excludes_whole_group_without_waiving_activity(source, tmp_path):
    original = tmp_path / "source"
    shutil.copytree(source.parent, original)
    path = original / "archive/real/u0.wav"
    write_audio(path, np.full(round(1.6 * ARCHIVE_SR), 0.05, dtype=np.float32), ARCHIVE_SR)
    config = PartialAudioConfig(n_utterances=3, split_counts=(1, 1, 1), real_to_real_controls=True,
                                segment_seconds=0.3, boundary_margin_seconds=0.1)
    result = plan_partial_dataset(original / "manifest.csv", config)
    assert len(result["selected"]) == 2
    assert result["shortfalls_by_split"]["train"] == 1
    assert result["excluded"][0]["reason"] == "no_distinct_disjoint_real_donor_region"


@pytest.mark.parametrize("duration", [np.nan, np.inf, -np.inf, 0.0, "invalid"])
def test_nonfinite_or_invalid_source_duration_is_rejected(source, tmp_path, duration):
    original = tmp_path / "source"
    shutil.copytree(source.parent, original)
    frame = pd.read_csv(original / "manifest.csv")
    frame["duration_s"] = frame.duration_s.astype(object)
    frame.loc[(frame.utt_id == "u0") & (frame.condition == "vocos"), "duration_s"] = duration
    frame.to_csv(original / "manifest.csv", index=False)
    result = plan_partial_dataset(original / "manifest.csv", PartialAudioConfig(n_utterances=3))
    assert all(item["utt_id"] != "u0" for item in result["selected"])
    assert result["excluded"][0]["reason"] == "invalid_source_duration_record"


def test_nested_inventory_receipt_name_cannot_hide_files(package, tmp_path):
    source, original, _, _ = package
    root = tmp_path / "copy"
    shutil.copytree(original, root)
    (root / "demo/artifacts.sha256.json").write_text("unlisted", encoding="utf-8")
    assert "demo/artifacts.sha256.json" in artifact_inventory(root)
    with pytest.raises(ValueError, match="inventory/hash"):
        validate_partial_dataset(root, source)


@pytest.mark.parametrize("field,value", [
    ("processing", "class_specific_gain"), ("rounding", "truncate"),
    ("phase_a_changed", True), ("signal_gates_rerun", True),
    ("research_eligibility", "citable"), ("phase_a_changed", 0),
])
def test_contradictory_provenance_rejected_even_after_resealing(package, tmp_path, field, value):
    source, original, _, _ = package
    root = tmp_path / "copy"
    shutil.copytree(original, root)
    path = root / "provenance.json"
    document = json.loads(path.read_text())
    document[field] = value
    write_json(path, document)
    reseal(root)
    with pytest.raises(ValueError, match="provenance claim"):
        validate_partial_dataset(root, source)


def test_validation_uses_no_writer_or_generator_or_temporary_directories(package, monkeypatch):
    import experiments.partial_audio as module

    source, root, _, _ = package

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent validation must not call generation or write files")

    monkeypatch.setattr(module, "splice", forbidden)
    monkeypatch.setattr(module, "write_audio", forbidden)
    assert validate_partial_dataset(root, source)["status"] == "passed"


def test_independent_reconstruction_rejects_generator_fade_bug(source, tmp_path, monkeypatch):
    import experiments.partial_audio as module

    original = module.splice

    def bad_splice(real, donor, start, end, fade, **kwargs):
        out = original(real, donor, start, end, fade, **kwargs)
        if not np.array_equal(real, donor):
            out[start + 1] += 2 / 32768
        return out

    monkeypatch.setattr(module, "splice", bad_splice)
    with pytest.raises(ValueError, match="does not reproduce"):
        build_partial_dataset(source, tmp_path / "bad-generator",
                              PartialAudioConfig(n_utterances=1))
    assert not (tmp_path / "bad-generator").exists()


def test_pcm16_quantization_oracle_matches_signed_grid_and_edges(tmp_path):
    from experiments.partial_audio import expected_pcm16

    x = np.array([0, .1, .5, .9, 1, 1.1, -.1, -.5, -.9, -1, -1.1, 0], dtype=np.float32) / 32768
    path = tmp_path / "grid.wav"
    write_audio(path, x, ARCHIVE_SR)
    np.testing.assert_array_equal(expected_pcm16(x, x, 2, 10, 1, 2),
                                   read_processed(path, ARCHIVE_SR))


@pytest.mark.parametrize("field,value", [
    ("has_synthetic", True), ("source_donor_condition", "vocos"),
    ("donor_start_sample", 0), ("same_source_null_edit", True),
    ("source_synthetic_id", "wrong"), ("intentional_excerpt_relocation", False),
])
def test_real_control_corrupt_semantics_rejected(reviewed, tmp_path, field, value):
    source, original, _, _ = reviewed
    root = tmp_path / "copy"
    shutil.copytree(original, root)
    path = root / "manifest.json"
    document = json.loads(path.read_text())
    row = next(row for row in document["examples"]
               if row["example_class"] == "real_to_real_control")
    row[field] = value
    write_json(path, document)
    reseal(root)
    with pytest.raises(ValueError, match="Manifest differs"):
        validate_partial_dataset(root, source)


def test_cli_explicit_counts_and_control_flag(source, tmp_path, capsys):
    from experiments.cli import main

    root = tmp_path / "cli"
    assert main(["partial", "build", "--manifest", str(source), "--out-dir", str(root),
                 "--split-count", "train=1", "--split-count", "dev=1", "--split-count", "eval=1",
                 "--real-to-real-controls", "--segment-seconds", "0.3"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["total_generated_files"] == 15 and summary["real_to_real_controls"] == 3


@pytest.mark.parametrize("specifications", [["train=1", "train=1"], ["unknown=1"],
                                            ["dev=1.5"], ["eval=-1"], ["train"]])
def test_bad_cli_split_requests_are_not_silently_interpreted(source, tmp_path, specifications):
    from experiments.cli import main

    args = ["partial", "build", "--manifest", str(source), "--out-dir", str(tmp_path / "out")]
    for item in specifications:
        args += ["--split-count", item]
    with pytest.raises(ValueError):
        main(args)
    assert not (tmp_path / "out").exists()
