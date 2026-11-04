"""Complete frozen-source census and local-only upload preparation regressions."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import pandas as pd
import pytest
from test_partial_audio import source as source

from data.partial_audio import PartialAudioConfig
from experiments.partial_audio import (
    FLAGS,
    artifact_inventory,
    build_partial_dataset,
    config_record,
    plan_partial_dataset,
    validate_partial_dataset,
    write_json,
)
from experiments.partial_audio_upload import (
    prepare_upload_directory,
    validate_upload_directory,
)


@pytest.fixture(scope="module")
def expanded(source, tmp_path_factory):
    root = tmp_path_factory.mktemp("expansion") / "dataset"
    cfg = PartialAudioConfig(n_utterances=3, all_utterances=True,
                             real_to_real_controls=True, segment_seconds=.3,
                             boundary_margin_seconds=.1, demo_examples=3)
    summary = build_partial_dataset(source, root, cfg)
    return source, root, cfg, summary


@pytest.mark.parametrize("changes", [
    {"all_utterances": 1}, {"all_utterances": "true"}, {"real_to_real_controls": False},
    {"variants": 2}, {"split_counts": (1, 1, 1)}, {"n_utterances": 201},
])
def test_expansion_requires_explicit_bounded_controlled_census(changes):
    cfg = PartialAudioConfig(n_utterances=3, all_utterances=True, real_to_real_controls=True)
    with pytest.raises(ValueError):
        replace(cfg, **changes).validate()


def test_pilot_limit_and_serialization_preserved():
    with pytest.raises(ValueError, match="Pilot permits"):
        PartialAudioConfig(n_utterances=13).validate()
    assert "all_utterances" not in config_record(PartialAudioConfig())
    assert "all_utterances" not in config_record(PartialAudioConfig(real_to_real_controls=True))


def test_complete_census_inherits_each_original_split(expanded):
    source, root, _, summary = expanded
    assert validate_partial_dataset(root, source) == summary
    assert summary["source_census_complete"] is True
    assert summary["split_coverage_complete"] is True
    assert summary["total_generated_files"] == 15
    assert summary["available_utterances_by_split"] == {"train": 1, "dev": 1, "eval": 1}
    assert summary["inspected_utterances_by_split"] == summary["available_utterances_by_split"]
    assert set(summary["excluded_utterances_by_split"].values()) == {0}
    originals = pd.read_csv(source).groupby("utt_id").split.first().to_dict()
    rows = json.loads((root / "manifest.json").read_text())["examples"]
    for utt, split in originals.items():
        group = [r for r in rows if r["group_id"] == utt]
        assert len(group) == 5 and {r["split"] for r in group} == {split}
        assert sum(r["has_synthetic"] for r in group) == 3
    assert all(r["citable"] is False for r in rows)


def copy_source(source, destination):
    shutil.copytree(source.parent, destination)
    frame = pd.read_csv(destination / "manifest.csv")
    frame["path"] = frame.path.str.replace(str(source.parent), str(destination), regex=False)
    frame.to_csv(destination / "manifest.csv", index=False)
    return destination / "manifest.csv"


def test_every_exclusion_counted_without_backfill_or_reassignment(expanded, tmp_path):
    source, _, cfg, _ = expanded
    manifest = copy_source(source, tmp_path / "source")
    (manifest.parent / "archive/vocos/u2.wav").unlink()
    selection = plan_partial_dataset(manifest, cfg)
    assert selection["source_census_complete"] is True
    assert selection["selected_utterances_by_split"] == {"train": 1, "dev": 1, "eval": 0}
    assert selection["excluded_utterances_by_split"] == {"train": 0, "dev": 0, "eval": 1}
    assert selection["exclusion_reasons_by_split"]["eval"] == {"missing_source_file": 1}
    assert selection["shortfalls_by_split"] == {"train": 0, "dev": 0, "eval": 1}
    result = build_partial_dataset(manifest, tmp_path / "output", cfg)
    assert result["source_census_complete"] is True
    assert result["split_coverage_complete"] is False
    assert result["total_generated_files"] == 10


def test_census_cannot_silently_request_a_subset(expanded):
    source, _, cfg, _ = expanded
    with pytest.raises(ValueError, match="original manifest utterance count"):
        plan_partial_dataset(source, replace(cfg, n_utterances=2))


def test_cross_split_identical_source_audio_blocks_expansion(expanded, tmp_path):
    source, _, cfg, _ = expanded
    manifest = copy_source(source, tmp_path / "source")
    shutil.copyfile(manifest.parent / "archive/vocos/u0.wav",
                    manifest.parent / "archive/vocos/u2.wav")
    with pytest.raises(ValueError, match="crosses frozen splits"):
        plan_partial_dataset(manifest, cfg)


def test_more_than_twelve_groups_are_scanned_in_explicit_mode(expanded, tmp_path):
    source, _, cfg, _ = expanded
    manifest = copy_source(source, tmp_path / "source")
    frame = pd.read_csv(manifest)
    frames = [frame]
    for i in range(1, 5):
        extra = frame.copy()
        extra["utt_id"] = extra.utt_id + f"-{i}"
        for row in extra.itertuples():
            new_path = manifest.parent / "archive" / row.condition / f"{row.utt_id}.wav"
            shutil.copyfile(manifest.parent / row.path, new_path)
            extra.loc[row.Index, "path"] = str(new_path)
        frames.append(extra)
    pd.concat(frames).to_csv(manifest, index=False)
    selection = plan_partial_dataset(manifest, replace(cfg, n_utterances=15))
    assert len(selection["selected"]) == 15
    assert selection["selected_utterances_by_split"] == {"train": 5, "dev": 5, "eval": 5}


def test_expansion_determinism_and_pilot_subset_compatibility(expanded, tmp_path):
    source, original, cfg, _ = expanded
    before = artifact_inventory(source.parent)
    repeated = tmp_path / "repeat"
    build_partial_dataset(source, repeated, cfg)
    assert artifact_inventory(original) == artifact_inventory(repeated)
    pilot = tmp_path / "pilot"
    build_partial_dataset(source, pilot, replace(cfg, all_utterances=False, split_counts=(1, 1, 1)))
    for name in ("manifest.json", "manifest.csv", "labels.json", "sources.json"):
        assert (original / name).read_bytes() == (pilot / name).read_bytes()
    for path in (original / "audio").glob("*.wav"):
        assert path.read_bytes() == (pilot / "audio" / path.name).read_bytes()
    assert before == artifact_inventory(source.parent)


@pytest.fixture(scope="module")
def upload(expanded, tmp_path_factory):
    source, dataset, _, _ = expanded
    folder = tmp_path_factory.mktemp("upload")
    card, protocol = folder / "card.md", folder / "protocol.md"
    card.write_text("Development only, non-citable. Card fixture.")
    protocol.write_text("Development only, non-citable. Protocol fixture.")
    root = folder / "prepared"
    before = artifact_inventory(dataset)
    summary = prepare_upload_directory(dataset, source, root, card, protocol)
    assert before == artifact_inventory(dataset)
    return source, dataset, root, card, protocol, summary


def test_upload_contains_all_original_bytes_and_no_executable_metadata(upload):
    source, dataset, root, _, _, summary = upload
    assert summary == validate_upload_directory(root, source)
    assert summary["primary_wavs"] == 15 and summary["publication_ready"] is False
    for name, digest in artifact_inventory(dataset).items():
        assert artifact_inventory(root)[name] == digest
    assert not (root / "dataset-metadata.json").exists()
    plan = json.loads((root / "upload_plan.json").read_text())
    assert plan["planned_visibility"] == "private" and plan["upload_authorized"] is False
    metadata = json.loads((root / "dataset-metadata.template.json").read_text())
    assert metadata["id"] is None and metadata["licenses"] == []


@pytest.mark.parametrize("kind", ["source", "dataset", "exists", "staging"])
def test_upload_refuses_unsafe_or_existing_paths(upload, tmp_path, kind):
    source, dataset, _, card, protocol, _ = upload
    target = {"source": source.parent / "upload", "dataset": dataset / "upload"}.get(
        kind, tmp_path / "new",
    )
    if kind in {"exists", "staging"}:
        (target if kind == "exists" else target.with_name(target.name + ".staging")).mkdir()
    with pytest.raises((ValueError, FileExistsError)):
        prepare_upload_directory(dataset, source, target, card, protocol)


@pytest.mark.parametrize("kind", ["audio", "private", "metadata", "documentation", "inventory"])
def test_upload_rejects_corruption(upload, tmp_path, kind):
    source, _, original, _, _, _ = upload
    root = tmp_path / "copy"
    shutil.copytree(original, root)
    if kind == "audio":
        next((root / "audio").glob("*.wav")).write_bytes(b"corrupt")
    elif kind == "private":
        path = root / "upload_plan.json"
        doc = json.loads(path.read_text())
        doc["planned_visibility"] = "public"
        write_json(path, doc)
    elif kind == "metadata":
        write_json(root / "dataset-metadata.json", {"id": "invented/owner"})
    elif kind == "documentation":
        (root / "PROTOCOL.md").write_text("wrong")
    else:
        (root / "unlisted.txt").write_text("extra")
    with pytest.raises((ValueError, RuntimeError)):
        validate_upload_directory(root, source)


def test_preparation_rejects_unexpected_even_resealed_files(upload, tmp_path):
    source, dataset, _, card, protocol, _ = upload
    root = tmp_path / "copy"
    shutil.copytree(dataset, root)
    (root / "kaggle.json").write_text("test-only placeholder, no credentials")
    write_json(root / "artifacts.sha256.json", {"files": artifact_inventory(root), **FLAGS})
    with pytest.raises(ValueError, match="Unexpected artifact"):
        prepare_upload_directory(root, source, tmp_path / "prepared", card, protocol)
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("field,value", [
    ("operations", "uploaded_publicly"), ("dataset_slug", "btpvf-v0"),
    ("generation_inventory_receipt_sha256", "0" * 64), ("documentation_sha256", {}),
])
def test_preparation_receipt_claims_rejected_even_after_resealing(upload, tmp_path, field, value):
    source, _, original, _, _, _ = upload
    root = tmp_path / "copy"
    shutil.copytree(original, root)
    document = json.loads((root / "preparation.json").read_text())
    document[field] = value
    write_json(root / "preparation.json", document)
    write_json(root / "artifacts.sha256.json", {"files": artifact_inventory(root), **FLAGS})
    with pytest.raises(ValueError, match="preparation|receipt|inventory|documentation"):
        validate_upload_directory(root, source)


def test_cli_all_source_expansion_and_count_conflicts(source, tmp_path, capsys):
    from experiments.cli import main

    args = ["partial", "build", "--manifest", str(source), "--all-utterances",
            "--real-to-real-controls", "--segment-seconds", ".3"]
    for conflicts in (["--n-utterances", "3"], ["--split-count", "train=3"]):
        with pytest.raises(ValueError, match="cannot be combined"):
            main([*args, "--out-dir", str(tmp_path / "bad"), *conflicts])
    assert main([*args, "--out-dir", str(tmp_path / "good")]) == 0
    assert json.loads(capsys.readouterr().out)["source_census_complete"] is True


def test_cli_prepare_and_validate_upload(upload, tmp_path, capsys):
    from experiments.cli import main

    source, dataset, _, card, protocol, _ = upload
    root = tmp_path / "prepared"
    assert main(["partial", "prepare-upload", "--dataset-dir", str(dataset),
                 "--source-manifest", str(source), "--out-dir", str(root),
                 "--dataset-card", str(card), "--protocol", str(protocol)]) == 0
    assert json.loads(capsys.readouterr().out)["publication_ready"] is False
    assert main(["partial", "validate-upload", "--dataset-dir", str(root),
                 "--source-manifest", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
