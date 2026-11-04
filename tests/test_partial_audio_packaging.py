"""Exact package membership is independent of self-authored inventory hashes."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from test_partial_audio import source as source
from test_partial_audio_expansion import expanded as expanded
from test_partial_audio_expansion import upload as upload

from experiments.partial_audio import FLAGS, artifact_inventory, build_partial_dataset, write_json
from experiments.partial_audio_upload import prepare_upload_directory, validate_upload_directory


def reseal(root):
    write_json(root / "artifacts.sha256.json", {"files": artifact_inventory(root), **FLAGS})


def copied(upload, tmp_path):
    source, _, original, _, _, _ = upload
    root = tmp_path / "package"
    shutil.copytree(original, root)
    return source, root


def test_valid_package_and_legitimate_demo_assets_pass_unchanged(upload):
    source, _, root, _, _, expected = upload
    before = artifact_inventory(root)
    assert validate_upload_directory(root, source) == expected
    assert before == artifact_inventory(root)
    assert list((root / "demo/audio").glob("*.wav"))
    assert list((root / "demo/sources").glob("*.wav"))


@pytest.mark.parametrize("name", ["unexpected.txt", "demo/unexpected.txt",
                                  "demo/audio/nested/unexpected.txt"])
@pytest.mark.parametrize("regenerate_inventory", [False, True])
def test_unexpected_files_rejected_recursively_even_when_resealed(
    upload, tmp_path, name, regenerate_inventory,
):
    source, root = copied(upload, tmp_path)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Unrelated benign fixture content.")
    if regenerate_inventory:
        reseal(root)
    before = artifact_inventory(root)
    with pytest.raises(ValueError, match="Unexpected artifact"):
        validate_upload_directory(root, source)
    assert before == artifact_inventory(root)  # No silent deletion, repair or filtering.


def test_preparer_rejects_resealed_demo_file_before_copying(upload, tmp_path):
    source, dataset, _, card, protocol, _ = upload
    root = tmp_path / "candidate"
    shutil.copytree(dataset, root)
    (root / "demo/unexpected.txt").write_text("Not a generated asset.")
    reseal(root)
    with pytest.raises(ValueError, match="Unexpected artifact"):
        prepare_upload_directory(root, source, tmp_path / "output", card, protocol)
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "output.staging").exists()
    assert (root / "demo/unexpected.txt").is_file()


@pytest.mark.parametrize("name", ["labels.json", "PROTOCOL.md", "demo/timestamps.csv",
                                  "primary_wav", "preview_wav"])
def test_missing_required_file_rejected_even_when_resealed(upload, tmp_path, name):
    source, root = copied(upload, tmp_path)
    path = (next((root / "audio").glob("*.wav")) if name == "primary_wav" else
            next((root / "demo/audio").glob("*.wav")) if name == "preview_wav" else root / name)
    path.unlink()
    reseal(root)
    with pytest.raises(ValueError, match="Missing required|missing generated"):
        validate_upload_directory(root, source)


@pytest.mark.parametrize("name", ["README.md", "labels.json", "PROTOCOL.md", "demo/README.md"])
def test_modified_required_bytes_rejected_by_actual_hashes(upload, tmp_path, name):
    source, root = copied(upload, tmp_path)
    with (root / name).open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_upload_directory(root, source)


def test_empty_required_file_rejected_even_with_matching_inventory(upload, tmp_path):
    source, root = copied(upload, tmp_path)
    (root / "labels.json").write_bytes(b"")
    reseal(root)
    with pytest.raises(ValueError, match="nonempty"):
        validate_upload_directory(root, source)


@pytest.mark.parametrize("name", ["../escape.wav", "/escape.wav", "C:/escape.wav",
                                  "audio/../escape.wav", "audio\\escape.wav"])
def test_manifest_paths_cannot_escape_or_alias_package_root(upload, tmp_path, name):
    source, root = copied(upload, tmp_path)
    document = json.loads((root / "manifest.json").read_text())
    document["examples"][0]["output_filename"] = name
    write_json(root / "manifest.json", document)
    reseal(root)
    with pytest.raises(ValueError, match="canonical relative paths"):
        validate_upload_directory(root, source)


def test_inventory_cannot_authorize_a_path_outside_root(upload, tmp_path):
    source, root = copied(upload, tmp_path)
    document = json.loads((root / "artifacts.sha256.json").read_text())
    document["files"]["../outside.txt"] = "0" * 64
    write_json(root / "artifacts.sha256.json", document)
    with pytest.raises(ValueError, match="inventory paths"):
        validate_upload_directory(root, source)
    assert not (tmp_path / "outside.txt").exists()


def test_unexpected_empty_directories_are_not_ignored(upload, tmp_path):
    source, root = copied(upload, tmp_path)
    (root / "demo/unexpected-empty").mkdir()
    reseal(root)
    with pytest.raises(ValueError, match="unexpected_directories"):
        validate_upload_directory(root, source)


@pytest.mark.parametrize("kind", ["file_symlink", "directory_symlink", "hardlink"])
def test_unsafe_links_rejected_before_following_targets(upload, tmp_path, kind):
    source, root = copied(upload, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "benign.txt"
    external.write_text("External data must not be read or copied.")
    path = root / "demo/unsafe"
    try:
        if kind == "hardlink":
            os.link(external, path)
        else:
            path.symlink_to(outside if kind == "directory_symlink" else external,
                            target_is_directory=kind == "directory_symlink")
    except OSError as exc:
        pytest.skip(f"OS cannot create this {kind}: {exc}")
    with pytest.raises(ValueError, match="Unsafe|hardlinks"):
        validate_upload_directory(root, source)
    assert external.read_text() == "External data must not be read or copied."


def test_portable_link_rejection_branch(upload, monkeypatch):
    source, _, root, _, _, _ = upload
    target = root / "demo"
    actual = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p == target or actual(p))
    with pytest.raises(ValueError, match="Unsafe package link"):
        validate_upload_directory(root, source)


def test_disabled_demo_configuration_has_no_implicit_demo_allowlist(expanded, tmp_path):
    source, _, config, _ = expanded
    dataset = tmp_path / "without-demo"
    build_partial_dataset(source, dataset, replace(config, demo_examples=0))
    card = tmp_path / "card.md"
    card.write_text("Development-only and non-citable.")
    output = tmp_path / "prepared"
    assert prepare_upload_directory(dataset, source, output, card, card)["status"] == "passed"
    assert not (output / "demo").exists()
    (output / "demo").mkdir()
    (output / "demo/README.md").write_text("Not expected when demo is disabled.")
    reseal(output)
    with pytest.raises(ValueError, match="Unexpected artifact"):
        validate_upload_directory(output, source)
