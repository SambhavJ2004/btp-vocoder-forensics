"""Prepare a local, private-upload review folder. No Kaggle API or subprocess I/O."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from data.partial_audio import PartialAudioConfig
from experiments.partial_audio import (
    FLAGS,
    artifact_inventory,
    check_flags,
    validate_partial_dataset,
    write_json,
)
from experiments.partial_audio_demo import preview_records
from vocoders.checkpoints import sha256_file

SLUG = "btpvf-partial-v1"
OPERATIONS = "byte_copy_and_local_validation_only_no_remote_calls"
PLAN = {
    "dataset_slug": SLUG,
    "planned_visibility": "private",
    "upload_authorized": False,
    "publication_ready": False,
    "pending": ["source_and_derivative_license_review", "authenticated_owner_review",
                "expanded_data_review", "explicit_user_upload_authorization"],
    "directory_handling": "future uploader must include audio/ and demo/; never default skip",
    "training_inventory": "manifest.json examples only; demo WAVs are duplicate review assets",
    **FLAGS,
}
TEMPLATE = {
    "id": None, "title": "VocoPatch partial v1 - development only",
    "licenses": [],
    "description": "Development-only, non-citable legacy-source partial-audio examples.",
    "_review_note": "NOT executable metadata. Owner and exactly one reviewed license required. "
                    "Privacy is controlled by the future upload command, not this template.",
    **FLAGS,
}
ROOT_FILES = {
    "manifest.json", "manifest.csv", "labels.json", "sources.json", "selection.json",
    "variant_exclusions.json", "provenance.json", "validation.json", "README.md",
    "artifacts.sha256.json",
}
PREPARATION_FILES = {"DATASET_CARD.md", "PROTOCOL.md", "preparation.json",
                     "upload_plan.json", "dataset-metadata.template.json"}
DEMO_FILES = {"examples.json", "timeline.svg", "timestamps.csv", "index.html", "README.md"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _package_name(name: str) -> str:
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("Package paths must be canonical relative paths without traversal.")
    return name


def _validate_package_files(root: Path, *, prepared: bool) -> dict[str, int]:
    """Schema defines membership; inventories bind bytes and cannot extend the schema."""
    if root.is_symlink() or root.resolve() != root or not root.is_dir():
        raise ValueError("Package root must be a real directory without unsafe links.")
    sizes, directories, pending = {}, set(), [root]
    while pending:
        for path in pending.pop().iterdir():
            # Inspect before descending: do not follow links/junctions or cycles.
            if (path.is_symlink() or getattr(path, "is_junction", lambda: False)()
                    or path.resolve() != path):
                raise ValueError("Unsafe package link or escaped path.")
            name = _package_name(path.relative_to(root).as_posix())
            if path.is_dir():
                directories.add(name)
                pending.append(path)
            elif path.is_file():
                stat = path.stat()
                if stat.st_nlink > 1 or stat.st_size <= 0:
                    raise ValueError(
                        "Package files must be nonempty regular files without hardlinks.",
                    )
                sizes[name] = stat.st_size
            else:
                raise ValueError("Unsafe non-regular package entry.")
    required = ROOT_FILES | (PREPARATION_FILES if prepared else set())
    if missing := required - sizes.keys():
        raise ValueError(f"Missing required package files: {sorted(missing)}")
    config = PartialAudioConfig(**_load(root / "provenance.json")["config"])
    config.validate()
    rows = _load(root / "manifest.json")["examples"]
    sources = _load(root / "sources.json")["sources"]
    expected = required | {_package_name(row["output_filename"]) for row in rows}
    if config.demo_examples:
        expected |= {f"demo/{name}" for name in DEMO_FILES}
        for example in preview_records(rows, sources, config.demo_examples):
            expected |= {"demo/" + _package_name(name) for name in example["file_sha256"]}
    expected_directories = {p.as_posix() for name in expected for p in Path(name).parents
                            if p != Path(".")}
    if set(sizes) != expected or directories != expected_directories:
        raise ValueError(
            f"Unexpected artifact or missing generated asset: "
            f"extra={sorted(set(sizes) - expected)}, missing={sorted(expected - sizes.keys())}, "
            f"unexpected_directories={sorted(directories - expected_directories)}",
        )
    inventory = _load(root / "artifacts.sha256.json")
    check_flags(inventory)
    hashes = inventory["files"]
    if set(hashes) != expected - {"artifacts.sha256.json"}:
        raise ValueError("Package inventory paths do not match the defined artifact schema.")
    for name, recorded_hash in hashes.items():
        if sha256_file(root / name) != recorded_hash:
            raise ValueError(f"Package artifact hash mismatch: {name}")
    return sizes


def validate_upload_directory(directory: str | Path, source_manifest: str | Path) -> dict:
    """Reconstruct all audio/labels and verify both generation and preparation inventories."""
    root = Path(directory).absolute()
    sizes = _validate_package_files(root, prepared=True)
    if _load(root / "upload_plan.json") != PLAN:
        raise ValueError("Upload plan must remain private, unauthorized and non-citable.")
    if _load(root / "dataset-metadata.template.json") != TEMPLATE:
        raise ValueError("Owner/license template must remain explicitly unresolved.")
    if (root / "dataset-metadata.json").exists():
        raise ValueError("Executable upload metadata is outside preparation scope.")
    receipt = _load(root / "preparation.json")
    check_flags(receipt)
    if receipt.get("dataset_slug") != SLUG or receipt.get("operations") != OPERATIONS:
        raise ValueError("Contradictory preparation identity or operations claim.")
    generation_files = receipt["generation_files_sha256"]
    if set(generation_files) != sizes.keys() - PREPARATION_FILES - {"artifacts.sha256.json"}:
        raise ValueError("Generation receipt paths do not match the defined artifact schema.")
    canonical = json.dumps({"files": generation_files, **FLAGS}, indent=2, sort_keys=True) + "\n"
    # write_json's text mode uses the host newline. Accept either sanctioned
    # encoding after relocation, while binding the recorded digest to this map.
    digests = {hashlib.sha256(text.encode("utf-8")).hexdigest()
               for text in (canonical, canonical.replace("\n", "\r\n"))}
    if receipt.get("generation_inventory_receipt_sha256") not in digests:
        raise ValueError("Generation receipt digest contradicts its complete inventory.")
    if set(receipt["documentation_sha256"]) != {"DATASET_CARD.md", "PROTOCOL.md"}:
        raise ValueError("Preparation receipt must bind both documentation artifacts.")
    for name, digest in generation_files.items():
        # Reject path traversal before reading any receipt-specified path.
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != digest:
            raise ValueError("Copied generation artifact changed.")
    for name, digest in receipt["documentation_sha256"].items():
        if name not in {"DATASET_CARD.md", "PROTOCOL.md"} or sha256_file(root / name) != digest:
            raise ValueError("Prepared dataset documentation changed.")
    summary = validate_partial_dataset(root, source_manifest)
    return {"status": "passed", "dataset_slug": SLUG,
            "primary_wavs": summary["total_generated_files"],
            "files": len(sizes), "total_bytes": sum(sizes.values()),
            "publication_ready": False, **FLAGS}


def prepare_upload_directory(dataset_dir: str | Path, source_manifest: str | Path,
                             out_dir: str | Path, dataset_card: str | Path,
                             protocol: str | Path) -> dict:
    """Copy a validated expansion and docs into fresh staging; preserve the input package."""
    dataset = Path(dataset_dir).absolute()
    source_root = Path(source_manifest).resolve().parent
    target = Path(out_dir).resolve()
    staging = target.with_name(target.name + ".staging")
    for protected in (dataset, source_root):
        if (target.is_relative_to(protected) or protected.is_relative_to(target)
                or staging.is_relative_to(protected) or protected.is_relative_to(staging)):
            raise ValueError("Upload preparation must be separate from source and generated data.")
    if target.exists() or staging.exists():
        raise FileExistsError("Upload output/staging exists; never overwrite or resume.")
    sizes = _validate_package_files(dataset, prepared=False)
    summary = validate_partial_dataset(dataset, source_manifest)
    if summary.get("selection_policy") != "all_frozen_source_utterances":
        raise ValueError("Upload preparation requires the explicit complete source census.")
    files = artifact_inventory(dataset)
    documents = {"DATASET_CARD.md": Path(dataset_card), "PROTOCOL.md": Path(protocol)}
    for path in documents.values():
        text = path.read_text(encoding="utf-8").lower()
        if "non-citable" not in text or "development" not in text:
            raise ValueError("Upload documentation must identify non-citable development status.")
    staging.mkdir(parents=True)
    try:
        for name in files:
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dataset / name, destination)
            if destination.stat().st_size != sizes[name]:
                raise ValueError(f"Copied artifact size mismatch: {name}")
        for name, path in documents.items():
            shutil.copyfile(path, staging / name)
        write_json(staging / "upload_plan.json", PLAN)
        write_json(staging / "dataset-metadata.template.json", TEMPLATE)
        write_json(staging / "preparation.json", {
            "dataset_slug": SLUG, "generation_files_sha256": files,
            "generation_inventory_receipt_sha256": sha256_file(dataset / "artifacts.sha256.json"),
            "documentation_sha256": {name: sha256_file(path) for name, path in documents.items()},
            "preparation_code_sha256": sha256_file(Path(__file__)),
            "operations": OPERATIONS, **FLAGS,
        })
        write_json(staging / "artifacts.sha256.json", {
            "files": artifact_inventory(staging), **FLAGS,
        })
        result = validate_upload_directory(staging, source_manifest)
        if files != artifact_inventory(dataset):
            raise ValueError("Generation package changed during preparation.")
        staging.rename(target)
        return result
    except Exception as exc:
        write_json(staging / "failure.json", {"status": "failed", "error": str(exc), **FLAGS})
        raise
