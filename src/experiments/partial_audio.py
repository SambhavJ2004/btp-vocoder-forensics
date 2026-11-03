"""Local VocoPatch development pilot; no Phase A changes, training or remote I/O."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from data.audio_io import probe, read_processed, write_audio
from data.invariants import ARCHIVE_SR, ARCHIVE_TIER, SPLIT_NAMES, SUBTYPE, check_pairing
from data.manifest import read_manifest, require_primary, select_tier, validate_manifest
from data.partial_audio import (
    DEFAULT_PARTIAL_CONFIG,
    FADE_METHOD,
    SCHEMA_VERSION,
    VOCODERS,
    PartialAudioConfig,
    UnsuitableUtterance,
    choose_real_donor,
    choose_region,
    interval_labels,
    splice,
)
from vocoders.checkpoints import sha256_file

FLAGS = {"development_only": True, "citable": False}
SOURCE_SIDECARS = {"manifest.provenance.json", "manifest.alignment.json", "gates.json",
                   "sanity.csv", "loudness_gain.csv"}
PROCESSING_CLAIMS = {
    "processing": "decode_archive_pcm16_splice_encode_pcm16_no_other_processing",
    "rounding": "nearest_integer_ties_to_even", "phase_a_changed": False,
    "signal_gates_rerun": False, "research_eligibility": "not_established",
}


def check_flags(document: dict) -> None:
    if document.get("development_only") is not True or document.get("citable") is not False:
        raise ValueError("Development-only/non-citable flags must be explicit booleans.")


def source_evidence(root: Path) -> dict:
    sidecars = {path.name: sha256_file(path) for path in root.iterdir()
                if path.is_file() and path.name in SOURCE_SIDECARS}
    path = root / "manifest.provenance.json"
    upstream = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    fields = ("git_commit", "code_snapshot", "dataset_version", "corpus", "seed", "conditions",
              "n_utterances", "archive_sr", "zeroshot_sr", "archive_band")
    return {"sidecars_sha256": sidecars,
            "upstream_recorded_fields": {key: upstream[key] for key in fields if key in upstream},
            "historical_generation_bytes_verified": False}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def config_record(config: PartialAudioConfig) -> dict:
    record = asdict(config)
    if config.schema_version == SCHEMA_VERSION:
        record.pop("real_to_real_controls")
        record.pop("split_counts")
    return record


def artifact_inventory(root: Path) -> dict[str, str]:
    """Exclude only the root receipt; never hide identically named nested files."""
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Package inventory contains a link or escaped path.")
        relative = path.relative_to(root).as_posix()
        if path.is_file() and relative != "artifacts.sha256.json":
            files[relative] = sha256_file(path)
    return files


def _token(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()[:24]


def package_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Package paths must remain inside the package.")
    return path


def source_frame(manifest: str | Path) -> pd.DataFrame:
    full = read_manifest(manifest)
    validate_manifest(full)  # Existing table gates remain binding; never caught or waived.
    frame = require_primary(select_tier(full, ARCHIVE_TIER), why="partial-audio source")
    required = ["utt_id", "condition", "split", "speaker", "path"]
    if frame[required].isna().any().any() or set(frame.condition) != {"real", *VOCODERS}:
        raise ValueError("Require complete v0 identities and exactly real plus the three vocoders.")
    if not set(frame.split) <= {"train", "dev", "eval"}:
        raise ValueError("Unknown frozen source split.")
    if frame.duplicated(["utt_id", "condition"]).any() or frame.path.duplicated().any():
        raise ValueError("Duplicate source rows or paths.")
    if (frame.groupby("utt_id").speaker.nunique() != 1).any():
        raise ValueError("Paired sources have conflicting speaker identities.")
    if frame.speaker.nunique() > 1 and (frame.groupby("speaker").split.nunique() > 1).any():
        raise ValueError("Multi-speaker source splits must be speaker-disjoint.")
    check_pairing({str(name): set(group.utt_id) for name, group in frame.groupby("condition")})
    return frame.sort_values(["utt_id", "condition"]).reset_index(drop=True)


def read_input(path: Path) -> np.ndarray:
    """Inspect the container before the sanctioned reader can downmix anything."""
    if not path.is_file():
        raise UnsuitableUtterance("missing_source_file")
    with path.open("rb") as stream:
        magic = stream.read(12)
    if path.suffix.lower() != ".wav" or magic[:4] != b"RIFF" or magic[8:] != b"WAVE":
        raise UnsuitableUtterance("source_is_not_supported_riff_wave")
    try:
        header = probe(path)
    except (OSError, RuntimeError) as exc:
        raise UnsuitableUtterance("unreadable_source_header") from exc
    if (header["sample_rate"] != ARCHIVE_SR or header["channels"] != 1
            or header["subtype"] != SUBTYPE or header["frames"] <= 0):
        raise UnsuitableUtterance("source_header_not_mono_archive_pcm16")
    # InvariantViolation from this call is deliberately not caught as an exclusion.
    return read_processed(path, expected_sr=ARCHIVE_SR)


def _read_group(group: pd.DataFrame, source_root: Path, manifest_digest: str):
    wavs, sources = {}, {}
    for row in group.itertuples():
        path = Path(row.path).resolve()
        if not path.is_relative_to(source_root):
            raise ValueError("Source path resolves outside its manifest dataset directory.")
        wavs[row.condition] = read_input(path)
        try:
            duration = float(row.duration_s)
        except (TypeError, ValueError) as exc:
            raise UnsuitableUtterance("invalid_source_duration_record") from exc
        if not np.isfinite(duration) or duration <= 0:
            raise UnsuitableUtterance("invalid_source_duration_record")
        if abs(len(wavs[row.condition]) / ARCHIVE_SR - duration) > 0.5 / ARCHIVE_SR:
            raise UnsuitableUtterance("source_duration_record_mismatch")
        record = {
            "source_id": _token(manifest_digest, row.utt_id, row.condition, ARCHIVE_TIER),
            "source_manifest_sha256": manifest_digest,
            "utt_id": str(row.utt_id), "condition": str(row.condition),
            "split": str(row.split), "speaker": str(row.speaker), "tier": ARCHIVE_TIER,
            "path": path.relative_to(source_root).as_posix(), "sha256": sha256_file(path),
            "num_samples": len(wavs[row.condition]), "sample_rate": ARCHIVE_SR,
            "channels": 1, "subtype": SUBTYPE, **FLAGS,
        }
        for field in ("vocoder_checkpoint", "vocoder_commit", "seed", "alignment_checked",
                      "alignment_peak_lag"):
            value = getattr(row, field)
            record[f"upstream_{field}_recorded"] = None if pd.isna(value) else value
        sources[row.condition] = record
    if len({len(wav) for wav in wavs.values()}) != 1:
        raise UnsuitableUtterance("paired_sample_count_mismatch")
    for condition in VOCODERS:
        if np.array_equal(wavs["real"], wavs[condition]):
            raise UnsuitableUtterance(f"identical_real_and_{condition}_recording")
    return wavs, sources


def plan_partial_dataset(manifest: str | Path, config: PartialAudioConfig) -> dict:
    """Deterministic whole-utterance eligibility scan. Writes no audio or metadata."""
    config.validate()
    manifest = Path(manifest).resolve()
    frame = source_frame(manifest)
    digest = sha256_file(manifest)
    if config.split_counts is None:
        order = np.random.default_rng(config.seed).permutation(sorted(set(frame.utt_id))).tolist()
        pools = [("all", config.n_utterances, order)]
    else:
        pools = []
        for split, count in zip(SPLIT_NAMES, config.split_counts, strict=True):
            ids = sorted(set(frame.loc[frame.split.eq(split), "utt_id"]))
            seed = int.from_bytes(hashlib.sha256(
                f"vocopatch-split-selection|{config.seed}|{split}".encode(),
            ).digest()[:8], "big")
            pools.append((split, count, np.random.default_rng(seed).permutation(ids).tolist()))
    selected, excluded = [], []
    for split, count, order in pools:
        if not count:
            continue
        selected_in_pool = 0
        for rank, utt_id in enumerate(order):
            group = frame[frame.utt_id.eq(utt_id)]
            try:
                wavs, sources = _read_group(group, manifest.parent, digest)
                start, end, activity = choose_region(wavs["real"], config, str(utt_id))
                real_donor = (choose_real_donor(wavs["real"], config, str(utt_id), 1, (start, end))
                              if config.real_to_real_controls else None)
            except UnsuitableUtterance as exc:
                exclusion = {"utt_id": str(utt_id), "candidate_rank": rank,
                             "reason": str(exc), **FLAGS}
                if config.schema_version != SCHEMA_VERSION:
                    exclusion["selection_split"] = split
                excluded.append(exclusion)
                continue
            item = {"utt_id": str(utt_id), "candidate_rank": rank,
                    "sources": sources, "first_region": [start, end], "activity": activity}
            if config.schema_version != SCHEMA_VERSION:
                item.update(selection_split=split, first_real_donor=real_donor)
            selected.append(item)
            selected_in_pool += 1
            if selected_in_pool == count:
                break
    result = {"source_manifest_sha256": digest, "selected": selected, "excluded": excluded, **FLAGS}
    if config.split_counts is None:
        result["candidate_order"] = pools[0][2]
    else:
        counts = Counter(item["sources"]["real"]["split"] for item in selected)
        requested = dict(zip(SPLIT_NAMES, config.split_counts, strict=True))
        result.update(
            selection_policy="explicit_frozen_split_counts",
            candidate_order_by_split={split: order for split, _, order in pools},
            requested_utterances_by_split=requested,
            selected_utterances_by_split={split: counts[split] for split in SPLIT_NAMES},
            shortfalls_by_split={
                split: max(0, n - counts[split]) for split, n in requested.items()
            },
        )
    if config.schema_version != SCHEMA_VERSION:
        result["requested_source_utterances"] = config.n_utterances
    return result


def _row(config: PartialAudioConfig, item: dict, condition: str, variant: int,
         start: int, end: int, activity: dict, *, real_donor: dict | None = None,
         real_to_real: bool = False) -> dict:
    sources = item["sources"]
    real, donor = sources["real"], sources[condition]
    synthetic = condition != "real"
    def edit_id(name: str) -> str:
        return _token(SCHEMA_VERSION, config.seed, item["utt_id"], variant, name,
                      real["sha256"], sources[name]["sha256"], start, end, config.fade_samples)

    example_id = edit_id(condition)
    fade = config.fade_samples
    ignore = [[start, start + fade], [end - fade, end]] if fade else []
    row = {
        "example_id": example_id, "output_filename": f"audio/{example_id}.wav",
        "original_utterance_id": item["utt_id"], "group_id": item["utt_id"],
        "split": real["split"], "speaker": real["speaker"], "variant": variant,
        "source_dataset_name": "btpvf-v0",
        "source_dataset_version": f"manifest_sha256:{real['source_manifest_sha256']}",
        "source_condition": condition, "source_real_id": real["source_id"],
        "source_donor_id": donor["source_id"],
        "source_synthetic_id": donor["source_id"] if synthetic else None,
        "source_real_sha256": real["sha256"], "source_donor_sha256": donor["sha256"],
        "same_source_null_edit": not synthetic, "has_synthetic": synthetic,
        "matched_control_id": edit_id("real"),
        "start_sample": start, "end_sample": end,
        "start_seconds": start / ARCHIVE_SR, "end_seconds": end / ARCHIVE_SR,
        "transition_ignore_ranges": ignore, "ignore_sample_count": 2 * fade,
        "pure_synthetic_ranges": [[start + fade, end - fade]] if synthetic else [],
        "num_samples": real["num_samples"], "sample_rate": ARCHIVE_SR,
        "channels": 1, "subtype": SUBTYPE, "tier": "archive_derived_partial_edit",
        "crossfade_method": FADE_METHOD, "crossfade_samples_per_boundary": fade,
        "crossfade_requested_ms": config.crossfade_ms,
        "crossfade_actual_ms": 1000 * fade / ARCHIVE_SR,
        "seed": config.seed, "activity": activity,
        "interval_labels": interval_labels(real["num_samples"], start, end, fade,
                                           synthetic=synthetic), **FLAGS,
    }
    if config.schema_version != SCHEMA_VERSION:
        kind = ("real_to_real_control" if real_to_real else
                "null_control" if not synthetic else f"mixed_{condition}")
        donor_start = real_donor["start_sample"] if real_to_real else start
        donor_end = donor_start + end - start

        def review_id(name: str, control_kind: str, origin: int) -> str:
            return _token(config.schema_version, config.seed, item["utt_id"], variant, name,
                          real["sha256"], sources[name]["sha256"], start, end, fade,
                          control_kind, origin, origin + end - start)

        row.update(
            example_id=review_id(condition, kind, donor_start), example_class=kind,
            same_source_null_edit=not synthetic and not real_to_real,
            source_donor_condition=condition,
            donor_start_sample=donor_start, donor_end_sample=donor_end,
            donor_start_seconds=donor_start / ARCHIVE_SR,
            donor_end_seconds=donor_end / ARCHIVE_SR,
            donor_activity=real_donor["activity"] if real_to_real else None,
            content_policy=("different_real_region_not_content_matched" if real_to_real else
                            "same_index_paired_vocoder" if synthetic else "same_source_same_index"),
            intentional_excerpt_relocation=real_to_real,
            matched_control_id=review_id("real", "null_control", start),
            matched_real_to_real_control_id=(
                review_id("real", "real_to_real_control", real_donor["start_sample"])
                if real_donor is not None else None
            ),
        )
        row["output_filename"] = f"audio/{row['example_id']}.wav"
    return row


def variant_recipe(wavs: dict, config: PartialAudioConfig, item: dict, variant: int,
                   previous: tuple[int, int] | None):
    start, end, activity = choose_region(wavs["real"], config, item["utt_id"],
                                         variant, previous=previous)
    real_donor = (choose_real_donor(wavs["real"], config, item["utt_id"], variant, (start, end))
                  if config.real_to_real_controls else None)
    return start, end, activity, real_donor


def example_rows(config: PartialAudioConfig, item: dict, variant: int, start: int, end: int,
                 activity: dict, real_donor: dict | None):
    for condition in ("real", *VOCODERS):
        yield _row(config, item, condition, variant, start, end, activity, real_donor=real_donor)
    if real_donor is not None:
        yield _row(config, item, "real", variant, start, end, activity,
                   real_donor=real_donor, real_to_real=True)


def expected_pcm16(real: np.ndarray, donor: np.ndarray, start: int, end: int,
                   fade: int, donor_start: int) -> np.ndarray:
    """Independent scalar fade oracle; never calls the generator or audio writer.

    The sanctioned float32->PCM_16 path floors to the 1/32768 grid. Regression
    calibration checks this contract against libsndfile at both signs/boundaries.
    """
    expected = real.copy()
    expected[start:end] = donor[donor_start:donor_start + end - start]
    for offset in range(fade):
        weight = (offset + 1) / (fade + 1)
        for target, origin in ((start + offset, donor_start + offset),
                               (end - 1 - offset, donor_start + end - start - 1 - offset)):
            base = float(real[target])
            expected[target] = base + weight * (float(donor[origin]) - base)
    return (np.floor(expected.astype(np.float64) * 32768) / 32768).astype(np.float32)


def validate_written_example(root: Path, row: dict, real: np.ndarray, donor: np.ndarray) -> dict:
    """Validate decoded samples with an independent oracle and no filesystem writes."""
    path = package_path(root, row["output_filename"])
    output = read_input(path)
    n, start, end, fade = (row[key] for key in (
        "num_samples", "start_sample", "end_sample", "crossfade_samples_per_boundary",
    ))
    if n != len(real) or len(output) != n or len(donor) != n:
        raise ValueError("Output/source sample count mismatch.")
    if row["start_seconds"] != start / ARCHIVE_SR or row["end_seconds"] != end / ARCHIVE_SR:
        raise ValueError("Timestamp/sample indexing mismatch.")
    if row["interval_labels"] != interval_labels(n, start, end, fade,
                                                synthetic=row["has_synthetic"]):
        raise ValueError("Labels do not match exact sample regions.")
    if (not np.array_equal(output[:start], real[:start])
            or not np.array_equal(output[end:], real[end:])):
        raise ValueError("Samples changed outside the declared edit interval.")
    donor_start = row.get("donor_start_sample", start)
    expected = expected_pcm16(real, donor, start, end, fade, donor_start)
    if not np.array_equal(output, expected):
        raise ValueError("Output does not reproduce the declared splice/PCM_16 operation.")
    core = slice(start + fade, end - fade)
    donor_core = slice(donor_start + fade, donor_start + end - start - fade)
    if not np.array_equal(output[core], donor[donor_core]):
        raise ValueError("Pure-source core is not exactly the paired donor PCM_16 samples.")
    if row["same_source_null_edit"] and not np.array_equal(output, real):
        raise ValueError("Same-source null edit changed decoded real samples.")
    changes = int(np.count_nonzero(output != real))
    core_changes = int(np.count_nonzero(output[core] != real[core]))
    if row["has_synthetic"] and not core_changes:
        raise ValueError("Synthetic-labeled core has no changed decoded samples.")
    if row.get("example_class") == "real_to_real_control":
        donor_end = donor_start + end - start
        if (row["source_donor_condition"] != "real" or row["has_synthetic"]
                or row["same_source_null_edit"] or row["source_synthetic_id"] is not None
                or not (donor_end <= start or donor_start >= end) or not core_changes):
            raise ValueError("Invalid/non-distinct real-to-real control semantics.")
    digest = sha256_file(path)
    if "generated_sha256" in row and row["generated_sha256"] != digest:
        raise ValueError("Generated audio hash mismatch.")
    return {"validation_status": "passed", "generated_sha256": digest,
            "changed_sample_count": changes, "changed_core_sample_count": core_changes}


def _csv_record(row: dict) -> dict:
    return {key: json.dumps(value, sort_keys=True, separators=(",", ":"))
            if isinstance(value, (dict, list, bool)) or value is None else str(value)
            for key, value in row.items()}


def write_manifest_files(root: Path, rows: list[dict],
                         schema_version: str = SCHEMA_VERSION) -> None:
    write_json(root / "manifest.json", {
        "schema_version": schema_version, "examples": rows, **FLAGS,
    })
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(_csv_record(row) for row in rows)
    write_json(root / "labels.json", {
        "schema_version": schema_version, "indexing": "zero_based_half_open",
        "label_values": {"-1": "transition_ignore", "0": "real", "1": "synthetic"},
        "examples": {row["example_id"]: row["interval_labels"] for row in rows}, **FLAGS,
    })


def code_provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    paths = ["src/data/partial_audio.py", "src/experiments/partial_audio.py",
             "src/experiments/partial_audio_demo.py", "src/experiments/cli.py",
             "src/data/audio_io.py", "src/data/manifest.py", "src/data/invariants.py",
             "src/vocoders/checkpoints.py", "CLAUDE.md", "pyproject.toml"]
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                text=True, check=False)
        head = result.stdout.strip() if result.returncode == 0 else "unavailable"
    except OSError:
        head = "unavailable"
    return {"head": head,
            "source_sha256": {path: sha256_file(root / path) for path in paths},
            "python": platform.python_version(), "platform": platform.platform(),
            "environment": {name: version(name) for name in ("numpy", "pandas", "soundfile")},
            **FLAGS}


def _summary(rows: list[dict], selection: dict, variant_exclusions: list[dict]) -> dict:
    mixed = [row for row in rows if row["has_synthetic"]]
    controls = [row for row in rows if not row["has_synthetic"]]
    summary = {
        "status": "passed", "utterances_used": len({row["group_id"] for row in rows}),
        "mixed_files": len(mixed), "real_only_controls": len(controls),
        "total_generated_files": len(rows), "sample_rate": ARCHIVE_SR,
        "channels": 1, "subtype": SUBTYPE,
        "total_duration_seconds": sum(row["num_samples"] for row in rows) / ARCHIVE_SR,
        "mixed_duration_seconds": sum(row["num_samples"] for row in mixed) / ARCHIVE_SR,
        "control_duration_seconds": sum(row["num_samples"] for row in controls) / ARCHIVE_SR,
        "files_by_source_condition": dict(Counter(row["source_condition"] for row in rows)),
        "utterance_groups_by_split": dict(Counter(
            {row["group_id"]: row["split"] for row in rows}.values(),
        )),
        "exclusions": selection["excluded"], "excluded_utterances": len(selection["excluded"]),
        "variant_exclusions": variant_exclusions,
        "excluded_variants": len(variant_exclusions),
        "pure_synthetic_sample_count": sum(
            row["end_sample"] - row["start_sample"] - 2 * row["crossfade_samples_per_boundary"]
            for row in mixed
        ),
        "source_files_verified_unchanged": 4 * len(selection["selected"]),
        "scope": "local_sample_integrity_not_research_gate_admission", **FLAGS,
    }
    if rows and "example_class" in rows[0]:
        classes = ("null_control", "real_to_real_control", *(f"mixed_{name}" for name in VOCODERS))
        summary.update(
            null_controls=sum(row["same_source_null_edit"] for row in rows),
            real_to_real_controls=sum(
                row["example_class"] == "real_to_real_control" for row in rows
            ),
            files_by_class_and_split={split: {
                kind: sum(row["split"] == split and row["example_class"] == kind for row in rows)
                for kind in classes
            } for split in SPLIT_NAMES},
            requested_utterances_by_split=selection.get("requested_utterances_by_split"),
            split_shortfalls=selection.get("shortfalls_by_split", {}),
            split_coverage_complete=(not any(selection.get("shortfalls_by_split", {}).values())
                                     and summary["utterances_used"] >=
                                     selection["requested_source_utterances"]),
        )
    return summary


def validate_partial_dataset(dataset_dir: str | Path, source_manifest: str | Path,
                             *, verify_inventory: bool = True) -> dict:
    root, source_manifest = Path(dataset_dir).resolve(), Path(source_manifest).resolve()
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    check_flags(provenance)
    config = PartialAudioConfig(**provenance["config"])
    config.validate()
    schema_version = config.schema_version
    if provenance.get("schema_version") != schema_version:
        raise ValueError("Unsupported provenance schema/configuration.")
    for key, value in PROCESSING_CLAIMS.items():
        if type(provenance.get(key)) is not type(value) or provenance[key] != value:
            raise ValueError(f"Contradictory or missing provenance claim: {key}")
    if provenance["source_manifest_sha256"] != sha256_file(source_manifest):
        raise ValueError("Original source manifest identity changed.")
    if provenance["source_evidence"] != source_evidence(source_manifest.parent):
        raise ValueError("Original provenance/gate metadata identity changed.")
    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    check_flags(document)
    if document.get("schema_version") != schema_version:
        raise ValueError("Unsupported manifest schema.")
    rows = document["examples"]
    for row in rows:
        check_flags(row)
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    # Recompute source eligibility/first interval from original frozen metadata/seed.
    expected_selection = plan_partial_dataset(source_manifest, config)
    if selection != expected_selection:
        raise ValueError("Source selection, hashes, splits or deterministic regions changed.")
    frame = source_frame(source_manifest)
    expected_rows, sources, variant_exclusions = [], {}, []
    for item in selection["selected"]:
        wavs, records = _read_group(frame[frame.utt_id.eq(item["utt_id"])],
                                    source_manifest.parent, selection["source_manifest_sha256"])
        if records != item["sources"]:
            raise ValueError("Source identities or original audio hashes changed.")
        sources.update({record["source_id"]: record for record in records.values()})
        previous = None
        for variant in range(1, config.variants + 1):
            try:
                start, end, activity, real_donor = variant_recipe(
                    wavs, config, item, variant, previous,
                )
            except UnsuitableUtterance as exc:
                if variant == 1:
                    raise
                variant_exclusions.append({"utt_id": item["utt_id"], "variant": variant,
                                           "reason": str(exc), **FLAGS})
                break
            for row in example_rows(config, item, variant, start, end, activity, real_donor):
                donor = wavs[row.get("source_donor_condition", row["source_condition"])]
                row.update(validate_written_example(root, row, wavs["real"], donor))
                expected_rows.append(row)
            previous = start, end
    if rows != expected_rows:
        raise ValueError("Manifest differs from recomputed source, labels or validation records.")
    catalog = json.loads((root / "sources.json").read_text(encoding="utf-8"))
    if catalog != {"sources": sources, **FLAGS}:
        raise ValueError("Source catalog mismatch.")
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    check_flags(labels)
    if (labels.get("schema_version") != schema_version
            or labels.get("indexing") != "zero_based_half_open"
            or labels.get("label_values") != {"-1": "transition_ignore", "0": "real",
                                               "1": "synthetic"}):
        raise ValueError("Invalid label schema.")
    if labels.get("examples") != {row["example_id"]: row["interval_labels"] for row in rows}:
        raise ValueError("Separate labels disagree with the manifest.")
    variants_document = json.loads((root / "variant_exclusions.json").read_text(encoding="utf-8"))
    if variants_document != {"exclusions": variant_exclusions, **FLAGS}:
        raise ValueError("Second-variant exclusion records differ.")
    with (root / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        if list(csv.DictReader(stream)) != [_csv_record(row) for row in rows]:
            raise ValueError("CSV and JSON manifest disagree.")
    if len({row["output_filename"] for row in rows}) != len(rows) or not rows:
        raise ValueError("Empty or duplicate output identities.")
    if {p.relative_to(root).as_posix() for p in (root / "audio").rglob("*") if p.is_file()} != {
        row["output_filename"] for row in rows
    }:
        raise ValueError("Generated audio inventory differs from the manifest.")
    if verify_inventory:
        actual = artifact_inventory(root)
        inventory = json.loads((root / "artifacts.sha256.json").read_text(encoding="utf-8"))
        if inventory != {"files": actual, **FLAGS}:
            raise ValueError("Artifact file inventory/hash mismatch.")
    from experiments.partial_audio_demo import validate_demo

    validate_demo(root, rows, sources, config.demo_examples)
    summary = _summary(rows, selection, variant_exclusions)
    if verify_inventory and json.loads((root / "validation.json").read_text("utf-8")) != summary:
        raise ValueError("Stored validation summary disagrees with recomputed results.")
    return summary


def build_partial_dataset(manifest: str | Path, out_dir: str | Path,
                          config: PartialAudioConfig = DEFAULT_PARTIAL_CONFIG) -> dict:
    """Preflight, edit, validate, then publish a new local folder. Never upload."""
    config.validate()
    manifest, target = Path(manifest).resolve(), Path(out_dir).resolve()
    source_root = manifest.parent
    staging = target.with_name(target.name + ".staging")
    if (target.is_relative_to(source_root) or source_root.is_relative_to(target)
            or staging.is_relative_to(source_root)):
        raise ValueError("Output must be separate from the original source dataset.")
    if target.exists() or staging.exists():
        raise FileExistsError("Output/staging directory exists; never overwrite or resume.")
    selection = plan_partial_dataset(manifest, config)
    if not selection["selected"]:
        raise ValueError(f"No eligible utterances; exclusions: {selection['excluded']}")
    frame = source_frame(manifest)
    staging.mkdir(parents=True)
    rows, sources, variant_exclusions = [], {}, []
    try:
        write_json(staging / "selection.json", selection)
        evidence = source_evidence(source_root)
        write_json(staging / "provenance.json", {
            "schema_version": config.schema_version, "config": config_record(config),
            "source_dataset_name": "btpvf-v0",
            "source_dataset_version": f"manifest_sha256:{selection['source_manifest_sha256']}",
            "source_manifest_sha256": selection["source_manifest_sha256"],
            "source_evidence": evidence, "code": code_provenance(),
            **PROCESSING_CLAIMS, **FLAGS,
        })
        for item in selection["selected"]:
            wavs, records = _read_group(frame[frame.utt_id.eq(item["utt_id"])], source_root,
                                        selection["source_manifest_sha256"])
            if records != item["sources"]:
                raise ValueError("Source files changed after preflight.")
            sources.update({record["source_id"]: record for record in records.values()})
            previous = None
            for variant in range(1, config.variants + 1):
                try:
                    start, end, activity, real_donor = variant_recipe(
                        wavs, config, item, variant, previous,
                    )
                except UnsuitableUtterance as exc:
                    if variant == 1:
                        raise
                    variant_exclusions.append({"utt_id": item["utt_id"], "variant": variant,
                                               "reason": str(exc), **FLAGS})
                    break
                for row in example_rows(config, item, variant, start, end, activity, real_donor):
                    donor = wavs[row.get("source_donor_condition", row["source_condition"])]
                    path = package_path(staging, row["output_filename"])
                    write_audio(path, splice(wavs["real"], donor, start, end, config.fade_samples,
                                            donor_start=row.get("donor_start_sample")), ARCHIVE_SR)
                    row.update(validate_written_example(
                        staging, row, wavs["real"], donor,
                    ))
                    rows.append(row)
                # Only reach the next variant after all four outputs passed validation.
                previous = start, end
        write_manifest_files(staging, rows, config.schema_version)
        write_json(staging / "sources.json", {"sources": sources, **FLAGS})
        write_json(staging / "variant_exclusions.json", {"exclusions": variant_exclusions, **FLAGS})
        from experiments.partial_audio_demo import write_demo

        write_demo(staging, rows, sources, source_root, config.demo_examples)
        summary = validate_partial_dataset(staging, manifest, verify_inventory=False)
        for name, digest in evidence["sidecars_sha256"].items():
            if sha256_file(source_root / name) != digest:
                raise ValueError("Source metadata changed during generation.")
        write_json(staging / "validation.json", summary)
        (staging / "README.md").write_text(
            "# VocoPatch partial-audio development pilot\n\n"
            "**Development only; non-citable. No model or localization predictions.**\n\n"
            "Existing paired archive speech is spliced at matching sample indices. "
            "All source files are unchanged. See provenance.json for parameters and code, "
            "selection.json for sampling/exclusions, manifest.json and labels.json for "
            "zero-based half-open sample labels, and validation.json for integrity results.\n\n"
            "Labels: 0 real, 1 pure synthetic core, -1 transition/ignore. Null controls "
            "have real audio everywhere and mirrored ignored editing boundaries. "
            "One shared null control per utterance/variant avoids duplicate real files.\n\n"
            "Open demo/index.html for offline audio and sample timelines. Preview copies "
            "are not additional training examples. Keep every source utterance and all "
            "its variants in the original frozen split; never split generated rows.\n\n"
            "Legacy provenance, RMS-based activity selection, splice boundaries, whole-file "
            "loudness drift, known full-band vocoder/resampler cues and same-source null "
            "controls limit scientific conclusions. Null edits do not eliminate real-to-"
            "synthetic boundary artifacts. No research-gate admission is claimed.\n\n"
            f"Validated mixed files: {summary['mixed_files']}; controls: "
            f"{summary['real_only_controls']}; utterances: {summary['utterances_used']}.\n",
            encoding="utf-8",
        )
        if config.schema_version != SCHEMA_VERSION:
            with (staging / "README.md").open("a", encoding="utf-8") as stream:
                stream.write(
                    "\n## Reviewed control protocol\n\n"
                    "Null controls remain exact identities. Real-to-real controls, when enabled, "
                    "replace the target with a disjoint active excerpt of the SAME real utterance. "
                    "This intentionally changes temporal/content correspondence and may repeat or "
                    "remove words. It is not alignment repair or content-matched resynthesis. "
                    "All contributing samples are real; core labels are 0 and fades are -1. "
                    "Donor/target indices and control classes are explicit in the manifest. "
                    "No control category eliminates all acoustic or splice confounds.\n\n"
                    "Explicit split counts select within frozen assignments, never redistribute "
                    "utterances. See selection.json and validation.json for requests, actual "
                    "counts and any shortfalls; passed integrity does not imply full coverage.\n\n"
                    f"Null controls: {summary['null_controls']}; real-to-real controls: "
                    f"{summary['real_to_real_controls']}. Parameters:\n\n"
                    f"```json\n{json.dumps(config_record(config), indent=2)}\n```\n"
                )
        write_json(staging / "artifacts.sha256.json", {
            "files": artifact_inventory(staging), **FLAGS,
        })
        validate_partial_dataset(staging, manifest)
        staging.rename(target)
        return summary
    except Exception as exc:
        # Preserve failed staging for diagnosis; never publish it or continue past a gate.
        write_json(staging / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
                                             "error": str(exc), **FLAGS})
        raise
