"""Freeze and validate VocoRoute-P speech releases without constructing a model."""

import hashlib
import io
import platform
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import scipy
import soundfile
import torch
from scipy.signal import resample_poly

from data.audio_io import probe, read_native, write_generator_pcm16

from .spectral import FRONTEND_V1, SpectralFrontend, frames_for_length
from .training_config import canonical_json, is_sha256, parse_json
from .training_data import (
    MANIFEST_SCHEMA,
    GeneratorManifest,
    GeneratorRow,
    TrainingAudio,
    load_manifest,
    prepare_example,
)
from .vctk_inventory import (
    MAX_RECORDING_BYTES,
    Inventory,
    SplitProtocol,
    Utterance,
    allocate_speakers,
    ranked_ids,
    sha256,
)

PROTOCOL_ID = "vocoroute-vctk092-n2b0-v1"
ARCHIVE_NAME = "VCTK-Corpus-0.92.zip"
NATIVE_RATE = 48_000
TARGET_RATE = FRONTEND_V1.sample_rate
RESAMPLE_PROFILE = "vocoroute-vctk092-polyphase-v1"
FREEZE_SCHEMA = "vocoroute-vctk092-freeze-v1"
OVERFIT_SCHEMA = "vocoroute-train-only-selection-v1"


@dataclass(frozen=True)
class AcquisitionPin:
    receipt_sha256: str = "b578c8878525ff3db87e25b4fd659d3962d75155295c15ed59e6ca07a34393fe"
    bytes: int = 11_747_302_977
    md5: str = "8a6ba2946b36fcbef0212cad601f4bfa"
    sha256: str = "f96258be9fdc2cbff6559541aae7ea4f59df3fcaf5cf963aae5ca647357e359c"


OFFICIAL_ACQUISITION = AcquisitionPin()


def write_json(path: Path, document: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json(document) + b"\n"
    path.write_bytes(data)
    return sha256(data)


def read_pinned_json(path: Path, expected_sha256: str) -> dict:
    data = path.read_bytes()
    if not is_sha256(expected_sha256) or sha256(data) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    return parse_json(data)


def verify_acquisition(
    archive: Path, pin: AcquisitionPin = OFFICIAL_ACQUISITION
) -> dict[str, bytes]:
    """Pin the receipt, attribution and full mounted archive bytes in a single scan."""
    directory = archive.parent
    receipt_path = directory / "acquisition-receipt.json"
    receipt = read_pinned_json(receipt_path, pin.receipt_sha256)
    if (
        archive.name != ARCHIVE_NAME
        or receipt.get("schema") != "vocoroute-vctk-acquisition-v1"
        or receipt.get("status") != "verified"
        or receipt.get("doi") != "10.7488/ds/2645"
        or receipt.get("license") != "CC BY 4.0"
        or receipt.get("archive")
        != {
            "name": ARCHIVE_NAME,
            "bytes": pin.bytes,
            "md5": pin.md5,
            "sha256": pin.sha256,
        }
        or archive.stat().st_size != pin.bytes
    ):
        raise ValueError("mounted VCTK acquisition does not match the approved archive identity")
    metadata = {receipt_path.name: receipt_path.read_bytes()}
    for name in ("README.txt", "license_text.txt"):
        data = (directory / name).read_bytes()
        if sha256(data) != receipt["metadata"][name]["sha256"]:
            raise ValueError(f"VCTK source metadata SHA-256 mismatch: {name}")
        metadata[name] = data
    md5, digest = hashlib.md5(usedforsecurity=False), hashlib.sha256()
    size = 0
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            md5.update(chunk)
            digest.update(chunk)
            size += len(chunk)
    if (size, md5.hexdigest(), digest.hexdigest()) != (pin.bytes, pin.md5, pin.sha256):
        raise ValueError("mounted VCTK archive checksum mismatch")
    return metadata


def resampling_document() -> dict:
    return {
        "schema": RESAMPLE_PROFILE,
        "decoder_encoder": {
            "soundfile": soundfile.__version__,
            "libsndfile": soundfile.__libsndfile_version__,
        },
        "runtime": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "input": {"encoding": "FLAC/PCM_16", "channels": 1, "rate": NATIVE_RATE},
        "filter": {
            "function": "scipy.signal.resample_poly",
            "up": 147,
            "down": 320,
            "window": ["kaiser", 5.0],
            "padtype": "constant",
            "cval": 0.0,
            "dtype": "float32",
        },
        "output": {"encoding": "WAV/PCM_16", "channels": 1, "rate": TARGET_RATE},
        "length": "ceil(native_samples * 147 / 320)",
        "gain_normalization": False,
        "silence_trimming": False,
    }


class InvalidRecording(ValueError):
    """A source recording that fails a predeclared audio eligibility gate."""


@dataclass(frozen=True)
class Prepared:
    row: GeneratorRow
    provenance: dict
    native: bytes
    prepared: bytes


def _signal_stats(signal: np.ndarray) -> dict:
    return {
        "min": float(signal.min()),
        "max": float(signal.max()),
        "rms": float(np.sqrt(np.mean(np.square(signal, dtype=np.float64)))),
        "mean": float(signal.mean(dtype=np.float64)),
        "zero_fraction": float(np.mean(signal == 0)),
    }


def prepare_recording(
    zipped: ZipFile, uid: str, item: Utterance, role: str, resample_sha256: str
) -> Prepared:
    with zipped.open(item.audio) as stream:
        native = stream.read(MAX_RECORDING_BYTES + 1)
    if len(native) != item.audio.file_size or not 0 < len(native) <= MAX_RECORDING_BYTES:
        raise ValueError(f"VCTK member length mismatch: {item.audio.filename}")
    stream = io.BytesIO(native)
    try:
        header = probe(stream)
        if native[:4] != b"fLaC" or (
            header["channels"],
            header["sample_rate"],
            header["subtype"],
        ) != (1, NATIVE_RATE, "PCM_16"):
            raise InvalidRecording("expected published mono 48 kHz PCM16 FLAC")
        if not 0 < header["frames"] <= 60 * NATIVE_RATE:
            raise InvalidRecording("native duration outside the 0–60 s decode bound")
        stream.seek(0)
        original, rate = read_native(stream)
    except (RuntimeError, ValueError) as exc:
        if isinstance(exc, InvalidRecording):
            raise
        raise InvalidRecording(f"native decode failed: {exc}") from exc
    if rate != NATIVE_RATE or original.shape != (header["frames"],) or not original.size:
        raise InvalidRecording("native sample count or sample rate mismatch")
    if not np.isfinite(original).all():
        raise InvalidRecording("nonfinite native samples")
    original_stats = _signal_stats(original)
    if original_stats["rms"] == 0:
        raise InvalidRecording("native digital silence")
    if original_stats["min"] <= -1 or original_stats["max"] >= 32767 / 32768:
        raise InvalidRecording("native PCM16 full-scale sample")
    converted = resample_poly(
        original, 147, 320, window=("kaiser", 5.0), padtype="constant", cval=0.0
    )
    expected_samples = (original.size * 147 + 319) // 320
    if converted.shape != (expected_samples,) or not np.isfinite(converted).all():
        raise ValueError("resampler violated the finite ceil-length contract")
    if converted.min() < -1 or converted.max() > 32767 / 32768:
        raise InvalidRecording("resampling would clip PCM16")
    stream = io.BytesIO()
    write_generator_pcm16(stream, converted, TARGET_RATE)
    prepared = stream.getvalue()
    stream.seek(0)
    header = probe(stream)
    stream.seek(0)
    decoded, decoded_rate = read_native(stream)
    if (
        prepared[:4] != b"RIFF"
        or header["subtype"] != "PCM_16"
        or header["channels"] != 1
        or decoded_rate != TARGET_RATE
        or decoded.shape != (expected_samples,)
        or np.max(np.abs(decoded - converted)) > 1 / 32768
    ):
        raise ValueError("prepared PCM16 failed its header, length or quantization check")
    prepared_stats = _signal_stats(decoded)
    if prepared_stats["rms"] == 0:
        raise InvalidRecording("prepared digital silence")
    row = GeneratorRow(
        utterance_id=uid,
        speaker_id=item.speaker,
        source_group=f"vctk-0.92/{uid}",
        role=role,
        native_path=f"native/{role}/{item.speaker}/{Path(item.audio.filename).name}",
        native_sha256=sha256(native),
        native_rate=rate,
        native_samples=original.size,
        prepared_path=f"prepared/{role}/{item.speaker}/{uid}.wav",
        prepared_sha256=sha256(prepared),
        prepared_samples=decoded.size,
        resample_steps=1,
        resampling_profile=RESAMPLE_PROFILE,
        resampling_receipt_sha256=resample_sha256,
    )
    return Prepared(
        row,
        {
            "utterance_id": uid,
            "role": role,
            "archive_member": item.audio.filename,
            "native_pcm_sha256": sha256(original.astype("<f4", copy=False).tobytes()),
            "transcript_sha256": item.transcript_sha256,
            "normalized_text_sha256": item.normalized_text_sha256,
            "native_duration_seconds": original.size / rate,
            "prepared_duration_seconds": decoded.size / TARGET_RATE,
            "native_bytes": len(native),
            "prepared_bytes": len(prepared),
            "native_stats": original_stats,
            "prepared_stats": prepared_stats,
            "flags": [
                name
                for name, flagged in (
                    (
                        "low_rms_below_minus60_dbfs",
                        min(original_stats["rms"], prepared_stats["rms"]) < 0.001,
                    ),
                    ("duration_outside_0_25_to_30_s", not 0.25 <= original.size / rate <= 30),
                    ("absolute_dc_above_0_01", abs(original_stats["mean"]) > 0.01),
                    (
                        "prepared_pcm16_full_scale",
                        prepared_stats["min"] <= -1 or prepared_stats["max"] >= 32767 / 32768,
                    ),
                )
                if flagged
            ],
        },
        native,
        prepared,
    )


def _save_recording(root: Path, recording: Prepared) -> None:
    for relative, content in (
        (recording.row.native_path, recording.native),
        (recording.row.prepared_path, recording.prepared),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)


def _clone_recording(source: Path, target: Path, row: GeneratorRow) -> None:
    for relative in (row.native_path, row.prepared_path):
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, dest)


def _manifest(name: str, rows: list[GeneratorRow]) -> dict:
    return {
        "schema": MANIFEST_SCHEMA,
        "source_kind": "native_generator",
        "target_rate": TARGET_RATE,
        "dataset_id": f"{PROTOCOL_ID}-{name}",
        "rows": [asdict(row) for row in sorted(rows, key=lambda row: row.utterance_id)],
    }


def _audit(rows: list[GeneratorRow], provenance: dict[str, dict]) -> dict:
    groups = {role: [row for row in rows if row.role == role] for role in ("train", "dev", "eval")}
    overlap, stats = {}, {}
    for role, group in groups.items():
        total_samples = sum(row.prepared_samples for row in group)
        stats[role] = {
            "speakers": len({row.speaker_id for row in group}),
            "utterances": len(group),
            "prepared_samples": total_samples,
            "prepared_seconds": total_samples / TARGET_RATE,
            "audio_bytes": sum(
                provenance[row.utterance_id][field]
                for row in group
                for field in ("native_bytes", "prepared_bytes")
            ),
        }
    for a, b in combinations(groups, 2):
        speakers = {r.speaker_id for r in groups[a]} & {r.speaker_id for r in groups[b]}
        sources = {r.source_group for r in groups[a]} & {r.source_group for r in groups[b]}
        if speakers or sources:
            raise ValueError(f"speaker/source leakage across {a} and {b}")
        left, right = (
            Counter(provenance[r.utterance_id]["normalized_text_sha256"] for r in groups[role])
            for role in (a, b)
        )
        shared = left.keys() & right.keys()
        overlap[f"{a}/{b}"] = {
            "speaker_overlap": 0,
            "source_overlap": 0,
            "distinct_shared_normalized_transcripts": len(shared),
            "left_utterances_with_shared_transcript": sum(left[key] for key in shared),
            "right_utterances_with_shared_transcript": sum(right[key] for key in shared),
        }
    return {"roles": stats, "cross_role": overlap}


def _check_examples(audio: TrainingAudio, rows: tuple[GeneratorRow, ...]) -> None:
    frontend = SpectralFrontend()
    with torch.inference_mode():
        for row in rows:
            waveform = audio.waveform(row)
            example = prepare_example(frontend, waveform)
            if (
                example.length != row.prepared_samples
                or not torch.equal(example.target[0], waveform)
                or example.mel.shape != (1, FRONTEND_V1.n_mels, frames_for_length(example.length))
                or not torch.isfinite(example.mel).all()
                or not torch.equal(
                    example.mel, frontend.condition(frontend.analyze(example.target))
                )
            ):
                raise ValueError("published audio failed frontend alignment")


def load_overfit_manifest(folder: Path, expected_sha256: str) -> GeneratorManifest:
    """Resolve the pinned diagnostic selection against its frozen pilot metadata."""
    selection = read_pinned_json(folder / "manifest.json", expected_sha256)
    if (
        set(selection) != {"schema", "parent_manifest_sha256", "train_utterance_ids"}
        or selection["schema"] != OVERFIT_SCHEMA
    ):
        raise ValueError("invalid train-only diagnostic selection")
    parent = load_manifest(folder / "parent-manifest.json", selection["parent_manifest_sha256"])
    ids = selection["train_utterance_ids"]
    if not isinstance(ids, list) or not ids or not all(isinstance(uid, str) for uid in ids):
        raise ValueError("diagnostic requires a list of train utterance IDs")
    train = {row.utterance_id: row for row in parent.examples("train")}
    if len(set(ids)) != len(ids) or set(ids) - train.keys():
        raise ValueError("diagnostic selection must be a unique subset of parent TRAIN")
    return replace(parent, rows=tuple(train[uid] for uid in ids))


def load_overfit(
    folder: Path, expected_sha256: str
) -> tuple[TrainingAudio, tuple[GeneratorRow, ...]]:
    """Read an explicitly pinned eight-utterance projection, without dev access."""
    manifest = load_overfit_manifest(folder, expected_sha256)
    return TrainingAudio(manifest, folder), manifest.rows


def _source_metadata(folder: Path, files: dict[str, bytes], profile: dict) -> str:
    folder.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        (folder / name).write_bytes(raw)
    return write_json(folder / "resampling-receipt.json", profile)


def _validate_training_release(root: Path, pin: str, *, probe_examples: bool) -> None:
    manifest = load_manifest(root / "manifest.json", pin)
    for row in manifest.rows:
        if row.role == "eval" and (
            (root / row.native_path).exists() or (root / row.prepared_path).exists()
        ):
            raise ValueError("training release contains held-out audio")
    audio = TrainingAudio(manifest, root)
    audio.validate_headers()
    if probe_examples:
        _check_examples(audio, tuple(manifest.examples(role)[0] for role in ("train", "dev")))


def prepare_train_dev(
    zipped: ZipFile,
    inventory: Inventory,
    protocol: SplitProtocol,
    output: Path,
    reports: Path,
    attribution: dict[str, bytes],
    *,
    max_audio_bytes: int,
) -> dict:
    """Build train/dev releases; compute held-out identities entirely in memory."""
    allocation = allocate_speakers(inventory, protocol)
    by_speaker = defaultdict(list)
    for uid, item in inventory.utterances.items():
        by_speaker[item.speaker].append(uid)
    allocation_sha = write_json(
        reports / "allocation.json",
        {
            "protocol": asdict(protocol),
            "speakers": allocation,
            "groups": {
                group: {
                    "gender": dict(Counter(inventory.speakers[sid].gender for sid in ids)),
                    "accent": dict(Counter(inventory.speakers[sid].accent for sid in ids)),
                    "ages": sorted(inventory.speakers[sid].age for sid in ids),
                }
                for group, ids in allocation.items()
            },
        },
    )
    expanded, pilot, overfit = (output / name for name in ("expanded", "pilot", "overfit"))
    profile = resampling_document()
    attribution = {**attribution, "speaker-info.txt": inventory.speaker_info}
    profile_sha = _source_metadata(expanded / "source", attribution, profile)
    rows = []
    selected_by_speaker = defaultdict(list)
    provenance = {}
    seen_bytes: dict[str, str] = {}
    projected_bytes = 0
    pilot_ids = set(allocation["pilot_train"] + allocation["pilot_dev"])
    for role, group, size in (
        ("train", "base_expert_fit", protocol.expanded_per_speaker),
        ("dev", "dev", protocol.expanded_per_speaker),
        ("eval", "eval", protocol.eval_per_speaker),
    ):
        for sid in allocation[group]:
            selected = selected_by_speaker[sid]
            candidates = ranked_ids(protocol.seed, f"utterance:{sid}", by_speaker[sid])
            for uid in candidates:
                try:
                    record = prepare_recording(
                        zipped, uid, inventory.utterances[uid], role, profile_sha
                    )
                except InvalidRecording as exc:
                    with (reports / "audio-rejections.jsonl").open("ab") as stream:
                        stream.write(
                            canonical_json({"utterance": uid, "role": role, "reason": str(exc)})
                            + b"\n"
                        )
                    continue
                row = record.row
                for digest in (
                    row.native_sha256,
                    row.prepared_sha256,
                    record.provenance["native_pcm_sha256"],
                ):
                    if seen_bytes.setdefault(digest, uid) != uid:
                        raise ValueError(
                            f"duplicate audio detected: {uid} and {seen_bytes[digest]}"
                        )
                if role != "eval":
                    copies = 1 + (sid in pilot_ids and len(selected) < protocol.pilot_per_speaker)
                    projected_bytes += copies * (len(record.native) + len(record.prepared))
                    if projected_bytes > max_audio_bytes:
                        raise ValueError("prepared training audio exceeds the approved storage cap")
                    _save_recording(expanded, record)
                rows.append(row)
                selected.append(row)
                provenance[uid] = record.provenance
                if len(selected) == size:
                    break
            if len(selected) < size:
                raise ValueError(
                    f"speaker {sid} supplied {len(selected)}/{size} eligible recordings"
                )

    eval_rows = [row for row in rows if row.role == "eval"]
    pilot_rows = [
        row
        for sid in allocation["pilot_train"] + allocation["pilot_dev"]
        for row in selected_by_speaker[sid][: protocol.pilot_per_speaker]
    ]
    diag_rows = [
        selected_by_speaker[sid][0]
        for sid in allocation["pilot_train"][: protocol.overfit_utterances]
    ]
    for root, subset in ((pilot, pilot_rows), (overfit, diag_rows)):
        _source_metadata(root / "source", attribution, profile)
        for row in subset:
            if root == overfit:
                projected_bytes += sum(
                    provenance[row.utterance_id][field]
                    for field in ("native_bytes", "prepared_bytes")
                )
            if projected_bytes > max_audio_bytes:
                raise ValueError("prepared training audio exceeds the approved storage cap")
            _clone_recording(expanded, root, row)
    expanded_sha = write_json(expanded / "manifest.json", _manifest("expanded", rows))
    pilot_sha = write_json(pilot / "manifest.json", _manifest("pilot", pilot_rows + eval_rows))
    (overfit / "parent-manifest.json").write_bytes((pilot / "manifest.json").read_bytes())
    overfit_sha = write_json(
        overfit / "manifest.json",
        {
            "schema": OVERFIT_SCHEMA,
            "parent_manifest_sha256": pilot_sha,
            "train_utterance_ids": [row.utterance_id for row in diag_rows],
        },
    )
    _validate_training_release(expanded, expanded_sha, probe_examples=True)
    _validate_training_release(pilot, pilot_sha, probe_examples=True)
    diagnostic_audio, diagnostic_rows = load_overfit(overfit, overfit_sha)
    diagnostic_audio.validate_headers()
    _check_examples(diagnostic_audio, diagnostic_rows)
    audit = {
        "expanded": _audit(rows, provenance),
        "pilot": _audit(pilot_rows + eval_rows, provenance),
        "overfit": _audit(diag_rows, provenance),
    }
    provenance_sha = write_json(output / "recording-provenance.json", provenance)
    audit_sha = write_json(output / "leakage-audit.json", audit)
    freeze = {
        "schema": FREEZE_SCHEMA,
        "protocol": asdict(protocol),
        "allocation_sha256": allocation_sha,
        "resampling_receipt_sha256": profile_sha,
        "provenance_sha256": provenance_sha,
        "leakage_audit_sha256": audit_sha,
        "manifests": {"expanded": expanded_sha, "pilot": pilot_sha, "overfit": overfit_sha},
        "archive_sha256": attribution_archive_sha(attribution),
    }
    freeze_sha = write_json(output / "freeze.json", freeze)
    return {"freeze_sha256": freeze_sha, "manifests": freeze["manifests"], "audit": audit}


def attribution_archive_sha(attribution: dict[str, bytes]) -> str:
    return parse_json(attribution["acquisition-receipt.json"])["archive"]["sha256"]


def prepare_held_out(
    zipped: ZipFile,
    inventory: Inventory,
    protocol: SplitProtocol,
    freeze_root: Path,
    freeze_sha256: str,
    output: Path,
    attribution: dict[str, bytes],
    *,
    max_audio_bytes: int,
) -> dict:
    """Replay and publish only held-out rows from the approved training freeze."""
    freeze = read_pinned_json(freeze_root / "freeze.json", freeze_sha256)
    if (
        freeze.get("schema") != FREEZE_SCHEMA
        or freeze.get("protocol") != asdict(protocol)
        or freeze.get("archive_sha256") != attribution_archive_sha(attribution)
    ):
        raise ValueError("held-out replay requires the approved source and split protocol")
    manifest_path = freeze_root / "expanded" / "manifest.json"
    manifest = load_manifest(manifest_path, freeze["manifests"]["expanded"])
    profile = resampling_document()
    profile_sha = sha256(canonical_json(profile) + b"\n")
    if profile_sha != freeze["resampling_receipt_sha256"]:
        raise ValueError("held-out replay requires the same resampling runtime as freeze")
    provenance = read_pinned_json(
        freeze_root / "recording-provenance.json", freeze["provenance_sha256"]
    )
    _source_metadata(
        output / "source", {**attribution, "speaker-info.txt": inventory.speaker_info}, profile
    )
    rows = [row for row in manifest.rows if row.role == "eval"]
    audio_bytes = 0
    for row in rows:
        record = prepare_recording(
            zipped, row.utterance_id, inventory.utterances[row.utterance_id], "eval", profile_sha
        )
        if record.row != row or record.provenance != provenance[row.utterance_id]:
            raise ValueError(
                f"held-out replay differs from the frozen manifest: {row.utterance_id}"
            )
        audio_bytes += len(record.native) + len(record.prepared)
        if audio_bytes > max_audio_bytes:
            raise ValueError("prepared evaluation audio exceeds the approved storage cap")
        _save_recording(output, record)
    for row in manifest.examples("train") + manifest.examples("dev"):
        if (output / row.native_path).exists() or (output / row.prepared_path).exists():
            raise ValueError("held-out release contains training-side audio")
    (output / "manifest.json").write_bytes(manifest_path.read_bytes())
    eval_pin = write_json(
        output / "eval-index.json",
        {
            "schema": "vocoroute-vctk092-held-out-v1",
            "freeze_sha256": freeze_sha256,
            "manifest_sha256": manifest.sha256,
            "audio_roles": ["eval"],
            "utterance_ids": [row.utterance_id for row in rows],
        },
    )
    return {
        "freeze_sha256": freeze_sha256,
        "manifest_sha256": manifest.sha256,
        "eval_index_sha256": eval_pin,
        "audit": _audit(rows, provenance),
    }
