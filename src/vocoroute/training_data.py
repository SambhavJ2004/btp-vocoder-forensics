"""Read-only intake for separately approved generator data and aligned tensors."""

import hashlib
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor

from .spectral import FRONTEND_V1, SpectralFrontend
from .training_config import DEV_ROLE, ROLES, TRAIN_ROLE, is_sha256, parse_json

MANIFEST_SCHEMA = "vocoroute-generator-manifest-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: str) -> None:
    if (
        not isinstance(value, str)
        or any(char in value for char in ("\\", ":", "\0"))
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError("source and prepared paths must be portable relative paths")


@dataclass(frozen=True)
class GeneratorRow:
    utterance_id: str
    speaker_id: str
    source_group: str
    role: str
    native_path: str
    native_sha256: str
    native_rate: int
    native_samples: int
    prepared_path: str
    prepared_sha256: str
    prepared_samples: int
    resample_steps: int
    resampling_profile: str
    resampling_receipt_sha256: str | None

    def __post_init__(self) -> None:
        labels = (self.utterance_id, self.speaker_id, self.source_group, self.resampling_profile)
        if any(
            not isinstance(value, str) or not value or value != value.strip() for value in labels
        ):
            raise ValueError("row IDs and resampling profile must be nonempty strings")
        if self.role not in ROLES:
            raise ValueError("invalid frozen generator role")
        _relative_path(self.native_path)
        _relative_path(self.prepared_path)
        if not is_sha256(self.native_sha256) or not is_sha256(self.prepared_sha256):
            raise ValueError("native and prepared recordings need SHA-256 digests")
        counts = (self.native_rate, self.native_samples, self.prepared_samples)
        if any(type(value) is not int or value < 1 for value in counts):
            raise ValueError("rates and sample counts must be positive integers")
        rate = FRONTEND_V1.sample_rate
        expected_samples = (self.native_samples * rate + self.native_rate - 1) // self.native_rate
        if self.prepared_samples != expected_samples:
            raise ValueError("prepared sample count disagrees with the ceil resampling contract")
        converted = self.native_rate != rate
        if type(self.resample_steps) is not int or self.resample_steps != int(converted):
            raise ValueError("resampling must be either identity or one native-to-generator step")
        if converted:
            if (
                self.resampling_profile == "identity"
                or not is_sha256(self.resampling_receipt_sha256)
                or self.native_sha256 == self.prepared_sha256
                or self.native_path.casefold() == self.prepared_path.casefold()
            ):
                raise ValueError("converted audio needs distinct bytes and pinned resampling")
        elif (
            self.resampling_profile != "identity"
            or self.resampling_receipt_sha256 is not None
            or self.native_sha256 != self.prepared_sha256
        ):
            raise ValueError("native-rate audio must use the identity recipe and identical bytes")


@dataclass(frozen=True)
class GeneratorManifest:
    dataset_id: str
    sha256: str
    rows: tuple[GeneratorRow, ...]

    def examples(self, role: str, *, seed: int = 0, epoch: int = 0) -> tuple[GeneratorRow, ...]:
        if role not in (TRAIN_ROLE, DEV_ROLE):
            raise ValueError("the training data interface only exposes train and dev roles")
        rows = sorted(
            (row for row in self.rows if row.role == role), key=lambda row: row.utterance_id
        )
        if role == TRAIN_ROLE:
            random.Random(f"{seed}:{epoch}").shuffle(rows)
        return tuple(rows)


def _validate_partition(rows: tuple[GeneratorRow, ...]) -> None:
    if {row.role for row in rows} != set(ROLES):
        raise ValueError("the frozen manifest must contain train, dev and eval rows")
    if len({row.utterance_id.casefold() for row in rows}) != len(rows):
        raise ValueError("utterance IDs must be unique")
    groups: dict[str, str] = {}
    speakers: dict[str, str] = {}
    paths: set[str] = set()
    digests: set[str] = set()
    multispeaker = len({row.speaker_id.casefold() for row in rows}) > 1
    for row in rows:
        if groups.setdefault(row.source_group.casefold(), row.role) != row.role:
            raise ValueError("native source groups must not cross roles")
        if multispeaker and speakers.setdefault(row.speaker_id.casefold(), row.role) != row.role:
            raise ValueError("multi-speaker data must be speaker-disjoint across roles")
        row_paths = {row.native_path.casefold(), row.prepared_path.casefold()}
        row_digests = {row.native_sha256, row.prepared_sha256}
        if paths & row_paths or digests & row_digests:
            raise ValueError("recording paths and byte identities must be unique across rows")
        paths.update(row_paths)
        digests.update(row_digests)


def load_manifest(path: str | Path, expected_sha256: str) -> GeneratorManifest:
    """Open the exact externally approved manifest bytes. Never infer new assignments."""
    if not is_sha256(expected_sha256):
        raise ValueError("an approved manifest SHA-256 pin is required")
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("generator manifest does not match the approved SHA-256")
    doc = parse_json(raw)
    if (
        set(doc) != {"schema", "dataset_id", "source_kind", "target_rate", "rows"}
        or doc["schema"] != MANIFEST_SCHEMA
        or doc["source_kind"] != "native_generator"
        or type(doc["target_rate"]) is not int
        or doc["target_rate"] != FRONTEND_V1.sample_rate
        or not isinstance(doc["dataset_id"], str)
        or not doc["dataset_id"].strip()
        or not isinstance(doc["rows"], list)
    ):
        raise ValueError("invalid native generator manifest schema")
    try:
        rows = tuple(GeneratorRow(**row) for row in doc["rows"])
    except TypeError as exc:
        raise ValueError("invalid generator manifest row fields") from exc
    _validate_partition(rows)
    return GeneratorManifest(doc["dataset_id"], digest, rows)


class TrainingExample(NamedTuple):
    mel: Tensor
    target: Tensor
    length: int


def prepare_example(frontend: SpectralFrontend, waveform: Tensor) -> TrainingExample:
    """Use the same untrimmed mono samples as target and mel source."""
    if waveform.ndim != 1 or waveform.dtype not in (torch.float32, torch.float64):
        raise ValueError("expected one float32 or float64 mono waveform")
    if not waveform.numel() or not torch.isfinite(waveform).all():
        raise ValueError("expected a finite, nonempty waveform")
    target = waveform.detach().clone().unsqueeze(0)
    mel = frontend.condition(frontend.analyze(target))
    return TrainingExample(mel, target, target.shape[-1])


def _recording_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"recording must be a file within the approved data root: {relative}")
    return path


def _check_header(stream: io.BytesIO, rate: int, samples: int, relative: str) -> None:
    from data.audio_io import probe

    info = probe(stream)
    if (info["channels"], info["sample_rate"], info["frames"]) != (1, rate, samples):
        raise ValueError(f"recording header disagrees with approved manifest: {relative}")
    stream.seek(0)


class TrainingAudio:
    """Verify train/dev bytes and expose only those recordings to the runner."""

    def __init__(self, manifest: GeneratorManifest, root: str | Path) -> None:
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("approved generator data root must be a directory")
        self._rows = dict.fromkeys(manifest.examples(TRAIN_ROLE) + manifest.examples(DEV_ROLE))
        for row in self._rows:
            for relative, expected in (
                (row.native_path, row.native_sha256),
                (row.prepared_path, row.prepared_sha256),
            ):
                if sha256_file(_recording_path(self.root, relative)) != expected:
                    raise ValueError(f"recording SHA-256 mismatch: {relative}")

    def _snapshot(self, relative: str, expected: str) -> io.BytesIO:
        raw = _recording_path(self.root, relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"recording SHA-256 mismatch after preflight: {relative}")
        return io.BytesIO(raw)

    def validate_headers(self) -> None:
        for row in self._rows:
            for relative, digest, rate, samples in (
                (row.native_path, row.native_sha256, row.native_rate, row.native_samples),
                (
                    row.prepared_path,
                    row.prepared_sha256,
                    FRONTEND_V1.sample_rate,
                    row.prepared_samples,
                ),
            ):
                with self._snapshot(relative, digest) as stream:
                    _check_header(stream, rate, samples, relative)

    def waveform(self, row: GeneratorRow) -> Tensor:
        if row not in self._rows:
            raise ValueError("only approved train/dev rows can be decoded")
        from data.audio_io import read_native

        rate = FRONTEND_V1.sample_rate
        with self._snapshot(row.prepared_path, row.prepared_sha256) as stream:
            _check_header(stream, rate, row.prepared_samples, row.prepared_path)
            audio, decoded_rate = read_native(stream)
        if audio.shape != (row.prepared_samples,) or decoded_rate != rate:
            raise ValueError("decoded waveform disagrees with approved manifest")
        return torch.from_numpy(audio)
