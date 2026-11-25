"""The only sanctioned audio read/write path.

Every module in this project reads and writes audio through here. Direct calls
to ``soundfile.write`` / ``librosa.load`` elsewhere bypass the invariant checks
and are treated as bugs in review (see CLAUDE.md, INV-05).

``sample_rate`` is a required argument on every write. It used to default to the
single delivery rate; with two tiers (INV-01) a default is exactly the kind of
silent mistake this module exists to prevent.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf

from .invariants import (
    CONTAINER,
    DTYPE,
    SUBTYPE,
    InvariantViolation,
    check_waveform,
    tier_for_rate,
)


def read_native(path: str | Path | BinaryIO) -> tuple[np.ndarray, int]:
    """Read a file at its own native sample rate, downmixed to mono.

    Native-rate reading is deliberate: the resample happens exactly once, in
    :func:`data.preprocess.resample_once`, so that no file is ever passed
    through two anti-aliasing filters (INV-01).
    """
    source = str(path) if isinstance(path, (str, Path)) else path
    wav, sr = sf.read(source, dtype=DTYPE, always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return np.ascontiguousarray(wav, dtype=np.float32), int(sr)


def read_processed(path: str | Path, expected_sr: int | None = None) -> np.ndarray:
    """Read a file that is already through the pipeline; assert it still complies.

    Pass ``expected_sr`` whenever the caller knows which tier it wants. Reading
    an archive file where a zero-shot file was intended is silent otherwise --
    the audio is valid, just at the wrong rate for the consumer.
    """
    wav, sr = read_native(path)
    check_waveform(wav, sr, where=str(path), expected_sr=expected_sr)
    return wav


def write_audio(
    path: str | Path,
    wav: np.ndarray,
    sample_rate: int,
    *,
    measured_lufs: float | None = None,
) -> Path:
    """Write a processed waveform, refusing anything that violates the contract.

    ``sample_rate`` also selects the tier, via
    :func:`data.invariants.tier_for_rate`, so an unsanctioned rate is rejected
    before any bytes are written.
    """
    path = Path(path)
    if path.suffix.lower() != CONTAINER:
        raise InvariantViolation(
            f"INV-05: refusing to write '{path.suffix}'. Every condition is "
            f"{CONTAINER}/{SUBTYPE}; a container difference is a per-condition label."
        )
    tier_for_rate(sample_rate)  # raises on an unsanctioned rate
    wav = np.ascontiguousarray(wav, dtype=np.float32)
    check_waveform(
        wav, sample_rate, where=str(path), expected_sr=sample_rate, measured_lufs=measured_lufs
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sample_rate, subtype=SUBTYPE)
    return path


def probe(path: str | Path | BinaryIO) -> dict[str, object]:
    """Cheap header read: rate, channels, subtype, duration. Used by audits."""
    source = str(path) if isinstance(path, (str, Path)) else path
    info = sf.info(source)
    return {
        "path": str(path),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "frames": info.frames,
        "duration_s": info.duration,
    }


def write_generator_pcm16(stream: BinaryIO, wav: np.ndarray, sample_rate: int) -> None:
    """Encode native generator targets without applying forensic loudness policy."""
    if (
        type(sample_rate) is not int
        or sample_rate <= 0
        or wav.ndim != 1
        or not wav.size
        or not np.isfinite(wav).all()
        or wav.min() < -1.0
        or wav.max() > 32767 / 32768
    ):
        raise ValueError("expected finite mono samples in the PCM16 range and a valid rate")
    sf.write(stream, wav, sample_rate, format="WAV", subtype="PCM_16")
