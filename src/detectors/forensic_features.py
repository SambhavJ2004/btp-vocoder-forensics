"""Shared, controlled archive-tier features for the non-citable ML pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import get_window

from data.audio_io import probe, read_processed, write_audio
from data.invariants import (
    ARCHIVE_SR,
    LADDER_FMAX,
    REAL_CONDITION,
    SUBTYPE,
    InvariantViolation,
    check_band_limit,
)
from data.preprocess import band_limit_comparison_set
from detectors.base import crop_samples_for_rate, fixed_length_crop


@dataclass(frozen=True)
class FeatureConfig:
    """Versioned protocol definition, not tunable per condition or dataset."""

    version: str = "archive_log_spectral_v1"
    sample_rate: int = ARCHIVE_SR
    band_limit_hz: float = LADDER_FMAX
    analysis_encoding: str = SUBTYPE
    crop_samples: int = crop_samples_for_rate(ARCHIVE_SR)
    crop_policy: str = "first_crop_repeat_pad"
    n_fft: int = 1024
    hop_samples: int = 256
    window: str = "hann_periodic"
    center: bool = False
    n_bands: int = 32
    log_floor: float = 1e-12
    statistics: str = "mean_then_population_std_log10_band_power"
    max_input_seconds: float = 60.0

    def validate(self) -> None:
        if asdict(self) != asdict(FeatureConfig()):
            raise ValueError("Unsupported feature/preprocessing configuration; use a new protocol.")

    @property
    def feature_names(self) -> list[str]:
        return [f"{stat}_log_power_band_{i:02d}"
                for stat in ("mean", "std") for i in range(self.n_bands)]


DEFAULT_FEATURE_CONFIG = FeatureConfig()


def read_archive_input(path: str | Path, config: FeatureConfig) -> np.ndarray:
    """Fail closed on unsupported inputs; do not fix rates, channels or gain."""
    config.validate()
    path = Path(path)
    if path.suffix.lower() != ".wav" or not path.is_file():
        raise ValueError(f"Expected an existing local WAV file: {path}")
    try:
        header = probe(path)
    except (RuntimeError, OSError) as exc:
        raise ValueError(f"Cannot read WAV header: {path}") from exc
    if (header["sample_rate"] != config.sample_rate or header["channels"] != 1
            or header["subtype"] != config.analysis_encoding):
        raise ValueError("Pilot input must be mono PCM_16 WAV at 22050 Hz; no implicit conversion.")
    if not 0.1 <= header["duration_s"] <= config.max_input_seconds:
        raise ValueError("Pilot WAV duration must be between 0.1 and 60 seconds.")
    return read_processed(path, expected_sr=config.sample_rate)


def spectral_statistics(wav: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """64 features from PCM_16 analysis audio; no metadata enters the vector."""
    config.validate()
    wav = fixed_length_crop(wav, config.crop_samples).astype(np.float64)
    frames = np.lib.stride_tricks.sliding_window_view(wav, config.n_fft)[::config.hop_samples]
    window = get_window("hann", config.n_fft, fftbins=True)
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2 / np.sum(window ** 2)
    frequencies = np.fft.rfftfreq(config.n_fft, 1 / config.sample_rate)
    edges = np.linspace(0, config.band_limit_hz, config.n_bands + 1)
    # Upper edges excluded: no above-band bin, resampler cue or format metadata.
    bands = np.column_stack([
        power[:, (frequencies >= lo) & (frequencies < hi)].mean(axis=1)
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ])
    logs = np.log10(np.maximum(bands, config.log_floor))
    features = np.concatenate([logs.mean(axis=0), logs.std(axis=0, ddof=0)])
    if not np.isfinite(features).all():
        raise ValueError("Non-finite forensic features.")
    return features


def extract_comparison_features(
    wavs: dict[str, np.ndarray], work_dir: str | Path,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> dict[str, np.ndarray]:
    """Process one complete utterance group, with bounded audio memory.

    The sanctioned comparison-set filter includes real. Materialize PCM_16 and
    reread it before cropping/features, rather than scoring Phase A float arrays.
    Temporary filenames use ordinals, never untrusted utterance/condition text.
    """
    config.validate()
    if len({len(wav) for wav in wavs.values()}) != 1:
        raise InvariantViolation("INV-06: pilot utterance group has unequal sample lengths.")
    limited = band_limit_comparison_set(
        wavs, sr=config.sample_rate, cutoff_hz=config.band_limit_hz,
    )
    features = {}
    for i, (condition, wav) in enumerate(limited.items()):
        path = Path(work_dir) / f"analysis_{i}.wav"
        write_audio(path, wav, config.sample_rate)
        decoded = read_processed(path, expected_sr=config.sample_rate)
        check_band_limit(decoded, config.sample_rate, cutoff_hz=config.band_limit_hz,
                         where="pilot PCM_16 analysis input")
        features[condition] = spectral_statistics(decoded, config)
    return features


def extract_file_features(
    path: str | Path, config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    *, work_dir: str | Path | None = None,
) -> np.ndarray:
    """The same transform for inference, with no class information available.

    A singleton occupies the transform API's reference slot, regardless of its
    unknown class. The filter is independent per waveform: this is the exact
    processing used for EVERY member of a training comparison set.
    """
    wav = read_archive_input(path, config)
    with TemporaryDirectory(prefix="btpvf-analysis-", dir=work_dir) as tmp:
        return extract_comparison_features({REAL_CONDITION: wav}, tmp, config)[REAL_CONDITION]
