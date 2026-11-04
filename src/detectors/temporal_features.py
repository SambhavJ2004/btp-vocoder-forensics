"""Common PCM16 analysis and acoustic subframes for development localization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import get_window, sosfiltfilt

from data.audio_io import read_processed, write_audio
from data.invariants import ARCHIVE_SR, LADDER_FMAX, check_band_limit
from data.preprocess import band_limit_comparison_set, band_limit_sos
from detectors.base import crop_samples_for_rate, fixed_length_crop

CONTEXT_SAMPLES = crop_samples_for_rate(ARCHIVE_SR)


@dataclass(frozen=True)
class WindowConfig:
    window_ms: float = 32.0
    hop_ms: float = 8.0
    filter_guard_ms: float = 150.0
    n_bands: int = 24
    log_floor: float = 1e-12
    sample_rate: int = ARCHIVE_SR
    band_limit_hz: float = LADDER_FMAX
    version: str = "temporal_spectral_pcm16_v1"

    def validate(self):
        values = (self.window_ms, self.hop_ms, self.filter_guard_ms, self.log_floor)
        if any(
            isinstance(x, bool) or not isinstance(x, (float, int)) or not np.isfinite(x)
            for x in values
        ):
            raise ValueError("Temporal feature parameters must be finite numbers.")
        if not 4 <= self.hop_ms <= self.window_ms <= 256:
            raise ValueError("Require 4 <= hop_ms <= window_ms <= 256.")
        if (
            self.filter_guard_ms != 150
            or self.n_bands != 24
            or self.log_floor != 1e-12
            or self.sample_rate != ARCHIVE_SR
            or self.band_limit_hz != LADDER_FMAX
            or self.version != "temporal_spectral_pcm16_v1"
        ):
            raise ValueError("Unsupported temporal analysis/guard protocol.")

    @property
    def window_samples(self):
        return round(self.window_ms * self.sample_rate / 1000)

    @property
    def hop_samples(self):
        return round(self.hop_ms * self.sample_rate / 1000)

    @property
    def guard_samples(self):
        return round(self.filter_guard_ms * self.sample_rate / 1000)

    @property
    def feature_names(self):
        return [f"log10_mean_psd_band_{i:02d}" for i in range(self.n_bands)] + [
            "log10_inband_power",
            "centroid_fraction",
            "spread_fraction",
            "rolloff85_fraction",
            "spectral_flatness",
            "log_spectral_slope",
            "zero_crossing_rate",
        ]


DEFAULT_WINDOW_CONFIG = WindowConfig()


def filter_support_receipt(config: WindowConfig = DEFAULT_WINDOW_CONFIG) -> dict:
    """Numerical effective support, not a claim that an IIR has finite support.

    A centered impulse spans ~5.94 s; compute the zero-phase absolute tail.
    Multiplying by 2 bounds convolution change for a bounded difference <=2
    within this measured support. End effects/infinite tails remain limitations.
    """
    config.validate()
    n = 2**17
    impulse = np.zeros(n)
    impulse[n // 2] = 1
    response = sosfiltfilt(band_limit_sos(), impulse)
    outside = np.abs(np.arange(n) - n // 2) > config.guard_samples
    tail = float(np.abs(response[outside]).sum())
    bound_steps = 2 * tail * 32768
    if not np.isfinite(bound_steps) or bound_steps > 0.01:
        raise ValueError("Pinned filter effective support exceeds the fixed guard tolerance.")
    return {
        "filter_guard_samples": config.guard_samples,
        "impulse_samples": n,
        "measured_absolute_tail_sum": tail,
        "bounded_delta_pcm16_steps": bound_steps,
        "acceptance_tolerance_pcm16_steps": 0.01,
        "finite_support": False,
        "scope": "numerical_effective_support_not_exact_purity_or_authentication",
    }


def analysis_comparison(wavs: dict[str, np.ndarray], work_dir: str | Path) -> dict:
    """Same verified filter/PCM16 path for every member, including the real null.

    Short inference inputs repeat-pad to the common context before filtering;
    return only original samples. No source file is written or renormalized.
    """
    lengths = {k: len(w) for k, w in wavs.items()}
    if not lengths or min(lengths.values()) <= 0 or len(set(lengths.values())) != 1:
        raise ValueError("Analysis needs nonempty, equally long paired waveforms.")
    short = next(iter(lengths.values())) < 256
    prepared = {k: fixed_length_crop(w, CONTEXT_SAMPLES) if short else w for k, w in wavs.items()}
    limited = band_limit_comparison_set(prepared)
    result = {}
    with TemporaryDirectory(prefix="temporal-pcm16-", dir=work_dir) as temp:
        for i, (name, wav) in enumerate(limited.items()):
            path = Path(temp) / f"{i}.wav"
            write_audio(path, wav, ARCHIVE_SR)
            decoded = read_processed(path, ARCHIVE_SR)
            check_band_limit(decoded, ARCHIVE_SR, where="temporal PCM16 analysis")
            result[name] = decoded[: lengths[name]]
    return result


def window_starts(n: int, config: WindowConfig) -> np.ndarray:
    config.validate()
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 0:
        raise ValueError("Recording length must be a nonnegative integer.")
    if n == 0:
        return np.empty(0, dtype=np.int64)
    last = max(0, n - config.window_samples)
    starts = np.arange(0, last + 1, config.hop_samples, dtype=np.int64)
    return np.r_[starts, last] if starts[-1] != last else starts


def extract_windows(wav: np.ndarray, config: WindowConfig) -> tuple[np.ndarray, np.ndarray]:
    """31 acoustic values; metadata and window position never enter the vector.

    Local spectral frames sit inside fixed 4.0375 s repeat-padded contexts.
    Overlap of window_length-1 between contexts preserves the global grid.
    Padded features occur only for an input shorter than one local frame.
    """
    config.validate()
    wav = np.asarray(wav)
    if wav.ndim != 1 or np.iscomplexobj(wav) or not np.isfinite(wav).all():
        raise ValueError("Expected finite mono analysis samples.")
    starts = window_starts(len(wav), config)
    length = config.window_samples
    if not len(starts):
        return starts, np.empty((0, len(config.feature_names)))
    frames = np.empty((len(starts), length), dtype=np.float64)
    stride = CONTEXT_SAMPLES - length + 1
    contexts = starts // stride * stride
    for origin in np.unique(contexts):
        take = np.flatnonzero(contexts == origin)
        context = fixed_length_crop(wav[origin : origin + CONTEXT_SAMPLES], CONTEXT_SAMPLES)
        frames[take] = np.lib.stride_tricks.sliding_window_view(context, length)[
            starts[take] - origin
        ]
    n_fft = 1 << (length - 1).bit_length()
    hann = get_window("hann", length, fftbins=True)
    power = np.abs(np.fft.rfft(frames * hann, n=n_fft, axis=1)) ** 2 / (
        config.sample_rate * np.sum(hann**2)
    )
    power[:, 1:-1] *= 2  # One-sided PSD; DC and Nyquist retain their original weights.
    frequencies = np.fft.rfftfreq(n_fft, 1 / config.sample_rate)
    keep = frequencies < config.band_limit_hz
    p, freq = power[:, keep], frequencies[keep] / config.band_limit_hz
    edges = np.linspace(0, 1, config.n_bands + 1)
    bands = np.column_stack(
        [
            p[:, (freq >= lo) & (freq < hi)].mean(axis=1)
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        ]
    )
    total = p.sum(axis=1)
    safe = np.maximum(total, config.log_floor)
    centroid = (p * freq).sum(axis=1) / safe
    spread = np.sqrt(np.maximum((p * (freq - centroid[:, None]) ** 2).sum(axis=1) / safe, 0))
    rolloff = freq[np.argmax(np.cumsum(p, axis=1) >= 0.85 * total[:, None], axis=1)]
    logs = np.log10(np.maximum(p, config.log_floor))
    flatness = np.where(
        total > config.log_floor,
        10 ** logs.mean(axis=1) / np.maximum(p.mean(axis=1), config.log_floor),
        0,
    )
    centered = freq - freq.mean()
    slope = (logs * centered).sum(axis=1) / np.sum(centered**2)
    features = np.column_stack(
        [
            np.log10(np.maximum(bands, config.log_floor)),
            np.log10(np.maximum(total * config.sample_rate / n_fft, config.log_floor)),
            centroid,
            spread,
            rolloff,
            flatness,
            slope,
            np.mean(np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]), axis=1),
        ]
    )
    if not np.isfinite(features).all():
        raise ValueError("Non-finite temporal acoustic features.")
    return starts, features


def sample_labels(intervals: list[dict], n: int) -> np.ndarray:
    labels = np.full(n, -1, dtype=np.int8)
    prior = 0
    for item in intervals:
        s, e, label = item["start_sample"], item["end_sample"], item["label"]
        if (
            not all(isinstance(v, int) and not isinstance(v, bool) for v in (s, e, label))
            or s != prior
            or not s < e <= n
            or label not in (-1, 0, 1)
        ):
            raise ValueError("Labels must exactly cover zero-based half-open sample intervals.")
        labels[s:e] = label
        prior = e
    if prior != n:
        raise ValueError("Incomplete sample labels.")
    return labels


def guarded_labels(labels: np.ndarray, guard: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, (-1, 0, 1)).all() or guard < 0:
        raise ValueError("Invalid dense labels or guard.")
    ignored = labels == -1
    if guard:
        # Hard splices also have temporal support even without a raw -1 fade.
        ignored[1:] |= labels[1:] != labels[:-1]
    prefix = np.r_[0, np.cumsum(ignored)]
    positions = np.arange(len(labels))
    mask = (
        prefix[np.minimum(len(labels), positions + guard + 1)]
        - prefix[np.maximum(0, positions - guard)]
        > 0
    )
    result = labels.copy()
    result[mask] = -1
    if guard:
        result[:guard] = result[max(0, len(labels) - guard) :] = -1
    return result


def assign_window_labels(labels: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """Require 100% one class. Any ignore, mixture or repeat padding is unscored."""
    starts = np.asarray(starts, dtype=np.int64)
    if length <= 0 or (starts < 0).any():
        raise ValueError("Invalid local window geometry.")
    result = np.full(len(starts), -1, dtype=np.int8)
    full = starts + length <= len(labels)
    for label in (0, 1):
        prefix = np.r_[0, np.cumsum(labels == label)]
        take = np.flatnonzero(full)
        good = prefix[starts[take] + length] - prefix[starts[take]] == length
        result[take[good]] = label
    return result


def smooth_probabilities(probabilities: np.ndarray, width: int) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or width not in (1, 3):
        raise ValueError("Invalid probabilities or supported smoothing width.")
    if not len(p):
        return p.copy()
    return np.convolve(np.pad(p, width // 2, mode="edge"), np.ones(width) / width, mode="valid")


def predicted_intervals(
    starts: np.ndarray,
    probabilities: np.ndarray,
    n: int,
    config: WindowConfig,
    threshold: float,
    merge_samples: int,
) -> list[list[int]]:
    """Project center scores to midpoint cells; merge short gaps without consulting truth."""
    starts = np.asarray(starts)
    p = np.asarray(probabilities)
    config.validate()
    if (
        starts.ndim != 1
        or p.shape != starts.shape
        or not np.isfinite(p).all()
        or not 0 <= threshold <= 1
        or merge_samples < 0
        or n < 0
        or starts.dtype.kind not in "iu"
        or (starts < 0).any()
        or (starts >= max(n, 1)).any()
        or ((p < 0) | (p > 1)).any()
        or (len(starts) > 1 and (np.diff(starts) <= 0).any())
    ):
        raise ValueError("Invalid temporal prediction geometry or scores.")
    if not len(starts):
        return []
    centers = np.minimum(starts + config.window_samples / 2, n)
    edges = np.r_[0, np.floor((centers[:-1] + centers[1:]) / 2).astype(int), n]
    intervals = []
    for i in np.flatnonzero(p >= threshold):
        s, e = int(edges[i]), int(edges[i + 1])
        if e <= s:
            continue
        if intervals and s - intervals[-1][1] <= merge_samples:
            intervals[-1][1] = e
        else:
            intervals.append([s, e])
    return intervals
