"""Four fixed CPU acoustic ablations and explicit numeric inference bundles.

Analysis filtering and context/cell policies are reused unchanged from Phase 3A.
These offline features use only waveform samples, never annotation metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import get_window
from scipy.special import expit

from data.audio_io import read_processed
from data.invariants import ARCHIVE_SR, LADDER_FMAX
from detectors.base import fixed_length_crop
from detectors.forensic_pilot import LinearHead, runtime_versions, sha256_file, write_json
from detectors.temporal_features import (
    CONTEXT_SAMPLES,
    WindowConfig,
    analysis_comparison,
    extract_windows,
    predicted_intervals,
    smooth_probabilities,
    window_starts,
)
from detectors.temporal_model import implementation_identity

PROTOCOL = "vocopatch_grouped_context_archive8k_v1"
REJECT_ALL_THRESHOLD = float(np.nextafter(1.0, np.inf))


@dataclass(frozen=True)
class ContextConfig(WindowConfig):
    window_ms: float = 64.0
    hop_ms: float = 16.0
    representation: str = "spectral31"

    def validate(self):
        # The Phase 3A schema stays unchanged; only this explicit extension
        # admits the predeclared 512 ms subframe inside the same 4.0375s context.
        base = WindowConfig(
            window_ms=64,
            hop_ms=16,
            filter_guard_ms=self.filter_guard_ms,
            n_bands=self.n_bands,
            log_floor=self.log_floor,
            sample_rate=self.sample_rate,
            band_limit_hz=self.band_limit_hz,
            version=self.version,
        )
        base.validate()
        if (
            isinstance(self.window_ms, bool)
            or isinstance(self.hop_ms, bool)
            or not isinstance(self.window_ms, (int, float))
            or not isinstance(self.hop_ms, (int, float))
            or self.hop_ms != 16
            or (self.window_ms, self.representation)
            not in {
                (64, "spectral31"),
                (256, "spectral31"),
                (512, "spectral31"),
                (256, "logmel24_mean_std"),
            }
        ):
            raise ValueError("Only the four predeclared context representations are supported.")

    @property
    def feature_names(self):
        if self.representation == "spectral31":
            return super().feature_names
        return [f"log10_mel_{stat}_{i:02d}" for stat in ("mean", "std") for i in range(24)]


CANDIDATES = {
    "A64": ContextConfig(),
    "B256": ContextConfig(window_ms=256),
    "C512": ContextConfig(window_ms=512),
    "Dmel256": ContextConfig(window_ms=256, representation="logmel24_mean_std"),
}


def mel_weights(n_fft: int):
    """24 HTK-spaced triangular filters; normalize discrete PSD weights to one."""
    frequencies = np.fft.rfftfreq(n_fft, 1 / ARCHIVE_SR)
    keep = frequencies < LADDER_FMAX
    mel_max = 2595 * np.log10(1 + LADDER_FMAX / 700)
    knots = 700 * (10 ** (np.linspace(0, mel_max, 26) / 2595) - 1)
    bank = []
    for lo, center, hi in zip(knots[:-2], knots[1:-1], knots[2:], strict=True):
        triangle = np.maximum(
            0,
            np.minimum(
                (frequencies[keep] - lo) / (center - lo), (hi - frequencies[keep]) / (hi - center)
            ),
        )
        if triangle.sum() <= 0:
            raise ValueError("Empty fixed mel filter.")
        bank.append(triangle / triangle.sum())
    return keep, np.array(bank)


def extract_context(wav, config: ContextConfig):
    config.validate()
    if config.representation == "spectral31":
        return extract_windows(wav, config)
    wav = np.asarray(wav)
    if wav.ndim != 1 or np.iscomplexobj(wav) or not np.isfinite(wav).all():
        raise ValueError("Expected finite mono context samples.")
    starts = window_starts(len(wav), config)
    if not len(starts):
        return starts, np.empty((0, 48))
    sub = WindowConfig(window_ms=32, hop_ms=8)
    offsets = window_starts(config.window_samples, sub)
    length = sub.window_samples
    n_fft = 1 << (length - 1).bit_length()
    hann = get_window("hann", length, fftbins=True)
    keep, bank = mel_weights(n_fft)
    result = []
    # Bound FFT temporary memory. Every subframe lies wholly in the declared
    # outer window, with the same repeat-padding/context policy as spectral31.
    stride = CONTEXT_SAMPLES - config.window_samples + 1
    for begin in range(0, len(starts), 32):
        frame_starts = starts[begin : begin + 32]
        outer = np.empty((len(frame_starts), config.window_samples))
        for i, start in enumerate(frame_starts):
            origin = int(start // stride * stride)
            context = fixed_length_crop(wav[origin : origin + CONTEXT_SAMPLES], CONTEXT_SAMPLES)
            outer[i] = context[start - origin : start - origin + config.window_samples]
        frames = np.lib.stride_tricks.sliding_window_view(outer, length, axis=1)[:, offsets, :]
        psd = np.abs(np.fft.rfft(frames * hann, n=n_fft, axis=-1)) ** 2 / (
            ARCHIVE_SR * np.sum(hann**2)
        )
        psd[..., 1:-1] *= 2
        log_mel = np.log10(np.maximum(psd[..., keep] @ bank.T, config.log_floor))
        result.append(np.c_[log_mel.mean(axis=1), log_mel.std(axis=1, ddof=0)])
    features = np.vstack(result)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite context features.")
    return starts, features


def context_intervals(starts, probabilities, n, config, threshold):
    if not np.isfinite(threshold) or not 0 <= threshold <= REJECT_ALL_THRESHOLD:
        raise ValueError("Invalid context threshold.")
    if threshold > 1:
        # Explicit deterministic all-negative operating point for saturated ties.
        return []
    return predicted_intervals(starts, probabilities, n, config, threshold, 0)


def context_identity():
    import detectors.forensic_pilot

    return {
        **implementation_identity(),
        __name__: sha256_file(__file__),
        "detectors.forensic_pilot": sha256_file(detectors.forensic_pilot.__file__),
    }


@dataclass
class ContextBundle:
    head: LinearHead
    config: ContextConfig
    thresholds: dict[str, float]
    metadata: dict

    def validate(self):
        self.config.validate()
        self.head.validate(len(self.config.feature_names), 1)
        if (
            not self.thresholds
            or any(
                not isinstance(k, str)
                or isinstance(v, bool)
                or not np.isfinite(v)
                or not 0 <= v <= REJECT_ALL_THRESHOLD
                for k, v in self.thresholds.items()
            )
            or self.metadata.get("protocol") != PROTOCOL
            or self.metadata.get("development_only") is not True
            or self.metadata.get("citable") is not False
            or self.metadata.get("score_direction") != "higher_is_synthetic"
            or self.metadata.get("feature_names") != self.config.feature_names
        ):
            raise ValueError("Unsupported context bundle schema/thresholds/disclosures.")

    def margins(self, x):
        self.validate()
        return self.head.decision_function(x)[:, 0]


def save_context_bundle(bundle, directory):
    bundle.validate()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    fields = ("mean", "scale", "coefficients", "intercept")
    np.savez_compressed(root / "model.npz", **{key: getattr(bundle.head, key) for key in fields})
    write_json(
        root / "bundle.json",
        {
            **bundle.metadata,
            "format_version": 1,
            "config": asdict(bundle.config),
            "thresholds": bundle.thresholds,
            "smoothing": 3,
            "merge_samples": 0,
            "environment": runtime_versions(),
            "implementation_identity": context_identity(),
        },
    )
    write_json(
        root / "checksums.json",
        {name: sha256_file(root / name) for name in ("bundle.json", "model.npz")},
    )


def load_context_bundle(directory):
    import json

    root = Path(directory).resolve()
    if {p.name for p in root.iterdir()} != {"model.npz", "bundle.json", "checksums.json"}:
        raise ValueError("Unexpected context bundle contents.")
    for path in root.iterdir():
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or getattr(path, "is_junction", lambda: False)()
        ):
            raise ValueError("Unsafe context bundle file.")
    checks = json.loads((root / "checksums.json").read_text())
    if set(checks) != {"bundle.json", "model.npz"} or any(
        sha256_file(root / name) != value for name, value in checks.items()
    ):
        raise ValueError("Context bundle integrity failure.")
    meta = json.loads((root / "bundle.json").read_text())
    if (
        meta.get("format_version") != 1
        or meta.get("environment") != runtime_versions()
        or meta.get("implementation_identity") != context_identity()
        or meta.get("smoothing") != 3
        or meta.get("merge_samples") != 0
    ):
        raise ValueError("Context bundle compatibility failure.")
    fields = ("mean", "scale", "coefficients", "intercept")
    with np.load(root / "model.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != set(fields):
            raise ValueError("Invalid context numeric state.")
        head = LinearHead(*(arrays[key].copy() for key in fields))
    bundle = ContextBundle(head, ContextConfig(**meta["config"]), meta["thresholds"], meta)
    bundle.validate()
    return bundle


def predict_context_file(bundle, wav_path, work_dir, operating_point="fpr_05"):
    if not isinstance(bundle, ContextBundle):
        bundle = load_context_bundle(bundle)
    bundle.validate()
    if operating_point not in bundle.thresholds:
        raise ValueError("Unknown declared operating point.")
    wav = read_processed(wav_path, ARCHIVE_SR)
    if not 0 < len(wav) <= 60 * ARCHIVE_SR:
        raise ValueError("Context inference requires nonempty audio of at most 60 seconds.")
    analysis = analysis_comparison({"real": wav}, work_dir)["real"]
    starts, features = extract_context(analysis, bundle.config)
    margin = bundle.margins(features)
    p = expit(margin)
    smoothed = smooth_probabilities(p, 3)
    intervals = context_intervals(
        starts, smoothed, len(wav), bundle.config, bundle.thresholds[operating_point]
    )
    return {
        "window_start_samples": starts.tolist(),
        "synthetic_margins": margin.tolist(),
        "uncalibrated_probabilities": p.tolist(),
        "smoothed_probabilities": smoothed.tolist(),
        "operating_point": operating_point,
        "threshold": bundle.thresholds[operating_point],
        "model_predicted_intervals": [
            {
                "start_sample": s,
                "end_sample": e,
                "start_seconds": s / ARCHIVE_SR,
                "end_seconds": e / ARCHIVE_SR,
            }
            for s, e in intervals
        ],
        "ground_truth_used": False,
        "sample_rate": ARCHIVE_SR,
        "num_samples": len(wav),
        "development_only": True,
        "citable": False,
    }
