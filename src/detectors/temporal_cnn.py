"""One small NumPy CNN and label-free log-mel inference, development only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import get_window
from scipy.special import expit

from data.audio_io import read_processed
from data.invariants import ARCHIVE_SR
from detectors.base import fixed_length_crop
from detectors.forensic_pilot import runtime_versions, sha256_file, write_json
from detectors.temporal_context import context_identity, context_intervals, mel_weights
from detectors.temporal_features import (
    CONTEXT_SAMPLES,
    WindowConfig,
    analysis_comparison,
    smooth_probabilities,
    window_starts,
)

GEOMETRY = WindowConfig(window_ms=128, hop_ms=16)
SUBFRAME = WindowConfig(window_ms=32, hop_ms=16)
FIELDS = ("kernel", "conv_bias", "head", "head_bias", "mean", "scale")
PARAMETERS = FIELDS[:4]
SHAPES = {"kernel": (9, 8), "conv_bias": (8,), "head": (440,), "head_bias": (1,)}
PROTOCOL = "vocopatch_numpy_cnn24x7_v1"
MODEL_SPEC = {
    "patch_shape": [24, 7],
    "outer_window_samples": 2822,
    "outer_hop_samples": 353,
    "subframe_samples": 706,
    "subframe_hop_samples": 353,
    "subframe_final_anchor": True,
    "fft_samples": 1024,
    "log_floor": 1e-12,
    "mel_weights": "HTK_discrete_PSD_mean_below_8000Hz",
    "conv_kernel": [3, 3],
    "conv_channels": 8,
    "activation": "ReLU",
    "pool": "mean_frequency_pairs",
    "head": "flatten_440_linear_logit",
    "normalization": "fit_sample_per_band_mean_std_time_and_observations",
    "parameters": 521,
    "smoothing_width": 3,
    "merge_samples": 0,
    "offline_lookahead": "zero_phase_filter_centered_smoothing_not_streaming",
}


def extract_logmel(wav):
    """Waveform only -> N x 24 frequency x 7 time, within each 128-ms window."""
    values = np.asarray(wav)
    if values.ndim != 1 or np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError("Require finite real mono samples.")
    starts = window_starts(len(values), GEOMETRY)
    if not len(starts):
        return starts, np.empty((0, 24, 7), dtype=np.float32)
    offsets = window_starts(GEOMETRY.window_samples, SUBFRAME)
    if len(offsets) != 7:
        raise ValueError("Unexpected fixed spectrogram geometry.")
    length = SUBFRAME.window_samples
    n_fft = 1 << (length - 1).bit_length()
    hann = get_window("hann", length, fftbins=True)
    keep, bank = mel_weights(n_fft)
    result = []
    stride = CONTEXT_SAMPLES - GEOMETRY.window_samples + 1
    for begin in range(0, len(starts), 64):
        current = starts[begin : begin + 64]
        outer = np.empty((len(current), GEOMETRY.window_samples))
        for i, start in enumerate(current):
            origin = int(start // stride * stride)
            context = fixed_length_crop(values[origin : origin + CONTEXT_SAMPLES], CONTEXT_SAMPLES)
            outer[i] = context[start - origin : start - origin + GEOMETRY.window_samples]
        frames = np.lib.stride_tricks.sliding_window_view(outer, length, axis=1)[:, offsets, :]
        psd = np.abs(np.fft.rfft(frames * hann, n=n_fft, axis=-1)) ** 2 / (
            ARCHIVE_SR * np.sum(hann**2)
        )
        psd[..., 1:-1] *= 2
        features = np.log10(np.maximum(psd[..., keep] @ bank.T, 1e-12))
        result.append(features.transpose(0, 2, 1).astype(np.float32))
    return starts, np.concatenate(result)


def checked_patches(x):
    values = np.asarray(x)
    if (
        np.iscomplexobj(values)
        or values.ndim != 3
        or values.shape[1:] != (24, 7)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Require finite N x 24 x 7 log-mel patches.")
    return values


@dataclass
class SmallCNN:
    kernel: np.ndarray
    conv_bias: np.ndarray
    head: np.ndarray
    head_bias: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def initialize(cls, x, seed):
        values = checked_patches(x).astype(np.float64)
        if not len(values):
            raise ValueError("Cannot fit normalization on empty training data.")
        rng = np.random.default_rng(seed)
        mean = values.mean(axis=(0, 2))
        scale = values.std(axis=(0, 2))
        scale[scale < 1e-8] = 1
        return cls(
            rng.normal(0, np.sqrt(2 / 9), (9, 8)),
            np.zeros(8),
            rng.normal(0, np.sqrt(1 / 440), 440),
            np.zeros(1),
            mean,
            scale,
        )

    def validate(self):
        for key, shape in {**SHAPES, "mean": (24,), "scale": (24,)}.items():
            value = getattr(self, key)
            if (
                not isinstance(value, np.ndarray)
                or value.shape != shape
                or value.dtype.kind != "f"
                or not np.isfinite(value).all()
            ):
                raise ValueError("Invalid CNN numeric state.")
        if (self.scale <= 0).any():
            raise ValueError("Invalid CNN normalization scale.")

    def forward(self, x, dropout_mask=None):
        self.validate()
        values = (checked_patches(x) - self.mean[None, :, None]) / self.scale[None, :, None]
        columns = np.lib.stride_tricks.sliding_window_view(values, (3, 3), axis=(1, 2))
        columns = columns.reshape(-1, 9)
        conv = (columns @ self.kernel + self.conv_bias).reshape(len(values), 22, 5, 8)
        pooled = np.maximum(conv, 0).reshape(len(values), 11, 2, 5, 8).mean(axis=2)
        flat = pooled.reshape(len(values), 440)
        if dropout_mask is not None:
            if dropout_mask.shape != flat.shape or not np.isfinite(dropout_mask).all():
                raise ValueError("Invalid dropout mask.")
            flat = flat * dropout_mask
        return flat @ self.head + self.head_bias[0], (columns, conv, flat, dropout_mask)

    def margins(self, x, batch=256):
        values = checked_patches(x)
        return (
            np.concatenate(
                [self.forward(values[i : i + batch])[0] for i in range(0, len(x), batch)]
            )
            if len(values)
            else np.empty(0)
        )

    def loss_grad(self, x, y, weight, *, l2=0.0001, dropout_mask=None):
        margin, (columns, conv, flat, mask) = self.forward(x, dropout_mask)
        y, weight = np.asarray(y), np.asarray(weight)
        if (
            not len(y)
            or y.shape != margin.shape
            or weight.shape != y.shape
            or not np.isin(y, (0, 1)).all()
            or not np.isfinite(weight).all()
            or (weight <= 0).any()
            or not np.isfinite(l2)
            or l2 < 0
        ):
            raise ValueError("Require binary labels and positive finite loss weights.")
        loss = np.mean(weight * (np.logaddexp(0, margin) - y * margin))
        loss += l2 / 2 * (np.sum(self.kernel**2) + np.sum(self.head**2))
        dm = weight * (expit(margin) - y) / len(y)
        grad = {"head": flat.T @ dm + l2 * self.head, "head_bias": np.array([dm.sum()])}
        dflat = dm[:, None] * self.head
        if mask is not None:
            dflat *= mask
        dp = dflat.reshape(len(y), 11, 5, 8)
        dc = np.repeat(dp, 2, axis=1) / 2 * (conv > 0)
        dc = dc.reshape(-1, 8)
        grad["kernel"] = columns.T @ dc + l2 * self.kernel
        grad["conv_bias"] = dc.sum(axis=0)
        return float(loss), grad


def loss_weights(y, groups):
    """Each source has equal total weight in each binary class; mean weight 1."""
    y, groups = np.asarray(y), np.asarray(groups)
    if not len(y) or groups.shape != y.shape or not np.isin(y, (0, 1)).all():
        raise ValueError("Require nonempty grouped binary samples.")
    weights = np.empty(len(y))
    unique = sorted(set(groups))
    for group in unique:
        for label in (0, 1):
            take = (groups == group) & (y == label)
            if not take.any():
                raise ValueError("Every training source must supply both classes.")
            weights[take] = len(y) / (2 * len(unique) * take.sum())
    return weights


def train_cnn(x, y, groups, seed, *, epochs=20, validation=None):
    """Fixed Adam recipe; optional inner validation selects an epoch, never outer data."""
    if not isinstance(epochs, int) or not 1 <= epochs <= 20:
        raise ValueError("CNN epochs must be within the fixed 20-epoch cap.")
    model = SmallCNN.initialize(x, seed)
    weights = loss_weights(y, groups)
    rng = np.random.default_rng(seed + 1)
    moment = {k: np.zeros_like(getattr(model, k)) for k in PARAMETERS}
    variance = {k: np.zeros_like(getattr(model, k)) for k in PARAMETERS}
    history, step, best, best_epoch, stale = [], 0, np.inf, 0, 0
    if validation is not None:
        vx, vy, vg = validation
        if set(groups) & set(vg):
            raise ValueError("Inner validation sources overlap training.")
        vw = loss_weights(vy, vg)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y))
        summed = 0.0
        for begin in range(0, len(y), 256):
            take = order[begin : begin + 256]
            mask = (rng.random((len(take), 440)) >= 0.1) / 0.9
            loss, gradient = model.loss_grad(x[take], y[take], weights[take], dropout_mask=mask)
            if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in gradient.values()):
                raise ValueError("Non-finite CNN loss/gradient; do not silently repair.")
            summed += loss * len(take)
            step += 1
            for key in PARAMETERS:
                moment[key] = 0.9 * moment[key] + 0.1 * gradient[key]
                variance[key] = 0.999 * variance[key] + 0.001 * gradient[key] ** 2
                update = (moment[key] / (1 - 0.9**step)) / (
                    np.sqrt(variance[key] / (1 - 0.999**step)) + 1e-8
                )
                getattr(model, key)[:] -= 0.001 * update
        item = {"epoch": epoch, "training_dropout_objective": summed / len(y)}
        if validation is not None:
            margin = model.margins(vx)
            val_loss = float(np.mean(vw * (np.logaddexp(0, margin) - vy * margin)))
            if not np.isfinite(val_loss):
                raise ValueError("Non-finite CNN validation loss.")
            item["inner_weighted_bce"] = val_loss
            if val_loss < best - 0.0001:
                best, best_epoch, stale = val_loss, epoch, 0
            else:
                stale += 1
        history.append(item)
        if validation is not None and stale >= 4:
            break
    model.validate()
    return model, {"history": history, "selected_epoch": best_epoch if validation else epochs}


def cnn_identity():
    return {**context_identity(), __name__: sha256_file(__file__)}


def save_cnn(model, directory, thresholds, **metadata):
    model.validate()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(root / "model.npz", **{k: getattr(model, k) for k in FIELDS})
    write_json(
        root / "bundle.json",
        {
            **metadata,
            "protocol": PROTOCOL,
            "model_spec": MODEL_SPEC,
            "thresholds": thresholds,
            "environment": runtime_versions(),
            "implementation_identity": cnn_identity(),
            "score_direction": "higher_is_synthetic",
            "development_only": True,
            "citable": False,
        },
    )
    write_json(
        root / "checksums.json",
        {name: sha256_file(root / name) for name in ("model.npz", "bundle.json")},
    )


def load_cnn(directory):
    root = Path(directory).resolve()
    expected = {"model.npz", "bundle.json", "checksums.json"}
    if {p.name for p in root.iterdir()} != expected:
        raise ValueError("Unexpected CNN bundle inventory.")
    if any(
        not p.is_file()
        or p.is_symlink()
        or p.stat().st_nlink != 1
        or getattr(p, "is_junction", lambda: False)()
        for p in root.iterdir()
    ):
        raise ValueError("Unsafe CNN bundle link/file.")
    meta = json.loads((root / "bundle.json").read_text())
    checks = json.loads((root / "checksums.json").read_text())
    if set(checks) != expected - {"checksums.json"} or any(
        sha256_file(root / n) != digest for n, digest in checks.items()
    ):
        raise ValueError("CNN bundle integrity failure.")
    if (
        meta.get("protocol") != PROTOCOL
        or meta.get("model_spec") != MODEL_SPEC
        or meta.get("environment") != runtime_versions()
        or meta.get("implementation_identity") != cnn_identity()
        or meta.get("development_only") is not True
        or meta.get("citable") is not False
        or meta.get("score_direction") != "higher_is_synthetic"
        or not meta.get("thresholds")
        or any(
            isinstance(v, bool) or not np.isfinite(v) or not 0 <= v <= np.nextafter(1.0, np.inf)
            for v in meta["thresholds"].values()
        )
    ):
        raise ValueError("CNN bundle schema/compatibility failure.")
    with np.load(root / "model.npz", allow_pickle=False) as state:
        if set(state.files) != set(FIELDS):
            raise ValueError("Unexpected CNN arrays.")
        model = SmallCNN(**{k: state[k] for k in FIELDS})
    model.validate()
    return model, meta


def predict_cnn_file(bundle, wav_path, work_dir, operating_point="fpr_05"):
    model, meta = load_cnn(bundle)
    if operating_point not in meta["thresholds"]:
        raise ValueError("Unknown CNN operating point.")
    wav = read_processed(wav_path, ARCHIVE_SR)
    if not 0 < len(wav) <= 60 * ARCHIVE_SR:
        raise ValueError("CNN inference needs nonempty audio of at most 60 seconds.")
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    analysis = analysis_comparison({"real": wav}, work_dir)["real"]
    starts, x = extract_logmel(analysis)
    margin = model.margins(x)
    p = expit(margin)
    smoothed = smooth_probabilities(p, 3)
    threshold = meta["thresholds"][operating_point]
    intervals = context_intervals(starts, smoothed, len(wav), GEOMETRY, threshold)
    return {
        "window_start_samples": starts.tolist(),
        "synthetic_margins": margin.tolist(),
        "uncalibrated_probabilities": p.tolist(),
        "smoothed_probabilities": smoothed.tolist(),
        "operating_point": operating_point,
        "threshold": threshold,
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
