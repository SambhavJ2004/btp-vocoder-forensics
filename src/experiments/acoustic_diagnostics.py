"""Pure CPU statistics for Phase 3D; oracle analyses never enter inference."""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window
from scipy.stats import spearmanr

from data.invariants import ARCHIVE_SR, GLOBAL_SEED
from detectors.temporal_features import WindowConfig, extract_windows
from experiments.temporal_review import distribution

GRID = WindowConfig(window_ms=64, hop_ms=16)
ACTIVITY_NAMES = ("low_energy", "active_quiet", "active_loud")
BOOTSTRAPS = 2000
BANDS = (
    (0, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 6000),
    (6000, 8000),
    (8000, 10900),
    (10900, 11025),
)


def mono(wav):
    values = np.asarray(wav)
    if (
        values.ndim != 1
        or not len(values)
        or np.iscomplexobj(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Require nonempty finite real mono diagnostic input.")
    return values.astype(np.float64)


def window_rms(wav, starts, length=GRID.window_samples):
    values = mono(wav)
    starts = np.asarray(starts)
    if (
        starts.ndim != 1
        or starts.dtype.kind not in "iu"
        or (starts < 0).any()
        or (starts + length > len(values)).any()
        or length <= 0
    ):
        raise ValueError("Diagnostic RMS requires complete in-range windows.")
    prefix = np.r_[0.0, np.cumsum(values**2)]
    return np.sqrt(np.maximum(0, (prefix[starts + length] - prefix[starts]) / length))


def activity_bins(levels, threshold):
    levels = np.asarray(levels)
    if (
        not np.isfinite(levels).all()
        or (levels < 0).any()
        or not np.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError("Invalid activity levels or real-reference threshold.")
    return np.where(levels < threshold, 0, np.where(levels < 10 * threshold, 1, 2)).astype(np.int8)


def rms_controlled_features(wav, starts):
    """Per-window RMS 0.03 in RAM, no clipping/WAV writes; silence stays unchanged.

    Batch independent frames through the existing extractor at hop=window.
    Batches are <4.0375 s; every returned frame lies wholly in original samples.
    """
    values = mono(wav)
    levels = window_rms(values, starts)
    gains = np.ones(len(starts))
    measurable = levels > 2 / 32768
    gains[measurable] = 0.03 / levels[measurable]
    frames = np.lib.stride_tricks.sliding_window_view(values, GRID.window_samples)[starts]
    batch_config = WindowConfig(window_ms=64, hop_ms=64)
    result, max_peak = [], 0.0
    for i in range(0, len(starts), 48):
        normalized = frames[i : i + 48] * gains[i : i + 48, None]
        max_peak = max(max_peak, float(np.max(np.abs(normalized))))
        _, features = extract_windows(normalized.ravel(), batch_config)
        if len(features) != len(normalized):
            raise ValueError("Independent normalized-frame geometry mismatch.")
        result.append(features)
    return (np.concatenate(result) if result else np.empty((0, 31))), {
        "unmeasurable_windows_unchanged": int((~measurable).sum()),
        "maximum_scaled_peak": max_peak,
        "gains": distribution(gains),
    }


def psd_frames(wav, starts):
    values = mono(wav)
    window_rms(values, starts)  # Validate full-frame support before indexing.
    hann = get_window("hann", GRID.window_samples, fftbins=True)
    frames = np.lib.stride_tricks.sliding_window_view(values, len(hann))[starts]
    psd = np.abs(np.fft.rfft(frames * hann, n=2048, axis=-1)) ** 2 / (ARCHIVE_SR * (hann @ hann))
    psd[:, 1:-1] *= 2
    return psd, np.fft.rfftfreq(2048, 1 / ARCHIVE_SR)


def smooth_log_psd(psd, quefrency_samples=22):
    """Real-cepstral low-quefrency envelope of log10 PSD; not a learned feature."""
    values = np.asarray(psd)
    if (
        values.ndim != 2
        or values.shape[1] != 1025
        or (values < 0).any()
        or not np.isfinite(values).all()
        or not 0 <= quefrency_samples <= 1024
    ):
        raise ValueError("Invalid diagnostic PSD/lifter geometry.")
    cepstrum = np.fft.irfft(np.log10(np.maximum(values, 1e-12)), n=2048, axis=1)
    if quefrency_samples < 1024:
        cepstrum[:, quefrency_samples + 1 : 2048 - quefrency_samples] = 0
    return np.fft.rfft(cepstrum, n=2048, axis=1).real


def periodicity(wav, starts):
    """Overlap-normalized autocorrelation maximum at 50–500 Hz lags, not HNR/F0."""
    values = mono(wav)
    window_rms(values, starts)
    frames = np.lib.stride_tricks.sliding_window_view(values, GRID.window_samples)[starts].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    frames *= get_window("hann", GRID.window_samples, fftbins=True)
    fft = np.fft.rfft(frames, n=4096, axis=1)
    ac = np.fft.irfft(np.abs(fft) ** 2, n=4096, axis=1)
    lags = np.arange(int(np.ceil(ARCHIVE_SR / 500)), int(np.floor(ARCHIVE_SR / 50)) + 1)
    prefix = np.c_[np.zeros(len(frames)), np.cumsum(frames**2, axis=1)]
    denominator = np.sqrt(
        prefix[:, GRID.window_samples - lags] * (prefix[:, [-1]] - prefix[:, lags])
    )
    result = np.full((len(frames), len(lags)), np.nan)
    np.divide(ac[:, lags], denominator, out=result, where=denominator > 1e-20)
    available = np.isfinite(result).any(axis=1)
    output = np.full(len(frames), np.nan)
    output[available] = np.nanmax(result[available], axis=1)
    return np.clip(output, -1, 1)


def spectral_pair(real, synthetic, *, threshold, allow_fullband_distance=False):
    """Matched full frames and a REAL-reference activity mask; no lag correction."""
    real, synthetic = mono(real), mono(synthetic)
    if len(real) != len(synthetic) or len(real) < GRID.window_samples:
        raise ValueError("Require equally long, complete paired diagnostic cores.")
    starts = np.arange(0, len(real) - GRID.window_samples + 1, GRID.hop_samples)
    active = window_rms(real, starts) >= threshold
    rp, freq = psd_frames(real, starts)
    sp, _ = psd_frames(synthetic, starts)
    difference = 10 * (np.log10(np.maximum(sp, 1e-12)) - np.log10(np.maximum(rp, 1e-12)))
    envelope = 10 * (smooth_log_psd(sp) - smooth_log_psd(rp))
    low = freq < 8000
    result = {
        "all_frames": len(starts),
        "matched_real_active_frames": int(active.sum()),
        "rms_ratio_db": float(
            20
            * np.log10(
                max(np.sqrt(np.mean(synthetic**2)), 1e-12) / max(np.sqrt(np.mean(real**2)), 1e-12)
            )
        ),
        "band_energy_ratio_db": [],
        "real_band_energy": [],
        "synthetic_band_energy": [],
    }
    centered_real, centered_syn = real - real.mean(), synthetic - synthetic.mean()
    denominator = np.linalg.norm(centered_real) * np.linalg.norm(centered_syn)
    result["zero_lag_waveform_correlation"] = (
        float(centered_real @ centered_syn / denominator) if denominator else None
    )
    ref_rms = np.sqrt(np.mean(real**2))
    result["relative_waveform_error_rms"] = (
        float(np.sqrt(np.mean((synthetic - real) ** 2)) / ref_rms) if ref_rms else None
    )
    for lo, hi in BANDS:
        select = (freq >= lo) & (freq < hi)
        rpower = float(rp[:, select].sum(axis=1).mean() * ARCHIVE_SR / 2048)
        spower = float(sp[:, select].sum(axis=1).mean() * ARCHIVE_SR / 2048)
        result["real_band_energy"].append(rpower)
        result["synthetic_band_energy"].append(spower)
        result["band_energy_ratio_db"].append(
            float(10 * np.log10(max(spower, 1e-12) / max(rpower, 1e-12)))
        )
    result["real_high_band_fraction"] = float(rp[:, freq >= 8000].sum() / max(rp.sum(), 1e-30))
    result["synthetic_high_band_fraction"] = float(sp[:, freq >= 8000].sum() / max(sp.sum(), 1e-30))
    if not active.any():
        result.update(
            lsd_below8k_db=None,
            envelope_below8k_db=None,
            shape_lsd_below8k_db=None,
            periodicity_difference=None,
            flatness_difference=None,
            fullband_lsd_db=None,
        )
        return result
    db = difference[active][:, low]
    result["lsd_below8k_db"] = float(np.sqrt(np.mean(db**2)))
    real_frames = np.lib.stride_tricks.sliding_window_view(real, GRID.window_samples)[starts]
    syn_frames = np.lib.stride_tricks.sliding_window_view(synthetic, GRID.window_samples)[starts]
    result["active_frame_exact_sample_fraction"] = float(
        np.mean(real_frames[active] == syn_frames[active])
    )
    result["shape_lsd_below8k_db"] = float(
        np.sqrt(np.mean((db - db.mean(axis=1, keepdims=True)) ** 2))
    )
    result["envelope_below8k_db"] = float(np.sqrt(np.mean(envelope[active][:, low] ** 2)))
    rperiod, speriod = periodicity(real, starts)[active], periodicity(synthetic, starts)[active]
    valid = np.isfinite(rperiod) & np.isfinite(speriod)
    result["periodicity_difference"] = (
        float(np.mean(speriod[valid] - rperiod[valid])) if valid.any() else None
    )
    result["periodicity_paired_frames"] = int(valid.sum())

    def flatness(power):
        p = np.maximum(power[active][:, low], 1e-12)
        return np.exp(np.log(p).mean(axis=1)) / p.mean(axis=1)

    result["flatness_difference"] = float(np.mean(flatness(sp) - flatness(rp)))
    result["fullband_lsd_db"] = (
        float(np.sqrt(np.mean(difference[active] ** 2))) if allow_fullband_distance else None
    )
    return result


def group_summary(values):
    """Percentile bootstrap of independent source statistics, fixed predictions.

    Each input is ONE statistic for ONE source. No window/file resampling and no
    model refitting; this does not capture training, speaker or corpus uncertainty.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or np.isinf(values).any():
        raise ValueError("Require one finite-or-missing statistic per source.")
    finite = values[np.isfinite(values)]
    result = {
        **distribution(finite),
        "missing_source_groups": int(len(values) - len(finite)),
        "source_mean_ci95": None,
        "ci_scope": "conditional_source_mean_frozen_predictions",
    }
    if len(finite) >= 10:
        rng = np.random.default_rng(GLOBAL_SEED + 304)
        draws = rng.integers(0, len(finite), size=(BOOTSTRAPS, len(finite)))
        result["source_mean_ci95"] = np.quantile(
            finite[draws].mean(axis=1), [0.025, 0.975]
        ).tolist()
    return result


def correlation(x, y):
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Require paired source-level covariates.")
    use = np.isfinite(a) & np.isfinite(b)
    rho = None
    if use.sum() >= 3 and np.ptp(a[use]) and np.ptp(b[use]):
        rho = float(spearmanr(a[use], b[use]).statistic)
    return {"source_groups": int(use.sum()), "spearman_rho": rho, "causal_claim": False}


def balanced_indices(records, *, activity_matched=False):
    """Train-only source/class balance; optional exact within-source activity-bin counts."""
    selected = [[] for _ in records]
    for group in sorted({r["row"]["group_id"] for r in records}):
        if any(r["row"]["split"] != "train" for r in records):
            raise ValueError("Ablation sampling is restricted to train.")
        buckets = range(3) if activity_matched else (-1,)
        count = 0
        for bucket in buckets:
            candidates = {0: [], 1: []}
            for i, record in enumerate(records):
                if record["row"]["group_id"] != group:
                    continue
                for label in (0, 1):
                    take = record["window_y"] == label
                    if activity_matched:
                        take &= record["own_activity"] == bucket
                    candidates[label].extend((i, int(j)) for j in np.flatnonzero(take))
            n = min(len(candidates[0]), len(candidates[1]), 24 if activity_matched else 72)
            for label in (0, 1):
                candidates[label].sort(
                    key=lambda item: (records[item[0]]["row"]["example_id"], item[1])
                )
            import hashlib

            seed = int.from_bytes(
                hashlib.sha256(f"{GLOBAL_SEED}:3d:{group}:{bucket}".encode()).digest()[:8], "little"
            )
            rng = np.random.default_rng(seed)
            for label in (0, 1):
                for index in rng.choice(len(candidates[label]), n, replace=False):
                    i, j = candidates[label][index]
                    selected[i].append(j)
            count += n
        if not count:
            raise ValueError(
                f"No balanced acoustic training support in {group}; do not silently drop it."
            )
    return [np.array(sorted(take), dtype=np.int64) for take in selected]
