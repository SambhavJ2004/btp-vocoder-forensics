"""Confound-leakage sanity checks — the gate before any result is believed.

From the plan's risk table: "sanity check that a detector cannot separate
conditions on silence or bandwidth alone". These checks are cheap, they run on
CPU, and they are the difference between a defensible result and a retracted
one. Run them on every dataset version before B or C consume it.

Each check trains a deliberately weak classifier on a feature that carries NO
vocoder artifact. If such a classifier separates real from vocoded, the
separation is a confound, not a finding.

Probes run per tier (INV-01) and the tier is reported with every row. The
bandwidth probe is ARCHIVE-only: on the derived 16 kHz set every condition has
passed through the same downsampler, so the probe would report a reassuring 0.5
while saying nothing about the archive the study actually measures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.audio_io import read_processed
from data.invariants import (
    ARCHIVE_SR,
    ARCHIVE_TIER,
    HIGH_BAND_ANALYSIS,
    REAL_CONDITION,
    SANITY_EER_FLOOR,
    TIER_RATES,
    TIERS,
    TRIM_FRAME_LENGTH,
    TRIM_HOP_LENGTH,
    TRIM_PAD_MS,
    TRIM_TOP_DB,
    InvariantViolation,
    is_excluded,
)
from data.manifest import select_tier
from detectors.eer import compute_eer


def _leading_trailing_silence(
    wav: np.ndarray, sr: int, threshold_db: float = -50.0
) -> tuple[float, float]:
    amp = np.abs(wav)
    thr = 10 ** (threshold_db / 20.0)
    voiced = np.where(amp > thr)[0]
    if voiced.size == 0:
        return len(wav) / sr, 0.0
    return voiced[0] / sr, (len(wav) - voiced[-1]) / sr


def _tier_rows(
    manifest: pd.DataFrame, tier: str, split: str | None
) -> tuple[pd.DataFrame, int]:
    """``split=None`` means every split. The probes train nothing -- each is a
    scalar feature scored directly -- so there is no held-out set to protect,
    and using every utterance is what keeps the EER's sampling noise below the
    floor at small dataset sizes."""
    sub = select_tier(manifest, tier)
    return (sub if split is None else sub[sub["split"] == split]), TIER_RATES[tier]


def silence_probe(
    manifest: pd.DataFrame, condition: str, split: str | None = "eval",
    tier: str = ARCHIVE_TIER,
) -> dict[str, object]:
    """INV-03 check. Can leading/trailing silence duration alone separate the
    condition from real? It must not: identical trim spans are applied to both.
    """
    sub, sr = _tier_rows(manifest, tier, split)
    feats: dict[str, np.ndarray] = {}
    for name in (REAL_CONDITION, condition):
        rows = sub[sub["condition"] == name]
        vals = [sum(_leading_trailing_silence(read_processed(p, sr), sr)) for p in rows["path"]]
        feats[name] = np.asarray(vals)

    eer, _ = compute_eer(feats[REAL_CONDITION], feats[condition])
    return _verdict("silence_duration", condition, eer, tier)


def loudness_probe(
    manifest: pd.DataFrame, condition: str, split: str | None = "eval",
    tier: str = ARCHIVE_TIER,
) -> dict[str, object]:
    """INV-04 check. Can broadband RMS alone separate the condition from real?

    Worth running on the derived tier too: the zero-shot set is deliberately not
    re-normalised, so this is the probe that would catch a condition whose
    high-band energy differed enough to shift its post-downsample level into a
    learnable cue.
    """
    sub, sr = _tier_rows(manifest, tier, split)
    feats: dict[str, np.ndarray] = {}
    for name in (REAL_CONDITION, condition):
        rows = sub[sub["condition"] == name]
        feats[name] = np.asarray(
            [float(np.sqrt(np.mean(read_processed(p, sr) ** 2))) for p in rows["path"]]
        )

    eer, _ = compute_eer(feats[REAL_CONDITION], feats[condition])
    return _verdict("rms_level", condition, eer, tier)


def duration_probe(
    manifest: pd.DataFrame, condition: str, split: str | None = "eval",
    tier: str = ARCHIVE_TIER,
) -> dict[str, object]:
    """INV-06 check. Can total duration alone separate the condition from real?
    Should be exactly at chance -- paired conditions have identical lengths.
    """
    sub, _ = _tier_rows(manifest, tier, split)
    real = sub[sub["condition"] == REAL_CONDITION]["duration_s"].to_numpy()
    other = sub[sub["condition"] == condition]["duration_s"].to_numpy()
    eer, _ = compute_eer(real, other)
    return _verdict("duration", condition, eer, tier)


def bandwidth_probe(
    manifest: pd.DataFrame, condition: str, split: str | None = "eval",
    tier: str = ARCHIVE_TIER,
) -> dict[str, object]:
    """INV-01 check. Can the highest frequency with meaningful energy separate
    the condition from real? A hit here usually means a resampling difference
    left a distinct anti-aliasing rolloff on one condition.

    ARCHIVE tier only. On the derived set every condition passed through the same
    downsampler, so the probe would measure that shared filter rather than the
    conditions and would report a falsely reassuring 0.5.

    Note this probe is expected to be less clean than the others: a genuine
    high-band vocoder artifact also moves effective bandwidth. Read a hit as
    "inspect the spectra", not automatically as "confound".
    """
    import librosa

    if tier != ARCHIVE_TIER:
        raise InvariantViolation(
            f"INV-01: the bandwidth probe is archive-only; '{tier}' has been "
            "low-passed by the shared downsampler and would always look clean."
        )
    sub, sr = _tier_rows(manifest, tier, split)
    feats: dict[str, np.ndarray] = {}
    for name in (REAL_CONDITION, condition):
        rows = sub[sub["condition"] == name]
        vals = []
        for p in rows["path"]:
            wav = read_processed(p, sr)
            rolloff = librosa.feature.spectral_rolloff(y=wav, sr=sr, roll_percent=0.995)
            vals.append(float(rolloff.mean()))
        feats[name] = np.asarray(vals)

    eer, _ = compute_eer(feats[REAL_CONDITION], feats[condition])
    return _verdict("bandwidth_rolloff", condition, eer, tier)


def _verdict(probe: str, condition: str, eer: float, tier: str) -> dict[str, object]:
    """One probe result.

    Some conditions are expected to separate on bandwidth *by design*, and
    flagging them as leaks would train the team to ignore the column. Those are
    exactly the conditions excluded from the high-band analysis (INV-18) --
    ``griffin_lim`` (mel-inversion cliff) and ``vocos`` (resampler edge) -- plus
    the paired control, trained at a wider fmax. They are reported as
    ``expected_by_design`` instead: visible, not a blocker.

    Every OTHER condition firing this probe is a genuine finding now that the
    archive is full-band: a time-domain vocoder should track real in the high
    band, and one that does not is telling you something.
    """
    from vocoders.registry import CONTROL_CONDITIONS

    by_design = probe == "bandwidth_rolloff" and (
        is_excluded(HIGH_BAND_ANALYSIS, condition) or condition in CONTROL_CONDITIONS
    )
    return {
        "probe": probe,
        "condition": condition,
        "tier": tier,
        "eer": eer,
        "leaked": bool(eer < SANITY_EER_FLOOR) and not by_design,
        "expected_by_design": by_design,
        "floor": SANITY_EER_FLOOR,
    }


def run_all(manifest: pd.DataFrame, split: str | None = "eval") -> pd.DataFrame:
    """Every probe against every non-real condition, on every tier present.

    A row with ``leaked == True`` blocks publication of that condition. The fix
    is to correct the pipeline and regenerate, never to relax SANITY_EER_FLOOR.
    """
    conditions = [c for c in manifest["condition"].unique() if c != REAL_CONDITION]
    tiers = [t for t in TIERS if t in set(manifest["tier"])]
    out: list[dict[str, object]] = []
    for tier in tiers:
        for cond in conditions:
            out.append(duration_probe(manifest, cond, split, tier))
            out.append(silence_probe(manifest, cond, split, tier))
            out.append(loudness_probe(manifest, cond, split, tier))
            if tier == ARCHIVE_TIER:
                out.append(bandwidth_probe(manifest, cond, split, tier))
    return pd.DataFrame(out)


# --- Diagnosing a silence-probe hit ------------------------------------------


SILENCE_DIAGNOSTIC_THRESHOLDS: tuple[float, ...] = (-60.0, -50.0, -40.0, -30.0, -20.0)


def reference_silent_mask(real: np.ndarray, sr: int = ARCHIVE_SR) -> np.ndarray:
    """Samples the REAL reference considers silent, by INV-03's own criterion.

    Frames more than ``TRIM_TOP_DB`` below the file's loudest frame -- the rule
    ``librosa.effects.trim`` applies, evaluated over the whole file instead of
    only its ends. Derived from real and applied by index to every condition, so
    every condition is measured over exactly the same samples (INV-03).
    """
    import librosa

    rms = librosa.feature.rms(
        y=real, frame_length=TRIM_FRAME_LENGTH, hop_length=TRIM_HOP_LENGTH, center=True
    )[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    silent_frames = db < -TRIM_TOP_DB
    mask = np.repeat(silent_frames, TRIM_HOP_LENGTH)[: len(real)]
    if len(mask) < len(real):
        mask = np.pad(mask, (0, len(real) - len(mask)), constant_values=False)
    return mask


def _dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(20.0 * np.log10(max(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))), 1e-12)))


def silence_diagnostic(
    manifest: pd.DataFrame,
    *,
    tier: str = ARCHIVE_TIER,
    split: str | None = None,
    thresholds: tuple[float, ...] = SILENCE_DIAGNOSTIC_THRESHOLDS,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Why a condition separates on silence duration. Diagnosis, not a gate.

    Returns two frames:

      floors   per condition: the level inside the REAL reference's silent
               regions, and how much of the file those regions are. A vocoder
               that reconstructs noise where the reference has near-silence
               reads higher here than real does.
      sweep    per condition and threshold: the silence probe's own feature
               (leading + trailing silence duration) recomputed at that
               threshold, and its EER against real. If the separation is
               reconstruction noise sitting at some level, it collapses once the
               threshold is well above that level; if it survives every
               threshold, the cause is not the noise floor.

    Neither frame decides anything. INV-12 says a leak blocks the condition, and
    whether this one is reconstruction noise (a vocoder artifact) or a pipeline
    fault is a judgement for a human to make on these numbers.
    """
    sub, sr = _tier_rows(manifest, tier, split)
    utts = sorted(sub[sub["condition"] == REAL_CONDITION]["utt_id"].unique())
    if limit is not None:
        utts = utts[:limit]
    conditions = [c for c in sorted(sub["condition"].unique()) if c != REAL_CONDITION]
    paths = {
        (str(r.condition), str(r.utt_id)): r.path
        for r in sub.itertuples()
        if str(r.utt_id) in set(utts)
    }

    floors: dict[str, list[float]] = {c: [] for c in (REAL_CONDITION, *conditions)}
    pads: dict[str, list[float]] = {c: [] for c in (REAL_CONDITION, *conditions)}
    silent_fraction: list[float] = []
    features: dict[float, dict[str, list[float]]] = {
        t: {c: [] for c in (REAL_CONDITION, *conditions)} for t in thresholds
    }

    for utt in utts:
        real = read_processed(paths[(REAL_CONDITION, utt)], sr)
        mask = reference_silent_mask(real, sr)
        silent_fraction.append(float(mask.mean()))
        for cond in (REAL_CONDITION, *conditions):
            wav = real if cond == REAL_CONDITION else read_processed(paths[(cond, utt)], sr)
            n = min(len(wav), len(mask))
            floors[cond].append(_dbfs(wav[:n][mask[:n]]))
            # The probe reads inward from both ends, so the trim pad is the
            # region it actually sees (INV-03 keeps TRIM_PAD_MS of context).
            pad = int(round(TRIM_PAD_MS * 1e-3 * sr))
            pads[cond].append(_dbfs(np.concatenate([wav[:pad], wav[-pad:]])))
            for t in thresholds:
                lead, trail = _leading_trailing_silence(wav, sr, threshold_db=t)
                features[t][cond].append(lead + trail)

    floor_rows = [
        {
            "condition": cond,
            "n": len(values),
            "silent_region_dbfs_median": float(np.nanmedian(values)),
            "silent_region_dbfs_p90": float(np.nanpercentile(values, 90)),
            "above_real_db": float(np.nanmedian(values) - np.nanmedian(floors[REAL_CONDITION])),
            "trim_pad_dbfs_median": float(np.nanmedian(pads[cond])),
            "trim_pad_above_real_db": float(
                np.nanmedian(pads[cond]) - np.nanmedian(pads[REAL_CONDITION])
            ),
            "reference_silent_fraction": float(np.mean(silent_fraction)),
        }
        for cond, values in floors.items()
    ]

    sweep_rows = []
    for t in thresholds:
        for cond in conditions:
            real_feature = np.asarray(features[t][REAL_CONDITION])
            cond_feature = np.asarray(features[t][cond])
            eer, _ = compute_eer(real_feature, cond_feature)
            sweep_rows.append(
                {
                    "threshold_dbfs": t,
                    "condition": cond,
                    "eer": eer,
                    "leaked_at_floor": bool(eer < SANITY_EER_FLOOR),
                    "real_silence_s_mean": float(real_feature.mean()),
                    "condition_silence_s_mean": float(cond_feature.mean()),
                }
            )
    return pd.DataFrame(floor_rows), pd.DataFrame(sweep_rows)
