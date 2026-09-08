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
    ARCHIVE_TIER,
    REAL_CONDITION,
    SANITY_EER_FLOOR,
    TIER_RATES,
    TIERS,
    InvariantViolation,
    is_correlation_excluded,
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


def _tier_rows(manifest: pd.DataFrame, tier: str, split: str) -> tuple[pd.DataFrame, int]:
    sub = select_tier(manifest, tier)
    return sub[sub["split"] == split], TIER_RATES[tier]


def silence_probe(
    manifest: pd.DataFrame, condition: str, split: str = "eval", tier: str = ARCHIVE_TIER
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
    manifest: pd.DataFrame, condition: str, split: str = "eval", tier: str = ARCHIVE_TIER
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
    manifest: pd.DataFrame, condition: str, split: str = "eval", tier: str = ARCHIVE_TIER
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
    manifest: pd.DataFrame, condition: str, split: str = "eval", tier: str = ARCHIVE_TIER
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

    Two conditions are expected to separate on bandwidth *by design*, and
    flagging them as leaks would train the team to ignore the column:
    ``griffin_lim`` (structurally cannot emit above its mel fmax) and
    ``bigvgan_v2_22khz_fullband`` (trained at a wider fmax than the ladder). Both
    are reported as ``expected_by_design`` instead -- visible, not a blocker.

    Every OTHER condition firing this probe is a genuine finding now that the
    archive is full-band: a time-domain vocoder should track real in the high
    band, and one that does not is telling you something.
    """
    by_design = is_correlation_excluded(condition) and probe == "bandwidth_rolloff"
    return {
        "probe": probe,
        "condition": condition,
        "tier": tier,
        "eer": eer,
        "leaked": bool(eer < SANITY_EER_FLOOR) and not by_design,
        "expected_by_design": by_design,
        "floor": SANITY_EER_FLOOR,
    }


def run_all(manifest: pd.DataFrame, split: str = "eval") -> pd.DataFrame:
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
