"""Batch quality evaluation over a manifest. Person B's entry point.

The X-axis straddles both tiers (INV-01), and it has to:

  ARCHIVE (22.05 kHz)   MCD, band-wise LSD, F0 error, and the high-band
                        distance. These measure the high band, which is the
                        whole mechanism argument, so they must not see the
                        derived set -- and since INV-17 moved band-limiting to
                        analysis time, the archive still HAS that band to
                        measure.
  ZEROSHOT (16 kHz)     UTMOS and PESQ-WB. Both are hard-locked to 16 kHz by
                        their own definitions -- UTMOS22 is a 16 kHz model and
                        PESQ-WB accepts nothing else.

Neither group resamples anything. Each reads the tier it needs, straight from
Phase A's output. A metric that fixed up its own input would be hiding a Phase A
bug, and would put a second filtering pass on the signal.

Results are joined on (utt_id, condition) into one per-utterance row, so the
correlation analysis downstream sees a single table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.audio_io import read_processed
from data.invariants import ARCHIVE_SR, ARCHIVE_TIER, REAL_CONDITION, ZEROSHOT_SR, ZEROSHOT_TIER
from data.manifest import require_primary, select_tier

from .f0 import f0_error
from .mcd import mcd
from .spectral import high_band_distance, log_spectral_distance_by_band
from .utmos import UTMOSScorer


def evaluate_archive(df: pd.DataFrame, *, with_f0: bool = True) -> pd.DataFrame:
    """Intrusive and band-wise metrics, on the primary artifact only."""
    archive = require_primary(select_tier(df, ARCHIVE_TIER), why="band-wise quality metrics")
    real = archive[archive["condition"] == REAL_CONDITION].set_index("utt_id")["path"].to_dict()

    rows: list[dict[str, object]] = []
    for row in tqdm(archive.itertuples(index=False), total=len(archive), desc="quality/archive"):
        if row.condition == REAL_CONDITION:
            continue
        wav = read_processed(row.path, ARCHIVE_SR)
        ref = read_processed(real[row.utt_id], ARCHIVE_SR)
        rec: dict[str, object] = {
            "utt_id": row.utt_id,
            "condition": row.condition,
            "split": row.split,
            "mcd": mcd(ref, wav, sr=ARCHIVE_SR),
        }
        rec.update(log_spectral_distance_by_band(ref, wav, sr=ARCHIVE_SR))
        # INV-17: the band the vocoder was never told about. Reported alongside
        # the energy fraction because the right amount of energy with the wrong
        # structure is a strong detection cue an energy measure cannot see.
        rec.update(high_band_distance(ref, wav, sr=ARCHIVE_SR))
        if with_f0:
            rec.update(f0_error(ref, wav, sr=ARCHIVE_SR))
        rows.append(rec)

    return pd.DataFrame(rows)


def evaluate_zeroshot(
    df: pd.DataFrame, *, with_utmos: bool = True, with_pesq: bool = True, device: str = "cpu"
) -> pd.DataFrame:
    """UTMOS and PESQ-WB, on the derived 16 kHz artifact.

    Both are blind above 8 kHz. That is a property of the metrics, not a choice,
    and it is recorded in each module's docstring for the limitations section.
    """
    zeroshot = select_tier(df, ZEROSHOT_TIER)
    real = zeroshot[zeroshot["condition"] == REAL_CONDITION].set_index("utt_id")["path"].to_dict()
    scorer = UTMOSScorer(device=device) if with_utmos else None

    rows: list[dict[str, object]] = []
    for row in tqdm(zeroshot.itertuples(index=False), total=len(zeroshot), desc="quality/zeroshot"):
        wav = read_processed(row.path, ZEROSHOT_SR)
        rec: dict[str, object] = {
            "utt_id": row.utt_id,
            "condition": row.condition,
            "split": row.split,
        }
        if scorer is not None:
            rec["utmos"] = scorer.score(wav, ZEROSHOT_SR)
        if with_pesq and row.condition != REAL_CONDITION:
            from .pesq_metric import pesq_wb

            ref = read_processed(real[row.utt_id], ZEROSHOT_SR)
            rec["pesq_wb"] = pesq_wb(ref, wav, sr=ZEROSHOT_SR)
        rows.append(rec)

    return pd.DataFrame(rows)


def evaluate_manifest(
    df: pd.DataFrame,
    *,
    with_utmos: bool = True,
    with_pesq: bool = True,
    with_f0: bool = True,
    device: str = "cpu",
) -> pd.DataFrame:
    """Both tiers, joined into one per-utterance table.

    An outer join on (utt_id, condition) rather than an inner one: a missing
    tier should surface as visible NaNs in the output, not as silently dropped
    utterances that would break the equal-n property the pairing invariant
    guarantees.
    """
    archive = evaluate_archive(df, with_f0=with_f0)
    zeroshot = evaluate_zeroshot(
        df, with_utmos=with_utmos, with_pesq=with_pesq, device=device
    )
    if archive.empty:
        return zeroshot
    if zeroshot.empty:
        return archive
    return archive.merge(zeroshot, on=["utt_id", "condition", "split"], how="outer")


def aggregate(per_utt: pd.DataFrame) -> pd.DataFrame:
    """Per-condition mean, std and n. The n column is not decorative -- it is
    how a reader confirms every condition was scored on the same utterance set.
    """
    numeric = per_utt.select_dtypes(include=[np.number]).columns.tolist()
    agg = per_utt.groupby("condition")[numeric].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def save(per_utt: pd.DataFrame, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / "quality_per_utterance.csv"
    p2 = out_dir / "quality_by_condition.csv"
    per_utt.to_csv(p1, index=False)
    aggregate(per_utt).to_csv(p2, index=False)
    return p1, p2
