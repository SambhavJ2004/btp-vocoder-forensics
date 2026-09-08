"""Phase B, Y-axis driver — both detection protocols. Person C.

The two protocols are separate entry points on purpose. There is no combined
"run detection" function, because there is no combined quantity: mismatched
answers "would a deployed detector catch this?", matched answers "is the
artifact information there at all?".

Both default to scoring ALL_CONDITIONS, the band-exempt control included: the
paired bandwidth contrast needs its EER, so it has to be measured. Filtering
happens later, at analysis time -- `spearman_headline` refuses an exempt
condition outright, and `analysis.main_plot` drops it via
`primary_ladder_frame` before correlating (INV-17).

They also default to different tiers (INV-01), and that is not incidental:

  mismatched -> ZEROSHOT (16 kHz)   the pretrained LA weights expect it
  matched    -> ARCHIVE  (22.05 kHz) trained from scratch, so it can see the
                                     high band the mechanism argument concerns

Which means the mismatched EER is measured on strictly less signal than the
matched EER. That is the honest arrangement -- a deployed detector really would
only get 16 kHz -- but the gap between the two protocols now has two causes
(blind spot AND bandwidth), and the thesis must say so rather than attributing
all of it to generalisation failure.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.invariants import ARCHIVE_TIER, ZEROSHOT_TIER
from data.manifest import read_manifest, validate_manifest
from detectors.protocols import Protocol, results_to_frame, score_condition
from vocoders.registry import ALL_CONDITIONS


def run_mismatched(
    manifest_path: str | Path,
    detector,
    out_path: str | Path,
    *,
    conditions: tuple[str, ...] = ALL_CONDITIONS,
    checkpoint: str = "",
    tier: str = ZEROSHOT_TIER,
) -> pd.DataFrame:
    """Zero-shot: an ASVspoof-2019-LA-trained detector, no adaptation.

    ``checkpoint`` must be an external, pretrained one. Passing a checkpoint that
    saw any of our audio makes this a matched result wearing the wrong label --
    DetectionResult refuses that combination.

    Defaults to the derived 16 kHz tier because that is what the released weights
    consume. Feeding them 22.05 kHz would be an unannounced domain shift on top
    of the one the protocol is meant to measure.
    """
    df = read_manifest(manifest_path)
    validate_manifest(df, require_zeroshot=(tier == ZEROSHOT_TIER))
    detector.load(checkpoint or None)

    results = [
        score_condition(
            detector, df, cond, protocol=Protocol.MISMATCHED, tier=tier,
            checkpoint=checkpoint,
            notes="zero-shot; no adaptation to project audio",
        )
        for cond in conditions
        if cond in set(df["condition"])
    ]
    out = results_to_frame(results)
    _write(out, out_path)
    return out


def run_matched(
    manifest_path: str | Path,
    detector_factory,
    out_path: str | Path,
    *,
    conditions: tuple[str, ...] = ALL_CONDITIONS,
    tier: str = ARCHIVE_TIER,
) -> pd.DataFrame:
    """One detector trained per condition: real vs vocoder-X.

    A fresh detector per condition, trained on the frozen splits (INV-09). One
    detector trained on all conditions at once would measure something else
    entirely -- a multi-class problem reported as a binary EER.

    Defaults to the archive tier: this protocol is the upper bound on how much
    artifact information the signal contains, and on the derived set that bound
    would be capped at whatever survives an 8 kHz low-pass.
    """
    df = read_manifest(manifest_path)
    validate_manifest(df)
    tier_df = df[df["tier"] == tier]

    results = []
    for cond in conditions:
        if cond not in set(df["condition"]):
            continue
        det = detector_factory()
        det.fit(
            tier_df[(tier_df["split"] == "train") & (tier_df["condition"].isin(["real", cond]))],
            tier_df[(tier_df["split"] == "dev") & (tier_df["condition"].isin(["real", cond]))],
        )
        results.append(
            score_condition(
                det, df, cond, protocol=Protocol.MATCHED, tier=tier,
                train_conditions=("real", cond),
            )
        )

    out = results_to_frame(results)
    _write(out, out_path)
    return out


def run_bandlimit_ablation(*_args, **_kwargs):
    """Sweep low-pass cutoffs, retraining at each one. ARCHIVE tier only.

    Retraining at every cutoff is the expensive part and the non-negotiable
    part: evaluating a full-band detector on filtered audio measures domain
    shift, not where the artifact information lives. Budget the GPU hours
    accordingly (len(cutoffs) x len(conditions) matched trainings).

    The archive rate and the full-band archive are what make this meaningful. At
    the old 16 kHz delivery rate there was no content above 8 kHz to remove; and
    while INV-17 band-limited at generation, there was none above LADDER_FMAX
    either. Now the archive carries the whole band, so cutoffs sit on both sides
    of LADDER_FMAX -- the boundary between what the vocoder was told and what it
    invented.
    """
    raise NotImplementedError(
        "Implement once run_matched works end to end. Use "
        "data.preprocess.band_limit_comparison_set, which applies the identical "
        "filter to real and every vocoded member at once (INV-17), over "
        "detectors.bandlimit.cutoffs_for_rate(ARCHIVE_SR). Record band_limit_hz "
        "and tier on every DetectionResult. Read audio from written PCM_16 files "
        "-- never from Phase A arrays (INV-17)."
    )


def _write(df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
