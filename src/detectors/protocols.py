"""The two detection protocols, kept structurally separate.

They measure different quantities and conflating them is a recurring weakness in
the literature. Separating them cleanly is a stated contribution of this work,
so the separation is enforced in code rather than left to discipline:

  MISMATCHED (zero-shot)
      A detector trained on ASVspoof 2019 LA, evaluated on our conditions with
      NO adaptation. Measures generalisation failure of deployed systems.
      Practical relevance. Fine-tuning here silently converts it into a matched
      result, so the runner refuses a checkpoint that was trained in-project.

  MATCHED
      A detector trained specifically on real-vs-vocoder-X. Measures how much
      artifact information the signal contains at all, independent of any single
      detector's blind spots. Upper bound on separability. Scientific relevance.

They also run on different TIERS (INV-01), which the result must record:
mismatched on the derived 16 kHz set (what the pretrained LA weights expect),
matched and the band-limited ablation on the 22.05 kHz archive (where the high
band the mechanism argument concerns actually exists).

INV-14: no number leaves this module without its protocol AND tier label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from data.invariants import (
    ARCHIVE_TIER,
    BAND_EXEMPT_CONDITIONS,
    REAL_CONDITION,
    TIER_RATES,
    TIERS,
    ZEROSHOT_TIER,
    InvariantViolation,
    is_band_exempt,
)

from .eer import bootstrap_eer_ci, compute_eer


def _specs():
    from vocoders.registry import SPECS

    return SPECS


class Protocol(str, Enum):
    MISMATCHED = "mismatched"
    MATCHED = "matched"


@dataclass
class DetectionResult:
    """One EER, and everything needed to interpret it."""

    protocol: Protocol
    detector: str
    condition: str
    eer: float
    threshold: float
    n_bonafide: int
    n_spoof: int
    tier: str = ZEROSHOT_TIER
    eer_ci_low: float | None = None
    eer_ci_high: float | None = None
    train_conditions: tuple[str, ...] = ()
    checkpoint: str = ""
    band_limit_hz: float | None = None  # set by the band-limited ablation
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.protocol is Protocol.MISMATCHED and self.train_conditions:
            raise InvariantViolation(
                "INV-14: a mismatched result cannot have in-project train conditions. "
                "If the detector saw our audio, it is a matched result -- label it so."
            )
        if self.protocol is Protocol.MATCHED and not self.train_conditions:
            raise InvariantViolation(
                "INV-14: a matched result must record what it was trained on."
            )
        if self.tier not in TIERS:
            raise InvariantViolation(
                f"INV-01: unknown tier '{self.tier}'. Every EER records which "
                f"artifact it was measured on. Known: {list(TIERS)}"
            )
        if self.band_limit_hz is not None:
            nyq = TIER_RATES[self.tier] / 2.0
            if self.band_limit_hz >= nyq:
                raise InvariantViolation(
                    f"INV-01: band limit {self.band_limit_hz} Hz is at or above "
                    f"Nyquist ({nyq}) for tier '{self.tier}'; it removes nothing and "
                    "would duplicate the full-band point under a different label."
                )
            if self.tier != ARCHIVE_TIER:
                raise InvariantViolation(
                    "INV-01: the band-limited ablation runs on the archive tier. On "
                    "the derived 16 kHz set everything above 8 kHz is already gone, "
                    "so the sweep would be measuring the downsampler."
                )

    def to_row(self) -> dict[str, object]:
        d = self.__dict__.copy()
        d.pop("extra")
        d["protocol"] = self.protocol.value
        d["train_conditions"] = "|".join(self.train_conditions)
        return d


def score_condition(
    detector,
    manifest: pd.DataFrame,
    condition: str,
    *,
    protocol: Protocol,
    split: str = "eval",
    tier: str = ZEROSHOT_TIER,
    with_ci: bool = True,
    **result_kwargs,
) -> DetectionResult:
    """Score real-vs-one-condition on one tier and return a labelled result.

    Real utterances come from the same split and the same tier, so the bona fide
    and spoof sets are the same content at the same rate -- which is the whole
    point of paired resynthesis: the detector cannot win on what was said, only
    on how it was rendered.
    """
    from data.audio_io import read_processed
    from data.manifest import select_tier

    expected_sr = TIER_RATES[tier]
    sub = select_tier(manifest, tier)
    sub = sub[sub["split"] == split]
    bona_paths = sub[sub["condition"] == REAL_CONDITION]["path"].tolist()
    spoof_paths = sub[sub["condition"] == condition]["path"].tolist()
    if len(bona_paths) != len(spoof_paths):
        raise InvariantViolation(
            f"INV-06: {len(bona_paths)} real vs {len(spoof_paths)} '{condition}' in "
            f"split '{split}'. Conditions must be paired."
        )

    bona = detector.score_batch([read_processed(p, expected_sr) for p in bona_paths])
    spoof = detector.score_batch([read_processed(p, expected_sr) for p in spoof_paths])
    eer, thr = compute_eer(bona, spoof)

    lo = hi = None
    if with_ci:
        lo, hi = bootstrap_eer_ci(bona, spoof)

    return DetectionResult(
        protocol=protocol,
        detector=detector.spec.key,
        condition=condition,
        tier=tier,
        eer=eer,
        threshold=thr,
        n_bonafide=len(bona),
        n_spoof=len(spoof),
        eer_ci_low=lo,
        eer_ci_high=hi,
        **result_kwargs,
    )


def results_to_frame(results: list[DetectionResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.to_row() for r in results])
    assert_protocols_not_pooled(df)
    return df


def assert_protocols_not_pooled(df: pd.DataFrame) -> None:
    """Guard against averaging matched and mismatched EERs into one number."""
    if "protocol" not in df.columns:
        raise InvariantViolation(
            "INV-14: results frame has no protocol column. Matched and mismatched "
            "EERs answer different questions and are never pooled or plotted "
            "together without the distinction being visible."
        )
    if "tier" not in df.columns:
        raise InvariantViolation(
            "INV-01/INV-14: results frame has no tier column. A 16 kHz zero-shot EER "
            "and a 22.05 kHz matched EER were measured on different audio; the "
            "distinction has to survive into the results table."
        )


def spearman_headline(
    quality_by_condition: pd.DataFrame,
    detection: pd.DataFrame,
    *,
    quality_col: str = "utmos_mean",
    protocol: Protocol = Protocol.MATCHED,
    detector: str | None = None,
) -> dict[str, float]:
    """The headline analysis: Spearman rank correlation, quality vs EER.

    Computed on the MATCHED protocol by default, because that is the one that
    measures the signal rather than a particular deployed detector's blind spots.

    With six conditions, Spearman has very little power: rho is reportable, the
    p-value is close to meaningless. Report both, and let the confidence
    intervals on the individual EERs carry the argument about whether the points
    are even distinguishable.
    """
    from scipy.stats import spearmanr

    assert_protocols_not_pooled(detection)
    det = detection[detection["protocol"] == protocol.value]
    if detector is not None:
        det = det[det["detector"] == detector]
    if det["detector"].nunique() > 1:
        raise InvariantViolation(
            "Correlate within one detector at a time; pooling architectures mixes "
            "two sources of variance into one rho."
        )
    if det["tier"].nunique() > 1:
        raise InvariantViolation(
            f"INV-01: correlating across tiers {sorted(set(det['tier']))}. EERs from "
            "different rates were measured on different audio and are not one series."
        )

    merged = quality_by_condition.merge(det, on="condition", how="inner")

    # INV-17. The primary correlation runs over the primary ladder only. Band
    # exempt conditions keep their full native band by design, so a rho computed
    # across them would be reading the exemption as a vocoder property -- which
    # is exactly the confound the ladder band exists to remove. This refuses
    # rather than silently filtering: a caller that passed an exempt condition
    # has a different question in mind and should say so.
    smuggled = sorted(set(merged["condition"]) & set(BAND_EXEMPT_CONDITIONS))
    if smuggled:
        raise InvariantViolation(
            f"INV-17: {smuggled} are band-exempt and cannot enter the primary "
            "correlation. They keep their full native band, so including them "
            "puts a bandwidth cliff back into the headline number. For the "
            "bandwidth-vs-architecture comparison use "
            "detectors.protocols.paired_bandwidth_contrast instead, or filter "
            "with data.manifest.primary_ladder_frame first."
        )

    if len(merged) < 3:
        raise InvariantViolation(f"Need >=3 conditions to correlate, got {len(merged)}.")

    rho, p = spearmanr(merged[quality_col], merged["eer"])
    return {
        "rho": float(rho),
        "p_value": float(p),
        "n_conditions": int(len(merged)),
        "quality_metric": quality_col,
        "protocol": protocol.value,
        "tier": det["tier"].iloc[0] if len(det) else "",
        "detector": det["detector"].iloc[0] if len(det) else "",
    }


# The bandwidth pairs, and what each one actually isolates. They are NOT two
# instances of the same comparison and must not be averaged.
BANDWIDTH_PAIRS: dict[str, tuple[str, str]] = {
    # Two separately trained checkpoints whose mel front-ends differ in fmax.
    # Isolates what a generator TRAINED to produce the high band does
    # differently from one that was not.
    "bigvgan": ("bigvgan_112m", "bigvgan_v2_22khz_fullband"),
    # One checkpoint, delivered with and without the ladder band. Isolates only
    # what the delivery filter removes -- the band-limited ablation at a single
    # cutoff, on a second architecture.
    "melgan": ("melgan", "melgan_fullband"),
}


def paired_bandwidth_contrast(
    detection: pd.DataFrame,
    *,
    pair: str | None = None,
    band_limited: str = "bigvgan_112m",
    full_band: str = "bigvgan_v2_22khz_fullband",
    protocol: Protocol = Protocol.MATCHED,
) -> dict[str, float]:
    """The comparison the primary correlation is not allowed to make.

    Pass ``pair`` to select from :data:`BANDWIDTH_PAIRS`, or name the two
    conditions directly.

    **The two pairs answer different questions.** ``bigvgan_112m`` and
    ``bigvgan_v2_22khz_fullband`` are separately trained checkpoints -- same
    architecture, same parameter count, same recipe -- differing in mel fmax
    alone, so their EER difference isolates what training bandwidth does to the
    artifact. ``melgan`` and ``melgan_fullband`` share one set of weights and
    differ only in whether the delivery filter ran, so that difference isolates
    what the filter removes: the band-limited ablation evaluated at one cutoff,
    on a second architecture.

    The second is the weaker of the two as a mechanism claim, and its value is
    breadth: without it the high-band argument rests on a single model. Report
    them side by side, never pooled.

    The band-limited ablation sweeps a filter over a fixed generator; the
    BigVGAN pair varies the generator's own band with the filter held fixed.
    They fail differently, which is why both are worth running.
    """
    if pair is not None:
        if pair not in BANDWIDTH_PAIRS:
            raise InvariantViolation(
                f"unknown bandwidth pair '{pair}'. Known: {sorted(BANDWIDTH_PAIRS)}"
            )
        band_limited, full_band = BANDWIDTH_PAIRS[pair]
    if not is_band_exempt(full_band):
        raise InvariantViolation(
            f"INV-17: '{full_band}' is not band-exempt, so this is not a bandwidth "
            "contrast -- both sides would be at the ladder band."
        )
    assert_protocols_not_pooled(detection)
    det = detection[detection["protocol"] == protocol.value]

    rows = {}
    for name in (band_limited, full_band):
        sub = det[det["condition"] == name]
        if sub.empty:
            raise InvariantViolation(f"INV-17: no {protocol.value} result for '{name}'.")
        if sub["tier"].nunique() > 1:
            raise InvariantViolation(
                f"INV-01: '{name}' has results on multiple tiers; pick one."
            )
        rows[name] = sub.iloc[0]

    if rows[band_limited]["tier"] != rows[full_band]["tier"]:
        raise InvariantViolation(
            "INV-01: the bandwidth pair must be compared on one tier. Measured on "
            f"'{rows[band_limited]['tier']}' and '{rows[full_band]['tier']}'."
        )

    specs = _specs()
    shared_weights = (
        specs[band_limited].checkpoint == specs[full_band].checkpoint
        if band_limited in specs and full_band in specs
        else None
    )
    return {
        "band_limited": band_limited,
        "full_band": full_band,
        "eer_band_limited": float(rows[band_limited]["eer"]),
        "eer_full_band": float(rows[full_band]["eer"]),
        "delta_eer": float(rows[full_band]["eer"] - rows[band_limited]["eer"]),
        "tier": str(rows[band_limited]["tier"]),
        "protocol": protocol.value,
        # True: the pair shares weights, so the contrast isolates the delivery
        # filter only. False: separately trained checkpoints, so it isolates
        # training bandwidth. The reader needs this to interpret delta_eer.
        "shared_weights": shared_weights,
    }


def orthogonality_verdict(rho: float, n_conditions: int) -> str:
    """Plain-language reading of the headline number, per the plan's outcome table."""
    strength = "strong" if abs(rho) >= 0.7 else "moderate" if abs(rho) >= 0.4 else "weak/absent"
    if strength == "strong" and rho > 0:
        reading = (
            "Quality and detectability are coupled: the field's working assumption "
            "holds, and the exchange rate is quantified."
        )
    elif strength == "weak/absent":
        reading = (
            "Quality and detectability look orthogonal: perceptual optimisation and "
            "forensic evasion are independent objectives, so an undetectable vocoder "
            "need not be a high-quality one."
        )
    else:
        reading = "Ambiguous at this sample size; lean on the per-condition CIs."
    return (
        f"rho={rho:+.3f} over {n_conditions} conditions ({strength}). {reading} "
        "Note that n=6 makes any p-value here close to uninformative."
    )


__all__ = [
    "BANDWIDTH_PAIRS",
    "DetectionResult",
    "Protocol",
    "paired_bandwidth_contrast",
    "assert_protocols_not_pooled",
    "orthogonality_verdict",
    "results_to_frame",
    "score_condition",
    "spearman_headline",
]
