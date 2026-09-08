"""The main plot and the headline number.

One figure carries this thesis: perceptual quality on X, matched EER on Y, one
point per vocoder. Per the plan, it should exist in rough form by December --
that buffer is what protects the project if something breaks in February.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.manifest import primary_ladder_frame
from detectors.protocols import (
    Protocol,
    assert_protocols_not_pooled,
    orthogonality_verdict,
    spearman_headline,
)


def main_plot(
    quality_by_condition: pd.DataFrame,
    detection: pd.DataFrame,
    out_path: str | Path,
    *,
    quality_col: str = "utmos_mean",
    protocol: Protocol = Protocol.MATCHED,
    detector: str | None = None,
):
    """Quality vs detectability, with EER confidence intervals.

    The error bars are not decoration. With six points, whether the trend is
    real turns on whether the points are separated by more than their own
    sampling noise, so a plot without CIs cannot support either conclusion.
    """
    import matplotlib.pyplot as plt

    assert_protocols_not_pooled(detection)
    # INV-17: the headline plot is the primary ladder. The band-exempt control
    # is measured and reported, but never plotted as a rung -- its bandwidth
    # differs by design, so a point for it on this axis would read as a vocoder
    # property.
    detection = primary_ladder_frame(detection, why="the main plot")
    quality_by_condition = primary_ladder_frame(
        quality_by_condition, why="the main plot"
    )
    det = detection[detection["protocol"] == protocol.value]
    if detector is not None:
        det = det[det["detector"] == detector]
    df = quality_by_condition.merge(det, on="condition", how="inner")

    fig, ax = plt.subplots(figsize=(7, 5))
    yerr = None
    if {"eer_ci_low", "eer_ci_high"}.issubset(df.columns) and df["eer_ci_low"].notna().all():
        yerr = [df["eer"] - df["eer_ci_low"], df["eer_ci_high"] - df["eer"]]

    ax.errorbar(df[quality_col], df["eer"], yerr=yerr, fmt="o", capsize=4, markersize=8)
    for _, r in df.iterrows():
        ax.annotate(r["condition"], (r[quality_col], r["eer"]), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)

    ax.set_xlabel(f"perceptual quality ({quality_col})")
    tier_label = df["tier"].iloc[0] if "tier" in df.columns and len(df) else "?"
    ax.set_ylabel(f"EER ({protocol.value} protocol, {tier_label} tier)")
    ax.set_title("Does better-sounding mean harder-to-detect?")
    ax.grid(alpha=0.3)

    stats = spearman_headline(
        quality_by_condition, detection, quality_col=quality_col,
        protocol=protocol, detector=detector,
    )
    caption = (
        f"Spearman rho = {stats['rho']:+.3f}  "
        f"(p = {stats['p_value']:.3f}, n = {stats['n_conditions']})"
    )
    ax.text(0.02, 0.02, caption, transform=ax.transAxes, fontsize=9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    return fig, stats


def summarise(quality_by_condition: pd.DataFrame, detection: pd.DataFrame, **kwargs) -> str:
    """One-paragraph reading of the result, for the thesis draft."""
    detection = primary_ladder_frame(detection, why="the headline summary")
    quality_by_condition = primary_ladder_frame(
        quality_by_condition, why="the headline summary"
    )
    stats = spearman_headline(quality_by_condition, detection, **kwargs)
    return orthogonality_verdict(stats["rho"], stats["n_conditions"])


def protocol_comparison(detection: pd.DataFrame) -> pd.DataFrame:
    """Matched vs mismatched EER side by side, per condition.

    Presented as two labelled columns rather than one merged number. The gap
    between them is itself a finding -- but with two causes now, not one: the
    deployed detector's blind spot AND the fact that the mismatched protocol runs
    on the band-limited 16 kHz tier. The tier is kept in the index so the
    comparison cannot be read as like-for-like by accident.
    """
    assert_protocols_not_pooled(detection)
    return (
        detection.pivot_table(
            index=["detector", "condition", "tier"],
            columns="protocol",
            values="eer",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
