"""INV-18 — committed measurements, and the resolver that holds claims to them.

``data/measurements/`` holds evidence produced by measurement runs outside this
repository (Kaggle, on the real corpus). It is committed deliberately -- see the
negation rule in .gitignore -- because invariants cite it by row. A citation
that points at a scratch disk is an argument, not a measurement.

The contract this module enforces is narrow and strict: a
:class:`data.invariants.MeasurementRef` names an artifact, a row selector, a
metric and a statistic, and records the value that statistic had. :func:`verify`
recomputes that statistic from the committed file and raises if it disagrees.
So an exclusion cannot quote a number the evidence does not contain, and the
evidence cannot change underneath an exclusion without the test suite noticing.

This is the same failure mode the enforcement audit catalogued -- one behaviour
described in two places, drifting apart -- applied to measurements instead of
constants. The prose-generation blocker in CLAUDE.md is the general fix; this is
the specific one for evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from .invariants import (
    LOUDNESS_COUPLING_OPERATING_PROBE,
    PROBE_CHANCE_MAX_DEVIATION,
    PROBE_CONTROL_MIN_AUROC,
    InvariantViolation,
    MeasurementRef,
)

# src/data/measurements.py -> parents[2] is the repository root.
MEASUREMENTS_DIR = Path(__file__).resolve().parents[2] / "data" / "measurements"

BAND_MEASUREMENTS = "band_measurements.csv"
META = "meta.json"
PROBES = "probes.json"

# `btpvf silence-diag` output from the first 200-file run whose INV-12 gate
# raised. Committed as the record of two things at once: the refutation of the
# reconstruction-noise hypothesis, and the pooled-percentile defect in the floor
# estimator that the pad-agreement check caught (INV-12; docs/measurements.md).
SILENCE_CALIBRATION = "silence_calibration.csv"
SILENCE_FLOORS = "silence_floors.csv"
SILENCE_SWEEP = "silence_threshold_sweep.csv"


def artifact_path(name: str) -> Path:
    path = MEASUREMENTS_DIR / name
    if not path.is_file():
        raise InvariantViolation(
            f"INV-18: measurement artifact {name!r} not found at {path}. Every "
            "exclusion must cite committed evidence; restore the file from history "
            "rather than removing the citation."
        )
    return path


@cache
def load_csv(name: str = BAND_MEASUREMENTS) -> pd.DataFrame:
    return pd.read_csv(artifact_path(name))


@cache
def load_meta(name: str = META) -> dict:
    return json.loads(artifact_path(name).read_text(encoding="utf-8"))


@cache
def load_probes(name: str = PROBES) -> dict:
    return json.loads(artifact_path(name).read_text(encoding="utf-8"))


def _recorded(ref: MeasurementRef) -> float:
    """A value the measurement run computed and the JSON artifact stores.

    Unlike a CSV citation this is not RECOMPUTED -- probes.json holds per-fold
    summaries, not per-file features -- so verify() checks the citation against
    the file, not the file against the audio.
    """
    if ref.statistic != "recorded":
        raise InvariantViolation(
            f"INV-18: {ref.artifact} stores results, not rows; statistic must be "
            f"'recorded', got {ref.statistic!r}."
        )
    probes = load_probes(ref.artifact)["probes"]
    if ref.condition not in probes:
        raise InvariantViolation(
            f"INV-18: no probe {ref.condition!r} in {ref.artifact}; available {sorted(probes)}."
        )
    fields = probes[ref.condition]
    if ref.metric not in fields:
        raise InvariantViolation(
            f"INV-18: probe {ref.condition!r} records no {ref.metric!r}; "
            f"available {sorted(fields)}."
        )
    return float(fields[ref.metric])


def _series(df: pd.DataFrame, condition: str, metric: str) -> np.ndarray:
    rows = df[df["condition"] == condition]
    if rows.empty:
        raise InvariantViolation(
            f"INV-18: no rows for condition {condition!r}; available "
            f"{sorted(df['condition'].unique())}."
        )
    if "-" in metric:
        left, right = (m.strip() for m in metric.split("-", 1))
        for col in (left, right):
            if col not in rows.columns:
                raise InvariantViolation(f"INV-18: unknown column {col!r}.")
        return (rows[left] - rows[right]).to_numpy(dtype=np.float64)
    if metric not in rows.columns:
        raise InvariantViolation(f"INV-18: unknown column {metric!r}.")
    return rows[metric].to_numpy(dtype=np.float64)


def resolve(ref: MeasurementRef) -> float:
    """Recompute ``ref``'s statistic from the committed artifact."""
    if ref.artifact.endswith(".json"):
        return _recorded(ref)
    df = load_csv(ref.artifact)
    values = _series(df, ref.condition, ref.metric)

    stat = ref.statistic
    if stat in ("mean", "median", "min", "max"):
        return float(getattr(np, stat)(values))
    if stat.startswith("auroc_vs:"):
        from sklearn.metrics import roc_auc_score

        other = _series(df, stat.split(":", 1)[1], ref.metric)
        labels = np.r_[np.zeros(len(values)), np.ones(len(other))]
        auc = roc_auc_score(labels, np.r_[values, other])
        # Orientation-free: which side is larger is a property of the feature,
        # not of how well it separates.
        return float(max(auc, 1.0 - auc))
    raise InvariantViolation(f"INV-18: unsupported statistic {stat!r}.")


def verify(ref: MeasurementRef) -> float:
    """Assert the recorded value is what the committed evidence says. Returns it."""
    actual = resolve(ref)
    if not np.isclose(actual, ref.value, rtol=ref.rel_tol, atol=0.0):
        raise InvariantViolation(
            f"INV-18: {ref.describe()} records {ref.value:.6g} but the committed "
            f"artifact gives {actual:.6g}. Either the evidence changed or the "
            "citation was wrong -- fix whichever it is, do not widen the tolerance."
        )
    return actual


# --- Paired probes: a below-band reading never travels without its control ---


@dataclass(frozen=True)
class ProbeReading:
    """One committed paired probe, read the only way it may be read.

    ``below_at_chance`` is meaningful ONLY because ``control_auroc`` shows the
    same features, labels, folds and classifier separating the same pair above
    the band. :func:`probe_reading` refuses to construct a reading whose control
    failed, so there is no object in this codebase that carries a below-band
    "at chance" without a passing control attached.
    """

    name: str
    auroc_below: float
    auroc_below_std: float
    control_auroc: float        # auroc_above: the positive control
    control_auroc_std: float
    ratio_highband: float
    loudness_normalised: bool = False
    mean_gain_delta_db: float | None = None   # recorded for normalised probes only

    @property
    def below_deviation(self) -> float:
        return abs(self.auroc_below - 0.5) + 2.0 * self.auroc_below_std

    @property
    def below_at_chance(self) -> bool:
        return self.below_deviation <= PROBE_CHANCE_MAX_DEVIATION

    @property
    def below_is_degenerate(self) -> bool:
        """Exactly 0.5 with zero fold spread: a pair-constant decision function.

        Not noisy chance. See docs/measurements.md, "Why 0.5000 is not a bug".
        """
        return self.auroc_below == 0.5 and self.auroc_below_std == 0.0


def probe_reading(name: str, *, artifact: str = PROBES) -> ProbeReading:
    """INV-18. Read committed probe ``name``; raise if its control is not separable."""
    fields = load_probes(artifact)["probes"].get(name)
    if fields is None:
        raise InvariantViolation(
            f"INV-18: no probe {name!r} in {artifact}; "
            f"available {sorted(load_probes(artifact)['probes'])}."
        )
    reading = ProbeReading(
        name=name,
        auroc_below=float(fields["auroc_below"]),
        auroc_below_std=float(fields["auroc_below_std"]),
        control_auroc=float(fields["auroc_above"]),
        control_auroc_std=float(fields["auroc_above_std"]),
        ratio_highband=float(fields["ratio_highband"]),
        loudness_normalised=bool(fields.get("loudness_normalised", False)),
        mean_gain_delta_db=(
            float(fields["mean_gain_delta_db"]) if "mean_gain_delta_db" in fields else None
        ),
    )
    if reading.loudness_normalised and reading.mean_gain_delta_db is None:
        raise InvariantViolation(
            f"INV-18: probe {name!r} is loudness-normalised but records no "
            "mean_gain_delta_db; the coupling cannot be read without it."
        )
    if reading.control_auroc < PROBE_CONTROL_MIN_AUROC:
        raise InvariantViolation(
            f"INV-18: probe {name!r} reads {reading.auroc_below:.4f} below LADDER_FMAX, "
            f"but its above-band control is only {reading.control_auroc:.4f} "
            f"(< {PROBE_CONTROL_MIN_AUROC}). Without a separable control a below-band "
            "0.5 cannot be told apart from a broken pipeline; do not read it."
        )
    return reading


# --- Loudness coupling: a bracket and an operating margin ---------------------


@dataclass(frozen=True)
class LoudnessCoupling:
    """What the committed normalised probes say about gain-induced leakage.

    Not a gate. ``highest_null_db`` is the largest gain delta at which the
    below-band probe stayed at chance; ``lowest_detected_db`` the smallest at
    which it did not. The onset lies between them and is not located more
    finely. ``margin`` is how far below the highest null the pinned resampler
    sits.
    """

    highest_null_db: float
    highest_null_probe: str
    lowest_detected_db: float
    lowest_detected_probe: str
    operating_db: float
    operating_probe: str

    @property
    def margin(self) -> float:
        return self.highest_null_db / self.operating_db

    @property
    def margin_to_detected(self) -> float:
        return self.lowest_detected_db / self.operating_db


def loudness_coupling(*, artifact: str = PROBES) -> LoudnessCoupling:
    """INV-18. Derive the coupling bracket from every normalised probe on file.

    Derived rather than hard-coded, so a re-run that adds a probe moves the
    bracket instead of leaving a stale constant. Raises if the evidence is not
    monotone -- a null above a detection would mean gain delta is not what
    drives the below-band leak, and the bracket would be meaningless.
    """
    readings = [
        probe_reading(name, artifact=artifact) for name in load_probes(artifact)["probes"]
    ]
    normalised = [r for r in readings if r.loudness_normalised]
    nulls = [r for r in normalised if r.below_at_chance]
    hits = [r for r in normalised if not r.below_at_chance]
    if not nulls or not hits:
        raise InvariantViolation(
            "INV-18: the loudness coupling needs at least one normalised probe at "
            "chance below the band and one that is not."
        )
    null = max(nulls, key=lambda r: r.mean_gain_delta_db)
    hit = min(hits, key=lambda r: r.mean_gain_delta_db)
    if null.mean_gain_delta_db >= hit.mean_gain_delta_db:
        raise InvariantViolation(
            f"INV-18: {null.name} is at chance at a gain delta of "
            f"{null.mean_gain_delta_db:.3g} dB but {hit.name} leaks at "
            f"{hit.mean_gain_delta_db:.3g} dB. Not monotone in gain delta."
        )
    op = probe_reading(LOUDNESS_COUPLING_OPERATING_PROBE, artifact=artifact)
    if not op.loudness_normalised or not op.below_at_chance:
        raise InvariantViolation(
            f"INV-18: the operating point {op.name} must be a normalised probe at "
            "chance below the band."
        )
    return LoudnessCoupling(
        highest_null_db=null.mean_gain_delta_db,
        highest_null_probe=null.name,
        lowest_detected_db=hit.mean_gain_delta_db,
        lowest_detected_probe=hit.name,
        operating_db=op.mean_gain_delta_db,
        operating_probe=op.name,
    )
