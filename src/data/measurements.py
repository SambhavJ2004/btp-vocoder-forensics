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
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from .invariants import InvariantViolation, MeasurementRef

# src/data/measurements.py -> parents[2] is the repository root.
MEASUREMENTS_DIR = Path(__file__).resolve().parents[2] / "data" / "measurements"

BAND_MEASUREMENTS = "band_measurements.csv"
META = "meta.json"


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
