"""Phase C driver — source tracing. Stretch goal.

Closed-set N-way, then leave-one-vocoder-out open-set, then the family-level
clustering question. Scope confirmation (committed deliverable vs stretch) is
one of the open questions for the supervisor.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.manifest import read_manifest, validate_manifest
from detectors.attribution import family_labels, leave_one_vocoder_out


def run_closed_set(manifest_path: str | Path, out_path: str | Path) -> pd.DataFrame:
    """N-way classification over the known vocoder set. Baseline, not a result."""
    df = read_manifest(manifest_path)
    validate_manifest(df)
    raise NotImplementedError(
        "Train an N-way head on the AASIST embedding. Report per-class accuracy and "
        "the confusion matrix -- the confusions are the interesting part, since they "
        "show which vocoders share a signature."
    )


def run_open_set(manifest_path: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Leave-one-vocoder-out with score-based rejection.

    The held-out vocoder is excluded from training AND from threshold
    calibration. Calibrating on the held-out class leaks the answer and is the
    standard way this experiment is quietly invalidated.
    """
    df = read_manifest(manifest_path)
    validate_manifest(df)
    folds = leave_one_vocoder_out()
    raise NotImplementedError(
        f"{len(folds)} folds to run. Use detectors.attribution.energy_score for "
        "rejection and report AUROC plus FPR@95TPR alongside closed-set accuracy on "
        "the accepted subset."
    )


def run_family_level(manifest_path: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Do GAN vocoders cluster distinctly from non-GAN approaches?

    If they do, family-level attribution may generalise to unseen family members
    even where instance-level attribution fails -- which would be the more
    practically useful result of the two.
    """
    labels = family_labels()
    raise NotImplementedError(
        f"Cluster embeddings and score against family labels {labels}. Report "
        "silhouette by family plus a 2-D projection for the thesis figure."
    )


def _write(df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
