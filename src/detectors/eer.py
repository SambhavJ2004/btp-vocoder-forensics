"""Equal Error Rate and friends.

EER rather than accuracy: it is threshold-free and invariant to the fake-to-real
ratio in the evaluation set, so a number from one condition is comparable with a
number from another even if the sets differ in size. Accuracy is neither.

Convention used throughout: higher score == more bona fide (real). Flip the sign
of the model output if it is trained the other way round; a sign error shows up
as an EER above 0.5 and is the first thing to check when one appears.
"""

from __future__ import annotations

import numpy as np


def _score_vector(scores: np.ndarray, name: str) -> np.ndarray:
    """Validate one class without silently flattening a batch/class dimension."""
    try:
        values = np.asarray(scores)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric score vector.") from exc
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {values.shape}.")
    if values.size == 0:
        raise ValueError("EER needs both bona fide and spoof scores.")
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real scores, not complex values.")
    try:
        values = values.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric scores.") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite scores.")
    return values


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> tuple[float, float]:
    """Return directional (eer, threshold), with higher scores meaning real.

    Inputs must be nonempty, finite, real one-dimensional vectors; the two
    classes may have different sizes. EER is a fraction in [0, 1].

    A deterministic decision accepts ``score > threshold``. Equal scores move
    together, producing only attainable ROC points. EER is the FRR == FAR
    crossing of the piecewise-linear curve between those points. Interpolation
    represents randomized threshold decisions, not a deterministic split of a
    tied group, and need not be attainable at any single threshold.

    The returned threshold is the score boundary at the upper end of the
    crossing segment (or the exact crossing point). It identifies the transition;
    applying it alone need not reproduce an interpolated EER. No score-direction
    optimization is performed: perfectly reversed scores have EER 1, not 0.
    """
    bona = _score_vector(bonafide_scores, "bonafide_scores")
    spoof = _score_vector(spoof_scores, "spoof_scores")

    scores = np.concatenate([bona, spoof])
    labels = np.concatenate([np.ones(bona.size, dtype=int), np.zeros(spoof.size, dtype=int)])
    order = np.argsort(scores, kind="mergesort")
    scores, labels = scores[order], labels[order]

    # Reject whole groups as the threshold rises. Comparing adjacent scores,
    # rather than subtracting them, also avoids overflow at float64 extremes.
    group_ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
    frr = np.cumsum(labels)[group_ends] / bona.size
    # Count remaining spoofs directly: 1 - rejected/total can round an exact
    # FRR == FAR point (e.g. 1/3) into an apparent interpolation segment.
    far = (spoof.size - np.cumsum(1 - labels)[group_ends]) / spoof.size
    frr = np.concatenate([[0.0], frr])
    far = np.concatenate([[1.0], far])
    thresholds = np.concatenate([[-np.inf], scores[group_ends]])

    # FRR - FAR increases strictly from -1 to +1, so the crossing is unique.
    difference = frr - far
    right = int(np.searchsorted(difference, 0.0))
    if difference[right] == 0.0:
        return float(frr[right]), float(thresholds[right])
    left = right - 1
    weight = -difference[left] / (difference[right] - difference[left])
    eer = frr[left] + weight * (frr[right] - frr[left])
    return float(eer), float(thresholds[right])


def compute_min_tdcf(*_args, **_kwargs) -> float:
    """t-DCF placeholder.

    t-DCF weights error types by their downstream cost to a speaker verification
    system, so it needs ASV scores this study does not produce. Report it only if
    an ASV system is added; until then, EER is the honest headline and the
    absence of t-DCF is a scope statement, not an omission to hide.
    """
    raise NotImplementedError(
        "t-DCF requires paired ASV scores (see the ASVspoof 2019/2021 evaluation "
        "package). Out of scope unless an ASV system is added to the pipeline."
    )


def bootstrap_eer_ci(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI on EER.

    Needed for the headline claim. With six vocoders, "weak correlation" versus
    "no correlation" turns on whether the per-condition EERs are separated by
    more than their own sampling noise -- a point estimate cannot settle that.
    Each draw uses the shared directional, tie-aware EER. The existing independent
    within-class sampling design is unchanged; this is not a paired bootstrap.
    """
    rng = np.random.default_rng(seed)
    bona = _score_vector(bonafide_scores, "bonafide_scores")
    spoof = _score_vector(spoof_scores, "spoof_scores")
    boots = np.empty(n_boot)
    for i in range(n_boot):
        b = rng.choice(bona, size=bona.size, replace=True)
        s = rng.choice(spoof, size=spoof.size, replace=True)
        boots[i] = compute_eer(b, s)[0]
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))
