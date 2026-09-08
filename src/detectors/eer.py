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


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> tuple[float, float]:
    """Return (eer, threshold). EER is a fraction in [0, 1], not a percentage."""
    bona = np.asarray(bonafide_scores, dtype=np.float64).ravel()
    spoof = np.asarray(spoof_scores, dtype=np.float64).ravel()
    if bona.size == 0 or spoof.size == 0:
        raise ValueError("EER needs both bona fide and spoof scores.")

    scores = np.concatenate([bona, spoof])
    labels = np.concatenate([np.ones_like(bona), np.zeros_like(spoof)])
    order = np.argsort(scores, kind="mergesort")
    scores, labels = scores[order], labels[order]

    # Sweeping the threshold upward through the sorted scores.
    tar_total, non_total = bona.size, spoof.size
    frr = np.cumsum(labels) / tar_total              # bona fide rejected
    far = 1.0 - (np.cumsum(1 - labels) / non_total)  # spoof accepted
    frr = np.concatenate([[0.0], frr])
    far = np.concatenate([[1.0], far])
    thresholds = np.concatenate([[scores[0] - 1e-6], scores])

    idx = int(np.nanargmin(np.abs(frr - far)))
    return float((frr[idx] + far[idx]) / 2.0), float(thresholds[idx])


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
    """
    rng = np.random.default_rng(seed)
    bona = np.asarray(bonafide_scores).ravel()
    spoof = np.asarray(spoof_scores).ravel()
    boots = np.empty(n_boot)
    for i in range(n_boot):
        b = rng.choice(bona, size=bona.size, replace=True)
        s = rng.choice(spoof, size=spoof.size, replace=True)
        boots[i] = compute_eer(b, s)[0]
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))
