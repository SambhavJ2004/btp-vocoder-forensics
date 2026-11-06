"""Phase C — source tracing. Which vocoder produced this audio?

Three settings, in increasing order of difficulty and of interest:

  CLOSED-SET
      N-way classification over the known vocoder set. Expected to work well;
      included as a baseline, not as a result.

  OPEN-SET (leave-one-vocoder-out)
      The system meets a generator absent from training. Softmax cannot express
      "none of these" and will assign a confident wrong label, so a score-based
      rejection mechanism is required: energy-based OOD scoring, or calibrated
      distance thresholds in embedding space. The held-out vocoder must be
      excluded from training AND from any threshold calibration -- calibrating
      the rejection threshold on the held-out class leaks the answer and is the
      standard way this experiment is accidentally invalidated.

  FAMILY-LEVEL
      Do GAN vocoders cluster coherently, distinct from non-GAN approaches? If
      so, family-level attribution may generalise to unseen family members even
      where instance-level attribution fails.

Stretch goal per the plan; scope confirmation is an open question for the
supervisor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vocoders.registry import LADDER, get_spec


@dataclass
class OpenSetSplit:
    """One leave-one-vocoder-out fold."""

    held_out: str
    train_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.held_out in self.train_conditions:
            raise ValueError(
                f"'{self.held_out}' is held out but also in training -- the fold is void."
            )


def leave_one_vocoder_out(conditions: tuple[str, ...] = LADDER) -> list[OpenSetSplit]:
    return [
        OpenSetSplit(held_out=c, train_conditions=tuple(x for x in conditions if x != c))
        for c in conditions
    ]


def family_labels(conditions: tuple[str, ...] = LADDER) -> dict[str, str]:
    """condition -> family, for the family-level clustering hypothesis."""
    return {c: get_spec(c).family for c in conditions}


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """E=-T*logsumexp(z/T), over the last axis; lower is nominally known-like.

    This preserves the original orientation/formula, not a calibrated unknown
    probability. Absolute logits require empirical OOD validation. Center before
    division to handle tiny T and large finite logits without ratio overflow.
    Unrepresentable float64 results raise rather than silently saturating.
    """
    t = np.asarray(temperature)
    if t.ndim or t.dtype.kind not in "iuf" or not np.isfinite(t) or t <= 0:
        raise ValueError("Temperature must be a finite positive numeric scalar.")
    t = float(t)
    if not np.isfinite(t) or t <= 0:
        raise ValueError("Temperature is outside positive float64 range.")
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim < 1 or not z.size or not np.isfinite(z).all():
        raise ValueError("Energy requires finite logits with a nonempty class axis.")
    maximum = z.max(axis=-1)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        centered = (z - maximum[..., None]) / t
        correction = np.log(np.exp(centered).sum(axis=-1))
        product = t * correction
        total = maximum + product
        # Product alone can overflow although its sum with a negative maximum fits.
        fallback = np.isinf(product) & (maximum < 0)
        if np.any(fallback):
            total = np.where(fallback, t * (maximum / t + correction), total)
    if not np.isfinite(total).all():
        raise ValueError("Energy is outside representable float64 range.")
    return -total


def open_set_metrics(*_args, **_kwargs) -> dict[str, float]:
    """AUROC / FPR@95TPR for known-vs-unknown, plus closed-set accuracy on the
    accepted subset. Report both alongside coverage. Rejecting everything at
    one cutoff does not establish score discrimination; constant scores give
    AUROC 0.5 when both populations are present.
    """
    raise NotImplementedError(
        "Implement once the closed-set classifier is trained. Calibrate any "
        "threshold on the TRAINING conditions' dev split only -- never on the "
        "held-out vocoder."
    )
