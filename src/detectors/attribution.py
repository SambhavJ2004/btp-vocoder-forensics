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
    """Free-energy OOD score. Lower energy == more in-distribution.

    E(x) = -T * logsumexp(logits / T). Preferred over max-softmax because it
    keeps the magnitude information softmax normalises away -- which is exactly
    the information that distinguishes "confidently one of these" from
    "confidently none of these".
    """
    from scipy.special import logsumexp

    return -temperature * logsumexp(np.asarray(logits, dtype=np.float64) / temperature, axis=-1)


def open_set_metrics(*_args, **_kwargs) -> dict[str, float]:
    """AUROC / FPR@95TPR for known-vs-unknown, plus closed-set accuracy on the
    accepted subset. Report both: a rejector that rejects everything scores
    perfectly on the first and uselessly on the second.
    """
    raise NotImplementedError(
        "Implement once the closed-set classifier is trained. Calibrate any "
        "threshold on the TRAINING conditions' dev split only -- never on the "
        "held-out vocoder."
    )
