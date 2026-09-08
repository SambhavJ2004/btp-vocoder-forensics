"""Vocoder adapter interface.

Every vocoder in the ladder is wrapped so Phase A can drive them all through
one loop. Adapters return audio at the vocoder's OWN native sample rate and do
nothing else to it -- no resampling, no trimming, no gain, no denoising.

That restriction is the point (INV-10): if one adapter quietly applied a peak
limiter or a de-esser, the measured difference between conditions would be
partly that post-processing rather than the vocoder. Normalisation happens once,
for every condition identically, in :mod:`data.preprocess`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from data.mel import MelConfig


@dataclass(frozen=True)
class VocoderSpec:
    """Static description of a condition, recorded into the manifest."""

    key: str                    # condition name in the manifest, e.g. "hifigan_v1"
    display_name: str
    family: str                 # "signal_processing" | "gan" | "diffusion" | "flow"
    year: int
    params_m: float | None      # millions; None where not applicable
    mel: MelConfig
    checkpoint: str = ""        # URL, HF repo id, or Kaggle Dataset path
    # INV-08 pinning. `checkpoint_revision` is the HuggingFace commit SHA: it is
    # content-addressed over the whole repo, verifiable without downloading the
    # weights, and therefore at least as strong a pin as a single file digest.
    # `checkpoint_sha256` is the weight-file digest, recorded on Kaggle at
    # download time. Either one satisfies the audit; both is better.
    checkpoint_revision: str = ""
    checkpoint_sha256: str = ""
    upstream_commit: str = ""
    isolated_env: bool = True   # INV-13: never co-installed with other vocoders
    # INV-17. False for paired controls: generated and reported, but not a rung.
    # Correlation exclusion is declared in invariants.CORRELATION_EXCLUDED, which
    # also covers rungs excluded for other reasons (griffin_lim).
    primary_ladder: bool = True
    notes: str = ""


class Vocoder(ABC):
    """Adapter around one upstream vocoder implementation."""

    spec: VocoderSpec

    @abstractmethod
    def load(self) -> None:
        """Materialise weights. Called once per session."""

    @abstractmethod
    def synthesize(self, mel: np.ndarray) -> np.ndarray:
        """mel -> waveform at ``self.spec.mel.sample_rate``. Mono, float32, no
        post-processing of any kind."""

    def resynthesize(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Real audio -> mel -> waveform, the Phase A operation.

        Default implementation goes through the vocoder's declared mel config;
        override where upstream ships its own extractor (preferred).
        """
        from data.mel import compute_mel

        if sr != self.spec.mel.sample_rate:
            raise ValueError(
                f"{self.spec.key}: expected input at {self.spec.mel.sample_rate} Hz "
                f"(its training rate), got {sr}. Resample the SOURCE to the vocoder's "
                "rate before analysis; the single output resample to 16 kHz happens "
                "later, in data.preprocess (INV-01)."
            )
        return self.synthesize(compute_mel(wav, self.spec.mel))

    def unload(self) -> None:  # noqa: B027 -- optional hook, not every adapter holds GPU state
        """Free GPU memory between conditions in a shared session."""


@dataclass
class ResynthesisResult:
    """What Phase A hands to the preprocessing stage."""

    utt_id: str
    condition: str
    wav_native: np.ndarray
    sample_rate: int
    meta: dict = field(default_factory=dict)
