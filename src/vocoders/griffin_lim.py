"""Griffin-Lim: the non-neural floor of the ladder.

Included because it anchors the low end of the quality axis with a method that
has no learned prior at all. Its artifacts are phase-reconstruction artifacts
by construction, which makes it the clearest positive control for the
phase-coherence cue the study expects detectors to exploit.

Fully implementable with no checkpoint, so this is the condition Person A ships
first in the v0 dataset. Having no checkpoint also means its mel front-end is a
project decision rather than a model property: it is set to LADDER_FMAX, so it
follows the analysis band instead of constraining it (INV-17).

**It is a floor reference, not a rung in the correlation.** Griffin-Lim
reconstructs by inverting the mel to a linear spectrogram and running ISTFT, so
it structurally cannot emit above fmax: measured high-band fraction 0.00000,
against 0.01803 for real. Every other condition is a time-domain upsampler that
produces full-band output regardless of what the mel carried. That makes
Griffin-Lim's detectability a bandwidth artifact rather than a reconstruction
artifact, and at the low-quality end of the ladder it would anchor a strong
positive Spearman for a reason unrelated to the hypothesis. It is generated and
reported.

Superseded in part by INV-18: the cliff lives ABOVE LADDER_FMAX, so griffin_lim
is back IN the band-limited primary correlation and excluded only from the
high-band analysis, with the measurement cited in ANALYSIS_EXCLUSIONS.
"""

from __future__ import annotations

import inspect

import librosa
import numpy as np

from data.invariants import GLOBAL_SEED
from data.mel import GRIFFIN_LIM_MEL, MelConfig

from .base import Vocoder, VocoderSpec

SPEC = VocoderSpec(
    key="griffin_lim",
    display_name="Griffin-Lim",
    family="signal_processing",
    year=1984,
    params_m=None,
    mel=GRIFFIN_LIM_MEL,
    checkpoint="",  # none required
    notes="No learned prior. Iterative phase reconstruction from magnitude only.",
)


class GriffinLim(Vocoder):
    """Mel -> NNLS linear magnitude -> Griffin-Lim, fixed iterations, fixed seed.

    INV-08 asks every stochastic condition to fix its sampler, and Griffin-Lim is
    stochastic: its phase starts random. ``librosa.feature.inverse.mel_to_audio``
    does not forward a ``random_state`` to ``griffinlim``, so it would draw from
    numpy's global generator and the audio for one utterance would depend on how
    many utterances came before it. The two steps are therefore called
    separately, with the seed passed explicitly -- identical to ``mel_to_audio``
    otherwise.
    """

    runtime_modules = ()

    def __init__(self, n_iter: int = 60, mel: MelConfig | None = None,
                 seed: int = GLOBAL_SEED):
        self.n_iter = n_iter
        self.seed = seed
        self.spec = SPEC if mel is None else VocoderSpec(**{**SPEC.__dict__, "mel": mel})

    def load(self) -> None:  # nothing to load
        return None

    def synthesize(self, mel: np.ndarray) -> np.ndarray:
        cfg = self.spec.mel
        linear = np.exp(mel) if cfg.log_base == "natural" else librosa.db_to_amplitude(mel)
        magnitude = librosa.feature.inverse.mel_to_stft(
            M=linear,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            power=1.0,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
        )
        wav = librosa.griffinlim(
            magnitude,
            n_iter=self.n_iter,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            center=cfg.center,
            **{_SEED_KWARG: self.seed},
        )
        return wav.astype(np.float32)


# librosa 0.11 renamed griffinlim's `random_state` to `rng` (the old name is a
# deprecated alias, removed in 1.2). Pass whichever the installed version takes.
# The seed fixes the phase init within one librosa version; librosa's version is
# recorded in the run provenance because the generator behind the seed may differ
# between versions.
_SEED_KWARG = (
    "rng" if "rng" in inspect.signature(librosa.griffinlim).parameters
    else "random_state"
)
