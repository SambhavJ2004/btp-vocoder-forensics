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
reported; `spearman_headline` refuses it (INV-17, CORRELATION_EXCLUDED).
"""

from __future__ import annotations

import librosa
import numpy as np

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
    def __init__(self, n_iter: int = 60, mel: MelConfig | None = None):
        self.n_iter = n_iter
        self.spec = SPEC if mel is None else VocoderSpec(**{**SPEC.__dict__, "mel": mel})

    def load(self) -> None:  # nothing to load
        return None

    def synthesize(self, mel: np.ndarray) -> np.ndarray:
        cfg = self.spec.mel
        linear = np.exp(mel) if cfg.log_base == "natural" else librosa.db_to_amplitude(mel)
        wav = librosa.feature.inverse.mel_to_audio(
            M=linear,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            power=1.0,
            n_iter=self.n_iter,
            center=cfg.center,
        )
        return wav.astype(np.float32)
