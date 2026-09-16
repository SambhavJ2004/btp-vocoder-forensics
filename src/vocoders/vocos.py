"""Vocos — replaces SpecDiff-GAN in the primary ladder.

SpecDiff-GAN's weights are not publicly available (its repo ships configs but no
pretrained checkpoint), so the diffusion-adjacent rung was unfillable. Vocos is
the substitute named in the plan's risk table.

It earns the rung on family diversity, which is what SpecDiff-GAN was there for.
Vocos does not model the waveform in the time domain at all: it predicts STFT
magnitude and phase and reconstructs by inverse FFT. That is a genuinely
different generative mechanism from the time-domain GAN vocoders around it, and
it is the kind of difference the Phase C family-level hypothesis is about.

Two things about it differ from every other condition and both are recorded
rather than smoothed over:

  Native rate 24 kHz, not 22.05. Its output takes one resample to the archive
  rate (INV-01 permits exactly that: native -> archive, single step). Every
  other condition is already at 22.05 kHz and takes none.

  100 mel bands, not 80, and fmax 12000, not 8000. Its front-end is genuinely
  unlike the rest of the ladder -- see the INV-02 note in CLAUDE.md, which the
  Vocos substitution partly un-narrows.

Neither the band count nor the fmax moves LADDER_FMAX -- 12000 was never the
minimum. The RESAMPLE does matter: it leaves a signature in [rt_edge, Nyquist),
which is why vocos is excluded from the high-band analysis while staying in the
band-limited primary one (INV-18), and why its resampler is pinned below.
"""

from __future__ import annotations

import numpy as np

from data.invariants import MeasurementRef, ResamplerSpec
from data.mel import VOCOS_MEL

from .base import Vocoder, VocoderSpec

_CSV = "band_measurements.csv"
_PROBES = "probes.json"

# --- INV-18  The pinned resampler ---------------------------------------------
# Vocos is 24 kHz native and is resampled to the 22.05 kHz archive on the way in
# (analysis) and on the way out (delivery). Both use THIS resampler.
#
# Why not the torchaudio default: its round trip keeps only 81.98% of the energy
# above LADDER_FMAX -- an 18% cost to exactly the band the high-band analysis
# measures. That number is committed (justification below: real_roundtrip was
# produced with the default resampler; its mean high-band fraction over real's is
# 0.8198).
#
# Why these kaiser parameters (probes.json, 200 LJSpeech files, GroupKFold(5) by
# utterance, PCM_16):
#
#     round trip vs real      high-band ratio   AUROC below LADDER_FMAX   above
#     torchaudio default           0.8005          0.5291 +- 0.0024       1.0000
#     this kaiser pin              0.9997          0.5000 +- 0.0000       0.9935
#
# Two separate benefits, both committed. The pin keeps the high band (0.9997,
# against 0.8005). And it is what makes Vocos admissible in the PRIMARY analysis:
# the default round trip is weakly but consistently separable BELOW the band
# (0.5291), the kaiser round trip is not. The earlier 60-file figure for this pin
# (ratio 0.9998) is not in any committed artifact; 0.9997 replaces it.
#
# Pinning does not remove Vocos's resampler signature ABOVE the band (0.9935),
# and the uniform round trip that would have equalised it was rejected -- see
# CLAUDE.md INV-18.
VOCOS_RESAMPLER = ResamplerSpec(
    backend="torchaudio",
    resampling_method="sinc_interp_kaiser",
    lowpass_filter_width=64,
    rolloff=0.9999,
    beta=14.769656459379492,
    justification=(
        # the default's cost, 60 files
        MeasurementRef(_CSV, "real_roundtrip", "hb_8k_nyq", "mean", 1.653462e-02),
        MeasurementRef(_CSV, "real", "hb_8k_nyq", "mean", 2.016939e-02),
        # the pin's benefit, 200 files
        MeasurementRef(_PROBES, "real_vs_rt_kaiser", "ratio_highband", "recorded",
                       0.9997367196968923),
        MeasurementRef(_PROBES, "real_vs_rt_kaiser", "auroc_below", "recorded", 0.5),
        MeasurementRef(_PROBES, "real_vs_rt_default", "auroc_below", "recorded", 0.5291),
    ),
)

SPEC = VocoderSpec(
    key="vocos",
    display_name="Vocos",
    family="flow",  # neither pure GAN nor diffusion: inverse-FFT spectral synthesis
    year=2023,
    params_m=13.5,
    mel=VOCOS_MEL,
    checkpoint="charactr/vocos-mel-24khz",
    checkpoint_revision="0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21",
    resampler=VOCOS_RESAMPLER,
    notes=(
        "Substitutes specdiff_gan, whose weights are unavailable. Predicts STFT "
        "coefficients and reconstructs by inverse FFT rather than modelling the "
        "waveform directly. 24 kHz native, 100 mel bands, fmax 12000."
    ),
)


class Vocos(Vocoder):
    spec = SPEC

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    def load(self) -> None:
        raise NotImplementedError(
            "Load via vocos.Vocos.from_pretrained(spec.checkpoint) in its own "
            "isolated session (INV-13). Pass the pinned revision -- the HF helper "
            "tracks the branch head otherwise (INV-08). Note this pulls its own "
            "torch/torchaudio pins, which is exactly why conditions do not share "
            "an environment."
        )

    def synthesize(self, mel: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Use vocos.feature_extractor (MelSpectrogramFeatures) for analysis so "
            "the front-end matches training -- it passes neither f_min nor f_max to "
            "torchaudio, so the defaults 0 / sr/2 apply and must not be overridden. "
            "Return the model's native 24 kHz, float32 mono, no post-processing "
            "(INV-10). The single resample to the 22.05 kHz archive happens later, "
            "in data.preprocess (INV-01)."
        )

    def unload(self) -> None:
        self._model = None
