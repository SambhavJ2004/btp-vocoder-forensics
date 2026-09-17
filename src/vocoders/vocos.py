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

from data.invariants import InvariantViolation, MeasurementRef, ResamplerSpec
from data.mel import VOCOS_MEL, MelConfig

from .base import Vocoder, VocoderSpec, resolve_torch_device

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
# and with the pipeline's loudness normalisation (keys ending _norm):
#
#     torchaudio default           gain +0.02986 dB  0.5348 +- 0.0018     1.0000
#     this kaiser pin              gain +4.62e-05 dB 0.5000 +- 0.0000     0.9989
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
        MeasurementRef(_PROBES, "real_vs_rt_kaiser", "ratio_highband", "recorded", 0.99974),
        MeasurementRef(_PROBES, "real_vs_rt_kaiser", "auroc_below", "recorded", 0.5),
        MeasurementRef(_PROBES, "real_vs_rt_default", "auroc_below", "recorded", 0.5291),
        # ...and through the pipeline's loudness normalisation
        MeasurementRef(_PROBES, "real_vs_rt_kaiser_norm", "auroc_below", "recorded", 0.5),
        MeasurementRef(_PROBES, "real_vs_rt_kaiser_norm", "mean_gain_delta_db", "recorded",
                       4.62e-05),
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


def check_feature_extractor(feature_extractor, mel: MelConfig) -> None:
    """INV-02. The loaded Vocos front-end must be the audited one, field by field.

    Upstream's ``MelSpectrogramFeatures`` wraps a torchaudio ``MelSpectrogram``
    and passes neither ``f_min`` nor ``f_max``, so torchaudio's defaults (0 and
    sr/2) apply. The audit recorded those resolved values; this compares the
    object actually loaded against them rather than trusting the config file
    and the audit to agree.
    """
    spec = getattr(feature_extractor, "mel_spec", None)
    if spec is None:
        raise InvariantViolation(
            f"INV-02: vocos loaded a {type(feature_extractor).__name__} front-end, not "
            "MelSpectrogramFeatures. This is not the audited mel checkpoint."
        )
    sr = int(spec.sample_rate)
    f_max = spec.f_max if spec.f_max is not None else sr / 2.0
    loaded = {
        "sample_rate": sr,
        "n_fft": int(spec.n_fft),
        "hop_length": int(spec.hop_length),
        "win_length": int(spec.win_length),
        "n_mels": int(spec.n_mels),
        "fmin": float(spec.f_min),
        "fmax": float(f_max),
    }
    for field, value in loaded.items():
        expected = getattr(mel, field)
        if expected is None or float(expected) != float(value):
            raise InvariantViolation(
                f"INV-02: vocos front-end {field} is {value}, audited {expected}. The "
                "checkpoint behind the pinned revision is not the one docs/mel_configs.md "
                "describes."
            )
    padding = getattr(feature_extractor, "padding", "center")
    if (padding == "center") != mel.center:
        raise InvariantViolation(
            f"INV-02/INV-16: vocos front-end padding is {padding!r}, audited "
            f"center={mel.center}. Padding decides frame alignment."
        )


class Vocos(Vocoder):
    """Vocos, loaded from the pinned revision with its own mel front-end.

    ``resynthesize`` runs upstream's copy-synthesis ``model(audio)``: its own
    ``MelSpectrogramFeatures`` then backbone and ISTFT head. Input is 24 kHz
    (the caller resamples the source with the pinned kaiser resampler, INV-18);
    output is the model's native 24 kHz, float32 mono, untouched (INV-10).
    """

    spec = SPEC
    runtime_modules = ("torch", "torchaudio", "vocos", "huggingface_hub")
    required_pins = ("checkpoint_revision",)

    def __init__(self, device: str | None = None):
        self.device = device
        self._model = None

    def load(self) -> None:
        if not self.spec.checkpoint_revision:
            raise InvariantViolation(
                "INV-08: vocos has no checkpoint_revision; from_pretrained would track "
                "the branch head."
            )
        from vocos import Vocos as UpstreamVocos

        self.device = resolve_torch_device(self.device)
        model = UpstreamVocos.from_pretrained(
            self.spec.checkpoint, revision=self.spec.checkpoint_revision
        )
        check_feature_extractor(model.feature_extractor, self.spec.mel)
        self._model = model.eval().to(self.device)

    def _require_loaded(self):
        if self._model is None:
            raise RuntimeError("vocos: call load() first.")
        return self._model

    def synthesize(self, mel: np.ndarray) -> np.ndarray:
        import torch

        model = self._require_loaded()
        features = torch.from_numpy(np.ascontiguousarray(mel, dtype=np.float32))
        if features.ndim == 2:
            features = features.unsqueeze(0)
        with torch.inference_mode():
            out = model.decode(features.to(self.device))
        return out.squeeze(0).float().cpu().numpy().astype(np.float32)

    def resynthesize(self, wav: np.ndarray, sr: int) -> np.ndarray:
        import torch

        if sr != self.spec.mel.sample_rate:
            raise ValueError(
                f"vocos: expected input at {self.spec.mel.sample_rate} Hz, got {sr}. "
                "Resample the source with VOCOS_RESAMPLER first (INV-18)."
            )
        model = self._require_loaded()
        audio = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            out = model(audio.to(self.device))
        return out.squeeze(0).float().cpu().numpy().astype(np.float32)

    def unload(self) -> None:
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass
