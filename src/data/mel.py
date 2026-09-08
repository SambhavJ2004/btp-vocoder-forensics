"""Per-vocoder mel front-ends (INV-02).

Each vocoder is trained against a specific mel configuration; feeding it a
foreign mel puts the model out of distribution and measures a broken vocoder
rather than the real one. So the design keeps each vocoder's native front-end,
records it in the manifest, and reports the difference as a stated limitation.

**Scope note (docs/mel_configs.md).** Five of the six ladder configs are
numerically identical -- n_fft 1024, hop 256, win 1024, 80 mels, fmin 0,
22050 Hz -- differing only in `fmax`, which INV-17 then equalises at delivery.
For those five, what remains of INV-02 is implementation difference rather than
parameter difference: MelGAN's `Audio2Mel` differs from HiFi-GAN's
`mel_spectrogram` in windowing and log convention, and BigVGAN's is a descendant
of HiFi-GAN's.

**Vocos is the exception, and it re-widens INV-02.** Replacing SpecDiff-GAN put a
genuinely different front-end into the ladder: 24 kHz, 100 mel bands, fmax
12000. So "same numbers, different extractors" is no longer true of the ladder
as a whole -- it is true of five conditions and false of one. INV-02's original
framing applies to Vocos in full.

Every `fmax` here comes from :data:`data.invariants.AUDITED_MEL_FMAX` rather
than a literal, so the audit table and the code cannot drift apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import librosa
import numpy as np

from .invariants import ARCHIVE_SR, AUDITED_MEL_FMAX, LADDER_FMAX


@dataclass(frozen=True)
class MelConfig:
    """A vocoder's native mel-spectrogram front-end.

    Values must be copied from the vocoder's own released config, not guessed.
    An incorrect mel is not a small error: it silently degrades that condition's
    quality score and inflates its detectability, which is exactly the
    correlation the study is trying to measure.

    ``fmax = None`` in a released config means "librosa default", which resolves
    to ``sr / 2`` -- not "unbounded". Store the resolved value here so nothing
    downstream has to re-derive it.
    """

    sample_rate: int
    n_fft: int
    hop_length: int
    win_length: int
    n_mels: int
    fmin: float
    fmax: float | None
    log_base: str = "natural"  # "natural" (log) or "db"
    center: bool = True
    normalized: bool = False
    source: str = ""  # URL or repo path the numbers were copied from

    def as_manifest_fields(self) -> dict[str, object]:
        d = asdict(self)
        return {
            "mel_n_fft": d["n_fft"],
            "mel_hop": d["hop_length"],
            "mel_win": d["win_length"],
            "mel_n_mels": d["n_mels"],
            "mel_fmin": d["fmin"],
            "mel_fmax": d["fmax"],
        }


def compute_mel(wav: np.ndarray, cfg: MelConfig) -> np.ndarray:
    """Reference mel implementation.

    Use ONLY where a vocoder does not ship its own extractor. Where it does,
    call the vocoder's own code -- small differences in windowing, padding or
    the log floor shift the input distribution enough to matter.
    """
    spec = librosa.feature.melspectrogram(
        y=wav,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        center=cfg.center,
        power=1.0,
    )
    if cfg.log_base == "db":
        return librosa.power_to_db(spec**2, ref=1.0)
    return np.log(np.clip(spec, a_min=1e-5, a_max=None))


def _lj_config(fmax: float, source: str, **overrides) -> MelConfig:
    """The shape every audited config in this ladder turned out to share.

    Only ``fmax`` and the source differ, which is precisely the finding the
    Step-1 audit produced. Expressing it this way makes a future divergence
    obvious: a checkpoint that needs an override is a checkpoint that broke the
    pattern, and the override says so at the call site.
    """
    return MelConfig(
        sample_rate=ARCHIVE_SR,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        fmin=0.0,
        fmax=fmax,
        source=source,
        **overrides,
    )


# --- Audited configurations (docs/mel_configs.md) ----------------------------

MELGAN_MEL = _lj_config(
    AUDITED_MEL_FMAX["melgan"],
    "descriptinc/melgan-neurips mel2wav/modules.py Audio2Mel defaults "
    "(mel_fmax=None -> librosa sr/2 = 11025); scripts/train.py passes only "
    "n_mel_channels",
)

HIFIGAN_V1_MEL = _lj_config(
    AUDITED_MEL_FMAX["hifigan_v1"],
    "jik876/hifi-gan config_v1.json",
)

# Declared but unavailable -- kept so the audit trail survives the substitution
# and so Vocos can be compared against what it replaced (INV-08).
SPECDIFF_MEL = _lj_config(
    AUDITED_MEL_FMAX["specdiff_gan"],
    "SpecDiff-GAN/SpecDiff-GAN configs/config_ljspeech.json (weights unavailable)",
)

# Vocos breaks the pattern every other ladder config follows, which is why it is
# spelled out rather than built with _lj_config: 24 kHz native, 100 mel bands,
# and a front-end that passes neither f_min nor f_max to torchaudio, so the
# library defaults 0.0 and sr/2 apply. `win_length` is likewise unset and
# defaults to n_fft.
VOCOS_MEL = MelConfig(
    sample_rate=24_000,
    n_fft=1024,
    hop_length=256,
    win_length=1024,  # torchaudio default: win_length = n_fft
    n_mels=100,
    fmin=0.0,
    fmax=AUDITED_MEL_FMAX["vocos"],  # torchaudio default f_max = sr/2 = 12000
    source=(
        "charactr/vocos-mel-24khz config.yaml + gemelo-ai/vocos "
        "vocos/feature_extractors.py MelSpectrogramFeatures (f_min/f_max not "
        "passed -> torchaudio defaults 0.0 / sr/2)"
    ),
)

BIGVGAN_BASE_22K_MEL = _lj_config(
    AUDITED_MEL_FMAX["bigvgan_base"],
    "nvidia/bigvgan_base_22khz_80band config.json",
)

BIGVGAN_V2_22K_FMAX8K_MEL = _lj_config(
    AUDITED_MEL_FMAX["bigvgan_112m"],
    "nvidia/bigvgan_v2_22khz_80band_fmax8k_256x config.json",
)

BIGVGAN_V2_22K_FULLBAND_MEL = _lj_config(
    AUDITED_MEL_FMAX["bigvgan_v2_22khz_fullband"],
    "nvidia/bigvgan_v2_22khz_80band_256x config.json (fmax=null -> librosa "
    "sr/2 = 11025)",
)

# Griffin-Lim has no checkpoint, so its front-end is ours to choose. It is set
# to the analysis band rather than to any model's value: it neither constrains
# LADDER_FMAX nor escapes it.
#
# Unlike every other condition, this value is a HARD CEILING on its output.
# Griffin-Lim inverts the mel to a linear spectrogram and runs ISTFT, so it
# cannot emit above fmax at all -- measured high-band fraction 0.00000. Every
# other condition is a time-domain upsampler whose fmax constrains only what it
# is TOLD, not what it produces. That asymmetry is why griffin_lim is a floor
# reference rather than a rung in the correlation. See docs/mel_configs.md.
GRIFFIN_LIM_MEL = _lj_config(
    LADDER_FMAX,
    "no checkpoint; front-end is a project decision, set to LADDER_FMAX",
)
