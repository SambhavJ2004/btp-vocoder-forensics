"""AASIST — the primary detector.

Chosen because ~300k parameters trains on a single T4 within the Kaggle quota,
and because the ASVspoof 2019 LA pretrained model is the natural mismatched
(zero-shot) baseline: it is a deployed-system stand-in that has never seen any
of the six vocoders.

Graph attention over spectro-temporal sub-bands, raw-waveform front-end
(RawNet2-style sinc filters), so it can key on high-band detail directly --
which is why the band-limited ablation is informative with this model.

Tier discipline (INV-01): the MISMATCHED protocol runs on the derived 16 kHz
set, because that is what the pretrained LA weights expect. The MATCHED protocol
and the band-limited ablation train from scratch and run on the 22.05 kHz
archive, so that the high band the mechanism argument concerns is actually
present in the input. Both are legitimate; they are simply not the same input,
and the DetectionResult carries the tier so the two are never silently compared.
"""

from __future__ import annotations

import numpy as np

from .base import (
    ARCHIVE_CROP_SAMPLES,
    ASVSPOOF_CROP_SAMPLES,
    ASVSPOOF_CROP_SECONDS,
    Detector,
    DetectorSpec,
    crop_samples_for_rate,
    fixed_length_crop,
)

SPEC = DetectorSpec(
    key="aasist",
    display_name="AASIST",
    params_m=0.297,
    input_kind="raw",
    max_seconds=4.0375,  # 64600 samples at 16 kHz, the ASVspoof convention
    pretrained="TODO: clovaai/aasist AASIST.pth (LA-trained)",
    notes="Zero-shot baseline for the mismatched protocol; also retrained for matched.",
)

class AASIST(Detector):
    spec = SPEC

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    def load(self, checkpoint: str | None = None) -> None:
        raise NotImplementedError(
            "Clone clovaai/aasist, build Model(config['model_config']), load the "
            "state dict, eval(). For the MISMATCHED protocol use the released "
            "LA-trained weights untouched -- no fine-tuning, no adaptation, or the "
            "protocol stops measuring generalisation failure."
        )

    def score(self, wav: np.ndarray) -> float:
        raise NotImplementedError(
            "Crop with self.crop_for(wav, sr) -- never crop independently (INV-11) "
            "-- then forward and "
            "return the bona fide logit (index 1 in the upstream head). Confirm the "
            "sign on a held-out set: an EER above 0.5 usually means it is flipped."
        )

    def fit(self, train_manifest, dev_manifest, **kwargs) -> None:
        raise NotImplementedError(
            "Matched protocol: train real-vs-vocoder-X from scratch (or from LA "
            "init, stated either way). Use the frozen splits from INV-09 -- never "
            "re-split per condition, or the conditions stop being comparable."
        )


def n_crop_samples(seconds: float = SPEC.max_seconds, sr: int | None = None) -> int:
    from data.invariants import ZEROSHOT_SR

    return round(seconds * (ZEROSHOT_SR if sr is None else sr))


__all__ = [
    "AASIST",
    "ARCHIVE_CROP_SAMPLES",
    "ASVSPOOF_CROP_SAMPLES",
    "ASVSPOOF_CROP_SECONDS",
    "SPEC",
    "crop_samples_for_rate",
    "fixed_length_crop",
    "n_crop_samples",
]
