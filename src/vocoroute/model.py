"""Untrained mel-to-waveform backbone and selective refinement composition."""

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .refinement import SelectiveRefiner
from .spectral import FRONTEND_V1, SpectralFrontend, frames_for_length


class SpectralBlock(nn.Module):
    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.temporal = nn.Conv1d(
            128, 128, 7, padding=3 * dilation, dilation=dilation, groups=128
        )
        self.norm = nn.LayerNorm(128)
        self.expand = nn.Linear(128, 256)
        self.project = nn.Linear(256, 128)

    def forward(self, features: Tensor) -> Tensor:
        hidden = self.norm(self.temporal(features).transpose(1, 2))
        hidden = self.project(F.gelu(self.expand(hidden)))
        return features + hidden.transpose(1, 2)


class SpectralBackbone(nn.Module):
    """Predict magnitude, phase and two signed real endpoints per frame."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv1d(FRONTEND_V1.n_mels, 128, 3, padding=1)
        self.blocks = nn.Sequential(*(SpectralBlock(d) for d in (1, 2, 4, 1, 2, 4)))
        self.head = nn.Conv1d(128, 2 * (FRONTEND_V1.bins - 1), 1)

    def forward(self, mel: Tensor) -> Tensor:
        predicted = self.head(self.blocks(self.stem(mel)))
        interior = FRONTEND_V1.bins - 2
        dc, magnitude, phase, nyquist = predicted.split((1, interior, interior, 1), dim=1)
        return torch.cat((dc, torch.polar(magnitude.clamp(-12, 5).exp(), phase), nyquist), dim=1)


class Synthesis(NamedTuple):
    waveform: Tensor
    spectrum: Tensor
    patch_ids: Tensor


class VocoRoute(nn.Module):
    """Generate an unpadded mono batch with caller-supplied patch scores."""

    def __init__(self) -> None:
        super().__init__()
        self.frontend = SpectralFrontend()
        self.backbone = SpectralBackbone()
        self.refiner = SelectiveRefiner()

    def forward(
        self, mel: Tensor, length: int, *, scores: Tensor | None = None, k: int = 0
    ) -> Synthesis:
        frames = frames_for_length(length)
        if (
            mel.ndim != 3
            or mel.shape[0] < 1
            or mel.shape[1:] != (FRONTEND_V1.n_mels, frames)
            or mel.dtype not in (torch.float32, torch.float64)
        ):
            raise ValueError(f"expected floating-point mel [B, {FRONTEND_V1.n_mels}, {frames}]")
        spectrum = self.backbone(mel)
        refined, patch_ids = self.refiner(spectrum, scores, k)
        return Synthesis(self.frontend.synthesize(refined, length), refined, patch_ids)
