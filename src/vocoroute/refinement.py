"""Local complex-spectral refinement and actual selected-patch execution."""

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .spectral import FRONTEND_V1, validate_spectrum

TIME_CORE, FREQ_CORE = 16, 32
TIME_OVERLAP, FREQ_OVERLAP = 2, 4
HALO = 4  # Four unit-dilation 3x3 convolutions.
PREDICTED_BINS = FRONTEND_V1.bins - 2
FREQ_TILES = (PREDICTED_BINS + FREQ_CORE - 1) // FREQ_CORE
PHASE_FEATURE_FLOOR = 1e-5
LOG_GAIN_LIMIT = 0.35
PHASE_LIMIT = math.pi / 4


def patch_count(frames: int) -> int:
    return ((frames + TIME_CORE - 1) // TIME_CORE) * FREQ_TILES


def _check_budget(k: int, count: int) -> None:
    if type(k) is not int or not 0 <= k <= count:
        raise ValueError(f"patch budget must be an integer in [0, {count}]")


def top_k_patches(scores: Tensor, k: int) -> Tensor:
    """Rank finite scores per utterance, breaking ties by ascending tile ID."""
    if scores.ndim != 2 or min(scores.shape) < 1 or not scores.is_floating_point():
        raise ValueError("expected nonempty floating-point patch scores [B, P]")
    _check_budget(k, scores.shape[1])
    if not torch.isfinite(scores).all():
        raise ValueError("patch scores must be finite")
    if k == 0:
        return torch.empty((scores.shape[0], 0), device=scores.device, dtype=torch.long)
    return torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]


def _feature_canvas(spectrum: Tensor) -> Tensor:
    interior = spectrum[:, 1:-1].transpose(1, 2)
    magnitude = interior.abs()
    safe_magnitude = magnitude.clamp_min(PHASE_FEATURE_FLOOR)
    phasor = torch.where(magnitude >= PHASE_FEATURE_FLOOR, interior / safe_magnitude, 0.0)
    time_delta = F.pad(phasor[:, 1:] * phasor[:, :-1].conj(), (0, 0, 1, 0))
    freq_delta = F.pad(phasor[:, :, 1:] * phasor[:, :, :-1].conj(), (1, 0))
    frequency = torch.arange(1, FRONTEND_V1.bins - 1, device=interior.device)
    frequency = (frequency.to(magnitude.dtype) / (FRONTEND_V1.bins - 1)).expand_as(magnitude)
    maps = torch.stack(
        (
            safe_magnitude.log(),
            phasor.imag,
            phasor.real,
            time_delta.imag,
            time_delta.real,
            freq_delta.imag,
            freq_delta.real,
            frequency,
        ),
        dim=1,
    )
    # A common canvas includes the incomplete final cores, taper and halo. Dense
    # convolutions and selected patches see the same zeros beyond the valid grid.
    time_extra = (-interior.shape[1]) % TIME_CORE
    freq_extra = (-PREDICTED_BINS) % FREQ_CORE
    time_border = TIME_OVERLAP + HALO
    freq_border = FREQ_OVERLAP + HALO
    return F.pad(
        maps, (freq_border, freq_border + freq_extra, time_border, time_border + time_extra)
    )


class PhaseExpert(nn.Module):
    """Eight feature planes to bounded log-gain and phase-offset planes."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(8, 24, 1)
        self.layers = nn.ModuleList(nn.Conv2d(24, 24, 3, padding=1) for _ in range(HALO))
        self.head = nn.Conv2d(24, 2, 1)

    def forward(self, features: Tensor) -> Tensor:
        hidden = F.gelu(self.stem(features))
        for layer in self.layers:
            hidden = hidden + F.gelu(layer(hidden))
        gain, phase = self.head(hidden).tanh().unbind(dim=1)
        return torch.stack((LOG_GAIN_LIMIT * gain, PHASE_LIMIT * phase), dim=1)


def apply_residual(spectrum: Tensor, residual: Tensor) -> Tensor:
    """Apply [B, 2, T, 511] in polar residual coordinates on interior bins."""
    expected = (spectrum.shape[0], 2, spectrum.shape[-1], PREDICTED_BINS)
    if residual.shape != expected:
        raise ValueError(f"expected residual shape {expected}")
    gain, phase = residual.transpose(2, 3).unbind(dim=1)
    interior = spectrum[:, 1:-1] * torch.polar(gain.exp(), phase)
    return torch.cat((spectrum[:, :1], interior, spectrum[:, -1:]), dim=1)


def _axis_weights(
    tile: Tensor, count: int, core: int, overlap: int, dtype: torch.dtype
) -> Tensor:
    # Adjacent tapers are complementary at each half-cell. At a physical grid
    # edge, the missing neighbor's weight belongs to the edge tile itself.
    offsets = torch.arange(core + 2 * overlap, device=tile.device, dtype=dtype)
    angle = math.pi * ((offsets + 0.5) / (2 * overlap)).clamp_max(1.0)
    rise = 0.5 - 0.5 * angle.cos()
    left = torch.where(tile[:, None] > 0, rise, 1.0)
    right = torch.where(tile[:, None] < count - 1, rise.flip(0), 1.0)
    return left * right


def _fuse(
    residual: Tensor,
    batch: Tensor,
    time_tile: Tensor,
    freq_tile: Tensor,
    batch_size: int,
    frames: int,
) -> Tensor:
    time_tiles = patch_count(frames) // FREQ_TILES
    time_offsets = torch.arange(residual.shape[2], device=residual.device) - TIME_OVERLAP
    freq_offsets = torch.arange(residual.shape[3], device=residual.device) - FREQ_OVERLAP
    time = time_tile[:, None] * TIME_CORE + time_offsets
    freq = freq_tile[:, None] * FREQ_CORE + freq_offsets
    valid_time = (0 <= time) & (time < frames)
    valid_freq = (0 <= freq) & (freq < PREDICTED_BINS)
    valid = valid_time[:, :, None] & valid_freq[:, None, :]

    time_weights = _axis_weights(time_tile, time_tiles, TIME_CORE, TIME_OVERLAP, residual.dtype)
    freq_weights = _axis_weights(freq_tile, FREQ_TILES, FREQ_CORE, FREQ_OVERLAP, residual.dtype)
    weighted = residual.permute(0, 2, 3, 1) * (
        time_weights[:, :, None, None] * freq_weights[:, None, :, None]
    )
    indices = (batch[:, None, None] * frames + time[:, :, None]) * PREDICTED_BINS + freq[:, None, :]
    combined = residual.new_zeros((batch_size * frames * PREDICTED_BINS, 2))
    combined = combined.index_add(0, indices[valid], weighted[valid])
    return combined.reshape(batch_size, frames, PREDICTED_BINS, 2).permute(0, 3, 1, 2)


class Refinement(NamedTuple):
    spectrum: Tensor
    patch_ids: Tensor


class SelectiveRefiner(nn.Module):
    """A shared expert with a sparse dispatch path and a dense control."""

    def __init__(self) -> None:
        super().__init__()
        self.expert = PhaseExpert()

    def forward(self, spectrum: Tensor, scores: Tensor | None, k: int) -> Refinement:
        validate_spectrum(spectrum)
        batch_size, _, frames = spectrum.shape
        count = patch_count(frames)
        _check_budget(k, count)
        if k == 0:
            empty = torch.empty((batch_size, 0), device=spectrum.device, dtype=torch.long)
            return Refinement(spectrum, empty)
        if (
            scores is None
            or scores.shape != (batch_size, count)
            or scores.device != spectrum.device
        ):
            raise ValueError(f"expected patch scores [{batch_size}, {count}] on spectrum device")
        ids = top_k_patches(scores, k)
        return Refinement(self.refine_selected(spectrum, ids), ids)

    def refine_selected(self, spectrum: Tensor, ids: Tensor) -> Tensor:
        """Dispatch explicit, distinct tile IDs per utterance in time-major order."""
        validate_spectrum(spectrum)
        batch_size, _, frames = spectrum.shape
        if (
            ids.ndim != 2
            or ids.shape[0] != batch_size
            or ids.dtype != torch.long
            or ids.device != spectrum.device
        ):
            raise ValueError("expected int64 patch IDs [B, K] on the spectrum device")
        if ids.shape[1] == 0:
            return spectrum
        if ((ids < 0) | (ids >= patch_count(frames))).any():
            raise ValueError("patch ID is outside the valid grid")
        ordered = ids.sort(dim=1).values
        if (ordered[:, 1:] == ordered[:, :-1]).any():
            raise ValueError("patch IDs must be distinct within each utterance")

        canvas = _feature_canvas(spectrum).permute(0, 2, 3, 1)
        flat = ids.flatten()
        batch = torch.arange(batch_size, device=ids.device).repeat_interleave(ids.shape[1])
        time_tile = flat // FREQ_TILES
        freq_tile = flat % FREQ_TILES
        rows = time_tile[:, None] * TIME_CORE + torch.arange(
            TIME_CORE + 2 * (TIME_OVERLAP + HALO), device=ids.device,
        )
        cols = freq_tile[:, None] * FREQ_CORE + torch.arange(
            FREQ_CORE + 2 * (FREQ_OVERLAP + HALO), device=ids.device,
        )
        # Index directly into the cheap feature grid. Only these tiles enter the expert.
        tiles = canvas[batch[:, None, None], rows[:, :, None], cols[:, None, :]]
        refined = self.expert(tiles.permute(0, 3, 1, 2).contiguous())
        residual = refined[:, :, HALO:-HALO, HALO:-HALO]
        fused = _fuse(residual, batch, time_tile, freq_tile, batch_size, frames)
        return apply_residual(spectrum, fused)

    def dense(self, spectrum: Tensor) -> Tensor:
        """Run one expert call over the shared canvas and keep the valid grid."""
        validate_spectrum(spectrum)
        start_time = TIME_OVERLAP + HALO
        start_freq = FREQ_OVERLAP + HALO
        time = slice(start_time, start_time + spectrum.shape[-1])
        freq = slice(start_freq, start_freq + PREDICTED_BINS)
        residual = self.expert(_feature_canvas(spectrum))[:, :, time, freq]
        return apply_residual(spectrum, residual)
