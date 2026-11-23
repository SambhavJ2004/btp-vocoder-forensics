"""Differentiable losses measured on a synthesized, sample-aligned waveform."""

from typing import NamedTuple

import torch
from torch import Tensor, nn

from .training_config import SPECTRAL_FLOOR, STFT_RESOLUTIONS, LossWeights


class ReconstructionTerms(NamedTuple):
    total: Tensor
    waveform: Tensor
    magnitude: Tensor
    log_magnitude: Tensor
    phase_time: Tensor
    phase_frequency: Tensor


def _phase_pair_loss(
    predicted: Tensor, reference: Tensor, reference_magnitude: Tensor, dim: int
) -> Tensor:
    size = predicted.shape[dim] - 1
    before = predicted.narrow(dim, 0, size)
    after = predicted.narrow(dim, 1, size)
    ref_before = reference.narrow(dim, 0, size)
    ref_after = reference.narrow(dim, 1, size)
    error = after * before.conj() - ref_after * ref_before.conj()
    distance = (error.real.square() + error.imag.square()) * 0.5
    energy = torch.minimum(
        reference_magnitude.narrow(dim, 0, size), reference_magnitude.narrow(dim, 1, size)
    )
    weight = torch.where(energy >= SPECTRAL_FLOOR, energy.square(), 0.0)
    numerator = (distance * weight).sum(dim=(1, 2))
    denominator = weight.sum(dim=(1, 2)).clamp_min(SPECTRAL_FLOOR**2)
    return (numerator / denominator).mean()


class ReconstructionLoss(nn.Module):
    """Waveform L1, multi-resolution magnitude L1, and optional phase differences."""

    def __init__(self, weights: LossWeights) -> None:
        super().__init__()
        self.weights = weights
        for fft, _ in STFT_RESOLUTIONS:
            self.register_buffer(f"hann_{fft}", torch.hann_window(fft, dtype=torch.float64))

    def forward(self, predicted: Tensor, target: Tensor) -> ReconstructionTerms:
        if (
            predicted.ndim != 2
            or min(predicted.shape) < 1
            or predicted.shape != target.shape
            or predicted.dtype not in (torch.float32, torch.float64)
            or predicted.dtype != target.dtype
            or predicted.device != target.device
        ):
            raise ValueError("loss needs matching float32/float64 waveforms [B, L] on one device")
        target = target.detach()
        waveform = (predicted - target).abs().mean()
        magnitude = predicted.new_zeros(())
        log_magnitude = predicted.new_zeros(())
        for fft, hop in STFT_RESOLUTIONS:
            spectra = torch.stft(
                torch.cat((predicted, target)),
                n_fft=fft,
                hop_length=hop,
                window=getattr(self, f"hann_{fft}").to(predicted),
                center=True,
                pad_mode="constant",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            pred_stft, ref_stft = spectra.chunk(2)
            pred_mag, ref_mag = pred_stft.abs(), ref_stft.abs()
            magnitude = magnitude + (pred_mag - ref_mag).abs().mean()
            pred_log = pred_mag.clamp_min(SPECTRAL_FLOOR).log()
            ref_log = ref_mag.clamp_min(SPECTRAL_FLOOR).log()
            log_magnitude = log_magnitude + (pred_log - ref_log).abs().mean()

        magnitude = magnitude / len(STFT_RESOLUTIONS)
        log_magnitude = log_magnitude / len(STFT_RESOLUTIONS)
        phase_time = predicted.new_zeros(())
        phase_frequency = predicted.new_zeros(())
        if self.weights.phase_enabled:
            # The last resolution is the native frontend grid. Omit real-only endpoints.
            pred_phase = pred_stft[:, 1:-1] / pred_mag[:, 1:-1].clamp_min(SPECTRAL_FLOOR)
            ref_phase = ref_stft[:, 1:-1] / ref_mag[:, 1:-1].clamp_min(SPECTRAL_FLOOR)
            phase_time = _phase_pair_loss(pred_phase, ref_phase, ref_mag[:, 1:-1], dim=2)
            phase_frequency = _phase_pair_loss(pred_phase, ref_phase, ref_mag[:, 1:-1], dim=1)
        weights = self.weights
        total = (
            weights.waveform * waveform
            + weights.magnitude * magnitude
            + weights.log_magnitude * log_magnitude
            + weights.phase_time * phase_time
            + weights.phase_frequency * phase_frequency
        )
        return ReconstructionTerms(
            total, waveform, magnitude, log_magnitude, phase_time, phase_frequency
        )
