"""Versioned Torch frontend and inverse transform for unpadded mono batches."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchaudio.functional import melscale_fbanks


@dataclass(frozen=True)
class FrontendConfig:
    version: str = "vocoroute-p-frontend-v1"
    sample_rate: int = 22_050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    mel_floor: float = 1e-5

    @property
    def bins(self) -> int:
        return self.n_fft // 2 + 1


FRONTEND_V1 = FrontendConfig()


def frames_for_length(length: int) -> int:
    if type(length) is not int or length < 1:
        raise ValueError("waveform length must be a positive integer")
    return 1 + length // FRONTEND_V1.hop_length


def validate_spectrum(spectrum: Tensor) -> None:
    if (
        spectrum.ndim != 3
        or spectrum.shape[1] != FRONTEND_V1.bins
        or min(spectrum.shape) < 1
        or spectrum.dtype not in (torch.complex64, torch.complex128)
    ):
        raise ValueError("expected a nonempty complex64 or complex128 spectrum [B, 513, T]")


class SpectralFrontend(nn.Module):
    """Periodic Hann, centered zero padding, unnormalized one-sided FFT.

    The window length equals the FFT length. Mel is natural-log power with a
    Slaney-scale, area-normalized filter bank between DC and Nyquist.
    """

    def __init__(self) -> None:
        super().__init__()
        cfg = FRONTEND_V1
        self.register_buffer("window", torch.hann_window(cfg.n_fft, dtype=torch.float64))
        filters = melscale_fbanks(
            cfg.bins,
            f_min=0.0,
            f_max=cfg.sample_rate / 2,
            n_mels=cfg.n_mels,
            sample_rate=cfg.sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer("mel_filters", filters.transpose(0, 1).contiguous())

    def analyze(self, waveform: Tensor) -> Tensor:
        if (
            waveform.ndim != 2
            or min(waveform.shape) < 1
            or waveform.dtype not in (torch.float32, torch.float64)
        ):
            raise ValueError("expected a nonempty float32 or float64 waveform [B, L]")
        cfg = FRONTEND_V1
        return torch.stft(
            waveform,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            window=self.window.to(waveform),
            center=True,
            pad_mode="constant",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

    def condition(self, spectrum: Tensor) -> Tensor:
        validate_spectrum(spectrum)
        # Endpoints represent signed real coefficients in a one-sided real FFT.
        real = spectrum.real
        power = torch.cat(
            (real[:, :1].square(), spectrum[:, 1:-1].abs().square(), real[:, -1:].square()),
            dim=1,
        )
        mel = self.mel_filters.to(power) @ power
        return mel.clamp_min(FRONTEND_V1.mel_floor).log()

    def synthesize(self, spectrum: Tensor, length: int) -> Tensor:
        validate_spectrum(spectrum)
        if spectrum.shape[-1] != frames_for_length(length):
            raise ValueError("spectrum frames disagree with explicit waveform length")
        cfg = FRONTEND_V1
        spectrum = torch.cat(
            (spectrum[:, :1].real, spectrum[:, 1:-1], spectrum[:, -1:].real), dim=1
        )
        return torch.istft(
            spectrum,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            window=self.window.to(spectrum.real),
            center=True,
            normalized=False,
            onesided=True,
            length=length,
        )
