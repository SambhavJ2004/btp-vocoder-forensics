"""In-memory waveform counterfactuals for a pair of explicit patch IDs."""

from dataclasses import dataclass

import torch
from torch import Tensor

from .refinement import SelectiveRefiner
from .spectral import SpectralFrontend


@dataclass(frozen=True)
class PatchEffects:
    patch_ids: tuple[int, int]
    base_loss: float
    first_loss: float
    second_loss: float
    joint_loss: float
    max_round_trip_error: float

    @property
    def independent_benefits(self) -> tuple[float, float]:
        return self.base_loss - self.first_loss, self.base_loss - self.second_loss

    @property
    def joint_benefit(self) -> float:
        return self.base_loss - self.joint_loss

    @property
    def interaction(self) -> float:
        return self.joint_benefit - sum(self.independent_benefits)


@torch.no_grad()
def measure_patch_effects(
    frontend: SpectralFrontend,
    refiner: SelectiveRefiner,
    base: Tensor,
    reference: Tensor,
    patch_ids: tuple[int, int],
) -> PatchEffects:
    """Score base, A, B and A+B by full-length waveform MSE on one example.

    A positive interaction means the joint benefit exceeds the independent
    benefits' sum. Compare neighboring and distant pairs by choosing their IDs.
    Reanalysis checks the waveform round trip, since arbitrary modified spectra
    need not themselves be consistent STFTs.
    """
    if reference.ndim != 2 or reference.shape[0] != 1 or base.ndim != 3 or base.shape[0] != 1:
        raise ValueError("patch-effect diagnostics require exactly one utterance")
    if (
        len(patch_ids) != 2
        or any(type(patch) is not int for patch in patch_ids)
        or patch_ids[0] == patch_ids[1]
    ):
        raise ValueError("patch-effect diagnostics require two distinct integer patch IDs")
    if reference.device != base.device or reference.dtype != base.real.dtype:
        raise ValueError("reference must match the spectrum device and real dtype")
    length = reference.shape[-1]
    first, second = patch_ids
    losses = []
    max_error = 0.0
    for selection in ((), (first,), (second,), patch_ids):
        ids = torch.tensor([selection], dtype=torch.long, device=base.device)
        spectrum = refiner.refine_selected(base, ids)
        waveform = frontend.synthesize(spectrum, length)
        losses.append((waveform - reference).square().mean().item())
        reconstructed = frontend.synthesize(frontend.analyze(waveform), length)
        max_error = max(max_error, (reconstructed - waveform).abs().max().item())
    return PatchEffects(patch_ids, *losses, max_error)
