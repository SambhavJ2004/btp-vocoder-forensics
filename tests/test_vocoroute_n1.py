"""Synthetic contracts for VocoRoute-P Phase N1."""

import math
from dataclasses import FrozenInstanceError

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from vocoroute.diagnostics import measure_patch_effects  # noqa: E402
from vocoroute.model import VocoRoute  # noqa: E402
from vocoroute.refinement import (  # noqa: E402
    SelectiveRefiner,
    apply_residual,
    patch_count,
    top_k_patches,
)
from vocoroute.spectral import FRONTEND_V1, SpectralFrontend  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def signal(kind, length, dtype=torch.float32):
    samples = torch.arange(length, dtype=dtype)
    if kind == "sinusoid":
        return (0.3 * torch.sin(2 * math.pi * 440 * samples / 22_050))[None]
    if kind == "transient":
        burst = (0.25 * torch.cos(samples) * torch.exp(-((samples - length / 2) / 20) ** 2))
        return burst[None]
    waveform = torch.zeros((1, length), dtype=dtype)
    if kind == "impulse":
        waveform[0, [0, length - 1]] = 0.75
    return waveform


def spectrum(frames, dtype=torch.float32, batch=1):
    generator = torch.Generator().manual_seed(19)
    interior = torch.randn((batch, 511, frames, 2), generator=generator, dtype=dtype)
    endpoints = torch.full((batch, 1, frames), -0.25, dtype=dtype).to(
        torch.complex128 if dtype == torch.float64 else torch.complex64
    )
    return torch.cat((endpoints, torch.view_as_complex(interior), -endpoints), dim=1)


@pytest.mark.parametrize("kind", ["impulse", "sinusoid", "silence", "transient"])
@pytest.mark.parametrize("length", [1, 255, 256, 1023, 1024, 1025, 4097])
def test_frontend_round_trip_frame_grid_and_length(kind, length):
    frontend = SpectralFrontend()
    waveform = signal(kind, length)
    analysis = frontend.analyze(waveform)
    frames = 1 + length // 256
    assert analysis.shape == (1, 513, frames)
    assert torch.count_nonzero(analysis[:, [0, -1]].imag) == 0
    mel = frontend.condition(analysis)
    assert mel.shape == (1, 80, frames)
    assert torch.isfinite(mel).all()
    if kind == "silence":
        torch.testing.assert_close(mel, torch.full_like(mel, math.log(1e-5)))
    reconstructed = frontend.synthesize(analysis, length)
    assert reconstructed.shape == (1, length)
    torch.testing.assert_close(reconstructed, waveform, atol=3e-7, rtol=2e-6)


def test_frontend_scaling_endpoint_policy_and_frozen_config():
    frontend = SpectralFrontend()
    waveform = signal("sinusoid", 2049)
    analysis = frontend.analyze(waveform)
    interior = frontend.condition(analysis) > math.log(1e-5) + 0.1
    doubled = frontend.condition(frontend.analyze(2 * waveform))
    expected = frontend.condition(analysis) + math.log(4)
    torch.testing.assert_close(doubled[interior], expected[interior], atol=2e-6, rtol=0)
    invalid_endpoints = analysis.clone()
    invalid_endpoints[:, [0, -1]] += 12j
    torch.testing.assert_close(
        frontend.synthesize(invalid_endpoints, 2049), frontend.synthesize(analysis, 2049)
    )
    with pytest.raises(ValueError, match="frames"):
        frontend.synthesize(analysis, 256)
    with pytest.raises(ValueError, match="length"):
        frontend.synthesize(analysis, 0)
    with pytest.raises(FrozenInstanceError):
        FRONTEND_V1.sample_rate = 16_000


def test_ties_budgets_and_score_validation():
    scores = torch.tensor([[0.5, -1.0, 0.5, 3.0], [1.0, 1.0, -0.5, 1.0]])
    expected = [[3, 0, 2], [0, 1, 3]]
    assert top_k_patches(scores, 3).tolist() == expected
    assert top_k_patches(scores, 3).tolist() == expected
    assert top_k_patches(scores, 0).shape == (2, 0)
    assert top_k_patches(scores, 4).tolist() == [[3, 0, 2, 1], [0, 1, 3, 2]]
    for bad in (-1, 5, True, 1.5):
        with pytest.raises(ValueError, match="budget"):
            top_k_patches(scores, bad)
    with pytest.raises(ValueError, match="finite"):
        top_k_patches(torch.tensor([[1.0, float("nan")]]), 1)


def test_dispatch_executes_only_ranked_tiles_for_each_batch_row():
    base = spectrum(17, batch=2)
    refiner = SelectiveRefiner()
    seen = []
    hook = refiner.expert.register_forward_pre_hook(lambda _, args: seen.append(args[0].shape))
    scores = torch.arange(patch_count(17), dtype=torch.float32).repeat(2, 1)
    scores[1].neg_()
    assert refiner(base, None, 0).spectrum is base
    assert seen == []
    sparse = refiner(base, scores, 2)
    assert sparse.patch_ids.tolist() == [[31, 30], [0, 1]]
    assert seen == [(4, 8, 28, 48)]
    torch.testing.assert_close(sparse.spectrum[0, 1:100], base[0, 1:100], rtol=0, atol=0)
    torch.testing.assert_close(sparse.spectrum[1, 400:500], base[1, 400:500], rtol=0, atol=0)
    with pytest.raises(ValueError, match="scores"):
        refiner(base, scores[:, :-1], 2)
    with pytest.raises(ValueError, match="distinct"):
        refiner.refine_selected(base, torch.tensor([[1, 1], [0, 1]]))
    hook.remove()


def test_selected_expert_gets_true_neighbors_and_zeros_only_outside_canvas():
    refiner = SelectiveRefiner().double()
    base = torch.ones((1, 513, 17), dtype=torch.complex128)
    base[0, 32, 15] = -2j
    base[0, 33, 15] = 3j
    base[0, 33, 16] = -4 + 0j
    seen = []
    hook = refiner.expert.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    with torch.no_grad():
        refiner.refine_selected(base, torch.tensor([[0, 17]]))
    hook.remove()
    first, later = seen[0].unbind()
    assert first[:, :6].count_nonzero() == 0
    assert first[:, :, :8].count_nonzero() == 0
    assert later[:, 7:].count_nonzero() == 0
    # Frame 15 is in the leading context of the next time tile. Its neighbor
    # in bin 32 provides the previous phase for bin 33 across the frequency seam.
    assert later[0, 5, 7] == pytest.approx(math.log(2))
    assert later[0, 5, 8] == pytest.approx(math.log(3))
    assert later[6, 5, 8] == pytest.approx(-1)
    assert later[3, 6, 8] == pytest.approx(1)
    torch.testing.assert_close(first[:, 21:23, 39:41], later[:, 5:7, 7:9])


@pytest.mark.parametrize("frames", [1, 16, 17, 31, 33])
def test_dense_tiled_parity_at_both_edges_and_interior(frames):
    generator = torch.Generator().manual_seed(22)
    refiner = SelectiveRefiner().double()
    with torch.no_grad():
        for param in refiner.parameters():
            param.uniform_(-0.11, 0.11, generator=generator)
        base = spectrum(frames, torch.float64, batch=2)
        scores = torch.rand((2, patch_count(frames)), generator=generator, dtype=torch.float64)
        tiled = refiner(base, scores, scores.shape[1]).spectrum
        dense = refiner.dense(base)
    torch.testing.assert_close(tiled, dense, rtol=2e-12, atol=2e-12)
    assert (dense - base).abs().max() > 1e-4
    torch.testing.assert_close(tiled[:, [0, -1]], base[:, [0, -1]], rtol=0, atol=0)


class ConstantExpert(torch.nn.Module):
    def forward(self, features):
        gain = torch.full_like(features[:, 0], 0.2)
        phase = torch.full_like(gain, 0.35)
        return torch.stack((gain, phase), dim=1)


def test_taper_is_partition_of_unity_with_fade_to_base_and_no_grid_seams():
    frames = 33  # one real frame in the last time tile
    refiner = SelectiveRefiner()
    refiner.expert = ConstantExpert()
    base = torch.ones((1, 513, frames), dtype=torch.complex64)
    all_ids = torch.arange(patch_count(frames))[None]
    combined = refiner.refine_selected(base, all_ids)
    polar = torch.polar(torch.tensor(math.exp(0.2)), torch.tensor(0.35))
    torch.testing.assert_close(combined[:, 1:-1], torch.full_like(base[:, 1:-1], polar))
    torch.testing.assert_close(combined, refiner.dense(base))
    single = refiner.refine_selected(base, torch.tensor([[0]]))
    time_ramp = (1 + torch.cos(math.pi * (torch.arange(4) + 0.5) / 4)) / 2
    freq_ramp = (1 + torch.cos(math.pi * (torch.arange(8) + 0.5) / 8)) / 2
    time_fade = torch.cat((torch.ones(14), time_ramp))
    freq_fade = torch.cat((torch.ones(28), freq_ramp))
    weights = freq_fade[:, None] * time_fade[None]
    torch.testing.assert_close(
        single[0, 1:37, :18], torch.polar((0.2 * weights).exp(), 0.35 * weights)
    )
    assert torch.equal(single[0, 37:, :], base[0, 37:, :])
    assert torch.equal(single[0, :, 18:], base[0, :, 18:])


def test_phase_only_preserves_spectral_magnitude_and_endpoints():
    base = spectrum(4, torch.float64)
    phase = torch.linspace(-math.pi / 4, math.pi / 4, 511, dtype=torch.float64)
    phase = phase.expand(1, 4, 511)
    corrected = apply_residual(base, torch.stack((torch.zeros_like(phase), phase), dim=1))
    torch.testing.assert_close(corrected.abs(), base.abs(), rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(corrected[:, [0, -1]], base[:, [0, -1]], rtol=0, atol=0)
    assert not torch.equal(base, corrected)
    with pytest.raises(ValueError, match="residual shape"):
        apply_residual(base, phase)


@pytest.mark.parametrize("kind,length", [("silence", 1), ("transient", 1025)])
def test_end_to_end_gradients_shapes_bounds_and_input_preservation(kind, length):
    model = VocoRoute()
    waveform = signal(kind, length).requires_grad_()
    original = waveform.detach().clone()
    analysis = model.frontend.analyze(waveform)
    mel = model.frontend.condition(analysis)
    mel.retain_grad()
    scores = torch.zeros((1, patch_count(mel.shape[-1])))
    expert_output = []
    hook = model.refiner.expert.register_forward_hook(
        lambda _, args, out: expert_output.append(out)
    )
    output = model(mel, length, scores=scores, k=2)
    hook.remove()
    assert output.waveform.shape == waveform.shape
    assert output.spectrum.shape == analysis.shape
    assert torch.count_nonzero(output.spectrum[:, [0, -1]].imag) == 0
    assert torch.isfinite(output.waveform).all()
    assert expert_output[0][:, 0].abs().max() <= 0.35
    assert expert_output[0][:, 1].abs().max() <= math.pi / 4
    output.waveform.square().mean().backward()
    for grad in [waveform.grad, mel.grad, *(param.grad for param in model.parameters())]:
        assert grad is not None
        assert torch.isfinite(grad).all()
    assert torch.count_nonzero(model.refiner.expert.head.weight.grad) > 0
    assert torch.equal(waveform.detach(), original)
    assert torch.equal(scores, torch.zeros_like(scores))


def test_selected_waveform_gradient_agrees_with_central_difference():
    frontend = SpectralFrontend()
    refiner = SelectiveRefiner().double()
    source = signal("transient", 1025, torch.float64)
    base = frontend.analyze(source)
    ids = torch.tensor([[4, 5]])
    direction = torch.zeros_like(base)
    direction[:, 164, 2] = 0.37 + 0.73j

    def value(shift):
        refined = refiner.refine_selected(base + shift * direction, ids)
        return frontend.synthesize(refined, 1025).square().mean()

    shift = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    analytic, = torch.autograd.grad(value(shift), shift)
    with torch.no_grad():
        finite_difference = (value(1e-5) - value(-1e-5)) / (2e-5)
    assert analytic.abs() > 1e-8
    torch.testing.assert_close(analytic, finite_difference, atol=2e-10, rtol=2e-5)


def test_all_tile_backward_matches_dense_backward():
    frontend = SpectralFrontend()
    refiner = SelectiveRefiner().double()
    base = spectrum(17, torch.float64).requires_grad_()
    ids = torch.arange(patch_count(17))[None]
    leaves = (base, *refiner.parameters())
    dense_waveform = frontend.synthesize(refiner.dense(base), 4097)
    tiled_waveform = frontend.synthesize(refiner.refine_selected(base, ids), 4097)
    dense_grads = torch.autograd.grad(dense_waveform.square().sum(), leaves)
    tiled_grads = torch.autograd.grad(tiled_waveform.square().sum(), leaves)
    for dense_grad, tiled_grad in zip(dense_grads, tiled_grads, strict=True):
        torch.testing.assert_close(dense_grad, tiled_grad, atol=2e-12, rtol=2e-10)


def test_interaction_diagnostic_uses_actual_waveforms_and_leaves_inputs_intact():
    length = 4097
    frontend = SpectralFrontend()
    refiner = SelectiveRefiner().double()
    refiner.expert = ConstantExpert()
    reference = signal("transient", length, torch.float64)
    base = frontend.analyze(reference)
    source_before = base.clone()
    reference_before = reference.clone()
    # The burst is centered around FFT bin 163, near the boundary at bin 161.
    effects = measure_patch_effects(frontend, refiner, base, reference, (4, 5))
    expected_losses = []
    for selection in ((4,), (5,), (4, 5)):
        after = refiner.refine_selected(base, torch.tensor([selection]))
        waveform = frontend.synthesize(after, length)
        expected_losses.append((waveform - reference).square().mean().item())
    assert effects.patch_ids == (4, 5)
    assert effects.base_loss < 1e-25
    assert effects.first_loss == pytest.approx(expected_losses[0], abs=1e-15)
    assert effects.second_loss == pytest.approx(expected_losses[1], abs=1e-15)
    assert effects.joint_loss == pytest.approx(expected_losses[2], abs=1e-15)
    assert effects.joint_benefit == pytest.approx(-expected_losses[2], abs=1e-15)
    assert effects.interaction == pytest.approx(sum(expected_losses[:2]) - expected_losses[2])
    assert abs(effects.interaction) > 1e-10
    assert effects.max_round_trip_error < 2e-14
    assert torch.equal(base, source_before)
    assert torch.equal(reference, reference_before)
