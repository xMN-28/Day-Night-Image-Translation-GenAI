import torch

from daynight.sky_refinement import (
    SkyStudent,
    compose_night,
    connected_sky_support,
    phase_average,
    refine_sky_alpha,
)


def test_phase_average_has_no_spatial_shift():
    x = torch.rand(1, 3, 36, 52)
    assert torch.allclose(phase_average(torch.nn.Identity(), x), x, atol=2e-7)


def test_disconnected_reflection_is_excluded_without_removing_open_sky():
    p = torch.zeros(1, 1, 64, 96)
    p[:, :, :20] = 0.95
    p[:, :, 50:60, 70:80] = 0.95
    filtered = connected_sky_support(p)
    assert filtered[0, 0, 10, 10] > 0.9
    assert filtered[0, 0, 55, 75] == 0


def test_no_sky_is_exact_identity_to_existing_generator():
    source, base = torch.rand(1, 3, 64, 96), torch.rand(1, 3, 64, 96)
    alpha = refine_sky_alpha(source, torch.zeros(1, 1, 64, 96), torch.ones(1, 1, 64, 96))
    assert torch.count_nonzero(alpha) == 0
    assert torch.allclose(compose_night(source, base, alpha), base)


def test_vegetation_guidance_preserves_non_sky_facade_exactly():
    source = torch.rand(1, 3, 64, 96)
    base = torch.rand_like(source)
    alpha = torch.zeros(1, 1, 64, 96)
    alpha[..., :25, :48] = 1
    vegetation = torch.zeros_like(alpha)
    result = compose_night(source, base, alpha, vegetation=vegetation)
    # A facade directly beside the sky must not be darkened by sky dilation.
    assert torch.allclose(result[..., :25, 50:60], base[..., :25, 50:60])


def test_mixture_recovers_sky_holes_but_preserves_dark_wire():
    source = torch.full((1, 3, 96, 128), 0.9)
    source[:, :, :70, 60:62] = 0.1
    source[:, :, 70:] = 0.2
    semantic = torch.zeros(1, 1, 96, 128)
    semantic[:, :, :70, :40] = 1
    vegetation = torch.zeros_like(semantic)
    vegetation[:, :, :70, 40:] = 1
    alpha = refine_sky_alpha(source, semantic, vegetation)
    assert alpha[0, 0, 25, 85] > 0.8
    assert alpha[0, 0, 25, 60] < 0.2
    result = compose_night(source, source * 0.6, alpha)
    assert result[0, :, 25, 85].mean() < 0.15
    assert torch.isfinite(result).all()


def test_student_odd_shape_and_renderer_gradients():
    torch.set_num_threads(2)
    model = SkyStudent(channels=8)
    source = torch.rand(1, 3, 33, 49)
    logits = model(source)
    assert logits.shape == (1, 1, 33, 49)
    result = compose_night(source, source * 0.4, logits.sigmoid())
    result.mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
