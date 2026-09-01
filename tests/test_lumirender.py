from __future__ import annotations

import torch

from daynight.config import load_config
from daynight.lumirender_losses import paired_perceptual_loss
from daynight.models import LumiRender, gaussian_blur, linear_to_srgb, srgb_to_linear
from daynight.models.networks import build_models


def test_srgb_linear_round_trip() -> None:
    image = torch.linspace(0, 1, 257).view(1, 1, 1, -1).repeat(1, 3, 4, 1)
    restored = linear_to_srgb(srgb_to_linear(image))
    assert torch.allclose(restored, image, atol=2e-6)


def test_gaussian_blur_preserves_shape_and_constant_field() -> None:
    image = torch.ones(2, 3, 17, 19)
    blurred = gaussian_blur(image, sigma=2.0)
    assert blurred.shape == image.shape
    assert torch.allclose(blurred, image, atol=1e-6)


def test_lumirender_is_seeded_bounded_and_differentiable() -> None:
    torch.manual_seed(4)
    model = LumiRender(base_channels=8, lights=4, latent_dim=4)
    image = torch.rand(1, 3, 64, 64) * 2 - 1
    first, details = model(image, seed=17, return_details=True)
    repeated = model(image, seed=17)
    alternative = model(image, seed=18)

    assert first.shape == image.shape
    assert torch.isfinite(first).all()
    assert torch.equal(first, repeated)
    assert not torch.equal(first, alternative)
    assert details["correction"].abs().max() <= 0.030001
    assert ((details["light_centers"] >= -0.15) & (details["light_centers"] <= 1.15)).all()

    first.mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_lumirender_config_builds_one_generator_and_night_critic() -> None:
    config = load_config("configs/lumirender.yaml")
    models = build_models(config)
    assert set(models) == {"G_day_night", "D_night"}
    assert isinstance(models["G_day_night"], LumiRender)


def test_paired_perceptual_loss_respects_alignment_confidence() -> None:
    generated = torch.ones(1, 3, 32, 32)
    target = torch.zeros_like(generated)
    aligned = torch.ones(1)
    assert paired_perceptual_loss(
        generated, target, torch.zeros(1, 1, 32, 32), aligned
    ).item() == 0
    assert paired_perceptual_loss(
        generated, target, torch.ones(1, 1, 32, 32), aligned
    ).item() > 0
