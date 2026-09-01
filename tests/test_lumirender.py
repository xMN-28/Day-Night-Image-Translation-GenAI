from __future__ import annotations

import torch

from daynight.config import load_config
from daynight.lumirender_losses import paired_perceptual_loss, teacher_factorization_loss
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


def test_lumirender_dark_pixels_have_finite_bf16_gradients() -> None:
    if not torch.cuda.is_available():
        return
    model = LumiRender(base_channels=8, lights=4, latent_dim=4).cuda().train()
    image = -torch.ones(1, 3, 64, 64, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(image, seed=5000)
        loss = output.square().mean()
    loss.backward()
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


def test_teacher_factorization_loss_is_autocast_safe() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semantic_logits = torch.randn(1, 6, 16, 16, device=device, requires_grad=True)
    details = {
        "depth": torch.sigmoid(torch.randn(1, 1, 16, 16, device=device, requires_grad=True)),
        "semantic": torch.sigmoid(semantic_logits),
        "normals": torch.nn.functional.normalize(torch.randn(1, 3, 16, 16, device=device), dim=1),
    }
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        loss = teacher_factorization_loss(
            details,
            torch.rand(1, 1, 16, 16, device=device),
            torch.rand(1, 6, 16, 16, device=device),
            torch.ones(1, device=device),
        )
    loss.backward()
    assert torch.isfinite(loss)
    assert semantic_logits.grad is not None
