from __future__ import annotations

import torch

from daynight.losses import (
    SobelEdgeLoss,
    color_statistics_loss,
    lsgan_discriminator_loss,
    patch_nce_loss,
    regional_illumination_loss,
)
from daynight.models.networks import (
    LocalGlobalDiscriminator,
    MultiScaleDiscriminator,
    PatchDiscriminator,
    ResnetGenerator,
)


def test_generators_and_discriminators_have_expected_shapes() -> None:
    image = torch.randn(1, 3, 64, 64)
    generator = ResnetGenerator(base_channels=8, blocks=2, attention=True)
    translated, features = generator(image, return_features=True)
    assert translated.shape == image.shape
    assert len(features) >= 4
    assert PatchDiscriminator(8)(translated).shape[1] == 1
    outputs = MultiScaleDiscriminator(8)(translated)
    assert len(outputs) == 2
    local_global_outputs = LocalGlobalDiscriminator(8)(translated)
    assert len(local_global_outputs) == 3
    assert local_global_outputs[-1].shape == (1, 1, 1, 1)


def test_all_losses_are_finite_and_differentiable() -> None:
    source = torch.randn(1, 3, 64, 64, requires_grad=True)
    translated = torch.tanh(source * 0.8)
    edge = SobelEdgeLoss()(source, translated)
    features_a = [torch.randn(1, 8, 8, 8, requires_grad=True)]
    features_b = [torch.randn(1, 8, 8, 8, requires_grad=True)]
    nce = patch_nce_loss(features_a, features_b, patches=16)
    discriminator = lsgan_discriminator_loss(torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    total = edge + nce + discriminator
    assert torch.isfinite(total)
    total.backward()
    assert source.grad is not None


def test_color_statistics_loss_is_differentiable_and_domain_sensitive() -> None:
    generated = torch.zeros(1, 3, 16, 16, requires_grad=True)
    matching_target = torch.zeros_like(generated)
    darker_target = torch.full_like(generated, -0.8)
    matching = color_statistics_loss(generated, matching_target)
    different = color_statistics_loss(generated, darker_target)
    assert matching.item() == 0.0
    assert different.item() > matching.item()
    different.backward()
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()


def test_regional_illumination_targets_bright_upper_scene() -> None:
    source = torch.full((1, 3, 32, 32), -0.2)
    source[:, :, :16] = 1.0
    generated = source.clone().requires_grad_(True)
    target_night = torch.full_like(source, -0.7)
    loss = regional_illumination_loss(source, generated, target_night, "day_to_night")
    assert loss.item() > 0
    loss.backward()
    assert generated.grad is not None
    assert generated.grad[:, :, :16].abs().sum() > 0
