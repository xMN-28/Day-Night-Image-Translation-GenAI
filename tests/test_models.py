from __future__ import annotations

import torch

from daynight.losses import SobelEdgeLoss, lsgan_discriminator_loss, patch_nce_loss
from daynight.models.networks import MultiScaleDiscriminator, PatchDiscriminator, ResnetGenerator


def test_generators_and_discriminators_have_expected_shapes() -> None:
    image = torch.randn(1, 3, 64, 64)
    generator = ResnetGenerator(base_channels=8, blocks=2, attention=True)
    translated, features = generator(image, return_features=True)
    assert translated.shape == image.shape
    assert len(features) >= 4
    assert PatchDiscriminator(8)(translated).shape[1] == 1
    outputs = MultiScaleDiscriminator(8)(translated)
    assert len(outputs) == 2


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
