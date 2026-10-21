import pytest
import torch
from torch import nn

from daynight.perceptual_targets import feature_distance


def test_target_features_preserve_prediction_gradient_but_freeze_target():
    features = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 1), nn.ReLU()
    ).requires_grad_(False)
    p = torch.rand(1, 3, 16, 16, requires_grad=True)
    target = torch.zeros_like(p, requires_grad=True)
    loss = feature_distance(p, target, features, (1, 3))
    loss.backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert target.grad is None
    assert all(parameter.grad is None for parameter in features.parameters())
    assert feature_distance(p, p, features, (1, 3)) == 0


def test_target_features_reject_misaligned_images_and_invalid_layers():
    features = nn.Sequential(nn.Identity())
    p = torch.zeros(1, 3, 16, 16)
    with pytest.raises(ValueError):
        feature_distance(p, p[:, :, :8], features, (0,))
    with pytest.raises(ValueError):
        feature_distance(p, p, features, (9,))
