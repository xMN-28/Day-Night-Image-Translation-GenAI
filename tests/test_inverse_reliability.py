import pytest
import torch

from daynight.inverse_reliability import glare_reliability, reliability_weighted_loss
from daynight.train_night_to_day_pilot import pilot_loss


def test_uniform_reliability_matches_existing_loss_and_gradients():
    torch.manual_seed(1)
    p = torch.rand(2, 3, 32, 48, requires_grad=True)
    t = torch.rand_like(p)
    weighted = reliability_weighted_loss(p, t, torch.ones(2, 1, 32, 48))
    original = pilot_loss(p, t)
    assert weighted.item() == pytest.approx(original.item())
    assert torch.allclose(torch.autograd.grad(weighted, p)[0], torch.autograd.grad(original, p)[0])


def test_glare_relaxation_needs_source_core_and_retained_target_brightness():
    source = torch.full((1, 3, 48, 64), -1.)
    target = torch.ones_like(source)
    assert glare_reliability(source, target).min() == 1
    source[..., 24, 32] = 1
    weights = glare_reliability(source, target)
    assert weights[0, 0, 24, 32] == pytest.approx(.05, abs=1e-6)
    assert weights[0, 0, 0, 0] == 1
    assert not weights.requires_grad
    assert glare_reliability(source, -target).min() == 1


def test_weighted_supervision_is_finite_and_less_restrictive_not_zero():
    p = torch.zeros(1, 3, 32, 48, requires_grad=True)
    target = torch.ones_like(p)
    loss = reliability_weighted_loss(p, target, torch.full((1, 1, 32, 48), .05))
    assert loss.item() == pytest.approx(.6)
    loss.backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
