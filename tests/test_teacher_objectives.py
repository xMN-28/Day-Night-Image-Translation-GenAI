import pytest
import torch

from daynight.teacher_objectives import (
    conditioned_pair,
    highlight_fidelity,
    least_squares_critic,
    least_squares_generator,
    light_footprint_fidelity,
    masked_target_fidelity,
)


def test_highlight_fidelity_focus_and_empty_region():
    target = torch.full((1, 3, 8, 8), -1.0)
    prediction = target.clone().requires_grad_(True)
    assert highlight_fidelity(prediction, target) == 0
    target[:, :, 3, 3] = 1
    focused = highlight_fidelity(prediction, target)
    assert focused.item() == pytest.approx(2)
    focused.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, :, 3, 3].abs().sum() > 0
    assert prediction.grad[:, :, 0, 0].abs().sum() == 0


def test_conditional_pair_preserves_source_and_candidate_gradient():
    source = torch.randn(2, 3, 16, 16)
    candidate = torch.randn_like(source, requires_grad=True)
    pair = conditioned_pair(source, candidate)
    assert torch.equal(pair[:, :3], source)
    pair[:, 3:].sum().backward()
    assert torch.equal(candidate.grad, torch.ones_like(candidate))
    with pytest.raises(ValueError):
        conditioned_pair(source, candidate[:, :, :8])


def test_lsgan_optima_and_finite_gradient():
    assert least_squares_critic(torch.ones(2), torch.zeros(2)) == 0
    assert least_squares_generator(torch.ones(2)) == 0
    fake = torch.zeros(2, requires_grad=True)
    least_squares_generator(fake).backward()
    assert torch.isfinite(fake.grad).all() and fake.grad.abs().sum() > 0


def test_light_footprint_penalizes_blob_expansion_and_matches_exact_target():
    target = torch.full((1, 3, 16, 16), -1.0)
    target[:, :, 7:9, 7:9] = 1
    assert light_footprint_fidelity(target, target) == 0
    small = target.clone()
    small[:, :, 6:10, 6:10] = 0.8
    large = target.clone()
    large[:, :, 4:12, 4:12] = 0.8
    assert light_footprint_fidelity(large, target) > light_footprint_fidelity(small, target)
    large.requires_grad_(True)
    light_footprint_fidelity(large, target).backward()
    assert torch.isfinite(large.grad).all()
    assert large.grad[:, :, 4, 4].sum() > 0  # Descent suppresses false light.


def test_light_footprint_handles_no_target_lights():
    target = torch.full((2, 3, 8, 8), -1.0)
    prediction = target.clone().requires_grad_(True)
    loss = light_footprint_fidelity(prediction, target)
    assert loss == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_region_objective_moves_toward_night_not_daytime_and_freezes_mask():
    prediction = torch.ones(1, 3, 16, 16, requires_grad=True)
    target = torch.full_like(prediction, -1, requires_grad=True)
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, :, :8] = 1
    mask.requires_grad_(True)
    loss = masked_target_fidelity(prediction, target, mask)
    assert loss.item() == pytest.approx(2)
    loss.backward()
    assert prediction.grad[:, :, :, :8].min() > 0  # Descent darkens predicted daytime sky.
    assert prediction.grad[:, :, :, 8:].abs().sum() == 0
    assert mask.grad is None and target.grad is None
    assert masked_target_fidelity(prediction, target, torch.zeros_like(mask)) == 0
