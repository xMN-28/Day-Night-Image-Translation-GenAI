import torch

from daynight.train_night_to_day_pilot import noise_robust_loss, perturb_training_input, pilot_loss


def test_zero_sigma_control_same_supervision_and_gradient():
    clean = torch.randn(1, 3, 16, 16, requires_grad=True)
    target = torch.zeros_like(clean)
    total, consistency = noise_robust_loss(clean, clean, target, 2)
    original = pilot_loss(clean, target)
    assert torch.equal(total, original)
    assert consistency == 0
    assert torch.allclose(torch.autograd.grad(total, clean, retain_graph=True)[0],
        torch.autograd.grad(original, clean)[0])


def test_noise_deterministic_resume_independent_and_bounded():
    x = torch.zeros(1, 3, 16, 16)
    rng = torch.Generator().manual_seed(2033)
    saved = rng.get_state()
    first = perturb_training_input(x, 2 / 255, rng)
    rng.set_state(saved)
    assert torch.equal(first, perturb_training_input(x, 2 / 255, rng))
    assert first.min() >= -1 and first.max() <= 1
    assert abs(float(first.std()) - 4 / 255) < .003
    assert perturb_training_input(x, 0, rng) is x


def test_consistency_anchor_is_detached():
    clean = torch.zeros(1, 3, 16, 16, requires_grad=True)
    noisy = torch.ones(1, 3, 16, 16, requires_grad=True)
    _, consistency = noise_robust_loss(clean, noisy, clean.detach(), 2)
    clean_grad, noisy_grad = torch.autograd.grad(consistency, (clean, noisy), allow_unused=True)
    assert clean_grad is None
    assert torch.isfinite(noisy_grad).all() and noisy_grad.abs().sum() > 0
