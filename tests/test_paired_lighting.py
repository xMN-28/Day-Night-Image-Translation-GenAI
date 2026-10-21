import torch

from daynight.train_sky_gan import critic_real_example, lighting_pyramid_loss


def test_static_loss_ignores_masked_moving_objects_and_is_differentiable():
    prediction = torch.zeros(1, 3, 64, 64, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[..., :19, :13] = 1
    mask = torch.ones(1, 1, 64, 64)
    mask[..., :19, :13] = 0
    assert lighting_pyramid_loss(prediction, target, mask) == 0
    target[..., 32:, 32:] = 0.5
    loss = lighting_pyramid_loss(prediction, target, mask)
    assert loss > 0
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_paired_critic_uses_correct_target_domain_and_rgb_range():
    bdd = torch.full((1, 3, 8, 8), -0.2)
    webcam = torch.full_like(bdd, 0.7, requires_grad=True)
    assert critic_real_example(bdd, webcam, "bdd") is bdd
    assert critic_real_example(bdd, None, "paired") is bdd
    selected = critic_real_example(bdd, webcam, "paired")
    assert torch.allclose(selected, torch.full_like(selected, 0.4))
    assert not selected.requires_grad
