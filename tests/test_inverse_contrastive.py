import pytest
import torch

from daynight.train_inverse_contrastive import PatchProjection, patch_contrastive


def test_same_position_beats_permuted_keys():
    q = torch.eye(8)[None].requires_grad_()
    k = q.detach().clone().requires_grad_()
    loss = patch_contrastive(q, k)
    assert loss < patch_contrastive(q, k.roll(1, 1))
    loss.backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is None


def test_projection_shared_coordinates_and_normalization():
    torch.manual_seed(1)
    head = PatchProjection(channels=(4,), width=8)
    feature = torch.rand(1, 4, 3, 3)
    ids = [torch.tensor([0, 3, 5])]
    a = head([feature], ids)[0]
    assert a.shape == (1, 3, 8)
    torch.testing.assert_close(a.norm(dim=-1), torch.ones(1, 3))
    torch.testing.assert_close(a, head([feature], ids)[0])


def test_invalid_contrastive_temperature():
    with pytest.raises(ValueError):
        patch_contrastive(torch.ones(1, 2, 3), torch.ones(1, 2, 3), 0)


def test_default_projection_matches_selected_generator_levels():
    from daynight.models.networks import ResnetGenerator

    generator = ResnetGenerator(blocks=9, attention=True)
    _, features = generator.encode_features(torch.randn(1, 3, 32, 32))
    selected = [features[i] for i in (0, 2, 5)]
    output = PatchProjection()(selected, [torch.tensor([0, 1])] * 3)
    assert all(item.shape == (1, 2, 128) for item in output)


def test_same_group_exactly_recovers_uniform_loss_and_gradients():
    q = torch.randn(1, 8, 4, requires_grad=True)
    k = torch.randn_like(q)
    groups = torch.zeros(1, 8, 4)
    groups[..., 0] = 1
    plain = patch_contrastive(q, k)
    grouped = patch_contrastive(q, k, memberships=groups)
    torch.testing.assert_close(plain, grouped, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(plain, q)[0],
        torch.autograd.grad(grouped, q)[0], rtol=0, atol=0)


def test_cross_group_negatives_reduced_positive_unchanged():
    q = torch.zeros(1, 2, 4, requires_grad=True)
    k = torch.zeros_like(q, requires_grad=True)
    groups = torch.tensor([[[1., 0, 0, 0], [0, 1., 0, 0]]], requires_grad=True)
    grouped = patch_contrastive(q, k, memberships=groups)
    torch.testing.assert_close(grouped, torch.tensor(1.02).log())
    assert grouped < patch_contrastive(q, k)
    grouped.backward()
    assert torch.isfinite(q.grad).all()
    assert k.grad is None and groups.grad is None


def test_invalid_group_partition_rejected():
    with pytest.raises(ValueError):
        patch_contrastive(torch.ones(1, 2, 3), torch.ones(1, 2, 3),
            memberships=torch.zeros(1, 2, 4))
