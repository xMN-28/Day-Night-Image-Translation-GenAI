import torch

from daynight.audit_inverse_degradations import (
    color_invariant,
    degradation_memberships,
    illuminance_clusters,
)


def test_invariant_finite_black_and_uniform_color():
    for level in (0, .5, 1):
        response = color_invariant(torch.full((1, 3, 12, 16), float(level)))
        assert torch.isfinite(response).all()
        assert response.max() < 1e-4


def test_achromatic_scale_change_has_small_spectral_invariant():
    brightness = torch.linspace(.1, .8, 24).view(1, 1, 1, 24).expand(1, 1, 16, 24)
    rgb = brightness * torch.tensor([.8, .6, .4])[None, :, None, None]
    assert color_invariant(rgb).max() < 1e-4


def test_spectral_chromatic_change_has_nonzero_response():
    rgb = torch.full((1, 3, 16, 24), .4)
    rgb[:, 0, :, 12:] = .8
    assert color_invariant(rgb).max() > .01


def test_soft_memberships_partition_and_determinism():
    rgb = torch.rand((2, 3, 16, 24), generator=torch.Generator().manual_seed(12))
    masks, invariant, labels, centers, degenerate, _ = degradation_memberships(rgb)
    assert torch.allclose(masks.sum(1), torch.ones_like(masks[:, 0]), atol=1e-6)
    assert masks.min() >= 0 and masks.max() <= 1
    assert torch.equal(masks, degradation_memberships(rgb)[0])
    assert labels.shape == (2, 16, 24) and invariant.shape == (2, 1, 16, 24)
    assert (centers[:, 1:] >= centers[:, :-1]).all()
    assert degenerate == [False, False]


def test_constant_image_reports_degenerate_clustering():
    _, _, degenerate = illuminance_clusters(torch.zeros(1, 3, 12, 16))
    assert degenerate == [True]
