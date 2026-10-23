import torch

from daynight.align_lumirender_pairs import _warp
from daynight.audit_inverse_alignment import inverse_alignment_confidence


def test_day_is_sampled_in_night_coordinates_not_reversed():
    day = torch.arange(8).float()[None, None, None].expand(1, 1, 4, 8)
    forward = torch.zeros(1, 2, 4, 8)
    forward[:, 0] = 2
    warped, valid = _warp(day, forward)
    assert torch.allclose(warped[..., :6], day[..., 2:], atol=1e-5)
    assert valid[..., :6].all() and not valid[..., 6:].any()
    backward = -forward
    confidence, error = inverse_alignment_confidence(forward, backward)
    assert torch.allclose(confidence[..., :6], torch.ones_like(confidence[..., :6]))
    assert error[..., :6].max() == 0
    assert confidence[..., 6:].max() == 0


def test_inconsistent_flow_does_not_get_high_confidence():
    forward = torch.zeros(1, 2, 4, 8)
    backward = torch.ones_like(forward) * 10
    confidence, _ = inverse_alignment_confidence(forward, backward)
    assert confidence.max() < .001
