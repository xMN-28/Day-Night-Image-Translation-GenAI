import numpy as np
import pytest
import torch

from daynight.prepare_inverse_carla_regions import static_upper_mask
from daynight.train_night_to_day_pilot import static_region_coarse_loss


def test_roi_exact_and_aligned_with_pooling():
    mask = np.asarray(static_upper_mask())
    assert mask.shape == (288, 512)
    assert (mask[:176] == 255).all()
    assert (mask[176:] == 0).all()
    assert 176 % 8 == 0


def test_unreviewed_sizes_rejected():
    with pytest.raises(ValueError):
        static_upper_mask((512, 256))


def test_coarse_supervision_has_no_gradient_in_excluded_road():
    prediction = torch.ones(1, 3, 288, 512, requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.from_numpy(np.asarray(static_upper_mask()).copy())[None, None].float() / 255
    static_region_coarse_loss(prediction, target, mask).backward()
    assert (prediction.grad[..., 176:, :] == 0).all()
    assert (prediction.grad[..., :176, :] != 0).all()
