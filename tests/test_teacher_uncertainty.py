import pytest
import torch

from daynight.audit_teacher_uncertainty import variation_metrics


def test_teacher_variation_is_zero_for_identical_outputs():
    samples = torch.zeros(4, 3, 8, 8)
    result = variation_metrics(samples)
    assert result["pairwise_rgb_l1_0to1"] == 0
    assert result["pairwise_bright_mask_iou"] == 1
    assert result["mean_rgb_standard_deviation"] == 0


def test_teacher_variation_detects_different_light_placement():
    samples = torch.zeros(2, 3, 8, 8)
    samples[0, :, 1, 1] = 1
    samples[1, :, 6, 6] = 1
    result = variation_metrics(samples)
    assert result["pairwise_rgb_l1_0to1"] > 0
    assert result["pairwise_bright_mask_iou"] == 0
    with pytest.raises(ValueError):
        variation_metrics(samples[:1])
