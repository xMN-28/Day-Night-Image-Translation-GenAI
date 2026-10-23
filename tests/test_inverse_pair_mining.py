import pytest
import torch

from daynight.mine_inverse_pairs import deformation_weight, spaced_unique_day_pairs


def test_candidate_sampling_is_training_only_unique_day_and_deterministic():
    rows = [{"split": "train", "day_path": f"day{i//2}.png"} for i in range(12)]
    rows.append({"split": "val", "day_path": "heldout.png"})
    selected = spaced_unique_day_pairs(rows, 4)
    assert len(selected) == 4 and len({r["day_path"] for _, r in selected}) == 4
    assert selected == spaced_unique_day_pairs(rows, 4)
    assert all(r["split"] == "train" for _, r in selected)
    with pytest.raises(ValueError):
        spaced_unique_day_pairs([], 2)


def test_local_deformation_ranking_penalizes_waviness_not_constant_shift():
    flow = torch.ones(1, 2, 16, 24) * 12
    assert torch.equal(deformation_weight(flow), torch.ones(1, 1, 16, 24))
    flow[..., ::2] = -12
    assert deformation_weight(flow).mean() < .01
