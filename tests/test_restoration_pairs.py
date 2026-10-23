from pathlib import Path

import pytest

from daynight.audit_restoration_pairs import matching_pairs


def test_pair_matching_is_by_relative_name_not_list_position(tmp_path):
    for relative in ("train/low/a.jpg", "train/low/b.jpg", "train/high/a.jpg", "train/high/b.jpg"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    pairs = matching_pairs(tmp_path, "train/low", "train/high")
    assert [p[0] for p in pairs] == ["a.jpg", "b.jpg"]
    Path(tmp_path / "train/high/b.jpg").rename(tmp_path / "train/high/c.jpg")
    with pytest.raises(ValueError):
        matching_pairs(tmp_path, "train/low", "train/high")


@pytest.mark.parametrize("low", ["eval15/low", "test/low", "validation/train/low", "../train/low", "other/low"])
def test_rejects_nontraining_or_traversal_before_reading(tmp_path, low):
    with pytest.raises(ValueError):
        matching_pairs(tmp_path, low, "train/high")
