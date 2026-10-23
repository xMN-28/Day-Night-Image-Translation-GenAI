import pytest

from daynight.prepare_inverse_loli_regions import select_unique_training


def test_group_selection_does_not_duplicate_targets_and_excludes_holdout_similarity():
    rows = [{"split": "train", "target_sha256": str(i), "exclude_bdd_holdout_similarity": i == 0,
        "name": f"{level}_{i}.jpg"} for i in range(10) for level in ("dense", "light", "moderate")]
    index = {"audit_only": True, "records": rows}
    chosen = select_unique_training(index, 6)
    assert len({r["target_sha256"] for r in chosen}) == 6
    assert all(r["target_sha256"] != "0" for r in chosen)
    assert chosen == select_unique_training(index, 6)
    assert [r["name"].split("_")[0] for r in chosen] == ["dense", "moderate", "light"] * 2


def test_rejects_any_heldout_row_before_selection():
    with pytest.raises(ValueError):
        select_unique_training({"audit_only": True, "records": [{"split": "val"}]}, 1)
