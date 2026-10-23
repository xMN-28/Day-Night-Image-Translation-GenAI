import pytest

from daynight.audit_darkdriving_pairs import exact_train_pairs
from daynight.index_darkdriving_mirror import summarize_files


def test_explicit_train_pairing_not_val_and_not_order():
    paths = ["root/day/images/train/a.jpg", "root/night/images/train/a.jpg",
        "root/day/images/val/b.jpg", "root/night/images/val/b.jpg",
        "root/day/images/train/c.jpg", "root/night/images/train/d.jpg"]
    result = summarize_files([{"name": p, "bytes": 1} for p in paths])
    assert result["exact_path_matched_train_pairs"]["day_night"] == 1
    assert result["explicit_train_images"] == {"day": 2, "night": 2}
    assert result["total_listed_bytes"] == 6


def test_inventory_rejects_traversal():
    with pytest.raises(ValueError):
        summarize_files([{"name": "../day/images/train/x.jpg", "bytes": 1}])


def test_restoration_pairs_exclude_detection_and_test():
    files = [{"name": f"darkdriving_lle/darkdriving_lle/{split}/{domain}/x_1.jpg", "bytes": 1}
        for split in ("train", "test") for domain in ("day", "night")]
    files += [{"name": "darkdriving_det/Det_data/night/images/train/x_1.jpg", "bytes": 1}]
    result = exact_train_pairs({"complete_listing": True, "version": 2, "files": files})
    assert len(result) == 1 and "/train/" in result[0]["day"]["name"]
    with pytest.raises(ValueError):
        exact_train_pairs({"complete_listing": False, "version": 2, "files": files})
