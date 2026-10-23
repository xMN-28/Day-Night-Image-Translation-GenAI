import pytest

from daynight.prepare_inverse_timelapse_regions import STATIC_TOP, camera_training_subset


def test_timelapse_subset_is_camera_training_only_and_correct_clock():
    rows = [{"camera": camera, "split": "train", "day_clock": "2015-11-01T12:00:00",
        "night_clock": f"2015-11-0{day}T00:00:00"} for camera in STATIC_TOP for day in range(1, 7)]
    rows.append({"camera": "heldout", "split": "val"})
    selected = camera_training_subset(rows)
    assert len(selected) == 36
    assert {row["camera"] for row in selected} == set(STATIC_TOP)
    assert all(row["split"] == "train" for row in selected)
    assert selected == camera_training_subset(rows)
    rows[0]["split"] = "val"
    with pytest.raises(ValueError):
        camera_training_subset(rows)


def test_timelapse_subset_rejects_twilight_clock():
    rows = [{"camera": next(iter(STATIC_TOP)), "split": "train", "day_clock": "2015-11-01T18:00:00",
        "night_clock": "2015-11-01T00:00:00"}]
    with pytest.raises(ValueError):
        camera_training_subset(rows)
