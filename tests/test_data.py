from __future__ import annotations

from pathlib import Path

from daynight.data import UnpairedDayNightDataset
from daynight.prepare_data import Candidate, choose_splits, duplicate_groups


def test_dataset_loads_both_domains(tiny_dataset: Path) -> None:
    dataset = UnpairedDayNightDataset(tiny_dataset, split="train", image_size=64, resize_size=72)
    sample = dataset[0]
    assert sample["day"].shape == (3, 64, 64)
    assert sample["night"].shape == (3, 64, 64)
    assert -1 <= float(sample["day"].min()) <= float(sample["day"].max()) <= 1


def test_duplicate_groups_never_cross_splits() -> None:
    candidates = [
        Candidate(str(i), str(i), "day", "train", value, [])
        for i, value in enumerate([0, 0, 1 << 63, 1 << 63, 0xAAAA, 0x5555])
    ]
    groups = duplicate_groups(candidates, max_distance=0)
    splits = choose_splits(candidates, groups, (2, 2, 2), seed=1)
    location = {item.name: split for split, items in splits.items() for item in items}
    assert location["0"] == location["1"]
    assert location["2"] == location["3"]
