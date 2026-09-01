from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from daynight.data import UnpairedDayNightDataset
from daynight.lumirender_data import LumiRenderDataset
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


def test_lumirender_supports_external_unpaired_night(
    tiny_dataset: Path, tmp_path: Path
) -> None:
    night_path = tmp_path / "external_night.png"
    Image.new("RGB", (96, 80), color=(4, 8, 16)).save(night_path)
    manifest = tmp_path / "acdc.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "source": "acdc",
                "split": "train",
                "paired": False,
                "night_path": str(night_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = LumiRenderDataset(
        tiny_dataset,
        split="train",
        image_size=64,
        resize_size=72,
        source_manifests=[str(manifest)],
        paired_fraction=1.0,
    )
    sample = dataset[0]
    assert sample["source_name"] == "acdc"
    assert float(sample["aligned"]) == 0.0
    assert sample["night"].shape == (3, 64, 64)
