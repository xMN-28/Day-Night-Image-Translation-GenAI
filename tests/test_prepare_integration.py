from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from daynight.prepare_data import prepare


def test_prepare_bdd_style_tree(tmp_path: Path) -> None:
    bdd = tmp_path / "bdd"
    images = bdd / "images" / "100k" / "train"
    labels = bdd / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    rng = np.random.default_rng(101)
    annotations = []
    for domain, timeofday, offset in (("day", "daytime", 0), ("night", "night", 100)):
        for index in range(4):
            name = f"{domain}_{index}.jpg"
            array = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
            array[:, :, 0] = (array[:, :, 0].astype(np.uint16) + offset) % 256
            Image.fromarray(array).save(images / name)
            annotations.append({"name": name, "attributes": {"timeofday": timeofday}})
    (labels / "bdd100k_labels_images_train.json").write_text(
        json.dumps(annotations), encoding="utf-8"
    )
    output = tmp_path / "prepared"
    prepare(
        argparse.Namespace(
            bdd_root=str(bdd),
            output=str(output),
            train_count=2,
            val_count=1,
            test_count=1,
            hash_distance=0,
            seed=5,
        )
    )
    metadata = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    assert metadata["counts"] == {"train": 2, "val": 1, "test": 1}
    for split, count in (("train", 2), ("val", 1), ("test", 1)):
        for domain in ("day", "night"):
            lines = (output / "manifests" / f"{split}_{domain}.jsonl").read_text().splitlines()
            assert len(lines) == count
            assert (output / "review" / f"{split}_{domain}.jpg").exists()


def test_prepare_per_image_label_tree(tmp_path: Path) -> None:
    bdd = tmp_path / "bdd100k"
    images = bdd / "images" / "100k" / "train"
    labels = bdd / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    rng = np.random.default_rng(202)
    for domain, timeofday in (("day", "daytime"), ("night", "night")):
        for index in range(3):
            stem = f"{domain}_{index}"
            Image.fromarray(rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)).save(
                images / f"{stem}.jpg"
            )
            (labels / f"{stem}.json").write_text(
                json.dumps({"name": stem, "attributes": {"timeofday": timeofday}}),
                encoding="utf-8",
            )
    output = tmp_path / "prepared"
    prepare(
        argparse.Namespace(
            bdd_root=str(bdd),
            output=str(output),
            train_count=1,
            val_count=1,
            test_count=1,
            hash_distance=0,
            seed=7,
        )
    )
    metadata = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    assert metadata["counts"] == {"train": 1, "val": 1, "test": 1}
