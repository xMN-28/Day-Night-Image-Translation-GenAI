from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    dataset = tmp_path / "prepared"
    source.mkdir()
    (dataset / "manifests").mkdir(parents=True)
    rng = np.random.default_rng(7)
    records = {"day": [], "night": []}
    for domain, domain_records in records.items():
        for index in range(4):
            base = 190 if domain == "day" else 35
            array = np.clip(base + rng.normal(0, 25, size=(80, 96, 3)), 0, 255).astype(np.uint8)
            name = f"{domain}_{index}.jpg"
            Image.fromarray(array).save(source / name)
            domain_records.append({"name": name, "path": name, "quality_flags": []})
    for split in ("train", "val", "test"):
        for domain, items in records.items():
            (dataset / "manifests" / f"{split}_{domain}.jsonl").write_text(
                "\n".join(json.dumps(item) for item in items) + "\n", encoding="utf-8"
            )
    (dataset / "dataset.json").write_text(
        json.dumps({"source_root": str(source), "format_version": 1}), encoding="utf-8"
    )
    return dataset
