"""TRAIN-only LoLI photographic-target auxiliary cache; synthetic dark inputs.

No held-out image pixels, outside editor or automatic visual approval.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .turbo_pilot import digest
from .utils import atomic_json_dump


def select_unique_training(index, count):
    if not 1 <= count <= 256 or not index.get("audit_only"):
        raise ValueError("Require audited index and1..256groups")
    groups = defaultdict(list)
    for row in index["records"]:
        if row["split"] != "train":
            raise ValueError("TRAIN only")
        if not row["exclude_bdd_holdout_similarity"]:
            groups[row["target_sha256"]].append(row)
    keys = sorted(groups)
    if len(keys) < count:
        raise ValueError("Insufficient distinct target groups")
    # Sorted hash sampling is independent of appearance or development outputs.
    # One illumination variant per target; allthreevariants never cross splits.
    selected = []
    for number, i in enumerate(np.linspace(0, len(keys) - 1, count, dtype=int)):
        options = groups[keys[int(i)]]
        preferred = ("dense", "moderate", "light")[number % 3]
        selected.append(next((r for r in options if Path(r["name"]).stem.startswith(preferred + "_")), options[0]))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("E:/LumiCycleCache/teacher_targets/LoLI_train_index_20261022.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=96)
    parser.add_argument("--approve-reviewed", action="store_true")
    args = parser.parse_args()
    manifest_path = args.output / "regions.json"
    if args.approve_reviewed:
        cache = json.loads(manifest_path.read_text())
        if cache["approved_after_visual_review"]:
            raise FileExistsError("Keep earlier audit approval")
        if cache["test_used"] or cache["lockbox_used"]:
            raise ValueError("TRAIN only")
        for row in cache["records"]:
            for key in ("input", "target", "confidence"):
                if digest(Path(row[key])) != row[f"{key}_sha256"]:
                    raise ValueError("Reviewed bytes changed")
        cache["approved_after_visual_review"] = True
        cache["visual_review_scope"] = "All prepared pair rows opened before approval; predictions not yet accepted"
        atomic_json_dump(cache, manifest_path)
        return
    if args.output.exists():
        raise FileExistsError("Keep existing data")
    index = json.loads(args.index.read_text())
    # Verify the prior overlap metadata has not silently changed.
    for item in index["bdd_metadata_references"]:
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError("Protected split metadata changed")
    selected = select_unique_training(index, args.count)
    args.output.mkdir(parents=True)
    records, panels = [], []
    for number, row in enumerate(selected):
        if digest(Path(row["target"])) != row["target_sha256"]:
            raise ValueError("Original target changed")
        with Image.open(row["input"]) as original, Image.open(row["target"]) as reference:
            if original.size != reference.size:
                raise ValueError("Pair dimensions differ")
            size = (512, max(32, round(512 * original.height / original.width / 4) * 4))
            source = original.convert("RGB").resize(size, Image.Resampling.LANCZOS)
            target = reference.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        folder = args.output / f"pair_{number:03d}"
        folder.mkdir()
        paths = {"input": folder / "SYNTHETIC_low.png", "target": folder / "REAL_day.png", "confidence": folder / "ALL_VALID.png"}
        source.save(paths["input"])
        target.save(paths["target"])
        Image.new("L", size, 255).save(paths["confidence"])
        record = {"index": 400000 + number, "split": "train", "original_name": row["name"],
            "target_group_sha256": row["target_sha256"], "original_input_sha256": digest(Path(row["input"])),
            "original_input": row["input"], "original_target": row["target"], "size": list(size),
            "source_label": "SYNTHETIC darkened TRAIN street", "target_label": "REAL daytime TRAIN photograph"}
        for key, path in paths.items():
            record[key], record[f"{key}_sha256"] = str(path.resolve()), digest(path)
        records.append(record)
        panel = Image.new("RGB", (512, 278), "#101010")
        for j, (label, im) in enumerate((("SYNTH low TRAIN", source), ("REAL day TRAIN", target))):
            panel.paste(im.resize((256, 256), Image.Resampling.LANCZOS), (256 * j, 22))
            ImageDraw.Draw(panel).text((256 * j + 4, 4), f"{number}:{row['name']} {label}", fill="white")
        panels.append(panel)
    for start in range(0, len(panels), 16):
        page = panels[start:start + 16]
        sheet = Image.new("RGB", (2048, 278 * ((len(page) + 3) // 4)), "#101010")
        for j, panel in enumerate(page):
            sheet.paste(panel, (512 * (j % 4), 278 * (j // 4)))
        sheet.save(args.output / f"TRAIN_pairs_{start:03d}.jpg", quality=96)
    atomic_json_dump({"approved_after_visual_review": False, "test_used": False, "lockbox_used": False,
        "source": index["source"], "license": index["license"], "input_origin": "synthetic_darkening",
        "target_origin": "real street photograph, not external generator",
        "index_sha256": digest(args.index), "selection": "Evenly spaced sorted target SHA groups; cyclic dense/moderate/light preference",
        "confidence_semantics": "All pixels paired; NOT semantic confidence or proof of real-night correspondence",
        "records": records}, manifest_path)
    print(f"Prepared{len(records)}distinct target groups; inspect all sheets before approval")


if __name__ == "__main__":
    main()
