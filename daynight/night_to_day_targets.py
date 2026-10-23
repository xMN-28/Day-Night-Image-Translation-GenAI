"""Bounded EXTERNAL night→day pseudo-target pilot, kept apart from production."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import read_manifest
from .turbo_pilot import digest, fit_input, load_official
from .utils import atomic_json_dump, tensor_to_pil


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", default=64, type=int)
    parser.add_argument("--suite", default=Path("outputs/night_to_day_experimental/fixed_suite_v1/suite.json"), type=Path)
    parser.add_argument("--cache-root", default=Path("E:/LumiCycleCache"), type=Path)
    args = parser.parse_args()
    if args.count < 1 or args.count > 512:
        raise ValueError("Bounded pilot count must be1..512")
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Preserve previous cache; choose a new output directory")
    torch.set_num_threads(6)
    suite = json.loads(args.suite.read_text())
    if suite["test_used"] or suite["split"] != "validation night":
        raise ValueError("Require frozen validation-night suite")
    train = read_manifest("data/bdd100k_daynight", "train", "night")
    val = read_manifest("data/bdd100k_daynight", "val", "night")
    ids = np.linspace(0, len(train) - 1, args.count, dtype=int).tolist()
    items = [("train", i, train[i]) for i in ids]
    for row in suite["groups"]["development"]:
        i = row["index"]
        if Path(row["path"]).resolve() != val[i].path.resolve() or digest(val[i].path) != row["sha256"]:
            raise ValueError("Frozen source changed")
        items.append(("val", i, val[i]))
    identities = {split: {str(r.path.resolve()) for s, _, r in items if s == split} for split in ("train", "val")}
    if identities["train"] & identities["val"]:
        raise ValueError("Source split overlap")
    args.output.mkdir(parents=True, exist_ok=True)
    model, checkpoint, commit = load_official(args.cache_root, "night_to_day")
    manifest = {"created_utc": dt.datetime.now(dt.UTC).isoformat(), "direction": "night_to_day",
        "teacher": "Official EXTERNAL CycleGAN-Turbo night2day", "teacher_sha256": digest(checkpoint),
        "teacher_checkpoint": str(checkpoint), "source_commit": commit, "precision": "FP16", "edge": 512,
        "seed_base": 5500, "test_used": False, "suite_sha256": digest(args.suite),
        "teacher_pretraining_overlap": "unknown", "records": [],
        "warning": "Synthetic pseudo-pairs, not real aligned daytime truth. Training selection requires visual target audit."}
    panels = {"train": [], "val": []}
    for split, i, record in items:
        folder = args.output / split
        folder.mkdir(exist_ok=True)
        image, x, crop = fit_input(Image.open(record.path), 512)
        seed = 5500 + i + (10000 if split == "val" else 0)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        prediction = model(x.cuda().half())
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"Nonfinite target {split}:{i}")
        target = tensor_to_pil(prediction).crop(crop)
        source_path, target_path = folder / f"{i:05d}_source.png", folder / f"{i:05d}_EXTERNAL_target.png"
        image.save(source_path); target.save(target_path)
        manifest["records"].append({"split": split, "index": i, "source": str(record.path),
            "source_sha256": digest(record.path), "seed": seed, "input": str(source_path.resolve()),
            "target": str(target_path.resolve()), "target_sha256": digest(target_path), "size": target.size})
        small = Image.new("RGB", (512, 164), "#101010")
        for j, (label, im) in enumerate([("Night source", image), ("EXTERNAL day target", target)]):
            thumb = im.copy(); thumb.thumbnail((256, 144), Image.Resampling.LANCZOS)
            small.paste(thumb, (j * 256, 20))
            ImageDraw.Draw(small).text((j * 256 + 4, 3), f"{split}:{i} {label}", fill="white")
        panels[split].append(small)
        if len(manifest["records"]) % 16 == 0:
            print(f"Cached {len(manifest['records'])}/{len(items)} pseudo-pairs", flush=True)
    for split, entries in panels.items():
        for start in range(0, len(entries), 16):
            page = entries[start:start + 16]
            sheet = Image.new("RGB", (2048, 164 * ((len(page) + 3) // 4)), "#101010")
            for j, panel in enumerate(page):sheet.paste(panel, ((j % 4) * 512, (j // 4) * 164))
            sheet.save(args.output / f"{split}_targets_{start:03d}.jpg", quality=95)
    atomic_json_dump(manifest, args.output / "manifest.json")
    print(f"Cached {len(items)} pseudo-pairs. No training; audit target sheets before selecting samples.", flush=True)


if __name__ == "__main__":
    main()
