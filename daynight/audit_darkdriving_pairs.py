"""Private academic TRAIN-pair audit. Never publishes images or trains a model.

Canonical restrictive academic terms apply, not the mirror's Apache label.
Keep this cache and all outputs out of public demos, Git and redistribution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

import kagglehub
import numpy as np
import torch
from PIL import Image, ImageDraw

from .experiment_night_to_day import load_inverse
from .turbo_pilot import digest
from .utils import atomic_json_dump, pil_to_tensor, tensor_to_pil

PREFIX = "darkdriving_lle/darkdriving_lle/train"


def exact_train_pairs(index):
    if not index.get("complete_listing") or index.get("version") != 2:
        raise ValueError("Require complete pinned mirror inventory")
    domains = {"day": {}, "night": {}}
    for row in index["files"]:
        path = PurePosixPath(row["name"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe mirror path")
        if path.suffix.lower() != ".jpg" or str(path.parent.parent) != PREFIX:
            continue
        if path.parent.name in domains:
            domains[path.parent.name][path.name] = row
    names = sorted(set(domains["day"]) & set(domains["night"]))
    if not names:
        raise ValueError("No exact TRAIN restoration pairs")
    return [{"name": name, "day": domains["day"][name], "night": domains["night"][name],
        "sequence_prefix": Path(name).stem.rsplit("_", 1)[0]} for name in names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous audit")
    index = json.loads(args.index.read_text())
    pairs = exact_train_pairs(index)
    selected = [pairs[int(i)] for i in np.linspace(0, len(pairs) - 1, 12, dtype=int)]
    expected_bytes = sum(r[k]["bytes"] for r in selected for k in ("day", "night"))
    if expected_bytes > 200 * 1024**2:
        raise ValueError("Exceeds200MiBprivate audit budget")
    args.output.mkdir(parents=True)
    records = []
    for number, row in enumerate(selected):
        record = {"name": row["name"], "sequence_prefix": row["sequence_prefix"], "split": "train"}
        for domain in ("night", "day"):
            path = Path(kagglehub.dataset_download("benxelua/darkdriving/versions/2", path=row[domain]["name"]))
            if not path.is_file() or path.stat().st_size != row[domain]["bytes"]:
                raise ValueError("Audit file size mismatch")
            record[domain], record[f"{domain}_sha256"] = str(path), digest(path)
        records.append(record)
        print(f"Private TRAINpair{number + 1}/12 cached", flush=True)
    model, step = load_inverse(args.parent)
    torch.set_num_threads(6)
    panels = []
    for row in records:
        with Image.open(row["night"]) as x, Image.open(row["day"]) as y:
            if x.size != y.size:
                raise ValueError("Pair dimensions differ")
            row["native_size"] = list(x.size)
            size = (512, max(32, round(x.height * 512 / x.width / 4) * 4))
            x, y = x.convert("RGB").resize(size, Image.Resampling.LANCZOS), y.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(pil_to_tensor(x)[None].cuda())
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite frozen audit output")
        result = tensor_to_pil(prediction)
        panel = Image.new("RGB", (1536, size[1] + 40), "#101010")
        for j, (label, im) in enumerate((("PRIVATE TRAIN night", x),
            ("PRIVATE TRAIN day reference", y), (f"Local parent{step}", result))):
            panel.paste(im, (j * 512, 40))
            ImageDraw.Draw(panel).text((j * 512 + 4, 4), f"{row['name']}\n{label}", fill="white")
        panels.append(panel)
        folder = args.output / Path(row["name"]).stem
        folder.mkdir()
        result.save(folder / "parent.png")
    for start in range(0, len(panels), 3):
        batch = panels[start:start + 3]
        sheet = Image.new("RGB", (1536, sum(p.height for p in batch)), "#101010")
        top = 0
        for panel in batch:
            sheet.paste(panel, (0, top))
            top += panel.height
        sheet.save(args.output / f"PRIVATE_audit_{start:02d}.jpg", quality=95)
    atomic_json_dump({"dataset": "DarkDriving Kaggle mirror", "ref": "benxelua/darkdriving/versions/2",
        "inventory_sha256": digest(args.index), "origin": "Claimed real aligned day/night; visual verification pending",
        "license": index["canonical_license"], "mirror_authorization_verified": False,
        "use_scope": "Private academic audit only; NO redistribution/public-demo/model publication",
        "selected_image_bytes": expected_bytes, "records": records,
        "selection": "12evenly spaced sorted TRAIN-pair names; no quality cherry-picking",
        "parent": str(args.parent), "parent_sha256": digest(args.parent),
        "test_used": False, "lockbox_used": False, "training_performed": False,
        "approved_for_training": False, "vision_review": "pending", "public_demo_unchanged": True},
        args.output / "audit.json")


if __name__ == "__main__":
    main()
