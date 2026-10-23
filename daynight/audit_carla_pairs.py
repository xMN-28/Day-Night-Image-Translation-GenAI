"""Small author-linked Kaggle CARLA TRAIN audit; no training/held-out pixels."""
from __future__ import annotations

import argparse
import re
from pathlib import Path, PurePosixPath

import kagglehub
import requests
import torch
from PIL import Image, ImageDraw

from .experiment_night_to_day import load_inverse
from .turbo_pilot import digest
from .utils import atomic_json_dump, pil_to_tensor, tensor_to_pil

REF = "okarif/carla-parired-images"
VERSION = 3


def training_pairs(files):
    """Only exact matching names under day_train/night_train, never infer order."""
    domains = {"day_train": {}, "night_train": {}}
    for row in files:
        name = row["name"]
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe dataset path")
        if len(path.parts) != 3 or path.parts[0] != "data_carla":
            continue
        if path.parts[1] not in domains or path.suffix.lower() != ".png":
            continue
        if not re.fullmatch(r"recording\d+_\d+\.png", path.name):
            raise ValueError("Unknown training sequence naming")
        domains[path.parts[1]][path.name] = row
    common = sorted(set(domains["day_train"]) & set(domains["night_train"]))
    if not common:
        raise ValueError("No exact paired TRAIN names")
    return [{"name": name, "day": domains["day_train"][name],
        "night": domains["night_train"][name], "sequence": name.split("_")[0]} for name in common]


def select_sequences(pairs, count):
    """Round-robin evenly spaced frames across available training sequences."""
    groups = {}
    for pair in pairs:
        groups.setdefault(pair["sequence"], []).append(pair)
    for values in groups.values():
        values.sort(key=lambda x: int(x["name"].rsplit("_", 1)[1].split(".")[0]))
    selected = []
    names = sorted(groups)
    for position in range(count):
        sequence = names[position % len(names)]
        pool = groups[sequence]
        rank = position // len(names)
        visits = (count + len(names) - 1 - names.index(sequence)) // len(names)
        index = round(rank * (len(pool) - 1) / max(visits - 1, 1))
        if pool[index] not in selected:
            selected.append(pool[index])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.count <= 24:
        raise ValueError("New output and bounded1..24sample audit required")
    session = requests.Session()
    metadata = session.get(f"https://www.kaggle.com/api/v1/datasets/view/{REF}", timeout=30)
    metadata.raise_for_status()
    metadata = metadata.json()
    if metadata["currentVersionNumber"] != VERSION:
        raise ValueError("Version changed; re-review before download")
    files, token = [], None
    for _ in range(100):
        params = {"pageSize": 1000}
        if token:
            params["pageToken"] = token
        response = session.get(f"https://www.kaggle.com/api/v1/datasets/list/{REF}",
            params=params, timeout=30)
        response.raise_for_status()
        page = response.json()
        files.extend({"name": r["name"], "bytes": r["totalBytes"]} for r in page["datasetFiles"])
        token = page.get("nextPageToken") or page.get("nextPageTokenNullable")
        if not token:
            break
    else:
        raise RuntimeError("Incomplete listing; no sampling")
    pairs = training_pairs(files)
    selected = select_sequences(pairs, args.count)
    expected_bytes = sum(row[domain]["bytes"] for row in selected for domain in ("day", "night"))
    if expected_bytes > 300 * 1024**2:
        raise ValueError("Exceeds300MiBaudit download budget")
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True)
    atomic_json_dump({"ref": REF, "version": VERSION, "metadata": metadata,
        "files": files, "train_pairs": len(pairs), "selected": selected,
        "expected_download_bytes": expected_bytes}, args.cache / "index.json")
    local = []
    for number, row in enumerate(selected):
        record = {"name": row["name"], "sequence": row["sequence"], "split": "train"}
        for domain in ("night", "day"):
            # The user explicitly requested Kaggle-only dataset downloads.
            path = Path(kagglehub.dataset_download(f"{REF}/versions/{VERSION}", path=row[domain]["name"]))
            if not path.is_file() or path.stat().st_size != row[domain]["bytes"]:
                raise ValueError("Downloaded file missing/size mismatch")
            record[domain] = str(path)
            record[f"{domain}_sha256"] = digest(path)
        local.append(record)
        print(f"Downloaded audited TRAIN pair{number + 1}/{len(selected)}", flush=True)
    model, step = load_inverse(args.parent)
    torch.set_num_threads(6)
    panels = []
    for row in local:
        with Image.open(row["night"]) as night, Image.open(row["day"]) as day:
            if night.size != day.size:
                raise ValueError("Pair dimensions differ")
            row["native_size"] = list(night.size)
            x = night.convert("RGB").resize((512, 288), Image.Resampling.LANCZOS)
            y = day.convert("RGB").resize((512, 288), Image.Resampling.LANCZOS)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(pil_to_tensor(x)[None].cuda())
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite audit prediction")
        result = tensor_to_pil(prediction)
        panel = Image.new("RGB", (1536, 312), "#101010")
        for j, (label, im) in enumerate([("CARLA TRAIN night", x),
            ("CARLA TRAIN day reference", y), (f"Local parent{step}", result)]):
            panel.paste(im, (512 * j, 24))
            ImageDraw.Draw(panel).text((512 * j + 4, 5), f"{row['name']} {label}", fill="white")
        panels.append(panel)
        folder = args.output / row["name"].removesuffix(".png")
        folder.mkdir()
        x.save(folder / "night.png")
        y.save(folder / "day.png")
        result.save(folder / "parent.png")
    for start in range(0, len(panels), 4):
        sheet = Image.new("RGB", (1536, 312 * len(panels[start:start + 4])))
        for j, panel in enumerate(panels[start:start + 4]):
            sheet.paste(panel, (0, j * 312))
        sheet.save(args.output / f"audit_{start:02d}.jpg", quality=95)
    atomic_json_dump({"dataset": REF, "version": VERSION,
        "origin": "Simulator-rendered, NOT real photographic day/night ground truth",
        "kaggle_license": metadata["licenseName"], "records": local,
        "parent": str(args.parent), "parent_sha256": digest(args.parent),
        "split": "TRAIN audit only", "test_used": False, "lockbox_used": False,
        "training_performed": False, "vision_review": "pending",
        "public_demo_unchanged": True}, args.output / "audit.json")


if __name__ == "__main__":
    main()
