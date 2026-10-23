"""Index LoLI TRAIN targets; exclude near-duplicates using saved BDD hashes only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .audit_restoration_pairs import matching_pairs
from .turbo_pilot import digest
from .utils import atomic_json_dump


def image_dhash(path):
    with Image.open(path) as im:
        values = np.asarray(ImageOps.exif_transpose(im).convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    value = 0
    for bit in (values[:, 1:] > values[:, :-1]).flatten():
        value = (value << 1) | int(bit)
    return value


def near_any(values, reference, maximum_distance=4):
    if not reference:
        raise ValueError("Missing held-out hash metadata")
    values, reference = np.asarray(values, dtype=np.uint64), np.asarray(reference, dtype=np.uint64)
    popcount = np.asarray([i.bit_count() for i in range(256)], dtype=np.uint8)
    distances = []
    for start in range(0, len(values), 64):
        xor = np.bitwise_xor(values[start:start + 64, None], reference[None])
        distance = popcount[xor.view(np.uint8).reshape(*xor.shape, 8)].sum(-1).min(-1)
        distances.extend(int(v) for v in distance)
    return [v <= maximum_distance for v in distances], distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bdd", type=Path, default=Path("data/bdd100k_daynight/manifests"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep earlier data index")
    pairs = matching_pairs(args.root, "Train/low", "Train/high")
    reference = []
    reference_files = []
    for split in ("val", "test"):
        for domain in ("day", "night"):
            path = args.bdd / f"{split}_{domain}.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            reference.extend(int(r["dhash"], 16) for r in records)
            reference_files.append({"path": str(path), "sha256": digest(path), "count": len(records)})
    unique, records = {}, []
    for number, (name, low, high) in enumerate(pairs):
        sha = digest(high)
        if sha not in unique:
            unique[sha] = {"sha256": sha, "dhash": image_dhash(high), "representative": name}
        records.append({"name": name, "input": str(low.resolve()), "target": str(high.resolve()),
            "target_sha256": sha, "split": "train"})
        if (number + 1) % 5000 == 0:
            print(f"Indexed{number + 1}/{len(pairs)} TRAIN pairs;uniqueRGBfiles={len(unique)}", flush=True)
    exclusions, distances = near_any([r["dhash"] for r in unique.values()], reference)
    for row, exclude, distance in zip(unique.values(), exclusions, distances, strict=True):
        row["exclude_bdd_holdout_similarity"] = exclude
        row["nearest_bdd_heldout_dhash_distance"] = distance
        row["dhash"] = f"{row['dhash']:016x}"
    for row in records:
        row["exclude_bdd_holdout_similarity"] = unique[row["target_sha256"]]["exclude_bdd_holdout_similarity"]
    excluded = sum(r["exclude_bdd_holdout_similarity"] for r in records)
    atomic_json_dump({"dataset": "LoLI-Street", "root": str(args.root),
        "source": "https://www.kaggle.com/datasets/tanvirnwu/loli-street-low-light-image-enhancement-of-street",
        "license": "CC BY-NC4.0 as stated by author-linked Kaggle metadata; no image redistribution",
        "input_origin": "synthetic darkening, NOT real paired night photographs",
        "pair_count": len(pairs), "unique_target_file_hashes": len(unique), "excluded_pair_count": excluded,
        "bdd_metadata_references": reference_files, "bdd_heldout_images_opened": False,
        "test_metrics_computed": False, "loli_heldout_opened": False, "training_performed": False,
        "audit_only": True, "near_duplicate_threshold": 4,
        "limitations": "64bit dHash screening is not proof of no overlap; crops/nearby frames may be missed. Group targets before creating any new split. Only24 diagnostic pairs visually reviewed separately.",
        "records": records, "unique_targets": list(unique.values())}, args.output)
    print(json.dumps({"pairs": len(pairs), "unique_targets": len(unique), "excluded_pairs": excluded}))


if __name__ == "__main__":
    main()
