"""Inspect existing coarse paired data for the experimental inverse task; no training."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .utils import atomic_json_dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/lumirender_sources/dark_zurich.jsonl"))
    parser.add_argument("--native-pairs", type=Path, default=Path("D:/datasets/lumirender/dark_zurich_train_pairs.csv"))
    args = parser.parse_args()
    if (args.output / "audit.json").exists():
        raise FileExistsError("Keep existing audit evidence; choose a fresh folder")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    native = list(csv.DictReader(args.native_pairs.open(encoding="utf-8-sig", newline="")))
    if len(rows) != len(native):
        raise ValueError("Cannot assume correspondence order: native/cached lengths differ")
    if any(Path(a["day_path"]).resolve() != Path(b["day_path"]).resolve() for a, b in zip(rows, native)):
        raise ValueError("Native/cached pairing order differs")
    args.output.mkdir(parents=True, exist_ok=True)
    indices = [0, 400, 800, 1200, 1600, 2000]
    summaries = []
    sheet = Image.new("RGB", (512 * 4, 312 * len(indices)), "#101010")
    draw = ImageDraw.Draw(sheet)
    for j, i in enumerate(indices):
        row, original = rows[i], native[i]
        confidence = Image.open(row["confidence_path"]).convert("L")
        c = np.asarray(confidence, dtype=np.float32) / 255
        summary = {"index": i, "native_night": original["night_path"], "cached_warped_night": row["night_path"],
            "day": row["day_path"], "confidence": row["confidence_path"], "confidence_mean": float(c.mean()),
            "fraction_confidence_gt_0p5": float((c > 0.5).mean()), "fraction_confidence_gt_0p9": float((c > 0.9).mean()),
            "native_night_size": Image.open(original["night_path"]).size,
            "cached_night_size": Image.open(row["night_path"]).size, "day_size": Image.open(row["day_path"]).size}
        summaries.append(summary)
        columns = [("NATIVE night", original["night_path"]), ("CACHED night warped to day", row["night_path"]),
                   ("Real daytime coarse reference", row["day_path"]), ("Flow confidence, NOT dynamic mask", row["confidence_path"])]
        for k, (label, path) in enumerate(columns):
            image = Image.open(path).convert("RGB")
            image.thumbnail((512, 288), Image.Resampling.LANCZOS)
            sheet.paste(image, (k * 512, j * 312 + 24))
            draw.text((k * 512 + 5, j * 312 + 5), f"PAIR {i}: {label}", fill="white")
    for page in range(0, len(indices), 3):
        sheet.crop((0, page * 312, 2048, min(page + 3, len(indices)) * 312)).save(args.output / f"pairs_{page:02d}.jpg", quality=95)
    unique_days = Counter(r["day_path"] for r in rows)
    sequences = Counter(str(Path(r["night_path"]).parent) for r in native)
    atomic_json_dump({"pair_count": len(rows), "unique_day_references": len(unique_days),
        "max_reuse_of_day_reference": max(unique_days.values()), "native_night_sequence_counts": dict(sequences),
        "samples": summaries, "test_used": False, "alignment_direction": "night warped into day camera coordinates",
        "limitations": ["Coarse same-location references, not exact simultaneous ground truth.",
            "Cached confidence is optical-flow consistency/motion heuristic, NOT moving-object segmentation.",
            "Do not split adjacent paired frames randomly; group location/sequence and repeated day reference.",
            "Native night→day supervision requires day-to-native-night alignment, or confidence-aware coarse feature losses."]}, args.output / "audit.json")
    print(f"Audited {len(rows)} cached pairs, {len(unique_days)} unique day references; sample panels: {args.output}")


if __name__ == "__main__":
    main()
