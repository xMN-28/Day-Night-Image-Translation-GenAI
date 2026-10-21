"""Inspect available coarse day/night pairs before using correspondence losses.

No training, source dataset mutation, or redefinition of existing BDD splits.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .utils import atomic_json_dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/lumirender_sources/dark_zurich.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/sky_pilot/paired_lighting_audit")
    )
    args = parser.parse_args()
    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    groups = Counter()
    splits = Counter()
    day_paths = Counter()
    confidence_rows = []
    missing = []
    for index, row in enumerate(records):
        groups[Path(row["day_path"]).parent.name] += 1
        day_paths[row["day_path"]] += 1
        splits[row.get("split", "train")] += 1
        for field in ("day_path", "night_path", "confidence_path"):
            if not row.get(field) or not Path(row[field]).is_file():
                missing.append({"index": index, "field": field, "path": row.get(field)})
        if row.get("confidence_path") and Path(row["confidence_path"]).is_file():
            with Image.open(row["confidence_path"]) as image:
                conf = np.asarray(image.convert("L")) / 255
            confidence_rows.append(
                {
                    "index": index,
                    "mean": float(conf.mean()),
                    "fraction_over_0_7": float((conf > 0.7).mean()),
                }
            )
    selected = np.linspace(0, len(records) - 1, 8, dtype=int).tolist()
    sheet = Image.new("RGB", (1152, 240 * len(selected)), "#101010")
    draw = ImageDraw.Draw(sheet)
    inspected = []
    args.output.mkdir(parents=True, exist_ok=True)
    for j, index in enumerate(selected):
        row = records[index]
        if not all(
            Path(row.get(k, "missing")).is_file()
            for k in ("day_path", "night_path", "confidence_path")
        ):
            continue
        conf = np.asarray(Image.open(row["confidence_path"]).convert("L")) / 255
        inspected.append(
            {
                "index": index,
                "day_path": row["day_path"],
                "night_path": row["night_path"],
                "confidence_mean": float(conf.mean()),
                "fraction_confidence_over_0_7": float((conf > 0.7).mean()),
            }
        )
        for col, field in enumerate(("day_path", "night_path", "confidence_path")):
            with Image.open(row[field]) as image:
                image = image.convert("RGB").resize((384, 216), Image.Resampling.LANCZOS)
            draw.text((col * 384 + 4, j * 240 + 4), f"{index}: {field}", fill="white")
            sheet.paste(image, (col * 384, j * 240 + 24))
    sheet.save(args.output / "pair_contact_sheet.jpg", quality=95)
    report = {
        "manifest": str(args.manifest),
        "records": len(records),
        "missing_path_count": len(missing),
        "missing_examples": missing[:20],
        "day_sequence_groups": dict(groups),
        "unique_day_paths": len(day_paths),
        "max_reuse_of_one_day_image": max(day_paths.values(), default=0),
        "confidence_audit_count": len(confidence_rows),
        "median_high_confidence_coverage": float(
            np.median([r["fraction_over_0_7"] for r in confidence_rows])
        )
        if confidence_rows
        else None,
        "pairs_with_under_1pct_high_confidence": sum(
            r["fraction_over_0_7"] < 0.01 for r in confidence_rows
        ),
        "splits": dict(splits),
        "selected_indices": selected,
        "inspected": inspected,
        "warning": "Flow confidence is not semantic validity. Mask moving classes and inspect alignment before photometric supervision.",
    }
    atomic_json_dump(report, args.output / "audit.json")
    print({k: v for k, v in report.items() if k != "inspected"}, flush=True)


if __name__ == "__main__":
    main()
