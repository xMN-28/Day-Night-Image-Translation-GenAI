"""Fixed-camera real NIGHT->DAY sky/building training-region pilot.

Manual camera-level static-region annotations, not automatic scene segmentation.
Preserve existing camera-disjoint validation and avoid old dense-warp caches.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .turbo_pilot import digest
from .utils import atomic_json_dump

# Fractions selected after opening all14 TRAIN-camera contact sheets. Lower
# traffic/plaza regions are excluded; no held-out camera informed these choices.
STATIC_TOP = {"00000850": .70, "00004368": .42, "00005381": .48,
    "00008575": .45, "00009291": .70, "00009320": .62,
    "00009403": .60, "00021510": .65, "00023966": .56}
EXCLUSIONS = {"00001323": "severe night lens glare", "00003837": "night banding",
    "00019607": "night camera view/scale differs from daytime",
    "00010888": "traffic-heavy intersection, no sky in view",
    "00011331": "near-black geometry and variable river; defer from this bounded pilot"}
REVIEWED_PAIR_REJECTIONS = {300010: "Direct sun/flare in DAY target",
    300012: "Clipped sun/flare in DAY target", 300013: "Clipped sun/flare in DAY target",
    300019: "Severe backlit flare in DAY target", 300027: "Lens flare/washed out DAY target"}


def camera_training_subset(rows, per_camera=4):
    if per_camera < 1:
        raise ValueError("Positive per-camera count required")
    grouped = defaultdict(list)
    for row in rows:
        if row["camera"] in STATIC_TOP:
            if row["split"] != "train":
                raise ValueError("Selected camera must be TRAIN only")
            day_hour = datetime.fromisoformat(row["day_clock"]).hour
            night_hour = datetime.fromisoformat(row["night_clock"]).hour
            if not 10 <= day_hour <= 14 or not (night_hour <= 3 or night_hour >= 22):
                raise ValueError("Require corrected-clock midday/night candidates")
            grouped[row["camera"]].append(row)
    if set(grouped) != set(STATIC_TOP):
        raise ValueError("Missing selected training camera")
    selected = []
    for camera in sorted(grouped):
        records = sorted(grouped[camera], key=lambda r: r["night_clock"])
        for i in np.linspace(0, len(records) - 1, min(per_camera, len(records)), dtype=int):
            selected.append(records[int(i)])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("data/curated_timelapse_v1/candidates.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approve-reviewed", action="store_true")
    args = parser.parse_args()
    manifest_path = args.output / "regions.json"
    if args.approve_reviewed:
        saved = json.loads(manifest_path.read_text())
        if saved["approved_after_visual_review"]:
            raise FileExistsError("Already approved; preserve the existing training/rejection manifest")
        if saved["test_used"] or saved["lockbox_used"] or not saved["camera_disjoint_training_only"]:
            raise ValueError("No held-out training regions")
        for row in saved["records"]:
            for key in ("input", "target", "confidence"):
                if digest(Path(row[key])) != row[f"{key}_sha256"]:
                    raise ValueError("Reviewed bytes changed")
        saved["rejected_records"] = [row for row in saved["records"] if row["index"] in REVIEWED_PAIR_REJECTIONS]
        saved["records"] = [row for row in saved["records"] if row["index"] not in REVIEWED_PAIR_REJECTIONS]
        saved["reviewed_pair_rejections"] = REVIEWED_PAIR_REJECTIONS
        saved["approved_after_visual_review"] = True
        atomic_json_dump(saved, manifest_path)
        return
    if args.output.exists():
        raise FileExistsError("Keep previous pilot data")
    rows = [json.loads(line) for line in args.candidates.read_text().splitlines()]
    selected = camera_training_subset(rows)
    args.output.mkdir(parents=True)
    records, panels = [], defaultdict(list)
    for number, row in enumerate(selected):
        camera = row["camera"]
        with Image.open(row["night_path"]) as source, Image.open(row["day_path"]) as reference:
            if source.size != reference.size:
                raise ValueError("Source/reference native dimensions differ; audit before resizing")
            width = 512
            height = max(32, round(source.height * width / source.width / 4) * 4)
            night = source.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            day = reference.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        confidence = np.zeros((height, width), dtype=np.uint8)
        confidence[:int(height * STATIC_TOP[camera])] = 255
        folder = args.output / f"{camera}_{number:03d}"
        folder.mkdir()
        paths = {"input": folder / "REAL_night.png", "target": folder / "REAL_day_reference.png",
            "confidence": folder / "MANUAL_static_top.png"}
        night.save(paths["input"])
        day.save(paths["target"])
        Image.fromarray(confidence).save(paths["confidence"])
        record = {"index": 300000 + number, "camera": camera, "split": "train",
            "original_night": row["night_path"], "original_day": row["day_path"],
            "original_night_sha256": digest(Path(row["night_path"])),
            "original_day_sha256": digest(Path(row["day_path"])),
            "day_clock": row["day_clock"], "night_clock": row["night_clock"],
            "static_top_fraction": STATIC_TOP[camera], "size": [width, height]}
        for key, path in paths.items():
            record[key] = str(path.resolve())
            record[f"{key}_sha256"] = digest(path)
        records.append(record)
        # Review original photographic targets, not invented intermediate RGB.
        panel = Image.new("RGB", (1536, height + 24), "#101010")
        for j, (label, image) in enumerate((("REAL night TRAIN", night), ("REAL day static reference", day),
            ("WHITE=manual sky/building support", Image.fromarray(confidence).convert("RGB")))):
            panel.paste(image, (512 * j, 24))
            ImageDraw.Draw(panel).text((512 * j + 5, 5), f"{camera} pair{number} {label}", fill="white")
        panels[camera].append(panel)
    for camera, pages in panels.items():
        height = sum(page.height for page in pages)
        sheet = Image.new("RGB", (1536, height), "#101010")
        y = 0
        for page in pages:
            sheet.paste(page, (0, y))
            y += page.height
        sheet.save(args.output / f"review_camera_{camera}.jpg", quality=96)
    atomic_json_dump({"test_used": False, "lockbox_used": False, "camera_disjoint_training_only": True,
        "approved_after_visual_review": False, "candidates_sha256": digest(args.candidates),
        "records": records, "training_cameras": sorted(STATIC_TOP), "excluded_training_cameras": EXCLUSIONS,
        "supervision": "Original REAL daytime reference RGB on manually annotated sky/building regions; no RGB generator",
        "alignment": "Original fixed camera; no optical-flow warp; aspect preserved with <=2px dimension rounding",
        "limitations": "Manual upper-region masks imperfect; weather/season/clouds differ; small camera set; no exact unseen-night recovery claim"}, manifest_path)


if __name__ == "__main__":
    main()
