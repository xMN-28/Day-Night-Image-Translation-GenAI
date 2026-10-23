"""Four manually reviewed static-region candidates, NOT full-image paired truth.

Rectangles are explicit training annotations, never per-demo output editing.
Require visual inspection of generated panels before --approve-reviewed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .turbo_pilot import digest
from .utils import atomic_json_dump

STATIC_ROIS = {
    408: (5, 65, 185, 190),
    548: (340, 45, 505, 190),
    190: (350, 110, 505, 260),
    1739: (310, 25, 505, 115),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=Path("E:/LumiCycleCache/teacher_targets/night2day_real_pairs_scan96_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approve-reviewed", action="store_true")
    args = parser.parse_args()
    path = args.output / "regions.json"
    if args.approve_reviewed:
        saved = json.loads(path.read_text())
        if saved["test_used"] or saved["lockbox_used"]:
            raise ValueError("No held-out region training")
        for row in saved["records"]:
            for key in ("input", "target", "confidence"):
                if digest(Path(row[key])) != row[f"{key}_sha256"]:
                    raise ValueError("Audited region bytes changed")
        saved["approved_after_visual_review"] = True
        atomic_json_dump(saved, path)
        return
    if args.output.exists():
        raise FileExistsError("Keep previous evidence")
    scan = json.loads((args.scan / "candidates.json").read_text())
    if scan["test_used"] or scan["lockbox_used"]:
        raise ValueError("No held-out region selection")
    rows = {r["index"]: r for r in scan["ranked_candidates"]}
    args.output.mkdir(parents=True)
    records = []
    for index, roi in STATIC_ROIS.items():
        row = rows[index]
        folder = Path(row["folder"])
        source = Image.open(folder / "NATIVE_night.png").convert("RGB")
        target = Image.open(folder / "REAL_day_warped.png").convert("RGB")
        confidence = np.asarray(Image.open(folder / "heuristic_confidence.png"), dtype=np.float32) / 255
        rectangle = np.zeros_like(confidence)
        x1, y1, x2, y2 = roi
        rectangle[y1:y2, x1:x2] = 1
        confidence = np.where(confidence >= .6, confidence, 0) * rectangle
        if (confidence > 0).mean() < .001:
            raise ValueError(f"Insufficient static support in candidate{index}")
        mask_path = args.output / f"pair_{index:04d}_confidence.png"
        Image.fromarray(np.rint(confidence * 255).astype(np.uint8)).save(mask_path)
        masked = np.asarray(target).copy()
        masked[confidence == 0] = 0
        sheet = Image.new("RGB", (2048, 312), "#101010")
        for j, (label, image) in enumerate((("REAL training night", source), ("Warped REAL day reference", target),
            ("STATIC annotated ROI + confidence", Image.open(mask_path).convert("RGB")),
            ("ONLY these pixels supervised", Image.fromarray(masked)))):
            sheet.paste(image, (512 * j, 24))
            ImageDraw.Draw(sheet).text((512 * j + 5, 5), f"TRAIN PAIR{index} {label}", fill="white")
        sheet.save(args.output / f"review_pair_{index:04d}.jpg", quality=96)
        record = {"index": index, "split": "train", "roi_xyxy": roi,
            "input": str((folder / "NATIVE_night.png").resolve()),
            "target": str((folder / "REAL_day_warped.png").resolve()),
            "confidence": str(mask_path.resolve()), "supported_fraction": float((confidence > 0).mean()),
            "original_night": row["night_path"], "original_day": row["day_path"]}
        for key in ("input", "target", "confidence"):
            record[f"{key}_sha256"] = digest(Path(record[key]))
        records.append(record)
    atomic_json_dump({"test_used": False, "lockbox_used": False,
        "approved_after_visual_review": False, "scan_sha256": digest(args.scan / "candidates.json"),
        "records": records, "supervision": "REAL day RGB in manually curated static building/wall regions ONLY",
        "limitations": "Four training annotations, not automatic semantics; no sky/glare/object-removal truth; no full-image pairing claim"}, path)


if __name__ == "__main__":
    main()
