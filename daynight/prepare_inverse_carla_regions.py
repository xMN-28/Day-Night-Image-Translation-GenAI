"""Prepare reviewed simulator TRAIN upper-static regions; never whole-image GT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from .turbo_pilot import digest
from .utils import atomic_json_dump


def static_upper_mask(size=(512, 288), rows=176):
    if size != (512, 288) or rows != 176:
        raise ValueError("ROI coordinates are reviewed for512x288only")
    mask = Image.new("L", size)
    ImageDraw.Draw(mask).rectangle((0, 0, 511, rows - 1), fill=255)
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path,
        default=Path("outputs/night_to_day_experimental/20261023_CARLA_training_audit12/audit.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approve-reviewed", action="store_true")
    args = parser.parse_args()
    manifest = args.output / "regions.json"
    if args.approve_reviewed:
        cache = json.loads(manifest.read_text())
        if cache["approved_after_visual_review"] or cache["test_used"] or cache["lockbox_used"]:
            raise ValueError("Cannot approve this cache")
        for row in cache["records"]:
            for key in ("input", "target", "confidence"):
                if digest(Path(row[key])) != row[f"{key}_sha256"]:
                    raise ValueError("Reviewed bytes changed")
        cache["approved_after_visual_review"] = True
        cache["review_scope"] = "All12source/reference/mask panels opened; top176rows static-region hypothesis, not exact photometric correspondence"
        atomic_json_dump(cache, manifest)
        return
    if args.output.exists():
        raise FileExistsError("Preserve prior cache")
    audit = json.loads(args.audit.read_text())
    if audit["dataset"] != "okarif/carla-parired-images" or audit["test_used"] or audit["lockbox_used"]:
        raise ValueError("Require audited CARLA TRAIN subset")
    if audit["vision_review"] == "pending" or len(audit["records"]) != 12:
        raise ValueError("Require exact reviewed12sample audit")
    args.output.mkdir(parents=True)
    panels, records = [], []
    for number, row in enumerate(audit["records"]):
        if row["split"] != "train":
            raise ValueError("Training only")
        for domain in ("night", "day"):
            if digest(Path(row[domain])) != row[f"{domain}_sha256"]:
                raise ValueError("Audit original bytes changed")
        folder = args.output / f"pair_{number:02d}"
        folder.mkdir()
        with Image.open(row["night"]) as x, Image.open(row["day"]) as y:
            source = x.convert("RGB").resize((512, 288), Image.Resampling.LANCZOS)
            target = y.convert("RGB").resize((512, 288), Image.Resampling.LANCZOS)
        mask = static_upper_mask()
        paths = {"input": folder / "SIMULATOR_night.png", "target": folder / "SIMULATOR_day.png",
            "confidence": folder / "UPPER_STATIC.png"}
        for key, im in (("input", source), ("target", target), ("confidence", mask)):
            im.save(paths[key])
        record = {"index": 500000 + number, "split": "train", "sequence": row["sequence"],
            "original_name": row["name"], "source_label": "SIMULATOR night TRAIN",
            "target_label": "SIMULATOR day TRAIN, upper ROI only"}
        for key, path in paths.items():
            record[key], record[f"{key}_sha256"] = str(path.resolve()), digest(path)
        records.append(record)
        panel = Image.new("RGB", (1536, 312), "#101010")
        for j, (label, im) in enumerate((("SIM night TRAIN", source),
            ("SIM day TRAIN", target), ("WHITE=supervised, BLACK=excluded", mask.convert("RGB")))):
            panel.paste(im, (j * 512, 24))
            ImageDraw.Draw(panel).text((j * 512 + 4, 5), f"{row['name']} {label}", fill="white")
        panels.append(panel)
    for start in range(0, len(panels), 4):
        sheet = Image.new("RGB", (1536, 1248))
        for j, panel in enumerate(panels[start:start + 4]):
            sheet.paste(panel, (0, 312 * j))
        sheet.save(args.output / f"TRAIN_regions_{start:02d}.jpg", quality=95)
    atomic_json_dump({"approved_after_visual_review": False, "test_used": False, "lockbox_used": False,
        "source": audit["dataset"], "license": audit["kaggle_license"],
        "input_origin": "simulator-rendered night", "target_origin": "simulator-rendered day",
        "audit_sha256": digest(args.audit), "confidence_semantics": "Manually reviewed top176/288rows; lower112rows excluded, not semantic confidence",
        "selection": "All12previous audit pairs, no new visual cherry-picking", "records": records}, manifest)


if __name__ == "__main__":
    main()
