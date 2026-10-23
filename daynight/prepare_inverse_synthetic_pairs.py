"""Training-only synthetic-night / real-day pairs for an isolated inverse pilot.

Keeps frozen day-to-night weights and all public routes unchanged. Synthetic
sources are not real nighttime ground truth; validation remains real night.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import read_manifest
from .inference import ModelManager, _fit_and_pad
from .turbo_pilot import digest
from .utils import atomic_json_dump


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument(
        "--approve-reviewed",
        action="store_true",
        help="After actually opening all synthetic sheets, accept this bounded cache",
    )
    parser.add_argument(
        "--real-cache",
        type=Path,
        default=Path("E:/LumiCycleCache/teacher_targets/night2day_pilot128_v2"),
    )
    args = parser.parse_args()
    if args.approve_reviewed:
        manifest_path = args.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["test_used"] or manifest["lockbox_used"]:
            raise ValueError("Cannot approve held-out training data")
        selection = {
            "manifest_sha256": digest(manifest_path),
            "accepted_train_indices": [
                row["index"] for row in manifest["records"] if row["split"] == "train"
            ],
            "rejected_train_indices": [],
            "review": "All four 16-pair synthetic sheets opened by Codex vision; 64 accepted as a domain-gap control",
            "limitations": "Synthetic source light artifacts and sun/dusk/weather cues retained; not real-night generalization evidence",
            "previous_real_selection": "111 real-night sources inherit their earlier individual visual audit",
        }
        atomic_json_dump(selection, args.output / "visual_selection_v1.json")
        print(
            f"Recorded explicit sheet-review approval of {len(selection['accepted_train_indices'])} training pairs"
        )
        return
    if args.output.exists() or not 1 <= args.count <= 256:
        raise ValueError("Require a new directory and 1..256 training-day sources")
    manifest_path = args.real_cache / "manifest.json"
    parent_manifest = json.loads(manifest_path.read_text())
    audited = json.loads((args.real_cache / "visual_selection_v1.json").read_text())
    if parent_manifest["test_used"] or audited["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Require matching audited training/validation cache")
    accepted = set(audited["accepted_train_indices"])
    retained = [
        copy.deepcopy(row)
        for row in parent_manifest["records"]
        if row["split"] == "val" or row["index"] in accepted
    ]
    val_hashes = {row["index"]: row["target_sha256"] for row in retained if row["split"] == "val"}
    train = read_manifest("data/bdd100k_daynight", "train", "day")
    indices = np.linspace(0, len(train) - 1, args.count, dtype=int).tolist()
    args.output.mkdir(parents=True)
    pairs = args.output / "synthetic_train"
    pairs.mkdir()
    manager = ModelManager()
    torch.set_num_threads(6)
    panels = []
    for number, index in enumerate(indices):
        record = train[index]
        with Image.open(record.path) as original:
            padded, crop = _fit_and_pad(original, 512)
            real_day = padded.crop(crop)
        synthetic_night, metadata = manager.translate_custom(
            real_day, "day_to_night", "LumiCycle distilled student", 512
        )
        identifier = 100000 + index
        input_path = pairs / f"{identifier}_SYNTHETIC_night.png"
        target_path = pairs / f"{identifier}_REAL_day.png"
        synthetic_night.save(input_path)
        real_day.save(target_path)
        retained.append(
            {
                "split": "train",
                "index": identifier,
                "original_day_train_index": index,
                "source": str(record.path),
                "source_sha256": digest(record.path),
                "input": str(input_path.resolve()),
                "input_sha256": digest(input_path),
                "target": str(target_path.resolve()),
                "target_sha256": digest(target_path),
                "supervision": "SYNTHETIC night input / REAL training-day target",
                "day_to_night_metadata": metadata,
            }
        )
        panel = Image.new("RGB", (512, 166), "#101010")
        draw = ImageDraw.Draw(panel)
        for column, (label, image) in enumerate(
            (("SYNTH night", synthetic_night), ("REAL day target", real_day))
        ):
            thumb = image.copy()
            thumb.thumbnail((256, 144), Image.Resampling.LANCZOS)
            panel.paste(thumb, (column * 256, 22))
            draw.text((column * 256 + 4, 4), f"trainDay{index} {label}", fill="white")
        panels.append(panel)
        if (number + 1) % 16 == 0:
            print(f"Generated {number + 1}/{args.count} synthetic training pairs", flush=True)
    for start in range(0, len(panels), 16):
        page = panels[start : start + 16]
        sheet = Image.new("RGB", (2048, 166 * ((len(page) + 3) // 4)), "#101010")
        for number, panel in enumerate(page):
            sheet.paste(panel, ((number % 4) * 512, (number // 4) * 166))
        sheet.save(args.output / f"synthetic_training_{start:03d}.jpg", quality=96)
    manifest = {
        "direction": "night_to_day",
        "test_used": False,
        "lockbox_used": False,
        "teacher": "Mixed audited EXTERNAL night2day pseudo-targets and synthetic-night/real-day pairs",
        "teacher_sha256": parent_manifest["teacher_sha256"],
        "real_night_parent_manifest_sha256": digest(manifest_path),
        "retained_real_train_count": len(accepted),
        "synthetic_train_count": args.count,
        "validation_target_hashes_unchanged": val_hashes,
        "day_to_night_generator_sha256": digest(manager.checkpoint_path),
        "day_to_night_matte_sha256": digest(manager.sky_repair.checkpoint),
        "records": retained,
        "warning": "Audit synthetic pairs before selection; evaluate exclusively on frozen real-night validation.",
    }
    atomic_json_dump(manifest, args.output / "manifest.json")
    print(
        f"Prepared mixed cache with {len(accepted)} audited real and {args.count} unaudited synthetic sources."
    )


if __name__ == "__main__":
    main()
