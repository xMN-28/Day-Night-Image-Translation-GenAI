"""Training-only stationary-camera pilot cache, corrected clock and static masks.

Median real night targets reduce transient traffic; semantic teachers exclude
moving-object classes. Original dataset images/annotations are never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .prepare_amos import _records
from .utils import atomic_json_dump


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((384, 216), Image.Resampling.LANCZOS)).copy()


def main():
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("D:/datasets/kagglehub/datasets/stevemark/daynight-dataset/versions/1/DNIM"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/curated_timelapse_v1"))
    args = parser.parse_args()
    torch.set_num_threads(6)
    provenance = json.loads((args.output / "provenance.json").read_text())
    teacher_name = "facebook/mask2former-swin-small-cityscapes-semantic"
    processor = AutoImageProcessor.from_pretrained(teacher_name, local_files_only=True)
    teacher = (
        Mask2FormerForUniversalSegmentation.from_pretrained(teacher_name, local_files_only=True)
        .cuda()
        .eval()
    )
    rows = []
    exclusions = {
        "00001323": "severe lens glare in night review samples",
        "00003837": "strong horizontal night banding in review samples",
    }

    @torch.inference_mode()
    def static(array):
        inputs = processor(
            Image.fromarray(array),
            return_tensors="pt",
            size={"shortest_edge": 384, "longest_edge": 768},
        )
        result = teacher(**{k: v.cuda() for k, v in inputs.items()})
        labels = processor.post_process_semantic_segmentation(result, target_sizes=[(216, 384)])[0]
        # Cityscapes ids 11..18 are people/riders/vehicles. This is a pseudo-mask,
        # not a guarantee every moving object is found on unseen camera domains.
        mask = (labels < 11).float()[None, None]
        return (1 - F.max_pool2d(1 - mask, 5, 1, 2))[0, 0].cpu().numpy()

    for camera in provenance["cameras"]:
        cid, split = camera["camera"], camera["split"]
        if split == "train" and cid in exclusions:
            continue
        records = _records(args.root / "Image", args.root / "time_stamp" / f"{cid}.txt")
        days = [(t, p) for t, p in records if 10 <= t.hour <= 14]
        nights = [(t, p) for t, p in records if t.hour <= 3 or t.hour >= 22]
        if not days or not nights:
            continue
        target = np.median(np.stack([rgb(p) for _, p in nights]), axis=0).astype(np.uint8)
        target_mask = static(target)
        folder = args.output / "cache" / split
        folder.mkdir(parents=True, exist_ok=True)
        for i, (clock, path) in enumerate(days):
            output = folder / f"{cid}_{i:03d}.npz"
            image = rgb(path)
            if not output.exists():
                mask = static(image) * target_mask
                np.savez_compressed(
                    output, image=image, target=target, static_mask=(mask * 255).astype(np.uint8)
                )
            rows.append(
                {
                    "camera": cid,
                    "split": split,
                    "cache": str(output),
                    "day_path": str(path),
                    "day_clock": clock.isoformat(),
                    "night_paths": [str(p) for _, p in nights],
                }
            )
        # Review a target and its mask; do not silently omit validation failures.
        sample = np.load(folder / f"{cid}_000.npz")
        canvas = Image.new("RGB", (1152, 240), "#101010")
        draw = ImageDraw.Draw(canvas)
        for j, (array, label) in enumerate(
            zip(
                (sample["image"], target, np.repeat(sample["static_mask"][..., None], 3, axis=2)),
                (
                    f"{cid} source {split}",
                    "Real-night temporal median",
                    "Training-only static pseudo-mask",
                ),
                strict=True,
            )
        ):
            draw.text((j * 384 + 4, 4), label, fill="white")
            canvas.paste(Image.fromarray(array), (j * 384, 24))
        review = Path("outputs/sky_pilot/timelapse_corrected_cache")
        review.mkdir(parents=True, exist_ok=True)
        canvas.save(review / f"{cid}.jpg", quality=95)
        print(json.dumps({"camera": cid, "split": split, "day_frames": len(days)}), flush=True)
    atomic_json_dump(
        {
            "pairs": rows,
            "excluded_training_cameras": exclusions,
            "teacher": teacher_name,
            "teacher_training_only": True,
            "clock_labels_spot_reviewed": True,
            "validation_retained_even_when_banded": True,
            "target": "median of real corrected-clock night frames, no dense warp",
            "limitations": "17-camera dataset; masks imperfect; camera registration not guaranteed",
        },
        args.output / "cache_manifest.json",
    )


if __name__ == "__main__":
    main()
