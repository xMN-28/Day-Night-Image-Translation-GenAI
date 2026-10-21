"""Full camera-disjoint paired validation; no claim of exact pixel ground truth."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .compare_sky_gan import file_digest
from .sky_pilot import load_base
from .sky_refinement import gray
from .train_sky_gan import lighting_pyramid_loss, load_lighting_pair, masked_l1
from .utils import atomic_json_dump, tensor_to_pil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/paired_lighting_pilot/latest.pt")
    )
    parser.add_argument("--parent", type=Path, default=Path("runs/resize_decoder_pilot/best.pt"))
    parser.add_argument(
        "--cache", type=Path, default=Path("data/curated_timelapse_v1/cache_manifest.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/sky_pilot/paired_lighting_metrics")
    )
    args = parser.parse_args()
    torch.set_num_threads(6)
    parent, model = load_base(args.parent), load_base(args.checkpoint)
    manifest = json.loads(args.cache.read_text())
    records = [r for r in manifest["pairs"] if r["split"] == "val"]
    train_cameras = {r["camera"] for r in manifest["pairs"] if r["split"] == "train"}
    if train_cameras & {r["camera"] for r in records}:
        raise ValueError("Train/validation camera leakage")
    args.output.mkdir(parents=True, exist_ok=True)
    scores, by_camera, visualized = [], defaultdict(list), set()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in records:
            x, target, mask = load_lighting_pair(row["cache"])
            x, target, mask = x[None].cuda(), target[None].cuda(), mask[None].cuda()
            old, new = (parent(x * 2 - 1).float() + 1) / 2, (model(x * 2 - 1).float() + 1) / 2
            result = {
                "camera": row["camera"],
                "source": row["day_path"],
                "mask_coverage": float(mask.mean()),
            }
            for key, image in (("parent", old), ("candidate", new)):
                result[f"{key}_pyramid"] = float(lighting_pyramid_loss(image, target, mask))
                result[f"{key}_masked_rgb_l1"] = float(masked_l1(image, target, mask))
                result[f"{key}_luminance"] = float(gray(image).mean())
            result["target_luminance"] = float(gray(target).mean())
            scores.append(result)
            by_camera[row["camera"]].append(result)
            if row["camera"] not in visualized:
                visualized.add(row["camera"])
                canvas = Image.new("RGB", (1536, 240), "#101010")
                draw = ImageDraw.Draw(canvas)
                for j, (im, label) in enumerate(
                    zip(
                        (x, old, new, target),
                        ("Held-out camera", "Parent", "Paired pilot", "Night median target"),
                        strict=True,
                    )
                ):
                    draw.text((j * 384 + 4, 4), label, fill="white")
                    canvas.paste(tensor_to_pil(im * 2 - 1), (j * 384, 24))
                canvas.save(args.output / f"camera_{row['camera']}.jpg", quality=95)
    metrics = [k for k in scores[0] if k not in {"camera", "source"}]

    def mean(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in metrics}

    summary = {
        "split": "val",
        "count": len(scores),
        "camera_count": len(by_camera),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_digest(args.checkpoint),
        "parent_sha256": file_digest(args.parent),
        "metrics": mean(scores),
        "per_camera": {k: mean(v) for k, v in by_camera.items()},
        "limitation": "Raw fixed-camera approximate correspondence; temporal median and semantic pseudo-masks, not exact individual night ground truth. Banded validation camera retained.",
        "records": scores,
    }
    atomic_json_dump(summary, args.output / "metrics.json")
    print({k: v for k, v in summary.items() if k != "records"}, flush=True)


if __name__ == "__main__":
    main()
