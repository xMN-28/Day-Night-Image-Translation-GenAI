from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .data import eval_transform, read_manifest
from .utils import tensor_to_pil

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
SEGMENTATION_MODEL = "facebook/mask2former-swin-small-cityscapes-semantic"


def _semantic_groups(id2label: dict[int, str]) -> dict[str, set[int]]:
    groups = {
        "sky": {"sky"},
        "road": {"road", "sidewalk", "terrain"},
        "vehicle": {"car", "truck", "bus", "train", "motorcycle", "bicycle"},
        "glass": set(),
        "building": {"building", "wall", "fence"},
        "emitter": {"traffic light", "traffic sign"},
    }
    normalized = {int(key): str(value).lower() for key, value in id2label.items()}
    return {
        group: {index for index, label in normalized.items() if label in labels}
        for group, labels in groups.items()
    }


def prepare_targets(data_root: Path, output: Path, split: str, size: int, limit: int) -> int:
    try:
        from transformers import (
            AutoImageProcessor,
            AutoModelForDepthEstimation,
            Mask2FormerForUniversalSegmentation,
        )
    except ImportError as error:
        raise RuntimeError('Install teacher dependencies with: pip install -e ".[teachers]"') from error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(device).eval()
    segment_processor = AutoImageProcessor.from_pretrained(SEGMENTATION_MODEL)
    segment_model = Mask2FormerForUniversalSegmentation.from_pretrained(
        SEGMENTATION_MODEL
    ).to(device).eval()
    groups = _semantic_groups(segment_model.config.id2label)
    records = read_manifest(data_root, split, "day")[:limit]
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with torch.inference_mode():
        for record in tqdm(records, desc="Frozen teacher pseudo-labels"):
            with Image.open(record.path) as source:
                image = tensor_to_pil(eval_transform(source, size))
            depth_inputs = {
                key: value.to(device) for key, value in depth_processor(image, return_tensors="pt").items()
            }
            depth = depth_model(**depth_inputs).predicted_depth[:, None]
            depth = F.interpolate(depth, (size, size), mode="bicubic", align_corners=False)[0, 0]
            depth = depth - depth.amin()
            depth = depth / depth.amax().clamp_min(1e-6)

            segment_inputs = {
                key: value.to(device)
                for key, value in segment_processor(image, return_tensors="pt").items()
            }
            segment_output = segment_model(**segment_inputs)
            labels = segment_processor.post_process_semantic_segmentation(
                segment_output, target_sizes=[(size, size)]
            )[0]
            masks = []
            for name in ("sky", "road", "vehicle", "glass", "building", "emitter"):
                ids = groups[name]
                mask = torch.zeros_like(labels, dtype=torch.bool)
                for class_id in ids:
                    mask |= labels == class_id
                masks.append(mask)
            semantic = torch.stack(masks).to(torch.uint8).mul(255)
            # Cityscapes has no glass class; use conservative vehicle/building boundaries as candidates.
            semantic[3] = torch.maximum(semantic[2], semantic[4])

            digest = hashlib.sha1(str(record.path.resolve()).encode()).hexdigest()[:20]
            target = output / f"{digest}.npz"
            np.savez_compressed(
                target,
                depth=depth.mul(65535).round().to(torch.uint16).cpu().numpy(),
                semantic=semantic.cpu().numpy(),
            )
            rows.append(
                {"image_path": str(record.path.resolve()), "target_path": str(target.resolve())}
            )
    manifest = output / f"{split}.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache frozen Depth Anything V2 and Cityscapes Mask2Former supervision"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/bdd100k_daynight"))
    parser.add_argument("--output", type=Path, default=Path("data/lumirender_teachers"))
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()
    count = prepare_targets(args.data_root, args.output, args.split, args.size, args.limit)
    print(f"Cached {count} teacher targets. Teachers are not used during LumiRender inference.")


if __name__ == "__main__":
    main()
