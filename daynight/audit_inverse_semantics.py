"""Inspect cached segmentation teacher reliability before any inverse-region loss.

Segmentation visualizations are predicted labels, never manually supplied truth.
No inverse training, source edits, network downloads, or public route changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .experiment_night_to_day import visibility_control
from .turbo_pilot import digest
from .utils import atomic_json_dump

PALETTE = np.array(
    [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ],
    dtype=np.uint8,
)


@torch.inference_mode()
def main():
    import transformers
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--indices", default="12,105,142,214,283,0,78,426")
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("E:/LumiCycleCache/teacher_targets/night2day_pilot128_v2"),
    )
    parser.add_argument(
        "--local-outputs",
        type=Path,
        default=Path(
            "E:/LumiCycleCache/experiments/night2day_diverse111_transpose1000/validation_02000"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier audit")
    manifest = json.loads((args.targets / "manifest.json").read_text())
    if manifest["test_used"] or manifest["direction"] != "night_to_day":
        raise ValueError("Development validation only")
    records = {row["index"]: row for row in manifest["records"] if row["split"] == "val"}
    indices = [int(value) for value in args.indices.split(",")]
    if len(indices) != len(set(indices)) or any(i not in records for i in indices):
        raise ValueError("Require unique development indices; no lockbox access")
    name = "facebook/mask2former-swin-small-cityscapes-semantic"
    processor = AutoImageProcessor.from_pretrained(name, local_files_only=True)
    model, loading_info = Mask2FormerForUniversalSegmentation.from_pretrained(
        name, local_files_only=True, output_loading_info=True
    )
    if any(
        loading_info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise RuntimeError(
            "Teacher checkpoint/runtime mismatch; use the audited isolated Transformers 4.44.2 environment"
        )
    model.cuda().eval().requires_grad_(False)
    torch.set_num_threads(6)
    args.output.mkdir(parents=True)
    report = {
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "loading_info": loading_info,
        "external_teacher": name,
        "local_files_only": True,
        "checkpoint_revision": getattr(model.config, "_commit_hash", None),
        "label_names": model.config.id2label,
        "test_used": False,
        "lockbox_used": False,
        "source_gamma": 0.45,
        "records": [],
        "warning": "Predicted semantic classes, not ground truth or visual quality acceptance",
    }
    for index in indices:
        row = records[index]
        source = Image.open(row["input"]).convert("RGB")
        variants = [
            ("Original night", source),
            ("Gamma lifted night", visibility_control(source)),
            (
                "LOCAL inverse16000",
                Image.open(args.local_outputs / f"val_{index:04d}.png").convert("RGB"),
            ),
            ("EXTERNAL night2day", Image.open(row["target"]).convert("RGB")),
        ]
        folder = args.output / f"val_{index:04d}"
        folder.mkdir()
        panel = Image.new("RGB", (2048, 640), "#101010")
        draw = ImageDraw.Draw(panel)
        item = {"index": index, "source_sha256": digest(Path(row["input"])), "variants": []}
        for column, (label, image) in enumerate(variants):
            inputs = processor(
                image, return_tensors="pt", size={"shortest_edge": 384, "longest_edge": 768}
            )
            output = model(**{key: value.cuda() for key, value in inputs.items()})
            labels = (
                processor.post_process_semantic_segmentation(
                    output, target_sizes=[(image.height, image.width)]
                )[0]
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            painted = Image.fromarray(PALETTE[labels])
            image.save(folder / f"variant_{column}_rgb.png")
            Image.fromarray(labels).save(folder / f"variant_{column}_label_ids.png")
            Image.fromarray((labels == 10).astype(np.uint8) * 255).save(
                folder / f"variant_{column}_sky.png"
            )
            painted.save(folder / f"variant_{column}_segmentation.png")
            panel.paste(image, (column * 512, 28))
            panel.paste(painted, (column * 512, 344))
            draw.text((column * 512 + 5, 7), f"VAL{index} {label}", fill="white")
            draw.text(
                (column * 512 + 5, 323),
                "Predicted labels: blue=sky, green=vegetation, gray=building",
                fill="white",
            )
            item["variants"].append(
                {
                    "label": label,
                    "sky_fraction": float((labels == 10).mean()),
                    "vegetation_fraction": float((labels == 8).mean()),
                }
            )
        panel.save(args.output / f"audit_{index:04d}.jpg", quality=96)
        report["records"].append(item)
        atomic_json_dump(report, args.output / "evidence.json")
        print(f"Audited {index}; inspect sky/structure labels before using any mask", flush=True)


if __name__ == "__main__":
    main()
