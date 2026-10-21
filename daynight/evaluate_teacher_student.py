"""Validation-only pseudo-target fidelity; not a real-night quality benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

from .compare_sky_gan import file_digest
from .sky_pilot import load_base
from .teacher_objectives import highlight_fidelity, light_footprint_fidelity, masked_target_fidelity
from .train_resize_decoder import decoder_loss
from .train_teacher_student import load_pair, select_training_subset
from .utils import atomic_json_dump, tensor_to_pil


def load_reference_sky(root, row, size):
    provenance = json.loads((root / "provenance.json").read_text())
    matches = [r for r in provenance["splits"][row["split"]] if r["index"] == row["index"]]
    if len(matches) != 1 or Path(matches[0]["path"]).resolve() != Path(row["source"]).resolve():
        raise ValueError("Sky reference source identity does not match target manifest")
    with np.load(root / row["split"] / f"{row['index']:05d}.npz") as data:
        reference_image = data["image"]
        with Image.open(row["source"]) as source:
            current = np.asarray(
                ImageOps.exif_transpose(source)
                .convert("RGB")
                .resize(
                    (reference_image.shape[1], reference_image.shape[0]), Image.Resampling.LANCZOS
                )
            )
        if not np.array_equal(current, reference_image):
            raise ValueError("Sky reference pixels do not match current source/preprocessing")
        alpha = torch.from_numpy(data["alpha"].copy()).float()[None, None] / 255
    return F.interpolate(alpha, size, mode="bilinear", align_corners=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--subset-count", type=int, default=0)
    parser.add_argument("--sky-reference-cache", type=Path)
    parser.add_argument("--parent-label", default="Trained parent")
    parser.add_argument("--candidate-label", default="Warm-started candidate")
    args = parser.parse_args()
    torch.set_num_threads(6)
    manifest = json.loads((args.targets / "manifest.json").read_text())
    if manifest["test_images_used"]:
        raise ValueError("This diagnostic must not use project test images")
    rows = select_training_subset(
        [r for r in manifest["records"] if r["split"] == args.split], args.subset_count
    )
    models = {"parent": load_base(args.parent), "candidate": load_base(args.checkpoint)}
    results = {k: [] for k in models}
    panels = []
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, row in enumerate(rows):
            source, target = load_pair(row["cache"], row["source"])
            x, target = source[None].cuda() * 2 - 1, target[None].cuda() * 2 - 1
            sky_reference = (
                load_reference_sky(args.sky_reference_cache, row, x.shape[-2:]).cuda()
                if args.sky_reference_cache
                else None
            )
            predictions = {}
            for name, model in models.items():
                pred = model(x).float()
                if not torch.isfinite(pred).all():
                    raise FloatingPointError("Nonfinite candidate validation image")
                predictions[name] = pred
                bright = ((pred + 1) / 2).amax(1) > 0.7
                target_bright = ((target + 1) / 2).amax(1) > 0.7
                recall = (bright & target_bright).sum() / target_bright.sum().clamp_min(1)
                precision = (bright & target_bright).sum() / bright.sum().clamp_min(1)
                intersection_over_union = (bright & target_bright).sum() / (
                    bright | target_bright
                ).sum().clamp_min(1)
                results[name].append(
                    {
                        "source": row["source"],
                        "weighted_teacher_fidelity": float(decoder_loss(pred, target)),
                        "rgb_l1": float(F.l1_loss(pred, target)),
                        "teacher_highlight_l1": float(highlight_fidelity(pred, target)),
                        "teacher_bright_pixel_recall": float(recall),
                        "teacher_bright_pixel_precision": float(precision),
                        "teacher_bright_pixel_iou": float(intersection_over_union),
                        "teacher_light_footprint_loss": float(
                            light_footprint_fidelity(pred, target)
                        ),
                        "extra_bright_pixel_fraction": float(
                            (bright & ~target_bright).float().mean()
                        ),
                    }
                )
                if sky_reference is not None:
                    weights = pred.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
                    excess = (
                        ((pred - target) * weights).sum(1, keepdim=True) / 2 - 0.05
                    ).clamp_min(0)
                    results[name][-1].update(
                        {
                            "reference_sky_target_l1_0to1": float(
                                masked_target_fidelity(pred, target, sky_reference) / 2
                            ),
                            "reference_sky_excess_luminance": float(
                                (excess * sky_reference).sum() / sky_reference.sum().clamp_min(64)
                            ),
                        }
                    )
            indices = (
                (0, 6, 13, 20, 27, 34, 40, 47)
                if len(rows) == 48
                else {int(j * (len(rows) - 1) / 7) for j in range(8)}
            )
            if i in indices:
                height, width = x.shape[-2:]
                canvas = Image.new("RGB", (width * 4, height + 24), "#101010")
                draw = ImageDraw.Draw(canvas)
                for j, (label, pred) in enumerate(
                    (
                        (
                            f"{args.split.upper()} input"
                            + (" (memorization diagnostic)" if args.split == "train" else ""),
                            x,
                        ),
                        (args.parent_label, predictions["parent"]),
                        (args.candidate_label, predictions["candidate"]),
                        ("EXTERNAL teacher pseudo-target", target),
                    )
                ):
                    draw.text((j * width + 4, 4), label, fill="white")
                    canvas.paste(tensor_to_pil(pred), (j * width, 24))
                canvas.save(args.output / f"val_{i:02d}.jpg", quality=95)
                panels.append(canvas)
    gallery = Image.new("RGB", (panels[0].width, panels[0].height * len(panels)))
    for i, panel in enumerate(panels):
        gallery.paste(panel, (0, i * panel.height))
    gallery.save(args.output / "gallery.jpg", quality=95)
    summary = {
        name: {k: sum(r[k] for r in scores) / len(scores) for k in scores[0] if k != "source"}
        for name, scores in results.items()
    }
    atomic_json_dump(
        {
            "checkpoint": str(args.checkpoint),
            "sha256": file_digest(args.checkpoint),
            "parent": str(args.parent),
            "parent_sha256": file_digest(args.parent),
            "labels": {"parent": args.parent_label, "candidate": args.candidate_label},
            "manifest_sha256": file_digest(args.targets / "manifest.json"),
            "split": args.split,
            "training_memorization_diagnostic_only": args.split == "train",
            "sky_reference": str(args.sky_reference_cache) if args.sky_reference_cache else None,
            "sky_reference_provenance_sha256": file_digest(
                args.sky_reference_cache / "provenance.json"
            )
            if args.sky_reference_cache
            else None,
            "sky_reference_scope": "Separate frozen segmentation-teacher pseudo masks, source pixels checked; imperfect, not human ground truth",
            "count": len(rows),
            "resolution": list(manifest["recipe"]["student_size"]),
            "warning": "Teacher fidelity only, not realism or true paired accuracy. External pretraining overlap unaudited.",
            "summary": summary,
            "records": results,
        },
        args.output / "metrics.json",
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
