"""Audit TRAIN-only enhancement photographs and the frozen local inverse.

No dataset test image is opened. No training or external RGB model is invoked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .experiment_night_to_day import load_inverse
from .turbo_pilot import digest
from .utils import atomic_json_dump, pil_to_tensor, tensor_to_pil


def matching_pairs(root, low_relative, high_relative):
    """Only explicitly training paths; pair exact relative names, never zip lists."""
    for relative in (low_relative, high_relative):
        parts = Path(relative).parts
        if Path(relative).is_absolute() or ".." in parts:
            raise ValueError("Require dataset-relative training paths")
        if any(part.lower() in {"test", "val", "eval15", "validation", "test_rllt"} for part in parts):
            raise ValueError("Held-out paths forbidden")
        if not any("train" in part.lower() or part.lower() == "our485" for part in parts):
            raise ValueError("Explicit training directory required")
    low_root, high_root = root / low_relative, root / high_relative
    extensions = {".png", ".jpg", ".jpeg"}
    low = {p.relative_to(low_root).as_posix(): p for p in low_root.rglob("*") if p.suffix.lower() in extensions}
    high = {p.relative_to(high_root).as_posix(): p for p in high_root.rglob("*") if p.suffix.lower() in extensions}
    if not low or low.keys() != high.keys():
        raise ValueError(f"Pair names must match exactly; low={len(low)} high={len(high)}")
    return [(name, low[name], high[name]) for name in sorted(low)]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--low", required=True)
    parser.add_argument("--high", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license-status", required=True)
    parser.add_argument("--input-origin", choices=("real_lowlight", "synthetic_darkening"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("E:/LumiCycleCache/experiments/night2day_diverse111_transpose1000/adaptation_02000.pt"))
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.sample_count <= 64:
        raise ValueError("Require new output and1..64 diagnostic examples")
    pairs = matching_pairs(args.root, args.low, args.high)
    selected = [pairs[int(i)] for i in np.linspace(0, len(pairs) - 1, min(args.sample_count, len(pairs)), dtype=int)]
    torch.set_num_threads(6)
    model, step = load_inverse(args.checkpoint)
    args.output.mkdir(parents=True)
    records, panels = [], []
    for number, (name, low, high) in enumerate(selected):
        with Image.open(low) as original, Image.open(high) as reference:
            if original.size != reference.size:
                raise ValueError("Mismatched pair shape")
            native_size = list(original.size)
            height = max(32, round(original.height / original.width * 512 / 4) * 4)
            source = original.convert("RGB").resize((512, height), Image.Resampling.LANCZOS)
            target = reference.convert("RGB").resize((512, height), Image.Resampling.LANCZOS)
        x = pil_to_tensor(source)[None].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(x)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite inverse")
        prediction = tensor_to_pil(result)
        folder = args.output / f"pair_{number:03d}"
        folder.mkdir()
        source.save(folder / "TRAIN_input.png")
        target.save(folder / "TRAIN_reference.png")
        prediction.save(folder / "LOCAL_prediction.png")
        panel = Image.new("RGB", (1536, height + 24), "#101010")
        for j, (label, im) in enumerate(((f"{args.input_origin} TRAIN", source),
            ("REAL normal-light TRAIN reference", target), (f"LOCAL inverse global{step}", prediction))):
            panel.paste(im, (512 * j, 24))
            ImageDraw.Draw(panel).text((512 * j + 4, 5), f"{name} {label}", fill="white")
        panels.append(panel)
        source01 = np.asarray(source, dtype=np.float32) / 255
        target01 = np.asarray(target, dtype=np.float32) / 255
        output01 = np.asarray(prediction, dtype=np.float32) / 255
        records.append({"name": name, "split": "train", "input": str(low.resolve()),
            "target": str(high.resolve()), "input_sha256": digest(low), "target_sha256": digest(high),
            "native_size": native_size, "resized_size": list(source.size),
            "input_mean_rgb": source01.mean((0, 1)).tolist(), "target_mean_rgb": target01.mean((0, 1)).tolist(),
            "local_rgb_l1_to_real_reference": float(np.abs(output01 - target01).mean())})
    for start in range(0, len(panels), 4):
        page = panels[start:start + 4]
        sheet = Image.new("RGB", (1536, sum(p.height for p in page)), "#101010")
        y = 0
        for panel in page:
            sheet.paste(panel, (0, y))
            y += panel.height
        sheet.save(args.output / f"TRAIN_audit_{start:02d}.jpg", quality=96)
    atomic_json_dump({"dataset_root": str(args.root), "source_url": args.source_url,
        "license_status": args.license_status, "input_origin": args.input_origin,
        "train_pair_count_by_filenames": len(pairs), "diagnostic_sample_count": len(records),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": digest(args.checkpoint), "global_step": step,
        "records": records, "test_opened": False, "validation_opened": False, "training_performed": False,
        "external_rgb_generator_used": False, "public_demo_unchanged": True,
        "warning": "TRAIN data audit, not generalization. Normal exposure need not mean daytime relighting.",
        "vision_review": "pending"}, args.output / "audit.json")
    print(json.dumps({"train_pairs_by_filenames": len(pairs), "rendered_diagnostics": len(records)}))


if __name__ == "__main__":
    main()
