"""Explicitly heuristic teacher-glare reliability, not semantic segmentation.

Only relax supervision around bright original-night cores that remain bright in
the pseudo-target. Do not use this as an object-removal or sky correctness mask.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .utils import atomic_json_dump, pil_to_tensor


def glare_reliability(source, target, floor=.05):
    if source.shape != target.shape or source.ndim != 4 or source.shape[1] != 3:
        raise ValueError("Require matched NCHW RGB tensors")
    if not 0 < floor <= 1:
        raise ValueError("Reliability floor must be in (0,1]")
    with torch.no_grad():
        rgb = ((source.float() + 1) / 2).clamp(0, 1)
        day = ((target.float() + 1) / 2).clamp(0, 1)
        coefficients = rgb.new_tensor([.299, .587, .114])[None, :, None, None]
        luma = (rgb * coefficients).sum(1, keepdim=True)
        target_luma = (day * coefficients).sum(1, keepdim=True)
        core = ((rgb.amax(1, keepdim=True) > .95) & (luma > .6)).float()
        support = F.avg_pool2d(F.max_pool2d(core, 17, 1, 8), 9, 1, 4)
        retained_brightness = ((target_luma - .45) / .25).clamp(0, 1)
        return (1 - (1 - floor) * support * retained_brightness).detach()


def reliability_weighted_loss(prediction, target, reliability):
    """Full-image denominator prevents upweighting unrelated reliable pixels."""
    if reliability.shape != (prediction.shape[0], 1, *prediction.shape[-2:]):
        raise ValueError("Require N1HW weights matching predictions")
    p, t, w = prediction.float(), target.float(), reliability.float()
    rgb = ((p - t).abs() * w).mean()
    coarse = sum(((F.avg_pool2d(p, k) - F.avg_pool2d(t, k)).abs()
        * F.avg_pool2d(w, k)).mean() for k in (4, 16)) / 2
    gradients = []
    for axis in (-1, -2):
        wa, wb = (w[..., 1:], w[..., :-1]) if axis == -1 else (w[..., 1:, :], w[..., :-1, :])
        gradients.append(((torch.diff(p, dim=axis) - torch.diff(t, dim=axis)).abs()
            * torch.minimum(wa, wb)).mean())
    return 10 * rgb + 2 * coarse + sum(gradients) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=Path("E:/LumiCycleCache/teacher_targets/night2day_pilot128_v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--count", type=int, default=0, help="0=all; positive=evenly spaced audit subset")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous audit")
    manifest = json.loads((args.targets / "manifest.json").read_text())
    if manifest["test_used"] or manifest.get("lockbox_used", False):
        raise ValueError("No test or lockbox masks")
    rows = [row for row in manifest["records"] if row["split"] == args.split]
    if args.split == "train":
        selected = set(json.loads((args.targets / "visual_selection_v1.json").read_text())["accepted_train_indices"])
        rows = [row for row in rows if row["index"] in selected]
    if args.count < 0 or not rows:
        raise ValueError("Require nonnegative count and nonempty audited split")
    if args.count and len(rows) > args.count:
        rows = [rows[int(i)] for i in np.linspace(0, len(rows) - 1, args.count, dtype=int)]
    args.output.mkdir(parents=True)
    torch.set_num_threads(6)
    panels, records = [], []
    for row in rows:
        source, target = [Image.open(row[key]).convert("RGB") for key in ("input", "target")]
        x, y = pil_to_tensor(source)[None], pil_to_tensor(target)[None]
        weights = glare_reliability(x, y)[0, 0].numpy()
        freedom = Image.fromarray(np.rint((1 - weights) * 255).astype(np.uint8)).convert("RGB")
        overlay = np.asarray(source).copy().astype(np.float32)
        alpha = (1 - weights)[..., None] * .65
        overlay = Image.fromarray(np.rint(overlay * (1 - alpha) + np.array([255, 0, 0]) * alpha).astype(np.uint8))
        panel = Image.new("RGB", (2048, 312), "#101010")
        for j, (label, image) in enumerate((("Night source", source), ("EXTERNAL target", target),
            ("WHITE = relaxed target supervision", freedom), ("RED = affected regions, NOT semantics", overlay))):
            panel.paste(image, (512 * j, 24))
            ImageDraw.Draw(panel).text((512 * j + 5, 5), f"{args.split.upper()}{row['index']} {label}", fill="white")
        panels.append(panel)
        records.append({"index": row["index"], "fraction_weight_lt05": float((weights < .5).mean()),
            "fraction_weight_lt095": float((weights < .95).mean()), "mean_weight": float(weights.mean())})
    for start in range(0, len(panels), 4):
        page = panels[start:start + 4]
        sheet = Image.new("RGB", (2048, 312 * len(page)), "#101010")
        for j, panel in enumerate(page):
            sheet.paste(panel, (0, j * 312))
        sheet.save(args.output / f"regions_{start:02d}.jpg", quality=96)
    atomic_json_dump({"purpose": "Audit heuristic target reliability, not mask accuracy",
        "split": args.split, "test_used": False, "lockbox_used": False, "records": records,
        "limitations": "Bright reflective surfaces and lamps may both be selected; no hard semantic labels; no sky-repair claim"},
        args.output / "audit.json")


if __name__ == "__main__":
    main()
