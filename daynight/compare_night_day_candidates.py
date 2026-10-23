"""Matched experimental inverse comparisons with explicitly external teacher column."""
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


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate", action="append", required=True, help="Label=checkpoint path")
    parser.add_argument("--bypass-context-for", action="append", default=[], help="Experimental ablation label; all other trained weights unchanged")
    args = parser.parse_args()
    if (args.output / "evidence.json").exists():
        raise FileExistsError("Keep previous matched evidence")
    manifest = json.loads((args.targets / "manifest.json").read_text())
    if manifest["direction"] != "night_to_day" or manifest["test_used"]:
        raise ValueError("Require experimental inverse validation cache")
    rows = [r for r in manifest["records"] if r["split"] == "val"]
    if not rows:raise ValueError("No validation sources")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    candidates = [item.split("=", 1) for item in args.candidate]
    if len({label for label, _ in candidates}) != len(candidates):raise ValueError("Duplicate label")
    if not set(args.bypass_context_for) <= {label for label, _ in candidates}:
        raise ValueError("Bypass label has no candidate")
    results = {}; predictions = {}
    for label, path in candidates:
        model, step = load_inverse(Path(path))
        if label in args.bypass_context_for:
            if not hasattr(model, "scene_context"):
                raise ValueError("Cannot ablate a missing context module")
            model.scene_context = torch.nn.Identity()
        values = []
        for row in rows:
            source = Image.open(row["input"]).convert("RGB")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(pil_to_tensor(source)[None].cuda())
            if not torch.isfinite(output).all():raise FloatingPointError("Nonfinite comparison output")
            result = tensor_to_pil(output)
            folder = args.output / label; folder.mkdir(exist_ok=True)
            result.save(folder / f"val_{row['index']:04d}.png")
            predictions[(label, row["index"])] = result
            target = np.asarray(Image.open(row["target"]).convert("RGB"), dtype=np.float32) / 255
            array = np.asarray(result, dtype=np.float32) / 255
            values.append({"index": row["index"], "target_rgb_l1_0to1": float(np.abs(array - target).mean())})
        results[label] = {"checkpoint": path, "sha256": digest(path), "global_step": step,
            "scene_context_bypassed": label in args.bypass_context_for,
            "weights": "EMA G_night_day", "precision": "BF16", "per_image": values,
            "mean_target_rgb_l1_0to1": float(np.mean([r["target_rgb_l1_0to1"] for r in values]))}
        del model
        torch.cuda.empty_cache()
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]
        sheet = Image.new("RGB", (512 * (2 + len(candidates)), 312 * len(batch)), "#101010")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(batch):
            columns = [("Night source", Image.open(row["input"])), ("EXTERNAL teacher", Image.open(row["target"]))]
            columns += [(label, predictions[(label, row["index"])]) for label, _ in candidates]
            for k, (label, im) in enumerate(columns):
                sheet.paste(im, (512 * k, 312 * j + 24))
                draw.text((512 * k + 5, 312 * j + 5), f"VAL{row['index']} {label}", fill="white")
        sheet.save(args.output / f"comparison_{start:02d}.jpg", quality=95)
    atomic_json_dump({"split": "validation night", "count": len(rows), "test_used": False,
        "targets_manifest_sha256": digest(args.targets / "manifest.json"), "teacher_sha256": manifest["teacher_sha256"],
        "results": results, "warning": "Teacher agreement, not a visual/daytime acceptance score; inspect images."}, args.output / "evidence.json")
    print(json.dumps({key: value["mean_target_rgb_l1_0to1"] for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
