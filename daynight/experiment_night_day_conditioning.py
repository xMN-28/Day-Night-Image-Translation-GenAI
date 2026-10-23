"""Photometric input ablations of the EXTERNAL teacher, not learned local fixes."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import read_manifest
from .turbo_pilot import digest, fit_input, load_official
from .utils import atomic_json_dump, tensor_to_pil


def condition_source(image, method):
    if method == "raw":return image.copy()
    if method not in {"rolloff", "balanced_rolloff"}:raise ValueError("Unknown source conditioning")
    x = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    if method == "balanced_rolloff":x = x ** 0.7
    brightness = x.max(2, keepdims=True)
    # Monotone compressive curve, preserves per-pixel RGB ratios after optional gamma.
    # No deconvolution/inpainting: this cannot recover clipped light-source detail.
    x = x * 0.7 / (0.7 + brightness)
    return Image.fromarray(np.rint(x.clip(0, 1) * 255).astype(np.uint8))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--indices", default="0,12,22,105,142,171,283,441")
    parser.add_argument("--cache-root", type=Path, default=Path("E:/LumiCycleCache"))
    args = parser.parse_args()
    if (args.output / "evidence.json").exists():raise FileExistsError("Keep prior evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    records = read_manifest("data/bdd100k_daynight", "val", "night")
    model, checkpoint, commit = load_official(args.cache_root, "night_to_day")
    methods = ("raw", "rolloff", "balanced_rolloff")
    evidence = {"teacher": "EXTERNAL CycleGAN-Turbo with custom photometric input ablations",
        "teacher_sha256": digest(checkpoint), "source_commit": commit, "seed_base": 4400,
        "split": "validation night", "test_used": False, "precision": "FP16", "edge": 512, "records": []}
    for index in map(int, args.indices.split(",")):
        source, _, _ = fit_input(Image.open(records[index].path), 512)
        folder = args.output / f"val_{index:04d}"; folder.mkdir(exist_ok=True)
        source.save(folder / "source.png")
        panel = Image.new("RGB", (2048, 312), "#101010")
        panel.paste(source, (0, 24)); draw = ImageDraw.Draw(panel)
        draw.text((5, 5), f"VAL{index} Night source", fill="white")
        for j, method in enumerate(methods):
            conditioned = condition_source(source, method)
            conditioned.save(folder / f"{method}_input.png")
            _, x, crop = fit_input(conditioned, 512)
            torch.manual_seed(4400 + index); torch.cuda.manual_seed_all(4400 + index)
            output = model(x.cuda().half())
            if not torch.isfinite(output).all():raise FloatingPointError("Nonfinite reference ablation")
            result = tensor_to_pil(output).crop(crop)
            result.save(folder / f"EXTERNAL_{method}.png")
            panel.paste(result, ((j + 1) * 512, 24))
            draw.text(((j + 1) * 512 + 5, 5), f"EXTERNAL + {method}", fill="white")
        panel.save(args.output / f"comparison_{index:04d}.jpg", quality=95)
        evidence["records"].append({"index": index, "source": str(records[index].path), "source_sha256": digest(records[index].path)})
    atomic_json_dump(evidence, args.output / "evidence.json")
    print(f"Generated{len(evidence['records'])}matched conditioning ablations; no local training.")


if __name__ == "__main__":
    main()
