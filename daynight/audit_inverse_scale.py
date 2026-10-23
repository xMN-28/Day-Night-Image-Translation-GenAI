"""Local inverse scale/context diagnostic; no external RGB generator or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .experiment_night_to_day import load_inverse
from .turbo_pilot import digest
from .utils import atomic_json_dump, pil_to_tensor, tensor_to_pil


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=Path("E:/LumiCycleCache/teacher_targets/night2day_pilot128_v2"))
    parser.add_argument("--checkpoint", type=Path, default=Path("E:/LumiCycleCache/experiments/night2day_diverse111_transpose1000/adaptation_02000.pt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep earlier evidence")
    manifest = json.loads((args.targets / "manifest.json").read_text())
    if manifest["test_used"] or manifest.get("lockbox_used", False):
        raise ValueError("No test or lockbox evaluation")
    rows = [row for row in manifest["records"] if row["split"] == "val"]
    torch.set_num_threads(6)
    model, step = load_inverse(args.checkpoint)
    args.output.mkdir(parents=True)
    panels, records = [], []
    for row in rows:
        source = Image.open(row["input"]).convert("RGB")
        if source.size != (512, 288):
            raise ValueError("This controlled scale diagnostic requires512x288 cached inputs")
        folder = args.output / f"val_{row['index']:04d}"
        folder.mkdir()
        columns = [("Night source", source)]
        for width in (512, 256, 128):
            height = width * 288 // 512
            resized = source.resize((width, height), Image.Resampling.LANCZOS)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(pil_to_tensor(resized)[None].cuda())
            output = tensor_to_pil(prediction)
            output.save(folder / f"LOCAL_native_{width}.png")
            upsampled = output.resize(source.size, Image.Resampling.BICUBIC)
            columns.append((f"LOCAL global{step} input{width}x{height}", upsampled))
        panel = Image.new("RGB", (2048, 312), "#101010")
        for j, (label, image) in enumerate(columns):
            panel.paste(image, (512 * j, 24))
            ImageDraw.Draw(panel).text((512 * j + 5, 5), f"VAL{row['index']} {label}", fill="white")
        panels.append(panel)
        records.append({"index": row["index"], "source_sha256": digest(Path(row["input"]))})
    for start in range(0, len(panels), 4):
        page = panels[start:start + 4]
        sheet = Image.new("RGB", (2048, 312 * len(page)), "#101010")
        for j, panel in enumerate(page):
            sheet.paste(panel, (0, j * 312))
        sheet.save(args.output / f"scales_{start:02d}.jpg", quality=96)
    atomic_json_dump({"checkpoint": str(args.checkpoint), "sha256": digest(args.checkpoint), "global_step": step,
        "records": records, "test_used": False, "lockbox_used": False, "external_rgb_generator_used": False,
        "purpose": "Probe spatial context/scale dependence, NOT claim upsampled images have512px detail",
        "public_demo_unchanged": True}, args.output / "audit.json")


if __name__ == "__main__":
    main()
