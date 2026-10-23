"""Audit real DAY warped into native NIGHT coordinates; never creates training data.

Flow is a frozen external correspondence estimator, not an image generator.
Raw, gamma-lifted and our local inverse inputs test cross-lighting sensitivity.
Forward/backward confidence is NOT a semantic static-object or correctness mask.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

from .align_lumirender_pairs import _tensor, _warp
from .experiment_night_to_day import load_inverse
from .turbo_pilot import digest
from .utils import atomic_json_dump, tensor_to_pil


def inverse_alignment_confidence(forward, backward):
    """Forward is NIGHT->DAY sampling displacement in pixels, not reverse flow."""
    sampled_backward, valid = _warp(backward, forward)
    error = (forward + sampled_backward).square().sum(1, keepdim=True).sqrt()
    return torch.exp(-error / 2) * valid, error


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("D:/datasets/lumirender/dark_zurich_train_pairs.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", default="0,400,800,1200,1600,2000")
    parser.add_argument("--checkpoint", type=Path,
        default=Path("E:/LumiCycleCache/experiments/night2day_diverse111_transpose1000/adaptation_02000.pt"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous audit; select a new output directory")
    rows = list(csv.DictReader(args.pairs.open(encoding="utf-8-sig", newline="")))
    indices = [int(i) for i in args.indices.split(",")]
    if any(not 0 <= i < len(rows) or rows[i]["split"] != "train" for i in indices):
        raise ValueError("Only existing training pairs may be inspected")
    torch.set_num_threads(6)
    device = torch.device("cuda")
    local, step = load_inverse(args.checkpoint)
    weights = Raft_Small_Weights.DEFAULT
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not weight_path.exists():
        raise FileNotFoundError("Require the previously cached RAFT weights; no implicit download")
    flow_model = raft_small(weights=None).to(device).eval()
    flow_model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    args.output.mkdir(parents=True)
    records = []
    for index in indices:
        row = rows[index]
        night = _tensor(Path(row["night_path"]), (512, 288), device)
        day = _tensor(Path(row["day_path"]), (512, 288), device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            translated = ((local(night * 2 - 1).float() + 1) / 2).clamp(0, 1)
        modes = {"raw": night, "gamma045": night.pow(.45), "local_inverse": translated}
        folder = args.output / f"pair_{index:04d}"
        folder.mkdir()
        sheet = Image.new("RGB", (2560, 312 * len(modes)), "#101010")
        per_mode = {}
        for j, (mode, source) in enumerate(modes.items()):
            normalized_source, normalized_day = weights.transforms()(source, day)
            forward = flow_model(normalized_source, normalized_day, num_flow_updates=12)[-1]
            backward = flow_model(normalized_day, normalized_source, num_flow_updates=12)[-1]
            aligned, in_bounds = _warp(day, forward)
            confidence, error = inverse_alignment_confidence(forward, backward)
            confidence *= in_bounds
            images = [tensor_to_pil(x * 2 - 1) for x in (source, day, aligned)]
            confidence_image = Image.fromarray((confidence[0, 0].cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
            overlay = tensor_to_pil((.5 * source + .5 * aligned) * 2 - 1)
            columns = list(zip((f"NIGHT coordinates ({mode})", "Unaligned REAL day", "REAL day warped into NIGHT", "FB consistency, NOT correctness", "50/50 overlay: inspect geometry"),
                images + [confidence_image, overlay]))
            for k, (label, image) in enumerate(columns):
                sheet.paste(image, (512 * k, 312 * j + 24))
                ImageDraw.Draw(sheet).text((512 * k + 5, 312 * j + 5), f"PAIR{index} {label}", fill="white")
                image.save(folder / f"{mode}_{k}.png")
            per_mode[mode] = {"mean_fb_confidence": float(confidence.mean()),
                "fraction_confidence_gt05": float((confidence > .5).float().mean()),
                "mean_fb_error_pixels": float(error.mean()), "mean_displacement_pixels": float(forward.square().sum(1).sqrt().mean())}
        sheet.save(args.output / f"pair_{index:04d}.jpg", quality=96)
        records.append({"index": index, "night_path": row["night_path"], "day_path": row["day_path"],
            "night_sha256": digest(Path(row["night_path"])), "day_sha256": digest(Path(row["day_path"])), "modes": per_mode})
        print(json.dumps(records[-1]), flush=True)
    atomic_json_dump({"purpose": "Alignment audit only; no approved training data",
        "coordinate_system": "Original night camera at512x288; DAY sampled using NIGHT->DAY flow",
        "external_flow_weights": str(weight_path), "flow_sha256": digest(weight_path),
        "local_checkpoint": str(args.checkpoint), "local_sha256": digest(args.checkpoint), "local_global_step": step,
        "public_demo_unchanged": True, "test_used": False, "lockbox_used": False,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "limitations": "FB consistency can be confidently wrong; no moving-object segmentation; inspect all panels",
        "records": records}, args.output / "audit.json")


if __name__ == "__main__":
    main()
