"""Independent-mask diagnostics for the frozen V2 + semantic matte experiment."""

from __future__ import annotations

import argparse
import hashlib
import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .sky_pilot import load_base
from .sky_refinement import SkyStudent, compose_night, gray, phase_average, predict_matte
from .train_sky_student import CachedSkyDataset
from .utils import atomic_json_dump, tensor_to_pil


def run(args):
    torch.set_num_threads(6)
    base = load_base(Path("runs/lumicycle_v2_bdd100k/checkpoints/step_00004500.pt"))
    generator = load_base(args.generator_checkpoint) if args.generator_checkpoint else base
    checkpoint_bytes = args.student.read_bytes()
    state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    student = SkyStudent(outputs=state["model"]["head.weight"].shape[0]).cuda().eval()
    student.load_state_dict(state["model"])
    dataset = CachedSkyDataset(Path("data/sky_student_fullframe_v1"), "val")
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, (x, target) in enumerate(dataset):
            x, target = x[None].cuda(), target[None].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                old = (base(x * 2 - 1).float() + 1) / 2
                translated = (
                    phase_average(generator, x * 2 - 1)
                    if args.phase_average
                    else generator(x * 2 - 1)
                )
                raw = (translated.float() + 1) / 2
                a, vegetation = predict_matte(student, x)
            new = compose_night(x, raw, a, vegetation=vegetation)
            reference_sky = target > 0.9
            reference_foreground = target < 0.02
            nsky = int(reference_sky.sum())

            def masked(value, mask):
                return float((value * mask).sum() / mask.sum().clamp_min(1))

            rows.append(
                {
                    "index": index,
                    "sky_pixels": nsky,
                    "base_sky_luminance": masked(gray(old), reference_sky),
                    "new_sky_luminance": masked(gray(new), reference_sky),
                    "base_bright_sky_fraction": masked((gray(old) > 0.25).float(), reference_sky),
                    "new_bright_sky_fraction": masked((gray(new) > 0.25).float(), reference_sky),
                    "foreground_change_l1": masked(
                        (new - old).abs().mean(1, True), reference_foreground
                    ),
                    "alpha_mae": float((a - target).abs().mean()),
                }
            )
            if index < 16:
                canvas = Image.new("RGB", (384 * 3, 240), "#101010")
                draw = ImageDraw.Draw(canvas)
                for j, (im, label) in enumerate(
                    zip((x, old, new), ("Input", "V2 4.5k", "V2 + student matte"))
                ):
                    draw.text((j * 384 + 5, 5), label, fill="white")
                    canvas.paste(tensor_to_pil(im * 2 - 1), (j * 384, 24))
                canvas.save(args.output / f"val_{index:02d}.jpg", quality=94)
        # Prepared 512-pixel inference; network loading excluded, warm calls only.
        probe = F.interpolate(x, (288, 512), mode="bilinear", align_corners=False)
        timings = []
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(7):
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                translated = (
                    phase_average(generator, probe * 2 - 1)
                    if args.phase_average
                    else generator(probe * 2 - 1)
                )
                old = (translated.float() + 1) / 2
                a, vegetation = predict_matte(student, probe)
            compose_night(probe, old, a, vegetation=vegetation)
            torch.cuda.synchronize()
            if repeat >= 2:
                timings.append(time.perf_counter() - started)
    with_sky = [r for r in rows if r["sky_pixels"] > 100]
    summary = {
        "split": "val",
        "count": len(rows),
        "with_sky": len(with_sky),
        "head_step": state["step"],
        "head_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "generator_checkpoint": str(args.generator_checkpoint),
        "phase_average": args.phase_average,
        "reference_masks": "independent frozen teacher pseudo-labels; imperfect, not human ground truth",
        "base_bright_sky_fraction_mean": float(
            np.mean([r["base_bright_sky_fraction"] for r in with_sky])
        ),
        "new_bright_sky_fraction_mean": float(
            np.mean([r["new_bright_sky_fraction"] for r in with_sky])
        ),
        "median_512_tensor_inference_seconds": float(np.median(timings)),
        "timing_scope": "V2 day-to-night, sky student, and matting tensor operations; image I/O and loading excluded",
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "records": rows,
        "acceptance": "requires visual review, these diagnostics alone do not prove realism",
    }
    atomic_json_dump(summary, args.output / "metrics.json")
    print({k: v for k, v in summary.items() if k != "records"}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, default=Path("runs/sky_student_v1/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sky_pilot/validation_metrics"))
    parser.add_argument("--generator-checkpoint", type=Path)
    parser.add_argument("--phase-average", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
