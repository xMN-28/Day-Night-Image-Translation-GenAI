"""Paired decoder-fidelity/periodic-artifact diagnostics, not realism scores."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .sky_pilot import load_base
from .sky_refinement import box, gray, phase_average
from .train_sky_student import CachedSkyDataset
from .utils import atomic_json_dump


def periodic_bias(image, mask, period=4):
    """Phase-locked high-pass mean on smooth, eroded reference sky.

    Reports channel/phase dispersion, not total high-frequency energy. Also
    measure target gradient error: blurring all details can otherwise game this.
    """
    residual = image - box(image, 2)
    means = []
    for dy in range(period):
        for dx in range(period):
            weight = mask[..., dy::period, dx::period]
            values = residual[..., dy::period, dx::period]
            means.append((values * weight).sum((2, 3)) / weight.sum((2, 3)).clamp_min(1))
    return float(torch.stack(means).std(0, unbiased=False).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/resize_decoder_pilot/best.pt")
    )
    parser.add_argument(
        "--parent", type=Path, default=Path("runs/lumicycle_v2_skygan_pilot/latest.pt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/sky_pilot/resize_decoder_metrics")
    )
    args = parser.parse_args()
    torch.set_num_threads(6)
    teacher, candidate = load_base(args.parent), load_base(args.checkpoint)
    dataset = CachedSkyDataset(Path("data/sky_student_fullframe_v1"), "val")
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, (x, alpha) in enumerate(dataset):
            x, alpha = x[None].cuda(), alpha[None].cuda()
            source = x * 2 - 1
            raw = teacher(source).float()
            target = phase_average(teacher, source)
            new = candidate(source).float()
            # Sky-only mask excludes high-contrast source boundaries and a 5px border.
            eroded = 1 - F.max_pool2d(1 - (alpha > 0.98).float(), 11, 1, 5)
            smooth = (box((gray(x) - box(gray(x), 2)).abs(), 2) < 0.012).float()
            mask = eroded * smooth
            mask[..., :5, :] = mask[..., -5:, :] = 0
            mask[..., :, :5] = mask[..., :, -5:] = 0
            row = {"index": index, "smooth_sky_pixels": int(mask.sum())}
            for name, im in (("raw_parent", raw), ("phase_teacher", target), ("resize", new)):
                row[f"{name}_period4"] = periodic_bias((im + 1) / 2, mask)
                row[f"{name}_target_l1"] = float((im - target).abs().mean() / 2)
                row[f"{name}_target_gradient_l1"] = (
                    sum(
                        float((torch.diff(im, dim=d) - torch.diff(target, dim=d)).abs().mean() / 2)
                        for d in (-1, -2)
                    )
                    / 2
                )
                row[f"{name}_mean_luminance"] = float(gray((im + 1) / 2).mean())
            rows.append(row)
        # Identical prepared tensor; model loading/image IO excluded; time each separately.
        probe = F.interpolate(source, (288, 512), mode="bilinear", align_corners=False)
        latency = {}
        for name, fn in (
            ("phase_teacher", lambda: phase_average(teacher, probe)),
            ("resize", lambda: candidate(probe)),
        ):
            times = []
            for repeat in range(9):
                torch.cuda.synchronize()
                started = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                if repeat >= 3:
                    times.append(time.perf_counter() - started)
            latency[name] = float(np.median(times))
    eligible = [r for r in rows if r["smooth_sky_pixels"] >= 100]
    metrics = {
        key: float(np.mean([r[key] for r in eligible]))
        for key in rows[0]
        if key not in {"index", "smooth_sky_pixels"}
    }
    result = {
        "split": "val",
        "count": len(rows),
        "eligible_smooth_sky_count": len(eligible),
        "checkpoint": str(args.checkpoint),
        "parent": str(args.parent),
        "metrics_mean_eligible": metrics,
        "median_512_tensor_seconds": latency,
        "timing_scope": "generator only, prepared 512x288 tensor, warm CUDA calls",
        "limitation": "fidelity/phase-bias diagnostics, not night realism or detector retention",
        "records": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(result, args.output / "metrics.json")
    print({k: v for k, v in result.items() if k != "records"}, flush=True)


if __name__ == "__main__":
    main()
