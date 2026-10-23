"""Audit N2D3-inspired photometric regions before using them in any loss.

Implements its published Gaussian-color derivative equations with explicit
stabilization choices. These maps are NOT semantic labels or an exact N2D3
reproduction, and this command performs no model training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .turbo_pilot import digest
from .utils import atomic_json_dump, pil_to_tensor

PALETTE = np.asarray([(30, 55, 120), (60, 190, 220), (255, 220, 55), (75, 220, 90)], dtype=np.uint8)


def color_invariant(rgb01, sigma=1.0, energy_floor=.02):
    """Equations5–8;NCHW approximate sRGB, not measured spectral radiance."""
    if rgb01.ndim != 4 or rgb01.shape[1] != 3 or min(rgb01.shape[-2:]) < 4:
        raise ValueError("Require N3HW with sides >=4")
    if sigma <= 0 or energy_floor <= 0:
        raise ValueError("Positive stabilization settings required")
    rgb01 = rgb01.float()
    matrix = rgb01.new_tensor([[.06, .63, .27], [.30, .04, -.35], [.34, -.60, .17]])
    energy = F.conv2d(rgb01, matrix[:, :, None, None])
    axis = torch.arange(-3, 4, device=rgb01.device).float()
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    gaussian = torch.exp(-(xx.square() + yy.square()) / (2 * sigma ** 2))
    gaussian /= gaussian.sum()
    gx, gy = -xx * gaussian / sigma ** 2, -yy * gaussian / sigma ** 2
    padded = F.pad(energy, (3, 3, 3, 3), mode="reflect")
    derivatives = [F.conv2d(padded, kernel.expand(3, 1, 7, 7), groups=3) for kernel in (gx, gy)]
    e, el, ell = energy.split(1, 1)
    safe = e.clamp_min(energy_floor)
    terms = []
    for derivative in derivatives:
        ex, elx, ellx = derivative.split(1, 1)
        first = (elx * e - el * ex) / safe.square()
        second = (ellx * e.square() - ell * ex * e - 2 * elx * el * e + 2 * el.square() * ex) / safe.pow(3)
        terms.extend((first, second))
    return torch.sqrt(sum(term.square() for term in terms) + 1e-12)


def illuminance_clusters(rgb01, iterations=25):
    """Deterministic per-image3means on maxRGB; init0/.5/1, empty bins retained."""
    if rgb01.ndim != 4 or rgb01.shape[1] != 3:
        raise ValueError("Require N3HW")
    values = rgb01.float().amax(1)
    labels, all_centers, degenerate = [], [], []
    for image in values:
        flat = image.flatten()
        centers = flat.new_tensor([0, .5, 1])
        for _ in range(iterations):
            assignment = (flat[:, None] - centers[None]).square().argmin(1)
            updated = torch.stack([flat[assignment == i].mean() if (assignment == i).any() else centers[i]
                for i in range(3)]).sort().values
            if torch.max((updated - centers).abs()) < 1e-6:
                centers = updated
                break
            centers = updated
        assignment = (flat[:, None] - centers[None]).square().argmin(1)
        labels.append(assignment.reshape(image.shape))
        all_centers.append(centers)
        degenerate.append(int(assignment.unique().numel()) < 3)
    return torch.stack(labels), torch.stack(all_centers), degenerate


def degradation_memberships(rgb01, sigma=1.0, energy_floor=.02):
    labels, centers, degenerate = illuminance_clusters(rgb01)
    invariant = color_invariant(rgb01, sigma, energy_floor)
    mean = invariant.mean((-2, -1), keepdim=True)
    std = invariant.std((-2, -1), unbiased=False, keepdim=True).clamp_min(1e-6)
    z = ((invariant - mean) / std).relu()
    # PaperEq9 can exceed1; bounded memberships are our explicit audit choice.
    effects = z.clamp(0, 1) * (labels == 1)[:, None]
    masks = torch.cat(((labels == 0)[:, None].float(), (labels == 1)[:, None].float() - effects,
        (labels == 2)[:, None].float(), effects), 1)
    return masks, invariant, labels, centers, degenerate, z


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=Path("E:/LumiCycleCache/teacher_targets/night2day_pilot128_v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--count", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or args.count < 0:
        raise ValueError("Require new output and nonnegative count")
    manifest = json.loads((args.targets / "manifest.json").read_text())
    if manifest.get("test_used") or manifest.get("lockbox_used"):
        raise ValueError("No held-out evaluation")
    rows = [r for r in manifest["records"] if r["split"] == args.split]
    if args.split == "train":
        accepted = set(json.loads((args.targets / "visual_selection_v1.json").read_text())["accepted_train_indices"])
        rows = [r for r in rows if r["index"] in accepted]
    if args.count and args.count < len(rows):
        rows = [rows[int(i)] for i in np.linspace(0, len(rows) - 1, args.count, dtype=int)]
    if not rows:
        raise ValueError("No selected sources")
    torch.set_num_threads(6)
    args.output.mkdir(parents=True)
    panels, records = [], []
    for row in rows:
        source = Image.open(row["input"]).convert("RGB")
        x = (pil_to_tensor(source)[None] + 1) / 2
        masks, invariant, labels, centers, degenerate, z = degradation_memberships(x)
        if not torch.isfinite(masks).all() or not torch.isfinite(invariant).all():
            raise FloatingPointError("Nonfinite region calculation")
        selected = masks[0].argmax(0).numpy()
        colors = Image.fromarray(PALETTE[selected])
        cluster_colors = Image.fromarray(PALETTE[labels[0].numpy()])
        # Use fixed normalizedzscale for comparable display, not per-case tuning.
        effect = Image.fromarray(np.rint(masks[0, 3].numpy() * 255).astype(np.uint8)).convert("RGB")
        heatmap = Image.fromarray(np.rint((z[0, 0].clamp(0, 3) / 3).numpy() * 255).astype(np.uint8)).convert("RGB")
        overlay = Image.blend(source, colors, .4)
        folder = args.output / f"{args.split}_{row['index']:04d}"
        folder.mkdir()
        colors.save(folder / "PREDICTED_regions_NOT_semantics.png")
        effect.save(folder / "light_effect_membership.png")
        np.savez_compressed(folder / "photometric_memberships.npz", memberships=masks.numpy(), invariant=invariant.numpy())
        columns = [("Night source", source), ("3means maxRGB", cluster_colors), ("Invariant z/3", heatmap),
            ("Bounded effect membership", effect), ("4region overlay NOT semantics", overlay)]
        panel = Image.new("RGB", (512 * len(columns), source.height + 24), "#101010")
        for j, (label, im) in enumerate(columns):
            panel.paste(im, (512 * j, 24))
            ImageDraw.Draw(panel).text((512 * j + 5, 5), f"{args.split}{row['index']} {label}", fill="white")
        panels.append(panel)
        records.append({"index": row["index"], "input_sha256": digest(Path(row["input"])),
            "centers": centers[0].tolist(), "degenerate_clusters": degenerate[0],
            "region_mean_memberships": masks[0].mean((-2, -1)).tolist(),
            "welllit_membership_gt05": float((masks[0, 1] > .5).float().mean()),
            "effect_membership_gt05": float((masks[0, 3] > .5).float().mean()),
            "raw_effect_z_gt1": float((z > 1).float().mean())})
    for start in range(0, len(panels), 4):
        page = panels[start:start + 4]
        sheet = Image.new("RGB", (2560, sum(p.height for p in page)), "#101010")
        top = 0
        for panel in page:
            sheet.paste(panel, (0, top))
            top += panel.height
        sheet.save(args.output / f"photometry_{start:02d}.jpg", quality=96)
    atomic_json_dump({"paper": "https://arxiv.org/html/2411.14504v1", "split": args.split,
        "manifest_sha256": digest(args.targets / "manifest.json"), "records": records,
        "constants": {"derivative_kernel": 7, "sigma": 1, "energy_floor": .02,
            "kmeans_initial_centers": [0, .5, 1], "kmeans_iterations": 25, "effect_z_clamp": [0, 1]},
        "limitations": "Model-based illumination hypotheses, NOT semantic truth or exact N2D3 reproduction. Numerical floor, finite derivative support and boundedEq9 are explicit implementation choices.",
        "test_used": False, "lockbox_used": False, "training_performed": False,
        "public_demo_unchanged": True, "vision_review": "pending"}, args.output / "audit.json")


if __name__ == "__main__":
    main()
