"""Private bounded similarity registration from cached flow correspondences.

Locally implemented deterministic RANSAC/least-squares, no new model/download.
No training targets are automatically approved by geometric heuristics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .align_lumirender_pairs import _tensor, _warp
from .turbo_pilot import digest
from .utils import atomic_json_dump, tensor_to_pil


def solve_similarity(source, target):
    """Four-DOF least squares: u=ax-by+tx; v=bx+ay+ty."""
    x, y = source.T
    a = np.zeros((2 * len(x), 4))
    a[0::2] = np.stack((x, -y, np.ones_like(x), np.zeros_like(x)), 1)
    a[1::2] = np.stack((y, x, np.zeros_like(x), np.ones_like(x)), 1)
    params, _, rank, _ = np.linalg.lstsq(a, target.reshape(-1), rcond=None)
    if rank != 4:
        raise ValueError("Degenerate similarity fit")
    s, r, tx, ty = params
    return np.array([[s, -r, tx], [r, s, ty]])


def apply_similarity(points, matrix):
    return points @ matrix[:, :2].T + matrix[:, 2]


def bounded_similarity(source, target, size):
    """Reject rather than silently using an excessive/unsupported correction."""
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("Require matching Nx2points")
    info = {"points": len(source), "accepted_geometry": False}
    if len(source) < 40 or not np.isfinite(source).all() or not np.isfinite(target).all():
        return None, {**info, "reason": "insufficient/nonfinite support"}
    rng = np.random.default_rng(2038)
    best = np.zeros(len(source), dtype=bool)
    for _ in range(256):
        ids = rng.choice(len(source), 2, replace=False)
        if np.linalg.norm(source[ids[0]] - source[ids[1]]) < 32:
            continue
        matrix = solve_similarity(source[ids], target[ids])
        inliers = np.linalg.norm(apply_similarity(source, matrix) - target, axis=1) < 2
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 40:
        return None, {**info, "reason": "insufficient RANSAC inliers"}
    matrix = solve_similarity(source[best], target[best])
    error = np.linalg.norm(apply_similarity(source, matrix) - target, axis=1)
    best = error < 2
    w, h = size
    coverage = np.ptp(source[best], axis=0) / [w, h]
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    max_move = float(np.linalg.norm(apply_similarity(corners, matrix) - corners, axis=1).max())
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    degrees = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    info.update({"inliers": int(best.sum()), "inlier_fraction": float(best.mean()),
        "median_inlier_error": float(np.median(error[best])), "coverage_xy": coverage.tolist(),
        "max_corner_displacement": max_move, "scale": scale, "rotation_degrees": degrees,
        "night_to_day_matrix": matrix.tolist()})
    passed = (best.sum() >= 40 and best.mean() >= .5 and (coverage >= .3).all()
        and max_move <= 8 and .98 <= scale <= 1.02 and abs(degrees) <= 1)
    info.update({"accepted_geometry": bool(passed), "reason": "bounded fit" if passed else "fails fixed geometric gate"})
    return (matrix if passed else None), info


def similarity_flow(matrix, h, w):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    xy = torch.stack((x, y), 0).float()[None]
    m = torch.as_tensor(matrix, dtype=torch.float32)
    mapped = torch.einsum("ij,bjhw->bihw", m[:, :2], xy) + m[:, 2][None, :, None, None]
    return mapped - xy


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path,
        default=Path("E:/LumiCycleCache/experiments/DarkDriving_private_audit12_20261023/audit.json"))
    parser.add_argument("--alignment", type=Path,
        default=Path("E:/LumiCycleCache/experiments/DarkDriving_private_alignment12_20261023/audit.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Retain earlier evidence")
    original = json.loads(args.pairs.read_text())
    audit = json.loads(args.alignment.read_text())
    if audit["source_audit_sha256"] != digest(args.pairs) or audit["test_used"] or audit["lockbox_used"]:
        raise ValueError("Require exact corresponding TRAIN audit")
    torch.set_num_threads(6)
    args.output.mkdir(parents=True)
    records = []
    for number, (row, aligned) in enumerate(zip(original["records"], audit["records"], strict=True)):
        if row["name"] != aligned["name"] or row["split"] != "train":
            raise ValueError("Pair order/split mismatch")
        for key in ("night", "day"):
            if digest(Path(row[key])) != row[f"{key}_sha256"]:
                raise ValueError("Original changed")
        w, h = aligned["size"]
        night = _tensor(Path(row["night"]), (w, h), torch.device("cpu"))
        day = _tensor(Path(row["day"]), (w, h), torch.device("cpu"))
        folder = args.output / Path(row["name"]).stem
        folder.mkdir()
        modes = {}
        sheet = Image.new("RGB", (2560, 2 * (h + 40)), "#101010")
        for j, mode in enumerate(("raw", "gamma045")):
            data = np.load(Path(aligned["folder"]) / f"{mode}_geometry.npz")
            flow, confidence = data["forward"][0], data["reliability"][0, 0]
            yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            chosen = (confidence > .5) & (xx % 8 == 0) & (yy % 8 == 0)
            source = np.stack((xx[chosen], yy[chosen]), 1)
            target = source + flow[:, chosen].T
            matrix, info = bounded_similarity(source, target, (w, h))
            modes[mode] = info
            if matrix is None:
                # Display original only, explicitly mark fit rejected.
                result, valid = day, torch.ones(1, 1, h, w)
            else:
                result, valid = _warp(day, similarity_flow(matrix, h, w))
            overlay = tensor_to_pil((.5 * night.pow(.45) + .5 * result) * 2 - 1)
            columns = [("Gamma visibility, NOT daylight", tensor_to_pil(night.pow(.45) * 2 - 1)),
                ("UNWARPED day", tensor_to_pil(day * 2 - 1)),
                ("Bounded day" if matrix is not None else "FIT REJECTED: unwarped day", tensor_to_pil(result * 2 - 1)),
                ("NIGHT/day overlay", overlay),
                ("In-bounds ONLY, NOT confidence", Image.fromarray((valid[0, 0].numpy() * 255).astype(np.uint8)).convert("RGB"))]
            for k, (label, im) in enumerate(columns):
                sheet.paste(im, (k * 512, j * (h + 40) + 40))
                ImageDraw.Draw(sheet).text((k * 512 + 4, j * (h + 40) + 4), f"{mode} {row['name']}\n{label}", fill="white")
                im.save(folder / f"{mode}_{k}.png")
        sheet.save(args.output / f"PRIVATE_rigid_{number:02d}.jpg", quality=95)
        records.append({"name": row["name"], "modes": modes, "approved_for_training": False})
    atomic_json_dump({"records": records, "alignment_sha256": digest(args.alignment),
        "method": "Deterministic256trial two-pointRANSAC plus four-DOF leastsquares; raw/gammaflowFBsmooth>.5;8pxgrid",
        "fixed_gate": "40inliers;>=.5fraction;xycoverage>=.3;maxcorner8px;scale[.98,1.02];rotation<=1degree",
        "warning": "Geometric gate NOT semantic correctness, dynamic-object mask or training approval; rejectedfits displayedunwarped",
        "test_used": False, "lockbox_used": False, "training_performed": False,
        "scope": "Private academic only; no redistribution", "vision_review": "pending",
        "public_demo_unchanged": True}, args.output / "audit.json")


if __name__ == "__main__":
    main()
