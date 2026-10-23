"""Rank original real training pairs for VISUAL overlap review, never auto-approve.

Frozen external RAFT estimates correspondence, not daytime image generation.
Confidence and deformation scores are ranking heuristics, NOT ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

from .align_lumirender_pairs import _tensor, _warp
from .audit_inverse_alignment import inverse_alignment_confidence
from .turbo_pilot import digest
from .utils import atomic_json_dump, tensor_to_pil


def spaced_unique_day_pairs(rows, count):
    if count < 1:
        raise ValueError("Positive candidate count required")
    unique = {}
    for index, row in enumerate(rows):
        if row["split"] == "train":
            unique.setdefault(str(Path(row["day_path"]).resolve()), (index, row))
    candidates = list(unique.values())
    if not candidates:
        raise ValueError("No training pairs")
    return [candidates[int(i)] for i in np.linspace(0, len(candidates) - 1, min(count, len(candidates)), dtype=int)]


def deformation_weight(flow):
    """Downrank locally wavy displacement, not a certified geometric verifier."""
    dx = F.pad(torch.diff(flow, dim=-1), (0, 1, 0, 0), mode="replicate")
    dy = F.pad(torch.diff(flow, dim=-2), (0, 0, 0, 1), mode="replicate")
    magnitude = torch.maximum(dx.abs().amax(1, keepdim=True), dy.abs().amax(1, keepdim=True))
    return torch.exp(-magnitude / .5)


def render_candidates(output, results, start, count):
    for rank, row in enumerate(results[start:start + count], start=start):
        folder = Path(row["folder"])
        sheet = Image.new("RGB", (2560, 312), "#101010")
        names = ("NATIVE_night", "REAL_day_unaligned", "REAL_day_warped", "heuristic_confidence", "overlay")
        for j, name in enumerate(names):
            with Image.open(folder / f"{name}.png") as image:
                sheet.paste(image.convert("RGB"), (512 * j, 24))
            ImageDraw.Draw(sheet).text((512 * j + 5, 5), f"RANK{rank+1} PAIR{row['index']} {name}", fill="white")
        sheet.save(output / f"rank_{rank+1:02d}_pair_{row['index']:04d}.jpg", quality=96)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("D:/datasets/lumirender/dark_zurich_train_pairs.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=96)
    parser.add_argument("--review-count", type=int, default=12)
    parser.add_argument("--review-start", type=int, default=0)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        if args.review_start < 0 or args.review_count < 1:
            raise ValueError("Require positive rendering count and nonnegative start")
        saved = json.loads((args.output / "candidates.json").read_text())
        if saved["test_used"] or saved["lockbox_used"]:
            raise ValueError("No held-out pair rendering")
        render_candidates(args.output, saved["ranked_candidates"], args.review_start, args.review_count)
        return
    if args.output.exists() or not 1 <= args.count <= 256 or not 1 <= args.review_count <= args.count:
        raise ValueError("Require new output and bounded positive counts (<=256)")
    rows = list(csv.DictReader(args.pairs.open(encoding="utf-8-sig", newline="")))
    selected = spaced_unique_day_pairs(rows, args.count)
    torch.set_num_threads(6)
    weights = Raft_Small_Weights.DEFAULT
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not checkpoint.exists():
        raise FileNotFoundError("Use existing cached RAFT checkpoint; no implicit download")
    model = raft_small(weights=None).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    args.output.mkdir(parents=True)
    results = []
    for number, (index, row) in enumerate(selected):
        night = _tensor(Path(row["night_path"]), (512, 288), torch.device("cuda"))
        day = _tensor(Path(row["day_path"]), (512, 288), torch.device("cuda"))
        best, modes = None, {}
        for mode, source in (("raw", night), ("gamma045", night.pow(.45))):
            a, b = weights.transforms()(source, day)
            forward = model(a, b, num_flow_updates=12)[-1]
            backward = model(b, a, num_flow_updates=12)[-1]
            aligned, valid = _warp(day, forward)
            confidence, _ = inverse_alignment_confidence(forward, backward)
            confidence *= valid
            reliability = confidence * deformation_weight(forward)
            score = float((reliability > .5).float().mean())
            modes[mode] = {"fraction_fb_gt05": float((confidence > .5).float().mean()),
                "fraction_fb_smooth_gt05": score, "mean_fb_smooth": float(reliability.mean())}
            if best is None or score > best[0]:
                best = (score, mode, aligned.clone(), reliability.clone())
        score, mode, aligned, reliability = best
        folder = args.output / f"pair_{index:04d}"
        folder.mkdir()
        for name, tensor in (("NATIVE_night", night), ("REAL_day_unaligned", day), ("REAL_day_warped", aligned)):
            tensor_to_pil(tensor * 2 - 1).save(folder / f"{name}.png")
        Image.fromarray(np.rint(reliability[0, 0].cpu().numpy() * 255).astype(np.uint8)).save(folder / "heuristic_confidence.png")
        tensor_to_pil((night.pow(.45) * .5 + aligned * .5) * 2 - 1).save(folder / "overlay.png")
        results.append({"index": index, "split": "train", "night_path": row["night_path"],
            "day_path": row["day_path"], "night_sha256": digest(Path(row["night_path"])),
            "day_sha256": digest(Path(row["day_path"])), "selected_mode": mode, "rank_score": score,
            "modes": modes, "folder": str(folder.resolve()), "approved_for_training": False})
        if (number + 1) % 8 == 0:
            print(f"Audited {number + 1}/{len(selected)} original training pairs", flush=True)
    results.sort(key=lambda item: item["rank_score"], reverse=True)
    render_candidates(args.output, results, 0, args.review_count)
    atomic_json_dump({"purpose": "Candidate mining only; NONE approved as training targets",
        "test_used": False, "lockbox_used": False, "public_demo_unchanged": True,
        "external_rgb_generator_used": False, "raft_checkpoint_sha256": digest(checkpoint),
        "pairs_csv_sha256": digest(args.pairs), "unique_day_sampling": True, "candidate_count": len(results),
        "limitations": "FB+smoothness can be confidently wrong; moving objects not semantically excluded; native-night coordinates",
        "ranked_candidates": results}, args.output / "candidates.json")


if __name__ == "__main__":
    main()
