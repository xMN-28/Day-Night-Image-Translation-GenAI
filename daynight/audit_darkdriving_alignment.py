"""Private TRAIN-only geometric audit. No automatic pixel-GT approval/training.

Cached external RAFT is a correspondence estimator, not an RGB generator.
No derived images may be published under the dataset's restrictive terms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

from .align_lumirender_pairs import _tensor, _warp
from .audit_inverse_alignment import inverse_alignment_confidence
from .mine_inverse_pairs import deformation_weight
from .turbo_pilot import digest
from .utils import atomic_json_dump, tensor_to_pil


def pad_for_flow(image):
    """Reflect-pad, not resize; return original height/width for exact cropping."""
    h, w = image.shape[-2:]
    return F.pad(image, (0, (-w) % 8, 0, (-h) % 8), mode="reflect"), (h, w)


def sampling_jacobian(flow):
    """Local determinant of (x,y)->(x+u,y+v); geometric diagnostic only."""
    dx = F.pad(torch.diff(flow, dim=-1), (0, 1, 0, 0), mode="replicate")
    dy = F.pad(torch.diff(flow, dim=-2), (0, 0, 0, 1), mode="replicate")
    return (1 + dx[:, 0]) * (1 + dy[:, 1]) - dy[:, 0] * dx[:, 1]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path,
        default=Path("E:/LumiCycleCache/experiments/DarkDriving_private_audit12_20261023/audit.json"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep earlier evidence")
    audit = json.loads(args.audit.read_text())
    if audit["test_used"] or audit["lockbox_used"] or audit["vision_review"] == "pending":
        raise ValueError("Require reviewed TRAIN-only audit")
    for row in audit["records"]:
        if row["split"] != "train":
            raise ValueError("Training data only")
        for domain in ("day", "night"):
            if digest(Path(row[domain])) != row[f"{domain}_sha256"]:
                raise ValueError("Reviewed original changed")
    torch.set_num_threads(6)
    weights = Raft_Small_Weights.DEFAULT
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not checkpoint.is_file():
        raise FileNotFoundError("No implicit flow-weight download; use existing cache")
    model = raft_small(weights=None).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    args.output.mkdir(parents=True)
    records = []
    for number, row in enumerate(audit["records"]):
        with Image.open(row["night"]) as im:
            size = (512, max(32, round(im.height * 512 / im.width / 4) * 4))
        night = _tensor(Path(row["night"]), size, torch.device("cuda"))
        day = _tensor(Path(row["day"]), size, torch.device("cuda"))
        folder = args.output / Path(row["name"]).stem
        folder.mkdir()
        tensor_to_pil((night.pow(.45) * .5 + day * .5) * 2 - 1).save(folder / "UNWARPED_overlay.png")
        sheet = Image.new("RGB", (2560, 2 * (size[1] + 40)), "#101010")
        modes = {}
        for j, (mode, source) in enumerate((("raw", night), ("gamma045", night.pow(.45)))):
            a, (h, w) = pad_for_flow(source)
            b, _ = pad_for_flow(day)
            a, b = weights.transforms()(a, b)
            forward = model(a, b, num_flow_updates=12)[-1][..., :h, :w]
            backward = model(b, a, num_flow_updates=12)[-1][..., :h, :w]
            aligned, valid = _warp(day, forward)
            confidence, error = inverse_alignment_confidence(forward, backward)
            smooth = deformation_weight(forward)
            reliability = confidence * smooth * valid
            jacobian = sampling_jacobian(forward)
            panel_source = night.pow(.45)  # Display visibility control, NOT generated daylight.
            confidence_image = Image.fromarray(np.rint(reliability[0, 0].cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
            overlay = tensor_to_pil((panel_source * .5 + aligned * .5) * 2 - 1)
            columns = [("Gamma NIGHT visibility (NOT day)", tensor_to_pil(panel_source * 2 - 1)),
                ("REAL day UNWARPED", tensor_to_pil(day * 2 - 1)),
                ("REAL day warped into night", tensor_to_pil(aligned * 2 - 1)),
                ("FB+smooth heuristic, NOT static truth", confidence_image), ("Overlay: inspect deformation", overlay)]
            for k, (label, im) in enumerate(columns):
                sheet.paste(im, (k * 512, j * (h + 40) + 40))
                ImageDraw.Draw(sheet).text((k * 512 + 4, j * (h + 40) + 4), f"{mode} {row['name']}\n{label}", fill="white")
                im.save(folder / f"{mode}_{k}.png")
            np.savez_compressed(folder / f"{mode}_geometry.npz", forward=forward.cpu().numpy(),
                backward=backward.cpu().numpy(), reliability=reliability.cpu().numpy())
            modes[mode] = {"mean_displacement_pixels": float(forward.square().sum(1).sqrt().mean()),
                "mean_fb_error_pixels": float(error.mean()),
                "fraction_fb_gt05": float((confidence > .5).float().mean()),
                "fraction_fb_smooth_gt05": float((reliability > .5).float().mean()),
                "fold_fraction": float((jacobian <= 0).float().mean()),
                "mean_abs_jacobian_minus1": float((jacobian - 1).abs().mean())}
        sheet.save(args.output / f"PRIVATE_alignment_{number:02d}.jpg", quality=95)
        records.append({"name": row["name"], "split": "train", "size": list(size),
            "folder": str(folder), "modes": modes, "approved_for_training": False})
        print(f"Audited geometry {number + 1}/{len(audit['records'])}", flush=True)
    atomic_json_dump({"source_audit_sha256": digest(args.audit), "records": records,
        "coordinate_system": "Native night resized aspect-preserving512x428; day sampled by night->day flow",
        "flow_sha256": digest(checkpoint), "flow_precision": "FP32", "flow_updates": 12,
        "flow_padding": "Reflect to multiple8, crop displacement to original before warping unpaddedday",
        "external_rgb_generator_used": False, "training_performed": False,
        "test_used": False, "lockbox_used": False, "public_demo_unchanged": True,
        "scope": "PRIVATE academic diagnostics only, no redistribution or automatic approval",
        "limitations": "FB consistency/smoothness/Jacobian do not prove correspondence or exclude moving objects",
        "vision_review": "pending"}, args.output / "audit.json")


if __name__ == "__main__":
    main()
