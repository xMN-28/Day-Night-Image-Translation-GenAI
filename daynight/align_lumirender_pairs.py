from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
from tqdm import tqdm


def _tensor(path: Path, size: tuple[int, int], device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR)).copy()
    return torch.from_numpy(array).permute(2, 0, 1)[None].float().div(255).to(device)


def _warp(image: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=image.device),
        torch.arange(width, device=image.device),
        indexing="ij",
    )
    sample_x = x[None].float() + flow[:, 0]
    sample_y = y[None].float() + flow[:, 1]
    valid = (sample_x >= 0) & (sample_x <= width - 1) & (sample_y >= 0) & (sample_y <= height - 1)
    grid = torch.stack(
        (2 * sample_x / max(1, width - 1) - 1, 2 * sample_y / max(1, height - 1) - 1),
        dim=-1,
    )
    return F.grid_sample(image, grid, align_corners=True), valid[:, None]


def align_pairs(csv_path: Path, output: Path, width: int, height: int) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights, progress=True).to(device).eval()
    output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")))
    registered: list[dict[str, str]] = []
    with torch.inference_mode():
        for index, row in enumerate(tqdm(rows, desc="RAFT pair alignment")):
            day_path, night_path = Path(row["day_path"]), Path(row["night_path"])
            if not day_path.is_absolute():
                day_path = csv_path.parent / day_path
            if not night_path.is_absolute():
                night_path = csv_path.parent / night_path
            day = _tensor(day_path, (width, height), device)
            night = _tensor(night_path, (width, height), device)
            normalized_day, normalized_night = weights.transforms()(day, night)
            forward = model(normalized_day, normalized_night)[-1]
            backward = model(normalized_night, normalized_day)[-1]
            aligned, in_bounds = _warp(night, forward)
            sampled_backward, _ = _warp(backward, forward)
            consistency = (forward + sampled_backward).square().sum(dim=1, keepdim=True).sqrt()
            confidence = torch.exp(-consistency / 2.0) * in_bounds
            motion = forward.square().sum(dim=1, keepdim=True).sqrt()
            confidence = confidence * torch.exp(-torch.relu(motion - 12) / 8)

            aligned_path = output / f"{index:06d}_night.png"
            confidence_path = output / f"{index:06d}_confidence.png"
            aligned_array = aligned[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
            confidence_array = confidence[0, 0].clamp(0, 1).mul(255).byte().cpu().numpy()
            Image.fromarray(aligned_array).save(aligned_path)
            Image.fromarray(confidence_array).save(confidence_path)
            registered.append(
                {
                    "day_path": str(day_path.resolve()),
                    "night_path": str(aligned_path.resolve()),
                    "confidence_path": str(confidence_path.resolve()),
                    "confidence": row.get("confidence", "1.0"),
                    "split": row.get("split", "train"),
                }
            )
    registered_csv = output / "aligned_pairs.csv"
    with registered_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(registered[0]))
        writer.writeheader()
        writer.writerows(registered)
    print(f"Aligned {len(registered)} pairs; register {registered_csv} next.")
    return len(registered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align licensed coarse day/night pairs with RAFT")
    parser.add_argument("--pairs-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=432)
    args = parser.parse_args()
    align_pairs(args.pairs_csv, args.output, args.width, args.height)


if __name__ == "__main__":
    main()
