from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import UnpairedDayNightDataset, eval_transform
from .evaluate import load_generators
from .inference import _resolve_checkpoint
from .losses import DinoSemanticLoss, SobelEdgeLoss
from .lumirender_losses import luminance
from .models import build_models
from .night_classifier import load_night_classifier
from .utils import atomic_json_dump


def _load_lumirender(path: str | Path, device: torch.device):
    path = _resolve_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    models = build_models(payload["config"])
    model = models["G_day_night"]
    state = payload.get("ema", {}).get("G_day_night", {}).get(
        "shadow", payload["models"]["G_day_night"]
    )
    model.load_state_dict(state)
    return payload["config"], model.to(device).eval()


class FixedSuiteDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest: str | Path, size: int) -> None:
        self.items = [
            json.loads(line)
            for line in Path(manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.size = size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        with Image.open(item["image_path"]) as image:
            day = eval_transform(image, self.size)
        return {"day": day, "category": item["category"]}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, model = _load_lumirender(args.checkpoint, device)
    _, baseline_models, baseline_path = load_generators(args.baseline, device)
    baseline = baseline_models["G_day_night"]
    dataset = (
        FixedSuiteDataset(args.suite_manifest, args.image_size)
        if args.suite_manifest
        else UnpairedDayNightDataset(
            args.data_root or config["data"]["root"], "test", args.image_size, args.image_size
        )
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    edge = SobelEdgeLoss().to(device)
    dino = None if args.skip_dino else DinoSemanticLoss().to(device)
    night_classifier = load_night_classifier(args.night_classifier, device)
    totals = {
        "lumirender_edge": 0.0,
        "baseline_edge": 0.0,
        "lumirender_luminance": 0.0,
        "baseline_luminance": 0.0,
        "lumirender_dino": 0.0,
        "baseline_dino": 0.0,
        "emitter_validity": 0.0,
        "correction_max": 0.0,
        "latency": 0.0,
        "lumirender_night_confidence": 0.0,
        "baseline_night_confidence": 0.0,
        "sky_luminance": 0.0,
        "sky_variance": 0.0,
        "reflection_plausibility": 0.0,
    }
    count = 0
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    with torch.inference_mode():
        for batch in tqdm(loader, desc="LumiRender acceptance evaluation"):
            day = batch["day"].to(device)
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                generated, details = model(day, seed=0, return_details=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            totals["latency"] += time.perf_counter() - started
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                baseline_output = baseline(day)
            totals["lumirender_edge"] += float(edge(day, generated))
            totals["baseline_edge"] += float(edge(day, baseline_output))
            totals["lumirender_luminance"] += float(luminance(generated).mean())
            totals["baseline_luminance"] += float(luminance(baseline_output).mean())
            totals["lumirender_night_confidence"] += float(night_classifier(generated).sigmoid())
            totals["baseline_night_confidence"] += float(
                night_classifier(baseline_output).sigmoid()
            )
            if dino is not None:
                totals["lumirender_dino"] += float(dino(day, generated))
                totals["baseline_dino"] += float(dino(day, baseline_output))
            emitter_mask = details["semantic"][:, 5:6]
            light = details["gaussian_light"].mean(dim=1, keepdim=True)
            bright = (light > 0.08).float()
            totals["emitter_validity"] += float(
                (bright * emitter_mask).sum() / bright.sum().clamp_min(1)
            )
            output_luminance = luminance(generated)
            sky = details["semantic"][:, 0:1]
            sky_mean = (output_luminance * sky).sum() / sky.sum().clamp_min(1)
            totals["sky_luminance"] += float(sky_mean)
            totals["sky_variance"] += float(
                ((output_luminance - sky_mean).square() * sky).sum() / sky.sum().clamp_min(1)
            )
            plausible_surface = details["semantic"][:, 1:2] + details["semantic"][:, 3:4]
            reflection_energy = details["reflections"].abs().mean(dim=1, keepdim=True)
            totals["reflection_plausibility"] += float(
                (reflection_energy * plausible_surface.clamp(0, 1)).sum()
                / reflection_energy.sum().clamp_min(1e-6)
            )
            totals["correction_max"] = max(
                totals["correction_max"], float(details["correction"].abs().max())
            )
            count += 1
            if count >= args.limit:
                break
    metrics = {name: value / count for name, value in totals.items() if name != "correction_max"}
    metrics["correction_max"] = totals["correction_max"]
    human_preference = 0.0
    if args.human_review and Path(args.human_review).exists():
        review = json.loads(Path(args.human_review).read_text(encoding="utf-8"))
        human_preference = float(review.get("lumirender_preference_rate", 0.0))
    checks = {
        "night_confidence": metrics["lumirender_night_confidence"]
        > metrics["baseline_night_confidence"],
        "sky_not_uniformly_black": metrics["sky_luminance"] >= 0.03
        and metrics["sky_variance"] >= 1e-5,
        "edge_retention": metrics["lumirender_edge"] <= metrics["baseline_edge"],
        "semantic_retention": (
            args.skip_dino
            or metrics["lumirender_dino"] <= metrics["baseline_dino"] * 1.02
        ),
        "emitter_location": metrics["emitter_validity"] >= 0.90,
        "reflection_plausibility": metrics["reflection_plausibility"] >= 0.85,
        "bounded_correction": metrics["correction_max"] <= 0.0301,
        "latency": metrics["latency"] < 2.0,
        "human_preference": human_preference >= 0.70,
    }
    result = {
        "passed": all(checks.values()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "baseline": str(baseline_path),
        "samples": count,
        "metrics": metrics,
        "checks": checks,
        "human_preference_rate": human_preference,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0,
    }
    output = Path(args.output)
    atomic_json_dump(result, output)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LumiRender against its acceptance gate")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--baseline",
        default="runs/lumicycle_v2_bdd100k/checkpoints/step_00004500.pt",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--suite-manifest", default="data/lumirender_suite.jsonl")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument(
        "--night-classifier", default="runs/evaluators/night_classifier.pt"
    )
    parser.add_argument("--human-review", default="outputs/lumirender/human_review.json")
    parser.add_argument("--output", default="runs/lumirender_physics_bdd100k/acceptance.json")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
