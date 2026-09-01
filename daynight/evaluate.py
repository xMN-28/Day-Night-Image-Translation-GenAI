from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import UnpairedDayNightDataset
from .inference import _resolve_checkpoint
from .losses import DinoSemanticLoss, SobelEdgeLoss
from .models import build_models
from .utils import atomic_json_dump, tensor_to_pil


def load_generators(checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = _resolve_checkpoint(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    models = build_models(payload["config"])
    for name in ("G_day_night", "G_night_day"):
        state = payload["models"][name]
        if name in payload.get("ema", {}):
            state = payload["ema"][name].get("shadow", state)
        models[name].load_state_dict(state)
        models[name].to(device).eval()
    return payload["config"], models, checkpoint_path


def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, models, checkpoint_path = load_generators(args.checkpoint, device)
    root = args.data_root or config["data"]["root"]
    dataset = UnpairedDayNightDataset(root, "test", args.image_size, args.image_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    edge = SobelEdgeLoss().to(device)
    dino = None if args.skip_dino else DinoSemanticLoss().to(device)
    day_detector = night_detector = None
    if args.detector:
        from .detector import ObjectRetentionMetric

        day_detector = ObjectRetentionMetric(args.detector_weights)
        night_detector = ObjectRetentionMetric(args.detector_weights)
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance

        metrics = {
            "day_to_night_fid": FrechetInceptionDistance(feature=2048, normalize=True).to(device),
            "night_to_day_fid": FrechetInceptionDistance(feature=2048, normalize=True).to(device),
            "day_to_night_kid": KernelInceptionDistance(
                subset_size=min(50, args.limit), normalize=True
            ).to(device),
            "night_to_day_kid": KernelInceptionDistance(
                subset_size=min(50, args.limit), normalize=True
            ).to(device),
        }
    except (ImportError, ModuleNotFoundError, RuntimeError) as error:
        print(
            f"FID/KID disabled because optional image metric dependencies are unavailable: {error}"
        )
        metrics = {}

    totals = {
        "day_to_night_edge": 0.0,
        "night_to_day_edge": 0.0,
        "day_to_night_dino": 0.0,
        "night_to_day_dino": 0.0,
    }
    count = 0
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    latencies = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating"):
            day = batch["day"].to(device)
            night = batch["night"].to(device)
            started = time.perf_counter()
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                fake_night = models["G_day_night"](day)
                fake_day = models["G_night_day"](night)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) / day.shape[0])
            batch_count = day.shape[0]
            totals["day_to_night_edge"] += float(edge(day, fake_night)) * batch_count
            totals["night_to_day_edge"] += float(edge(night, fake_day)) * batch_count
            if dino is not None:
                totals["day_to_night_dino"] += float(dino(day, fake_night)) * batch_count
                totals["night_to_day_dino"] += float(dino(night, fake_day)) * batch_count
            if day_detector is not None and count < args.detector_limit:
                day_detector.update(day, fake_night)
                night_detector.update(night, fake_day)
            real_day = ((day + 1) / 2).clamp(0, 1)
            real_night = ((night + 1) / 2).clamp(0, 1)
            generated_day = ((fake_day + 1) / 2).clamp(0, 1)
            generated_night = ((fake_night + 1) / 2).clamp(0, 1)
            if metrics:
                metrics["day_to_night_fid"].update(real_night, real=True)
                metrics["day_to_night_fid"].update(generated_night, real=False)
                metrics["night_to_day_fid"].update(real_day, real=True)
                metrics["night_to_day_fid"].update(generated_day, real=False)
                metrics["day_to_night_kid"].update(real_night, real=True)
                metrics["day_to_night_kid"].update(generated_night, real=False)
                metrics["night_to_day_kid"].update(real_day, real=True)
                metrics["night_to_day_kid"].update(generated_day, real=False)
            for index in range(batch_count):
                sample_index = count + index
                if sample_index < args.save_samples:
                    tensor_to_pil(fake_night[index]).save(
                        output_dir / f"{sample_index:04d}_day_to_night.jpg"
                    )
                    tensor_to_pil(fake_day[index]).save(
                        output_dir / f"{sample_index:04d}_night_to_day.jpg"
                    )
            count += batch_count
            if count >= args.limit:
                break

    results = {
        "checkpoint": str(checkpoint_path),
        "samples": count,
        "image_size": args.image_size,
        "mean_bidirectional_latency_seconds": sum(latencies) / max(1, len(latencies)),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0,
    }
    results.update({name: value / max(1, count) for name, value in totals.items()})
    for name, metric in metrics.items():
        value = metric.compute()
        if "kid" in name:
            results[f"{name}_mean"] = float(value[0])
            results[f"{name}_std"] = float(value[1])
        else:
            results[name] = float(value)
    if day_detector is not None:
        results["day_to_night_detector"] = day_detector.compute()
        results["night_to_day_detector"] = night_detector.compute()
    atomic_json_dump(results, output_dir / "metrics.json")
    print(json.dumps(results, indent=2))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate perceptual and structural translation quality"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default="outputs/evaluation")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--save-samples", type=int, default=32)
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument(
        "--detector", action="store_true", help="Measure frozen-detector object retention"
    )
    parser.add_argument("--detector-weights", default="yolo11n.pt")
    parser.add_argument("--detector-limit", type=int, default=100)
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
