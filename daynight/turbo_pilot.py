from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import read_manifest
from .turbo_reference import TurboReference
from .utils import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Low-VRAM external CycleGAN-Turbo feasibility/reference pilot"
    )
    parser.add_argument("--data-root", default="data/bdd100k_daynight")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-vram-gb", type=float, default=11.5)
    parser.add_argument("--output", default="outputs/turbo_pilot")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = read_manifest(args.data_root, "train", "day")[: args.steps]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = TurboReference("day_to_night")
    latencies = []
    for index, record in enumerate(records):
        from PIL import Image

        with Image.open(record.path) as image:
            started = time.perf_counter()
            generated = model(image.convert("RGB"))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
        if index < 8:
            generated.save(output / f"{index:03d}.jpg")
        peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        if peak > args.max_vram_gb:
            raise RuntimeError(f"Turbo pilot exceeded VRAM limit: {peak:.2f} GB")
    result = {
        "label": "external pretrained CycleGAN-Turbo reference",
        "samples": len(records),
        "mean_latency_seconds": sum(latencies) / max(1, len(latencies)),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available()
        else 0,
        "note": "This is a feasibility/reference pilot, not LumiRender training or originality.",
    }
    atomic_json_dump(result, output / "metrics.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
