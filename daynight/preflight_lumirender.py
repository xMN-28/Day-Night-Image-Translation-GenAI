from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from .config import load_config
from .models import build_models


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify LumiRender data, teachers and GPU headroom")
    parser.add_argument("--config", default="configs/lumirender.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    checks: dict[str, object] = {}
    pseudo = Path(config["data"]["pseudo_manifest"])
    checks["teacher_targets"] = _line_count(pseudo) if pseudo.exists() else 0
    sources = {}
    for value in config["data"]["source_manifests"]:
        path = Path(value)
        sources[path.stem] = _line_count(path) if path.exists() else 0
    checks["registered_pairs"] = sources
    checks["free_disk_gb"] = round(shutil.disk_usage(Path.cwd().anchor).free / 1024**3, 2)
    checks["cuda_available"] = torch.cuda.is_available()
    checks["gpu"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "none"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        models = {name: model.cuda().train() for name, model in build_models(config).items()}
        sample = torch.randn(1, 3, 384, 384, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated = models["G_day_night"](sample, seed=1)
            loss = generated.mean() + sum(
                output.square().mean() for output in models["D_night"](generated)
            )
        loss.backward()
        checks["probe_peak_vram_gb"] = round(
            torch.cuda.max_memory_allocated() / 1024**3, 3
        )
        checks["probe_finite"] = bool(torch.isfinite(generated).all())
    print(json.dumps(checks, indent=2))
    failures = []
    if int(checks["teacher_targets"]) < 100:
        failures.append("fewer than 100 teacher targets (5,000 recommended)")
    if not any(sources.values()):
        failures.append("no licensed aligned day/night pairs registered")
    if float(checks["free_disk_gb"]) < 20:
        failures.append("less than 20 GB free disk reserve")
    if not checks["cuda_available"]:
        failures.append("CUDA unavailable")
    if float(checks.get("probe_peak_vram_gb", 99)) > float(config["train"]["max_vram_gb"]):
        failures.append("384 px probe exceeded VRAM ceiling")
    if failures:
        raise SystemExit("Preflight failed: " + "; ".join(failures))
    print("LumiRender preflight passed.")


if __name__ == "__main__":
    main()
