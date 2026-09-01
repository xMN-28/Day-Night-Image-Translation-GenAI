from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify LumiCycle training prerequisites")
    parser.add_argument("--data-root", default="data/bdd100k_daynight")
    args = parser.parse_args()
    workspace = Path.cwd()
    free_gb = shutil.disk_usage(workspace).free / 1024**3
    dataset = Path(args.data_root)
    checks = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        if torch.cuda.is_available()
        else 0,
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "workspace_free_gb": round(free_gb, 2),
        "dataset_prepared": (dataset / "dataset.json").exists(),
    }
    print(json.dumps(checks, indent=2))
    failures = []
    if not checks["cuda_available"]:
        failures.append("CUDA GPU is not available")
    if checks["gpu_vram_gb"] < 10:
        failures.append("At least 10 GB VRAM is recommended")
    if free_gb < 35:
        failures.append("At least 35 GB free workspace storage is required; 50 GB is recommended")
    if failures:
        print("\nWarnings:\n- " + "\n- ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
