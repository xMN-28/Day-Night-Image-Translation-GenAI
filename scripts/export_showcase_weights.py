"""Preserve all current demo generators before pruning training-only assets.

Exports are inference-only, not resumable training backups. Each exported tensor
is checked against the original inference state before this script succeeds.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from daynight.remote_worker import MODEL_CATALOG

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    destination = ROOT / "checkpoints/showcase"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    torch.set_num_threads(6)
    for model_id, spec in MODEL_CATALOG.items():
        output = destination / f"{model_id}.pt"
        if output.exists():
            raise FileExistsError(output)
        original = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
        name = "G_night_day" if model_id == "night_day_experimental" else "G_day_night"
        state = original["models"][name]
        if name in original.get("ema", {}):
            state = original["ema"][name].get("shadow", state)
        payload = {"config": original["config"], "step": original.get("step"),
                   "models": {name: state}, "ema": {name: {"shadow": state}},
                   "inference_only": True,
                   "original_checkpoint_sha256": sha(spec.checkpoint)}
        if "inverse_architecture" in original:
            payload["inverse_architecture"] = original["inverse_architecture"]
        torch.save(payload, output)
        loaded = torch.load(output, map_location="cpu", weights_only=False)
        assert state.keys() == loaded["models"][name].keys()
        assert all(torch.equal(t, loaded["models"][name][k]) for k, t in state.items())
        records.append({"id": model_id, "source": str(spec.checkpoint),
            "source_sha256": payload["original_checkpoint_sha256"], "output": str(output),
            "sha256": sha(output), "bytes": output.stat().st_size,
            "step": original.get("step"), "all_inference_tensors_identical": True})
        del loaded, payload, original, state
        print(f"Preserved {model_id}", flush=True)
    source = ROOT / "runs/sky_student_regions_v2/best.pt"
    original = torch.load(source, map_location="cpu", weights_only=False)
    output = destination / "sky_matte.pt"
    payload = {"model": original["model"], "step": original.get("step"),
               "inference_only": True, "original_checkpoint_sha256": sha(source)}
    torch.save(payload, output)
    loaded = torch.load(output, map_location="cpu", weights_only=False)
    assert all(torch.equal(t, loaded["model"][k]) for k, t in original["model"].items())
    records.append({"id": "sky_matte", "output": str(output), "sha256": sha(output),
                    "bytes": output.stat().st_size, "step": original.get("step"),
                    "all_inference_tensors_identical": True})
    (destination / "manifest.json").write_text(json.dumps({"inference_only": True,
        "exact_training_resume_available": False, "models": records}, indent=2), encoding="utf-8")
    print(json.dumps({"models": len(records), "MiB": round(sum(r["bytes"] for r in records)/2**20, 2)}))


if __name__ == "__main__":
    main()
