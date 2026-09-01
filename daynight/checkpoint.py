from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import config_hash
from .utils import atomic_json_dump, atomic_torch_save


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


class CheckpointManager:
    def __init__(self, output_dir: str | Path, keep: int = 4) -> None:
        self.directory = Path(output_dir) / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep = keep

    @property
    def latest_pointer(self) -> Path:
        return self.directory / "latest.json"

    def resolve(self, resume: str | None) -> Path | None:
        if resume in {None, "", "none"}:
            return None
        if resume == "auto":
            if not self.latest_pointer.exists():
                return None
            pointer = json.loads(self.latest_pointer.read_text(encoding="utf-8"))
            path = self.directory / pointer["filename"]
            return path if path.exists() else None
        path = Path(resume)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def resolve_best(self) -> Path | None:
        pointer = self.directory / "best.json"
        if not pointer.exists():
            return None
        data = json.loads(pointer.read_text(encoding="utf-8"))
        path = self.directory / data["filename"]
        return path if path.exists() else None

    def save(self, step: int, payload: dict[str, Any], best: bool = False) -> Path:
        filename = f"step_{step:08d}.pt"
        path = self.directory / filename
        atomic_torch_save(payload, path)
        atomic_json_dump({"filename": filename, "step": step}, self.latest_pointer)
        if best:
            atomic_json_dump({"filename": filename, "step": step}, self.directory / "best.json")
        self._prune()
        return path

    def _prune(self) -> None:
        checkpoints = sorted(self.directory.glob("step_*.pt"))
        protected = set()
        for pointer_name in ("latest.json", "best.json"):
            pointer = self.directory / pointer_name
            if pointer.exists():
                protected.add(json.loads(pointer.read_text(encoding="utf-8"))["filename"])
        removable = [path for path in checkpoints if path.name not in protected]
        while len(checkpoints) > self.keep and removable:
            path = removable.pop(0)
            path.unlink(missing_ok=True)
            checkpoints.remove(path)


def build_checkpoint(
    *,
    step: int,
    config: dict[str, Any],
    models: dict[str, torch.nn.Module],
    ema: dict[str, Any],
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler],
    replay: dict[str, Any],
    training_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "step": step,
        "config": config,
        "config_hash": config_hash(config),
        "models": {name: model.state_dict() for name, model in models.items()},
        "ema": ema,
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "schedulers": {name: scheduler.state_dict() for name, scheduler in schedulers.items()},
        "replay": replay,
        "training_state": training_state,
        "rng": capture_rng_state(),
    }
