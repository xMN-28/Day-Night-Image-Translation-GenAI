from __future__ import annotations

from pathlib import Path

import torch

from daynight.checkpoint import CheckpointManager


def test_checkpoint_manager_resolves_and_prunes(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, keep=2)
    for step in (1, 2, 3):
        manager.save(step, {"step": step, "tensor": torch.tensor(step)}, best=step == 2)
    assert manager.resolve("auto") is not None
    assert manager.resolve("auto").name == "step_00000003.pt"
    assert (manager.directory / "step_00000002.pt").exists()
    assert len(list(manager.directory.glob("step_*.pt"))) == 2
