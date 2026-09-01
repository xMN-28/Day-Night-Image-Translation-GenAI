from __future__ import annotations

from pathlib import Path

import torch

from daynight.checkpoint import CheckpointManager
from daynight import utils


def test_checkpoint_manager_resolves_and_prunes(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, keep=2)
    for step in (1, 2, 3):
        manager.save(step, {"step": step, "tensor": torch.tensor(step)}, best=step == 2)
    assert manager.resolve("auto") is not None
    assert manager.resolve("auto").name == "step_00000003.pt"
    assert (manager.directory / "step_00000002.pt").exists()
    assert len(list(manager.directory.glob("step_*.pt"))) == 2


def test_atomic_json_dump_retries_windows_reader_lock(
    tmp_path: Path, monkeypatch,
) -> None:
    destination = tmp_path / "status.json"
    real_replace = utils.os.replace
    calls = 0

    def briefly_locked(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("simulated Windows reader lock")
        real_replace(source, target)

    monkeypatch.setattr(utils.os, "replace", briefly_locked)
    utils.atomic_json_dump({"step": 123}, destination)

    assert destination.read_text(encoding="utf-8").find('"step": 123') >= 0
    assert calls == 3
