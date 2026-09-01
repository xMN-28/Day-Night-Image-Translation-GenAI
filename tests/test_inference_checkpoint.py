from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from daynight.inference import ModelManager
from daynight.models import build_models


def test_checkpoint_load_and_bidirectional_inference(tmp_path: Path, monkeypatch) -> None:
    config = {
        "experiment": {"name": "inference-test", "output_dir": str(tmp_path)},
        "data": {},
        "model": {
            "kind": "cyclegan",
            "generator_blocks": 1,
            "base_channels": 8,
            "attention": True,
            "multiscale_discriminator": False,
        },
        "loss": {},
        "train": {},
    }
    models = build_models(config)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "config": config,
            "models": {name: model.state_dict() for name, model in models.items()},
            "ema": {},
        },
        checkpoint,
    )
    manager = ModelManager()
    # Exercise the same loader used by the public API without changing repository paths.
    monkeypatch.setenv("LUMICYCLE_CHECKPOINT", str(checkpoint))
    image = Image.new("RGB", (67, 65), "#7890a8")
    output, metadata = manager.translate_custom(image, "day_to_night", "LumiCycle", 128)
    assert output.size == image.size
    assert metadata["model"] == "LumiCycle"
    assert metadata["direction"] == "day_to_night"
