from __future__ import annotations

import pytest
from PIL import Image

from daynight.inference import _fit_and_pad


def test_direction_guards_keep_day_models_out_of_inverse():
    from daynight.inference import ModelManager

    manager = ModelManager()
    with pytest.raises(ValueError, match="does not support"):
        manager.translate_custom(Image.new("RGB", (32, 32)), "night_to_day", "LumiCycle", 512)
    with pytest.raises(ValueError, match="does not support"):
        manager.translate_custom(Image.new("RGB", (32, 32)), "day_to_night",
                                 "LumiCycle night-to-day experimental", 512)


def test_inverse_adapter_dispatches_correct_generator_without_sky_repair():
    import torch

    from daynight.inference import ModelManager

    manager = ModelManager()
    manager.device = torch.device("cpu")
    manager._load_custom = lambda *args: None
    manager.models = {"G_night_day": torch.nn.Identity()}
    manager.global_step = 16500
    result, metadata = manager.translate_custom(
        Image.new("RGB", (33, 35)), "night_to_day",
        "LumiCycle night-to-day experimental", 512)
    assert result.size == (33, 35)
    assert metadata["experimental"] and not metadata["sky_repair"]
    assert metadata["global_step"] == 16500


def test_inference_preprocessing_preserves_aspect_and_multiple_of_four() -> None:
    image = Image.new("RGB", (1001, 503), "white")
    padded, crop = _fit_and_pad(image, 768)
    assert padded.width % 4 == 0
    assert padded.height % 4 == 0
    assert crop[2] <= 768
    assert crop[2] / crop[3] == pytest.approx(1001 / 503, rel=0.01)
