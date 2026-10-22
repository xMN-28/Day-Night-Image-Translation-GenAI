from pathlib import Path

import pytest
import torch
from PIL import Image

from daynight.inference import ModelManager
from daynight.sky_refinement import compose_night, refine_sky_alpha
from daynight.sky_repair import SkyRepair


def test_missing_matte_is_not_a_silent_raw_fallback(tmp_path):
    repair = SkyRepair(tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="matte weights"):
        repair.apply(torch.zeros(1, 3, 32, 48), torch.zeros(1, 3, 32, 48))


def test_recover_unlabeled_sky_inside_nonvegetation_cable_bundle():
    source = torch.full((1, 3, 96, 128), 0.85)
    source[..., 70:, :] = 0.2
    source[..., :70, 60:62] = 0.08
    probability = torch.zeros(1, 1, 96, 128)
    probability[..., :70, :50] = 0.98
    vegetation = torch.zeros_like(probability)  # cables are not vegetation
    alpha = refine_sky_alpha(source, probability, vegetation, recover_color_holes=True)
    assert alpha[0, 0, 25, 75] > 0.8
    assert alpha[0, 0, 25, 60] < 0.2
    result = compose_night(source, source * 0.7, alpha, vegetation=vegetation,
                           recover_thin_edges=True)
    assert result[0, :, 25, 75].mean() < 0.15
    assert result[0, :, 25, 60].mean() < 0.1
    assert torch.isfinite(result).all()


def test_repair_with_no_sky_keeps_prediction_exact():
    source = torch.rand(1, 3, 64, 96)
    base = torch.rand_like(source)
    alpha = refine_sky_alpha(source, torch.zeros(1, 1, 64, 96),
                             recover_color_holes=True)
    result = compose_night(source, base, alpha, recover_thin_edges=True)
    assert torch.equal(result, base)


def test_remote_manager_path_applies_repair_by_default_and_can_ablate(monkeypatch):
    manager = ModelManager()
    manager.device = torch.device("cpu")
    manager.models = {"G_day_night": torch.nn.Identity()}
    manager.checkpoint_path = Path("frozen_parent.pt")
    monkeypatch.setattr(manager, "_load_custom", lambda *args: None)

    class SpyRepair:
        calls = 0

        def apply(self, source, prediction):
            self.calls += 1
            return prediction * 0, {"sky_repair": True, "matte_step": 7800}

    spy = SpyRepair()
    manager.sky_repair = spy
    image = Image.new("RGB", (48, 32), "white")
    repaired, metadata = manager.translate_custom(
        image, "day_to_night", "LumiCycle distilled student", 512
    )
    assert spy.calls == 1 and metadata["sky_repair"] is True
    assert repaired.getpixel((0, 0))[0] in (127, 128)
    raw, metadata = manager.translate_custom(
        image, "day_to_night", "LumiCycle distilled student", 512, sky_repair=False
    )
    assert spy.calls == 1 and metadata["sky_repair"] is False
    assert raw.getpixel((0, 0)) == (255, 255, 255)
    manager.translate_custom(image, "day_to_night", "LumiCycle V2", 512)
    assert spy.calls == 1  # historical V2 remains unmodified
