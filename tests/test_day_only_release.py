import pytest
from PIL import Image

from daynight.inference import ModelManager, translate
from daynight.remote_worker import MODEL_CATALOG, RemoteWorker


@pytest.mark.parametrize("direction", ["night_to_day", "Night → Day", "night2day"])
def test_day_model_rejects_inverse_direction_before_loading(direction):
    with pytest.raises(ValueError, match="does not support"):
        translate(Image.new("RGB", (32, 32)), direction)


def test_invalid_direction_rejected():
    with pytest.raises(ValueError, match="Choose Day"):
        translate(Image.new("RGB", (32, 32)), "invalid")


def test_manager_rejects_inverse_before_checkpoint_access():
    with pytest.raises(ValueError, match="does not support"):
        ModelManager().translate_custom(Image.new("RGB", (32, 32)), "night_to_day", "LumiCycle V2", 512)


def test_catalog_restores_only_explicit_experimental_inverse():
    assert MODEL_CATALOG["night_day_experimental"].directions == ("night_to_day",)
    assert all(spec.directions == ("day_to_night",)
               for key, spec in MODEL_CATALOG.items() if key != "night_day_experimental")


def test_worker_rejects_inverse_without_downloading(monkeypatch):
    monkeypatch.setattr("pathlib.Path.is_file", lambda self: True)
    worker = RemoteWorker("http://127.0.0.1:3000", "test")
    with pytest.raises(ValueError, match="does not support"):
        worker.process({"id": "test", "modelId": "v2_best", "direction": "night_to_day"})
