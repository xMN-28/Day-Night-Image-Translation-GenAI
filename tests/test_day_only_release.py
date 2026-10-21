import pytest
from PIL import Image

from daynight.inference import ModelManager, translate
from daynight.remote_worker import MODEL_CATALOG, RemoteWorker


@pytest.mark.parametrize("direction", ["night_to_day", "Night → Day", "night2day", "invalid"])
def test_public_api_rejects_removed_direction_before_loading(direction):
    with pytest.raises(ValueError, match="Only Day"):
        translate(Image.new("RGB", (32, 32)), direction)


def test_manager_rejects_inverse_before_checkpoint_access():
    with pytest.raises(ValueError, match="withdrawn"):
        ModelManager().translate_custom(Image.new("RGB", (32, 32)), "night_to_day", "LumiCycle V2", 512)


def test_catalog_day_only():
    assert all(spec.directions == ("day_to_night",) for spec in MODEL_CATALOG.values())


def test_worker_rejects_inverse_without_downloading(monkeypatch):
    monkeypatch.setattr("pathlib.Path.is_file", lambda self: True)
    worker = RemoteWorker("http://127.0.0.1:3000", "test")
    with pytest.raises(ValueError, match="does not support"):
        worker.process({"id": "test", "modelId": "v2_best", "direction": "night_to_day"})
