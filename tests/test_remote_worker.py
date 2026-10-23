from pathlib import Path

import pytest

from daynight.remote_worker import MODEL_CATALOG, RemoteWorker


def test_remote_catalog_uses_project_relative_checkpoint_paths():
    assert "v2_best" in MODEL_CATALOG
    assert MODEL_CATALOG["v2_best"].checkpoint.is_absolute()
    assert MODEL_CATALOG["lumirender_final"].directions == ("day_to_night",)
    assert all(isinstance(spec.checkpoint, Path) for spec in MODEL_CATALOG.values())
    inverse = MODEL_CATALOG["night_day_experimental"]
    assert inverse.directions == ("night_to_day",)
    assert "experimental" in inverse.label


def test_remote_worker_rejects_insecure_nonlocal_server():
    with pytest.raises(ValueError, match="HTTPS"):
        RemoteWorker("http://example.com", "secret")


def test_remote_worker_allows_local_development_server():
    worker = RemoteWorker("http://127.0.0.1:3000/", "secret")
    assert worker.server == "http://127.0.0.1:3000"
