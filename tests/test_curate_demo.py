import json
import sys

import pytest

from daynight import curate_demo


def setup_selection(tmp_path, monkeypatch, indices="1"):
    screening = tmp_path / "screening"
    screening.mkdir()
    (screening / "manifest.json").write_text(json.dumps({
        "records": [{"validation_index": 1}], "checkpoint_sha256": "stale-hash"
    }))
    showcase = tmp_path / "showcase"
    monkeypatch.setattr(sys, "argv", [
        "curate_demo", "--output", str(screening), "--showcase", str(showcase),
        "--select", indices,
    ])
    return showcase


def test_selection_refuses_duplicate_indices(tmp_path, monkeypatch):
    setup_selection(tmp_path, monkeypatch, "1,1")
    with pytest.raises(ValueError, match="duplicate"):
        curate_demo.main()


def test_selection_refuses_unscreened_indices(tmp_path, monkeypatch):
    setup_selection(tmp_path, monkeypatch, "2")
    with pytest.raises(ValueError, match="Only screened"):
        curate_demo.main()


def test_selection_refuses_stale_showcase_folder(tmp_path, monkeypatch):
    showcase = setup_selection(tmp_path, monkeypatch)
    showcase.mkdir()
    rejected = showcase / "rejected.jpg"
    rejected.write_bytes(b"retain me")
    with pytest.raises(FileExistsError, match="Archive"):
        curate_demo.main()
    assert rejected.read_bytes() == b"retain me"


def test_selection_refuses_changed_checkpoint(tmp_path, monkeypatch):
    setup_selection(tmp_path, monkeypatch)
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"different weights")
    monkeypatch.setitem(curate_demo.MODEL_PATHS, "LumiCycle distilled student", checkpoint)
    with pytest.raises(ValueError, match="Checkpoint changed"):
        curate_demo.main()
