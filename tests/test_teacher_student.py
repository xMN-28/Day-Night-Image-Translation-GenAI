import copy
import json

import numpy as np
import pytest
import torch
from PIL import Image

from daynight.compare_sky_gan import file_digest
from daynight.train_teacher_student import (
    configure_execution,
    load_pair,
    select_training_subset,
    validate_resolution_curriculum,
)


def test_student_can_read_target_only_cache_without_copying_original_source(tmp_path):
    source, target = tmp_path / "source.jpg", tmp_path / "target.jpg"
    Image.new("RGB", (1280, 720), (180, 160, 140)).save(source)
    Image.new("RGB", (384, 216), (30, 25, 20)).save(target)
    x, y = load_pair(target, source)
    assert x.shape == y.shape == (3, 216, 384)
    assert x.mean() > y.mean()
    assert 0 <= float(y.min()) <= float(y.max()) <= 1


def test_original_lossless_cache_stays_compatible(tmp_path):
    path = tmp_path / "pair.npz"
    np.savez_compressed(
        path,
        image=np.full((216, 384, 3), 120, dtype=np.uint8),
        target=np.full((216, 384, 3), 30, dtype=np.uint8),
    )
    x, y = load_pair(path)
    assert x.shape == y.shape == (3, 216, 384)
    assert abs(float(x.mean()) - 120 / 255) < 1e-6


def test_native_teacher_resolution_is_not_silently_reduced(tmp_path):
    source, target = tmp_path / "source.jpg", tmp_path / "target.jpg"
    Image.new("RGB", (1280, 720), (180, 160, 140)).save(source)
    Image.new("RGB", (512, 288), (30, 25, 20)).save(target)
    x, y = load_pair(target, source)
    assert x.shape == y.shape == (3, 288, 512)


def test_deterministic_setting_is_applied_after_seed_helper(monkeypatch):
    previous = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
    )
    monkeypatch.setattr(
        "daynight.train_teacher_student.seed_everything",
        lambda seed: setattr(torch.backends.cudnn, "benchmark", True),
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        configure_execution(2030, deterministic=True)
        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.deterministic
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        (
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        ) = previous[:4]
        torch.use_deterministic_algorithms(previous[4])


def test_resolution_curriculum_accepts_only_same_teacher_sources_splits_and_seeds(tmp_path):
    previous = {
        "test_images_used": False,
        "recipe": {"student_size": [384, 216], "teacher_maximum_edge": 512, "teacher": "fixed"},
        "records": [
            {
                "split": "train",
                "index": 0,
                "source": "a.jpg",
                "source_sha256": "fixed",
                "seed": 12,
                "cache": "old.jpg",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(previous))
    current = copy.deepcopy(previous)
    current["recipe"]["student_size"] = [512, 288]
    current["records"][0]["cache"] = "new.jpg"
    validate_resolution_curriculum(path, current, file_digest(path))
    for field, value in (("split", "test"), ("source", "b.jpg"), ("seed", 99)):
        altered = copy.deepcopy(current)
        altered["records"][0][field] = value
        with pytest.raises(ValueError, match="identities"):
            validate_resolution_curriculum(path, altered, file_digest(path))
    altered = copy.deepcopy(current)
    altered["recipe"]["teacher"] = "different"
    with pytest.raises(ValueError, match="teacher"):
        validate_resolution_curriculum(path, altered, file_digest(path))
    with pytest.raises(ValueError, match="parent checkpoint"):
        validate_resolution_curriculum(path, current, "incorrect digest")


def test_training_subset_is_fixed_evenly_spaced_and_never_duplicates():
    records = list(range(2048))
    subset = select_training_subset(records, 32)
    assert len(subset) == len(set(subset)) == 32
    assert subset[0] == 0 and subset[-1] == 2047
    assert select_training_subset(records, 32) == subset
    assert select_training_subset(records, 0) is records
    with pytest.raises(ValueError):
        select_training_subset(records, -1)
