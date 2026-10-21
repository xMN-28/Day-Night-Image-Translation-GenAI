import json

import numpy as np
import pytest
from PIL import Image

from daynight.evaluate_teacher_student import load_reference_sky


def test_reference_sky_requires_matching_source_and_cached_pixels(tmp_path):
    source = tmp_path / "source.png"
    image = np.full((16, 32, 3), 150, dtype=np.uint8)
    Image.fromarray(image).save(source)
    row = {"split": "val", "index": 7, "source": str(source)}
    (tmp_path / "val").mkdir()
    (tmp_path / "provenance.json").write_text(
        json.dumps({"splits": {"val": [{"index": 7, "path": str(source)}]}})
    )
    np.savez_compressed(
        tmp_path / "val/00007.npz", image=image, alpha=np.full((16, 32), 255, dtype=np.uint8)
    )
    alpha = load_reference_sky(tmp_path, row, (32, 64))
    assert alpha.shape == (1, 1, 32, 64) and alpha.min() == 1
    Image.new("RGB", (32, 16), "black").save(source)
    with pytest.raises(ValueError, match="pixels"):
        load_reference_sky(tmp_path, row, (32, 64))
    with pytest.raises(ValueError, match="identity"):
        load_reference_sky(tmp_path, {**row, "source": str(tmp_path / "other.png")}, (32, 64))
