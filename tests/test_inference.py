from __future__ import annotations

import pytest
from PIL import Image

from daynight.inference import _fit_and_pad


def test_inference_preprocessing_preserves_aspect_and_multiple_of_four() -> None:
    image = Image.new("RGB", (1001, 503), "white")
    padded, crop = _fit_and_pad(image, 768)
    assert padded.width % 4 == 0
    assert padded.height % 4 == 0
    assert crop[2] <= 768
    assert crop[2] / crop[3] == pytest.approx(1001 / 503, rel=0.01)
