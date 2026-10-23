import numpy as np
import pytest
from PIL import Image

from daynight.experiment_inverse_restoration import glare_mask, preserve_outside


def test_dark_source_has_no_glare_mask():
    image = Image.new("RGB", (128, 72), (5, 8, 10))
    assert not np.asarray(glare_mask(image)).any()


def test_mask_is_automatic_bounded_and_deterministic():
    array = np.zeros((96, 160, 3), dtype=np.uint8)
    array[40:45, 70:75] = 255
    image = Image.fromarray(array)
    a = np.asarray(glare_mask(image))
    b = np.asarray(glare_mask(image))
    assert np.array_equal(a, b)
    assert a[42, 72] > 0
    assert a[0, 0] == a[-1, -1] == 0


def test_exact_outside_pixel_preservation():
    base = Image.new("RGB", (32, 32), (25, 60, 100))
    proposal = Image.new("RGB", (32, 32), (240, 220, 200))
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:20, 10:20] = 128
    output = np.asarray(preserve_outside(base, proposal, Image.fromarray(mask)))
    assert np.array_equal(output[mask == 0], np.asarray(base)[mask == 0])
    assert (output[15, 15] != np.asarray(base)[15, 15]).any()


def test_mismatched_restoration_size_rejected():
    with pytest.raises(ValueError, match="sizes"):
        preserve_outside(Image.new("RGB", (8, 8)), Image.new("RGB", (9, 8)), Image.new("L", (8, 8)))
