import pytest
from PIL import Image

from daynight.experiment_night_to_day import visibility_control
from daynight.turbo_pilot import fit_input, reference_spec


def test_experimental_reference_directions_are_explicit_and_default_unchanged():
    assert reference_spec()[0] == "day2night.pkl"
    assert reference_spec("night_to_day")[0] == "night2day.pkl"
    with pytest.raises(ValueError):
        reference_spec("unknown")


def test_visibility_control_is_bounded_and_preserves_dimensions():
    source = Image.new("RGB", (19, 11), (30, 80, 250))
    result = visibility_control(source)
    assert result.size == source.size
    assert result.mode == "RGB"
    assert all(a <= b <= 255 for a, b in zip(source.getpixel((0, 0)), result.getpixel((0, 0))))


def test_reference_preprocessing_preserves_aspect_and_only_pads_to_vae_grid():
    source = Image.new("RGB", (640, 362), (90, 120, 150))
    fitted, tensor, crop = fit_input(source, 512)
    assert fitted.size == (512, 290)
    assert tensor.shape == (1, 3, 296, 512)
    assert crop == (0, 0, 512, 290)
    assert tensor.min() >= -1 and tensor.max() <= 1
