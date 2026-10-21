from PIL import Image

from daynight.turbo_pilot import fit_input


def test_reference_preprocessing_preserves_aspect_and_only_pads_to_vae_grid():
    source = Image.new("RGB", (640, 362), (90, 120, 150))
    fitted, tensor, crop = fit_input(source, 512)
    assert fitted.size == (512, 290)
    assert tensor.shape == (1, 3, 296, 512)
    assert crop == (0, 0, 512, 290)
    assert tensor.min() >= -1 and tensor.max() <= 1
