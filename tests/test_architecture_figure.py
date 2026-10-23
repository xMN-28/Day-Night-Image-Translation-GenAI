"""Numerical checks for the real-tensor figure extractor, without checkpoints."""

import pytest
import torch

from daynight.sky_refinement import compose_night
from scripts.figures.extract_architecture_assets import compositor_components


@pytest.mark.parametrize("alpha_kind", ["zero", "one", "soft"])
@pytest.mark.parametrize("with_vegetation", [False, True])
def test_figure_intermediates_match_production_compositor(alpha_kind, with_vegetation):
    generator = torch.Generator().manual_seed(913)
    source = torch.rand((1, 3, 64, 96), generator=generator)
    base = torch.rand((1, 3, 64, 96), generator=generator) * 0.25
    alpha = torch.rand((1, 1, 64, 96), generator=generator)
    if alpha_kind == "zero":
        alpha.zero_()
    elif alpha_kind == "one":
        alpha.fill_(1)
    vegetation = torch.rand(alpha.shape, generator=generator) if with_vegetation else None
    actual = compositor_components(source, base, alpha, vegetation)
    expected = compose_night(source, base, alpha, vegetation=vegetation, recover_thin_edges=True)
    torch.testing.assert_close(actual["reconstructed"], expected, atol=1e-7, rtol=1e-6)
    assert set(actual) == {
        "repair_support",
        "unmixed_foreground",
        "sky_field",
        "sky_contribution",
        "reconstructed",
    }
    assert all(torch.isfinite(value).all() for value in actual.values())
    assert actual["repair_support"].shape == alpha.shape
    assert actual["reconstructed"].shape == source.shape
