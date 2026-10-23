import pytest
import torch

from daynight.audit_inverse_noise import bilateral, gaussian, validate_sources


def test_filters_preserve_constant_and_shape():
    x = torch.full((3, 12, 16), .37)
    for filter_fn in (bilateral, gaussian):
        result = filter_fn(x)
        assert result.shape == x.shape
        assert torch.allclose(result, x, atol=1e-6)


def test_bilateral_preserves_strong_boundary_better_than_gaussian():
    x = torch.zeros(3, 16, 16)
    x[:, :, 8:] = 1
    assert (bilateral(x) - x).abs().max() < .001
    assert (gaussian(x) - x).abs().max() > .01


def test_bilateral_deterministic_reduces_flat_noise():
    x = (.3 + torch.randn((3, 20, 20), generator=torch.Generator().manual_seed(5)) * .02).clamp(0, 1)
    first = bilateral(x)
    assert torch.equal(first, bilateral(x))
    assert first.std() < x.std()
    assert first.min() >= x.min() and first.max() <= x.max()


def test_development_gate_rejects_test_and_membership():
    suite = {"test_used": False, "groups": {"development": [{"index": i, "sha256": str(i)} for i in range(24)]}}
    manifest = {"direction": "night_to_day", "records": [
        {"split": "val", "index": i, "source_sha256": str(i)} for i in range(24)]}
    assert len(validate_sources(manifest, suite)) == 24
    with pytest.raises(ValueError):
        validate_sources({**manifest, "test_used": True}, suite)
    manifest["records"][0]["index"] = 999
    with pytest.raises(ValueError):
        validate_sources(manifest, suite)


def test_invalid_filter_parameters():
    with pytest.raises(ValueError):
        bilateral(torch.zeros(3, 8, 8), range_sigma=0)
