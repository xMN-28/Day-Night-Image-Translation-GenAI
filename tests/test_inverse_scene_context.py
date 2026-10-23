import pytest
import torch

from daynight.experiment_night_to_day import inverse_from_payload
from daynight.inverse_scene_context import (
    SceneContextInverse,
    SpatialSceneContext,
    add_context_preserving_weights,
)
from daynight.models.networks import ResnetGenerator


def test_migration_preserves_backbone_rng_and_initial_outputs():
    torch.set_num_threads(2)
    base = ResnetGenerator(base_channels=8, blocks=1, attention=True).eval()
    rng = torch.get_rng_state()
    changed = add_context_preserving_weights(base).eval()
    assert torch.equal(rng, torch.get_rng_state())
    x = torch.randn(1, 3, 32, 48)
    with torch.no_grad():
        assert torch.equal(base(x), changed(x))
    for name, value in base.state_dict().items():
        assert torch.equal(value, changed.state_dict()[name])


def test_context_has_initial_gradient_and_later_nonlocal_effect():
    block = SpatialSceneContext(8, width=16, grid=4, heads=4)
    x = torch.randn(1, 8, 16, 24, requires_grad=True)
    assert torch.equal(block(x), x)
    block(x).square().mean().backward()
    assert block.project.weight.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()
    with torch.no_grad():
        torch.nn.init.normal_(block.project.weight, std=.05)
        changed = x.detach().clone()
        changed[:, :, -4:, -4:] += 1
        # A far-away spatial change can affect the top-left correction.
        assert not torch.allclose(block(x.detach())[:, :, :2, :2], block(changed)[:, :, :2, :2], atol=1e-6)


def test_checkpoint_loader_roundtrip_for_context():
    model = SceneContextInverse(base_channels=8, blocks=1, attention=True)
    payload = {"config": {"model": {"base_channels": 8, "generator_blocks": 1, "attention": True}},
        "inverse_architecture": {"upsampling": "transpose", "scene_context": {"width": 128, "grid": 8, "heads": 4}},
        "ema": {"G_night_day": {"shadow": model.state_dict()}}}
    restored, architecture = inverse_from_payload(payload)
    assert isinstance(restored, SceneContextInverse)
    assert architecture["scene_context"]["grid"] == 8
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())


def test_unsupported_decoder_not_silently_converted():
    with pytest.raises(TypeError):
        add_context_preserving_weights(ResnetGenerator(base_channels=8, blocks=1, upsampling="resize"))
