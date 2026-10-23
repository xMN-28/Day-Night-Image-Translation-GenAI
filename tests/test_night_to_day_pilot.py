import pytest
import torch
from PIL import Image

from daynight.experiment_night_day_conditioning import condition_source
from daynight.experiment_night_to_day import inverse_from_payload
from daynight.models import ResnetGenerator
from daynight.models.networks import ResizeConv2d
from daynight.train_night_to_day_pilot import (
    adversarial_lsgan,
    pilot_loss,
    select_training_rows,
    static_region_coarse_loss,
    static_region_rgb_loss,
)


def test_real_coarse_region_loss_constant_offset_and_finite_gradient():
    p = torch.zeros(1, 3, 32, 48, requires_grad=True)
    target = torch.full_like(p, .5)
    confidence = torch.ones(1, 1, 32, 48)
    loss = static_region_coarse_loss(p, target, confidence)
    assert loss.item() == pytest.approx(.5)
    loss.backward()
    assert torch.isfinite(p.grad).all() and p.grad.mean() < 0


def test_static_real_region_loss_has_zero_gradient_outside_and_correct_normalization():
    prediction = torch.zeros(1, 3, 32, 48, requires_grad=True)
    target = torch.ones_like(prediction)
    weights = torch.zeros(1, 1, 32, 48)
    weights[..., :4, :4] = .75
    loss = static_region_rgb_loss(prediction, target, weights)
    assert loss.item() == pytest.approx(1)
    loss.backward()
    assert prediction.grad[..., 4:, :].abs().sum() == 0
    assert prediction.grad[..., :4, :4].abs().sum() > 0
    with pytest.raises(ValueError):
        static_region_rgb_loss(prediction, target, torch.zeros_like(weights))


def test_inverse_adversarial_targets_and_gradients():
    outputs = [torch.zeros(1, 1, 3, 4, requires_grad=True), torch.zeros(1, 1, 1, 1, requires_grad=True)]
    assert adversarial_lsgan(outputs, False).item() == 0
    loss = adversarial_lsgan(outputs, True)
    assert loss.item() == pytest.approx(1)
    loss.backward()
    assert all(torch.isfinite(output.grad).all() and output.grad.mean() < 0 for output in outputs)


def test_capacity_subset_is_synthetic_training_only_and_deterministic():
    rows = [{"split": "train", "index": i, "original_day_train_index": i} for i in range(64)]
    rows += [{"split": "val", "index": 999, "original_day_train_index": 999},
             {"split": "train", "index": 1000}]
    selected = select_training_rows(rows, "synthetic", 32)
    assert len(selected) == 32
    assert selected == select_training_rows(rows, "synthetic", 32)
    assert selected[0]["index"] == 0 and selected[-1]["index"] == 63
    assert all(r["split"] == "train" and "original_day_train_index" in r for r in selected)
    with pytest.raises(ValueError):
        select_training_rows(rows, "invalid")
    with pytest.raises(ValueError):
        select_training_rows([], "synthetic")


def test_teacher_conditioning_is_explicit_bounded_and_raw_is_unchanged():
    image = Image.new("RGB", (19, 13), (80, 140, 255))
    assert condition_source(image, "raw").tobytes() == image.tobytes()
    for method in ("rolloff", "balanced_rolloff"):
        result = condition_source(image, method)
        assert result.size == image.size and result.mode == "RGB"
        assert max(result.getpixel((0, 0))) <= 106
    with pytest.raises(ValueError):condition_source(image, "unknown")


def test_inverse_resize_warmstart_preserves_encoder_and_roundtrips_saved_topology():
    original = ResnetGenerator(base_channels=8, blocks=1, attention=True)
    payload = {"config": {"model": {"base_channels": 8, "generator_blocks": 1, "attention": True}},
        "ema": {"G_night_day": {"shadow": original.state_dict()}}}
    converted, architecture = inverse_from_payload(payload, "resize")
    assert isinstance(converted.up1[0], ResizeConv2d)
    assert architecture["converted"]
    for key, value in original.state_dict().items():
        if key not in {"up1.0.weight", "up2.0.weight"}:
            assert torch.equal(converted.state_dict()[key], value)
    payload["inverse_architecture"] = architecture
    payload["ema"]["G_night_day"]["shadow"] = converted.state_dict()
    restored, metadata = inverse_from_payload(payload)
    assert isinstance(restored.up1[0], ResizeConv2d)
    assert not metadata["converted"]
    x = torch.randn(1, 3, 32, 48)
    assert torch.equal(restored(x), converted(x))


def test_inverse_target_loss_exact_match_and_finite_gradient():
    target = torch.rand(1, 3, 32, 48) * 2 - 1
    prediction = target.clone().requires_grad_(True)
    loss = pilot_loss(prediction, target)
    assert loss.item() == pytest.approx(0)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_inverse_target_loss_charges_daylight_offset_not_source_structure():
    target = torch.full((1, 3, 32, 48), 0.5)
    prediction = torch.zeros_like(target, requires_grad=True)
    loss = pilot_loss(prediction, target)
    # RGB10 + coarse2, zero target-gradient loss for spatially constant offset.
    assert loss.item() == pytest.approx(6)
    loss.backward()
    assert prediction.grad.mean().item() < 0
