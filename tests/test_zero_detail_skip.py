import pytest
import torch

from daynight.models.networks import ResnetGenerator, build_models


@pytest.mark.parametrize("upsampling", ["transpose", "resize"])
def test_zero_detail_skip_keeps_trained_weights_and_function(upsampling):
    parent = ResnetGenerator(
        base_channels=8, blocks=1, attention=True, upsampling=upsampling
    ).eval()
    adapted = ResnetGenerator(
        base_channels=8, blocks=1, attention=True, upsampling=upsampling, detail_skip=True
    ).eval()
    missing, unexpected = adapted.load_state_dict(parent.state_dict(), strict=False)
    assert missing and all(key.startswith("detail_skip.") for key in missing)
    assert not unexpected
    assert all(
        torch.equal(adapted.state_dict()[key], value) for key, value in parent.state_dict().items()
    )
    for size in ((32, 48), (33, 49)):
        x = torch.randn(1, 3, *size)
        torch.testing.assert_close(parent(x), adapted(x), rtol=0, atol=0)


def test_detail_skip_can_learn_without_changing_frozen_base():
    model = ResnetGenerator(base_channels=8, blocks=1, upsampling="resize", detail_skip=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("detail_skip."))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    loss = model(torch.randn(1, 3, 32, 48)).square().mean()
    loss.backward()
    assert model.detail_skip[-1].weight.grad.abs().sum() > 0
    assert all(
        p.grad is None
        for name, p in model.named_parameters()
        if not name.startswith("detail_skip.")
    )
    optimizer.step()
    assert any(
        not torch.equal(value, before[key])
        for key, value in model.state_dict().items()
        if key.startswith("detail_skip.")
    )
    assert all(
        torch.equal(value, before[key])
        for key, value in model.state_dict().items()
        if not key.startswith("detail_skip.")
    )


def test_factory_keeps_identity_initialization_and_inverse_unchanged():
    models = build_models(
        {"model": {"base_channels": 8, "generator_blocks": 1, "day_detail_skip": True}}
    )
    assert models["G_day_night"].detail_skip[-1].weight.count_nonzero() == 0
    assert models["G_night_day"].detail_skip is None
