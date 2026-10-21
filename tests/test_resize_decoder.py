import torch

from daynight.evaluate_resize_decoder import periodic_bias
from daynight.models.networks import ResizeConv2d, ResnetGenerator, build_models
from daynight.train_resize_decoder import adapt_decoder, decoder_loss


def test_warmstart_keeps_every_encoder_weight_and_only_changes_decoder_layout():
    original = ResnetGenerator(base_channels=8, blocks=1, attention=True)
    adapted = ResnetGenerator(base_channels=8, blocks=1, attention=True, upsampling="resize")
    state = adapt_decoder(original.state_dict())
    adapted.load_state_dict(state)
    for key, value in original.state_dict().items():
        if key not in {"up1.0.weight", "up2.0.weight"}:
            assert torch.equal(adapted.state_dict()[key], value)
    x = torch.randn(1, 3, 32, 48)
    y = adapted(x)
    assert y.shape == x.shape
    loss = decoder_loss(y, original(x).detach())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapted.parameters())


def test_decoder_configuration_changes_day_direction_only():
    config = {"model": {"base_channels": 8, "generator_blocks": 1, "day_upsampling": "resize"}}
    models = build_models(config)
    assert isinstance(models["G_day_night"].up1[0], ResizeConv2d)
    assert isinstance(models["G_night_day"].up1[0], torch.nn.ConvTranspose2d)


def test_periodic_diagnostic_detects_grid_but_not_constant_sky():
    sky = torch.full((1, 3, 48, 64), 0.1)
    mask = torch.ones(1, 1, 48, 64)
    assert periodic_bias(sky, mask) < 1e-7
    sky[..., ::2, ::2] += 0.1
    assert periodic_bias(sky, mask) > 0.01
