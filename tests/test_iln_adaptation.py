import torch

from daynight.models.networks import InstanceLayerNorm, ResnetGenerator


def test_iln_warmstart_exactly_preserves_trained_generator_at_initialization():
    torch.manual_seed(314)
    parent = ResnetGenerator(base_channels=8, blocks=2, attention=True, upsampling="resize").eval()
    model = ResnetGenerator(
        base_channels=8, blocks=2, attention=True, upsampling="resize", residual_norm="iln"
    ).eval()
    missing, unexpected = model.load_state_dict(parent.state_dict(), strict=False)
    assert missing and not unexpected
    assert all(k.endswith((".rho", ".gamma", ".beta")) for k in missing)
    x = torch.randn(1, 3, 32, 48)
    with torch.no_grad():
        assert torch.allclose(parent(x), model(x), atol=1e-6, rtol=1e-6)
    for name, value in parent.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])


def test_iln_mixture_can_learn_from_identity_initialization():
    norm = InstanceLayerNorm(4)
    x = torch.randn(1, 4, 8, 8) + torch.arange(4)[None, :, None, None]
    target = x / 2
    loss = (norm(x) - target).square().mean()
    loss.backward()
    assert norm.rho.grad is not None and norm.rho.grad.abs().sum() > 0
    assert torch.isfinite(norm.rho.grad).all()
