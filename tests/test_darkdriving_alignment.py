import torch

from daynight.audit_darkdriving_alignment import pad_for_flow, sampling_jacobian


def test_flow_padding_preserves_source_geometry():
    x = torch.randn(1, 3, 428, 512)
    padded, shape = pad_for_flow(x)
    assert shape == (428, 512) and padded.shape[-2:] == (432, 512)
    torch.testing.assert_close(x, padded[..., :428, :512], atol=0, rtol=0)


def test_sampling_jacobian_translation_and_fold():
    flow = torch.ones(1, 2, 8, 16) * 2
    torch.testing.assert_close(sampling_jacobian(flow), torch.ones(1, 8, 16))
    flow[:, 0] = -2 * torch.arange(16)[None, None]
    assert (sampling_jacobian(flow) < 0).all()
