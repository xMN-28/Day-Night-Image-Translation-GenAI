import numpy as np
import torch

from daynight.align_lumirender_pairs import _warp
from daynight.audit_darkdriving_rigid import bounded_similarity, similarity_flow


def test_robust_small_translation_and_large_motion_rejection():
    source = np.random.default_rng(1).uniform([0, 0], [511, 427], size=(200, 2))
    target = source + [2, -1]
    target[:40] += 70
    matrix, info = bounded_similarity(source, target, (512, 428))
    assert info["accepted_geometry"]
    np.testing.assert_allclose(matrix, [[1, 0, 2], [0, 1, -1]], atol=1e-8)
    matrix, info = bounded_similarity(source, source + [20, 0], (512, 428))
    assert matrix is None and not info["accepted_geometry"]


def test_night_to_day_sampling_sign_and_identity():
    day = torch.arange(16).float()[None, None, None].expand(1, 1, 8, 16)
    flow = similarity_flow(np.array([[1, 0, 2], [0, 1, 0]]), 8, 16)
    warped, valid = _warp(day, flow)
    torch.testing.assert_close(warped[..., :14], day[..., 2:], atol=1e-5, rtol=0)
    assert not valid[..., 14:].any()


def test_insufficient_support_rejected():
    matrix, info = bounded_similarity(np.zeros((10, 2)), np.zeros((10, 2)), (512, 428))
    assert matrix is None and not info["accepted_geometry"]
