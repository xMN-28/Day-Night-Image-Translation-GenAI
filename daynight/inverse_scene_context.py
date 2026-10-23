"""Experimental spatial-context residual for a warm-started inverse only."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .models.networks import CBAM, ResnetGenerator


class SpatialSceneContext(nn.Module):
    """Mix8x8positioned scene tokens; zero output projection starts as identity.

    Not a semantic segmentation head, physical renderer or pretrained foundation
    model. The residual has no guaranteed preservation after training.
    """

    def __init__(self, channels, width=128, grid=8, heads=4):
        super().__init__()
        if width % heads or min(channels, width, grid, heads) < 1:
            raise ValueError("Positive dimensions and divisible attention width required")
        self.grid = grid
        self.reduce = nn.Conv2d(channels, width, 1)
        self.position = nn.Linear(2, width)
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.project = nn.Conv2d(width, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, (self.grid, self.grid))
        tokens = self.reduce(pooled).flatten(2).transpose(1, 2)
        axis = torch.linspace(-1, 1, self.grid, device=x.device, dtype=tokens.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        positions = torch.stack((xx, yy), -1).reshape(1, self.grid ** 2, 2)
        tokens = tokens + self.position(positions)
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        context = tokens.transpose(1, 2).reshape(x.shape[0], -1, self.grid, self.grid)
        correction = self.project(context)
        return x + F.interpolate(correction, size=x.shape[-2:], mode="bilinear", align_corners=False)


class SceneContextInverse(ResnetGenerator):
    def __init__(self, *, context_width=128, context_grid=8, context_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.scene_context = SpatialSceneContext(self.down2[0].out_channels,
            context_width, context_grid, context_heads)

    def encode_features(self, source):
        x, features = super().encode_features(source)
        x = self.scene_context(x)
        features[-1] = x
        return x, features


def add_context_preserving_weights(base):
    if type(base) is not ResnetGenerator or base.detail_skip is not None:
        raise ValueError("Only unwrapped standard inverse is supported")
    # Local CPU initializer does not alter main training RNG or primary samples.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2034)
        model = SceneContextInverse(base_channels=base.stem[1].out_channels,
            blocks=len(base.residuals), attention=isinstance(base.residuals[0].attention, CBAM),
            upsampling="transpose")
    # Explicitly reject an unsupported decoder/normalization instead of silently
    # mapping kernels or changing the existing function.
    if not isinstance(base.up1[0], nn.ConvTranspose2d):
        raise TypeError("Only original transpose decoder supported for this isolation")
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    if unexpected or not missing or any(not key.startswith("scene_context.") for key in missing):
        raise ValueError("Migration would discard or initialize backbone parameters")
    if any(not torch.equal(value, model.state_dict()[key]) for key, value in base.state_dict().items()):
        raise AssertionError("Backbone migration is not exact")
    return model
