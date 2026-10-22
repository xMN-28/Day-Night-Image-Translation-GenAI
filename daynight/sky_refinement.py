"""Experimental semantic sky correction on top of frozen LumiCycle V2 weights.

Images use [0, 1]. Semantic masks are supervision, never hand-drawn evaluation masks.
The renderer is an explicit compositing operation, not an invented GAN achievement.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def box(x: torch.Tensor, radius: int) -> torch.Tensor:
    radius = min(radius, (min(x.shape[-2:]) - 1) // 2)
    return F.avg_pool2d(F.pad(x, (radius,) * 4, mode="reflect"), 2 * radius + 1, 1)


def gray(x: torch.Tensor) -> torch.Tensor:
    return (x * x.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, True)


@torch.no_grad()
def connected_sky_support(probability: torch.Tensor) -> torch.Tensor:
    """Exclude disconnected dashboard reflections from a road-scene sky matte.

    Work on a small grid; seed confident upper-frame regions. This permits sky
    under bridges while rejecting lower-frame windshield/dashboard reflections.
    The road-domain prior is inappropriate for arbitrary indoor compositions.
    """
    size = (min(54, probability.shape[-2]), min(96, probability.shape[-1]))
    p = F.adaptive_max_pool2d(probability, size)
    allowed = p > 0.15
    reached = torch.zeros_like(allowed)
    seed_rows = max(1, round(size[0] * 0.65))
    reached[:, :, :seed_rows] = p[:, :, :seed_rows] > 0.65
    for _ in range(sum(size)):
        reached = (F.max_pool2d(reached.float(), 3, 1, 1) > 0) & allowed
    support = F.interpolate(reached.float(), probability.shape[-2:], mode="nearest")
    return probability * support


@torch.no_grad()
def phase_average(generator: nn.Module, source: torch.Tensor) -> torch.Tensor:
    """Average all 4x4 encoder phases with reflection context and exact alignment.

    This suppresses phase-dependent decoder checkerboard texture. It costs 16
    forward calls and is an inference ablation, not a newly learned decoder.
    """
    height, width = source.shape[-2:]
    padded = F.pad(source, (8,) * 4, mode="reflect")
    total = torch.zeros_like(source, dtype=torch.float32)
    for dy in range(4):
        for dx in range(4):
            window = padded[..., dy : dy + height + 8, dx : dx + width + 8]
            predicted = generator(window).float()
            total += predicted[..., 8 - dy : 8 - dy + height, 8 - dx : 8 - dx + width]
    return total / 16


def guided_filter(
    guide: torch.Tensor, target: torch.Tensor, radius: int = 8, epsilon: float = 0.001
) -> torch.Tensor:
    """Local linear luminance guidance; handles arbitrarily many target channels."""
    g = gray(guide)
    mean_g, mean_t = box(g, radius), box(target, radius)
    var = (box(g * g, radius) - mean_g.square()).clamp_min(0)
    cov = box(g * target, radius) - mean_g * mean_t
    a = cov / (var + epsilon)
    b = mean_t - a * mean_g
    return box(a, radius) * g + box(b, radius)


def refine_sky_alpha(
    source: torch.Tensor, probability: torch.Tensor, mixture: torch.Tensor | None = None,
    *, recover_color_holes: bool = False,
) -> torch.Tensor:
    """Guided soft mask plus local foreground/background color unmixing.

    `mixture` marks vegetation when teacher labels are available. Without it the
    semantic probability itself bounds the transition neighborhood.
    """
    p = probability.float().clamp(0, 1)
    reliable = (p > 0.75).float()
    bg_mass = box(reliable, 63)
    global_bg = (source * reliable).sum((2, 3), True) / reliable.sum((2, 3), True).clamp_min(1)
    # Smooth normalized convolution with a weak global fallback, avoiding seams
    # where a box window suddenly gains or loses its final reliable sky pixel.
    background = (box(source * reliable, 63) + 0.003 * global_bg) / (bg_mass + 0.003)
    dark = ((p < 0.25) & (gray(source) < 0.35)).float()
    fg_mass = box(dark, 31)
    foreground = (
        box(source * dark, 31) + 0.003 * source.new_tensor([0.12, 0.13, 0.12])[None, :, None, None]
    ) / (fg_mass + 0.003)
    delta = background - foreground
    alpha = ((source - foreground) * delta).sum(1, True) / delta.square().sum(1, True).clamp_min(
        0.01
    )
    neighborhood = (bg_mass / 0.008).clamp(0, 1)
    transition = neighborhood
    if mixture is not None:
        # A leafless tree is often one solid vegetation segment. Restricting
        # refinement to a narrow sky-edge band would miss its interior holes.
        rows = torch.linspace(0, 1, p.shape[-2], device=p.device)[None, None, :, None]
        horizon = (rows * reliable).amax((2, 3), True)
        above_horizon = ((horizon + 0.03 - rows) / 0.04).clamp(0, 1)
        has_sky = (reliable.sum((2, 3), True) > 16).float()
        transition = torch.maximum(neighborhood * (p > 0.1), mixture * above_horizon * has_sky)
    else:
        transition = transition * (F.max_pool2d(p, 17, 1, 8) > 0.2)
    if recover_color_holes:
        # The two-class semantic student can label an entire cable bundle as
        # foreground, including its bright interior sky. Recover only source
        # colors close to locally observed, confident sky above its horizon.
        # Continuous confidence avoids binary p>0.1 cutouts. Dark wires retain
        # low unmixing alpha; no correction is possible without reliable sky.
        color_error = (source.float() - background).square().mean(1, True)
        color_match = ((0.025 - color_error) / 0.020).clamp(0, 1)
        rows = torch.linspace(0, 1, p.shape[-2], device=p.device)[None, None, :, None]
        horizon = (rows * reliable).amax((2, 3), True)
        upper_support = ((horizon + 0.03 - rows) / 0.04).clamp(0, 1)
        has_sky = (reliable.sum((2, 3), True) > 16).float()
        recovery = color_match * (bg_mass / 0.04).clamp(0, 1) * upper_support * has_sky
        transition = torch.maximum(transition, recovery)
    guided = guided_filter(source, p).clamp(0, 1)
    result = alpha.clamp(0, 1) * transition + guided * (1 - transition)
    return result.clamp(0, 1)


def compose_night(
    source: torch.Tensor,
    base: torch.Tensor,
    alpha: torch.Tensor,
    strength: float = 1.0,
    vegetation: torch.Tensor | None = None,
    *, recover_thin_edges: bool = False,
) -> torch.Tensor:
    """Replace the daylight sky component and recover source fine edges nearby.

    Preserve V2 away from the sky. On the sky boundary use an achromatic local
    exposure transfer, so red/cyan GAN halos are not copied into thin objects.
    No moons, stars, lamps or scene geometry are pasted or synthesized here.
    """
    a = alpha.clamp(0, 1)
    # Smooth spatially varying exposure transfer retains source high frequencies.
    ratio = (box(gray(base), 15) / box(gray(source), 15).clamp_min(0.04)).clamp(0.12, 0.85)
    foreground_ceiling = 0.12 if recover_thin_edges else 0.24
    clean_foreground = source * ratio.clamp_max(foreground_ceiling)
    near_sky = box(F.max_pool2d(a, 25, 1, 12), 5).clamp(0, 1)
    near = near_sky if vegetation is None else torch.maximum(a, near_sky * vegetation)
    if recover_thin_edges:
        # The vegetation class does not include cables. Recover source dark
        # details in a narrow sky boundary band instead of retaining bright GAN
        # streaks there. Keep distant facades and foreground lights unchanged.
        boundary = box(F.max_pool2d(a, 17, 1, 8), 1)
        # Selecting source-derived edge pixels fully is necessary: a fractional
        # blend would retain the very bright GAN cable halos being removed.
        near = torch.maximum(near, (boundary * 10).clamp(0, 1))
    # A dim blue-gray sky with a smooth horizon gradient and retained cloud detail.
    height = source.shape[-2]
    y = torch.linspace(0, 1, height, device=source.device)[None, None, :, None]
    sky_color = source.new_tensor([0.026, 0.035, 0.055])[None, :, None, None]
    sky = sky_color + y * source.new_tensor([0.035, 0.037, 0.038])[None, :, None, None]
    texture = (gray(source) - box(gray(source), 15)).clamp(-0.2, 0.2)
    sky = (sky + texture * 0.06).clamp(0, 0.15)
    # Foreground already contains the source sky mixed into anti-aliased pixels.
    # Remove that daylight contribution before inserting the new background.
    known = (a > 0.9).float()
    global_bg = (source * known).sum((2, 3), True) / known.sum((2, 3), True).clamp_min(1)
    daylight_bg = (box(source * known, 63) + 0.003 * global_bg) / (box(known, 63) + 0.003)
    old_bg = daylight_bg * ratio.clamp_max(foreground_ceiling)
    unmix = (clean_foreground - a * old_bg).clamp_min(0)
    mix_result = unmix + a * sky
    # Continuous support prevents small uncertain alpha values from selecting a
    # completely different foreground formula (which produced patch cutouts).
    result = base * (1 - near) + mix_result * near
    return (base + strength * (result - base)).clamp(0, 1)


class SkyStudent(nn.Module):
    """Small full-resolution U-Net with skip connections and resize-convolution."""

    def __init__(self, channels: int = 24, outputs: int = 1) -> None:
        super().__init__()

        def block(inc: int, out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(inc, out, 3, padding=1),
                nn.GroupNorm(4, out),
                nn.SiLU(),
                nn.Conv2d(out, out, 3, padding=1),
                nn.GroupNorm(4, out),
                nn.SiLU(),
            )

        self.enc1 = block(3, channels)
        self.enc2 = block(channels, channels * 2)
        self.enc3 = block(channels * 2, channels * 4)
        self.middle = block(channels * 4, channels * 8)
        self.dec3 = block(channels * 12, channels * 4)
        self.dec2 = block(channels * 6, channels * 2)
        self.dec1 = block(channels * 3, channels)
        self.head = nn.Conv2d(channels, outputs, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(F.avg_pool2d(x1, 2))
        x3 = self.enc3(F.avg_pool2d(x2, 2))
        z = self.middle(F.avg_pool2d(x3, 2))
        for skip, decoder in ((x3, self.dec3), (x2, self.dec2), (x1, self.dec1)):
            z = decoder(
                torch.cat(
                    [
                        F.interpolate(
                            z, size=skip.shape[-2:], mode="bilinear", align_corners=False
                        ),
                        skip,
                    ],
                    1,
                )
            )
        return self.head(z)


@torch.no_grad()
def predict_matte(student: SkyStudent, source: torch.Tensor, *, recover_color_holes: bool = False):
    low = F.interpolate(source, (216, 384), mode="bilinear", align_corners=False)
    probabilities = student(low).float().sigmoid()
    probabilities = F.interpolate(
        probabilities, source.shape[-2:], mode="bilinear", align_corners=False
    )
    sky = connected_sky_support(probabilities[:, :1])
    vegetation = probabilities[:, 1:2] if probabilities.shape[1] > 1 else None
    alpha = refine_sky_alpha(source, sky, vegetation, recover_color_holes=recover_color_holes)
    return alpha, vegetation
