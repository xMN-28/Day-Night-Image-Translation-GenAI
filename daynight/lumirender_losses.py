from __future__ import annotations

import torch
import torch.nn.functional as F

from .models.lumirender import gaussian_blur, srgb_to_linear


def total_variation(image: torch.Tensor) -> torch.Tensor:
    return (image[..., 1:, :] - image[..., :-1, :]).abs().mean() + (
        image[..., :, 1:] - image[..., :, :-1]
    ).abs().mean()


def linear_reconstruction_loss(details: dict[str, torch.Tensor]) -> torch.Tensor:
    return F.smooth_l1_loss(details["day_reconstruction"], details["source_linear"])


def reflectance_consistency_loss(details: dict[str, torch.Tensor]) -> torch.Tensor:
    source = details["source_linear"]
    source_chromaticity = source / source.sum(dim=1, keepdim=True).clamp_min(1e-3)
    reflectance = details["reflectance"]
    reflectance_chromaticity = reflectance / reflectance.sum(dim=1, keepdim=True).clamp_min(1e-3)
    chromaticity = F.smooth_l1_loss(reflectance_chromaticity, source_chromaticity.detach())
    illumination_smoothness = total_variation(details["day_illumination"])
    return chromaticity + 0.1 * illumination_smoothness


def confidence_photometric_loss(
    generated: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    aligned: torch.Tensor,
) -> torch.Tensor:
    generated_linear = srgb_to_linear(((generated + 1) / 2).clamp(0, 1))
    target_linear = srgb_to_linear(((target + 1) / 2).clamp(0, 1)).detach()
    mask = confidence * aligned.view(-1, 1, 1, 1)
    difference = torch.sqrt((generated_linear - target_linear).square() + 1e-6)
    return (difference * mask).sum() / (mask.sum() * generated.shape[1]).clamp_min(1)


def paired_perceptual_loss(
    generated: torch.Tensor, target: torch.Tensor, aligned: torch.Tensor
) -> torch.Tensor:
    mask = aligned.view(-1, 1, 1, 1)
    losses = []
    generated_level = generated
    target_level = target.detach()
    for _ in range(3):
        losses.append((F.smooth_l1_loss(generated_level, target_level, reduction="none") * mask).mean())
        generated_level = F.avg_pool2d(generated_level, 2)
        target_level = F.avg_pool2d(target_level, 2)
    return torch.stack(losses).mean()


def depth_normal_consistency_loss(details: dict[str, torch.Tensor]) -> torch.Tensor:
    depth = details["depth"]
    normals = details["normals"]
    dx = F.pad(depth[..., :, 1:] - depth[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(depth[..., 1:, :] - depth[..., :-1, :], (0, 0, 0, 1))
    implied = F.normalize(torch.cat((-dx, -dy, torch.ones_like(depth)), dim=1), dim=1)
    return (1 - (implied * normals).sum(dim=1).clamp(-1, 1)).mean()


def teacher_factorization_loss(
    details: dict[str, torch.Tensor],
    teacher_depth: torch.Tensor,
    teacher_semantic: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    mask = valid.view(-1, 1, 1, 1)
    if not bool(mask.any()):
        return details["depth"].new_zeros(())
    depth = F.smooth_l1_loss(details["depth"], teacher_depth, reduction="none")
    semantic = F.binary_cross_entropy(
        details["semantic"].clamp(1e-5, 1 - 1e-5), teacher_semantic, reduction="none"
    )
    dx = F.pad(teacher_depth[..., :, 1:] - teacher_depth[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(teacher_depth[..., 1:, :] - teacher_depth[..., :-1, :], (0, 0, 0, 1))
    teacher_normals = F.normalize(
        torch.cat((-dx, -dy, torch.ones_like(teacher_depth)), dim=1), dim=1
    )
    normal = 1 - (details["normals"] * teacher_normals).sum(dim=1, keepdim=True)
    return (depth * mask).mean() + (semantic * mask).mean() + 0.5 * (normal * mask).mean()


def emitter_validity_loss(details: dict[str, torch.Tensor]) -> torch.Tensor:
    emitter_mask = details["semantic"][:, 5:6]
    light = details["gaussian_light"].mean(dim=1, keepdim=True)
    bright_core = torch.relu(light - 0.08)
    outside = (bright_core * (1 - gaussian_blur(emitter_mask, 2.0))).mean()
    sparsity = emitter_mask.mean()
    return outside + 0.1 * sparsity


def physics_prior_loss(details: dict[str, torch.Tensor]) -> torch.Tensor:
    semantic = details["semantic"]
    overlap = (semantic * (semantic.sum(dim=1, keepdim=True) - semantic)).mean()
    normal_length = (details["normals"].norm(dim=1) - 1).abs().mean()
    roughness_extremes = torch.relu(0.03 - details["roughness"]).mean()
    return overlap + normal_length + roughness_extremes


def bounded_residual_loss(details: dict[str, torch.Tensor]) -> torch.Tensor:
    correction = details["correction"]
    coarse_leakage = gaussian_blur(correction, 4.0).abs().mean()
    return correction.abs().mean() + 4 * coarse_leakage


def luminance(image: torch.Tensor) -> torch.Tensor:
    image = ((image + 1) / 2).clamp(0, 1)
    return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
