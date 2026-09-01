from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .models.networks import discriminator_outputs, haar_high_frequency


def lsgan_generator_loss(output: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    losses = [
        F.mse_loss(prediction, torch.ones_like(prediction))
        for prediction in discriminator_outputs(output)
    ]
    return torch.stack(losses).mean()


def lsgan_discriminator_loss(
    real_output: torch.Tensor | list[torch.Tensor], fake_output: torch.Tensor | list[torch.Tensor]
) -> torch.Tensor:
    losses = []
    for real, fake in zip(
        discriminator_outputs(real_output), discriminator_outputs(fake_output), strict=True
    ):
        losses.append(
            0.5
            * (F.mse_loss(real, torch.ones_like(real)) + F.mse_loss(fake, torch.zeros_like(fake)))
        )
    return torch.stack(losses).mean()


def discriminator_accuracy(
    real_output: torch.Tensor | list[torch.Tensor], fake_output: torch.Tensor | list[torch.Tensor]
) -> torch.Tensor:
    scores = []
    for real, fake in zip(
        discriminator_outputs(real_output), discriminator_outputs(fake_output), strict=True
    ):
        scores.append(0.5 * ((real > 0.5).float().mean() + (fake < 0.5).float().mean()))
    return torch.stack(scores).mean()


def patch_nce_loss(
    source_features: list[torch.Tensor],
    target_features: list[torch.Tensor],
    patches: int = 128,
    temperature: float = 0.07,
) -> torch.Tensor:
    losses = []
    for source, target in zip(source_features, target_features, strict=True):
        batch, _channels, height, width = source.shape
        count = min(patches, height * width)
        indices = torch.randperm(height * width, device=source.device)[:count]
        source_vectors = source.flatten(2)[:, :, indices].transpose(1, 2)
        target_vectors = target.flatten(2)[:, :, indices].transpose(1, 2)
        source_vectors = F.normalize(source_vectors, dim=-1)
        target_vectors = F.normalize(target_vectors, dim=-1)
        for batch_index in range(batch):
            logits = target_vectors[batch_index] @ source_vectors[batch_index].transpose(0, 1)
            labels = torch.arange(count, device=source.device)
            losses.append(F.cross_entropy(logits / temperature, labels))
    return torch.stack(losses).mean() if losses else source_features[0].new_zeros(())


class SobelEdgeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        horizontal = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        vertical = horizontal.transpose(0, 1)
        self.register_buffer("horizontal", horizontal.view(1, 1, 3, 3))
        self.register_buffer("vertical", vertical.view(1, 1, 3, 3))

    def edges(self, image: torch.Tensor) -> torch.Tensor:
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        gx = F.conv2d(gray, self.horizontal, padding=1)
        gy = F.conv2d(gray, self.vertical, padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)

    def forward(self, source: torch.Tensor, translated: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.edges(source), self.edges(translated))


class DinoSemanticLoss(nn.Module):
    """Lazy frozen DINOv2-S feature loss. The first use downloads official weights."""

    def __init__(self) -> None:
        super().__init__()
        self.model: nn.Module | None = None

    def _load(self, device: torch.device) -> nn.Module:
        if self.model is None:
            torch.hub.set_dir(str(Path("artifacts/torch_cache/hub").resolve()))
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        return self.model.to(device)

    def _features(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(
            (image + 1) / 2, size=(224, 224), mode="bilinear", align_corners=False
        )
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return self._load(image.device)((image - mean) / std)

    def forward(self, source: torch.Tensor, translated: torch.Tensor) -> torch.Tensor:
        source_features = self._features(source).detach()
        translated_features = self._features(translated)
        return (1 - F.cosine_similarity(source_features, translated_features, dim=-1)).mean()


def diff_augment(image: torch.Tensor, enabled: bool = True) -> torch.Tensor:
    if not enabled:
        return image
    # Differentiable color jitter and integer translation, shared per mini-batch.
    brightness = torch.empty((), device=image.device).uniform_(-0.1, 0.1)
    saturation = torch.empty((), device=image.device).uniform_(0.8, 1.2)
    mean = image.mean(dim=1, keepdim=True)
    image = (image - mean) * saturation + mean + brightness
    shift_x = int(torch.randint(-8, 9, (), device=image.device).item())
    shift_y = int(torch.randint(-8, 9, (), device=image.device).item())
    return torch.roll(image, shifts=(shift_y, shift_x), dims=(2, 3)).clamp(-1, 1)


def output_variance(image: torch.Tensor) -> torch.Tensor:
    return image.detach().float().var(dim=(0, 2, 3)).mean()


def color_statistics_loss(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match unpaired target illumination without comparing unrelated pixels."""
    generated_01 = (generated + 1) / 2
    target_01 = ((target.detach() + 1) / 2).to(generated_01.dtype)

    def statistics(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_mean = image.mean(dim=(2, 3))
        rgb_std = image.std(dim=(2, 3), unbiased=False)
        luminance = (
            0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        )
        luminance_mean = luminance.mean(dim=(2, 3))
        luminance_std = luminance.std(dim=(2, 3), unbiased=False)
        return torch.cat((rgb_mean, luminance_mean), dim=1), torch.cat(
            (rgb_std, luminance_std), dim=1
        )

    generated_mean, generated_std = statistics(generated_01)
    target_mean, target_std = statistics(target_01)
    return F.l1_loss(generated_mean, target_mean) + F.l1_loss(generated_std, target_std)


def regional_illumination_loss(
    source: torch.Tensor,
    generated: torch.Tensor,
    target: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    """Match scene-level exposure and focus on likely sky in the upper frame."""

    def luminance(image: torch.Tensor) -> torch.Tensor:
        image = (image + 1) / 2
        return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]

    source_luminance = luminance(source)
    generated_luminance = luminance(generated)
    target_luminance = luminance(target.detach()).to(generated_luminance.dtype)

    # Horizontal bands capture sky/midground/road exposure without requiring paired scenes.
    generated_bands = F.adaptive_avg_pool2d(generated_luminance, (3, 1))
    target_bands = F.adaptive_avg_pool2d(target_luminance, (3, 1))
    band_loss = F.smooth_l1_loss(generated_bands, target_bands)

    height = source_luminance.shape[2]
    y = torch.linspace(0, 1, height, device=source.device, dtype=source_luminance.dtype)
    upper_prior = (1 - y / 0.65).clamp(0, 1).view(1, 1, height, 1)
    if direction == "day_to_night":
        candidate_mask = torch.sigmoid((source_luminance - 0.65) * 12) * upper_prior
    elif direction == "night_to_day":
        candidate_mask = torch.sigmoid((0.35 - source_luminance) * 12) * upper_prior
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    def weighted_mean(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (image * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3)).clamp_min(1e-4)

    generated_focus = weighted_mean(generated_luminance, candidate_mask)
    target_upper = weighted_mean(target_luminance, upper_prior.expand_as(target_luminance))
    focus_loss = F.smooth_l1_loss(generated_focus, target_upper)
    return band_loss + focus_loss


def wavelet_structure_loss(
    source: torch.Tensor, translated: torch.Tensor, levels: int = 2
) -> torch.Tensor:
    """Preserve normalized high-frequency structure across lighting changes."""
    losses = []
    source_level = source
    translated_level = translated
    for _ in range(levels):
        source_high = haar_high_frequency(source_level).detach()
        translated_high = haar_high_frequency(translated_level)
        source_scale = source_high.abs().mean(dim=(2, 3), keepdim=True).clamp_min(1e-3)
        translated_scale = translated_high.abs().mean(dim=(2, 3), keepdim=True).clamp_min(1e-3)
        difference = source_high / source_scale - translated_high / translated_scale
        losses.append(torch.sqrt(difference.square() + 1e-6).mean())
        source_level = F.avg_pool2d(source_level, 2, ceil_mode=True)
        translated_level = F.avg_pool2d(translated_level, 2, ceil_mode=True)
    return torch.stack(losses).mean()


def spatial_self_similarity_loss(
    source: torch.Tensor, translated: torch.Tensor, grid_size: int = 12
) -> torch.Tensor:
    """F/LSeSim-inspired local spatial-correlation preservation."""

    def descriptors(image: torch.Tensor) -> torch.Tensor:
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        pooled = F.adaptive_avg_pool2d(gray, (grid_size, grid_size))
        patches = F.unfold(pooled, kernel_size=3, padding=1).transpose(1, 2)
        return F.normalize(patches, dim=-1)

    source_descriptors = descriptors(source).detach()
    translated_descriptors = descriptors(translated)
    source_similarity = source_descriptors @ source_descriptors.transpose(1, 2)
    translated_similarity = translated_descriptors @ translated_descriptors.transpose(1, 2)
    return F.smooth_l1_loss(translated_similarity, source_similarity)


def refinement_residual_loss(coarse: torch.Tensor, refined: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(refined, coarse.detach())
