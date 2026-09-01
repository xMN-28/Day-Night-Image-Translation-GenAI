from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .models.networks import discriminator_outputs


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
