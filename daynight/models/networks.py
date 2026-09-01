from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average = self.mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        maximum = self.mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        return x * torch.sigmoid(average + maximum)


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat(
            [torch.mean(x, dim=1, keepdim=True), torch.amax(x, dim=1, keepdim=True)], dim=1
        )
        return x * torch.sigmoid(self.conv(pooled))


class CBAM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class ResnetBlock(nn.Module):
    def __init__(self, channels: int, attention: bool = False) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
        )
        self.attention = CBAM(channels) if attention else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attention(self.block(x))


class ResnetGenerator(nn.Module):
    """CycleGAN ResNet generator with optional LumiCycle bottleneck attention."""

    def __init__(self, base_channels: int = 64, blocks: int = 9, attention: bool = False) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, base_channels, 7, bias=False),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.residuals = nn.ModuleList(
            [ResnetBlock(base_channels * 4, attention=attention) for _ in range(blocks)]
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 4,
                base_channels * 2,
                3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 2,
                base_channels,
                3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(base_channels, 3, 7), nn.Tanh())

    def encode_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features: list[torch.Tensor] = []
        x = self.stem(x)
        features.append(x)
        x = self.down1(x)
        features.append(x)
        x = self.down2(x)
        features.append(x)
        for index, block in enumerate(self.residuals):
            x = block(x)
            if index in {0, len(self.residuals) // 2, len(self.residuals) - 1}:
                features.append(x)
        return x, features

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        x, features = self.encode_features(x)
        x = self.up1(x)
        x = self.up2(x)
        output = self.head(x)
        return (output, features) if return_features else output


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        use_spectral_norm: bool = False,
        input_channels: int = 3,
    ) -> None:
        super().__init__()

        def conv(*args: int, **kwargs: int | bool) -> nn.Module:
            layer = nn.Conv2d(*args, **kwargs)
            return spectral_norm(layer) if use_spectral_norm else layer

        layers: list[nn.Module] = [
            conv(input_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        channels = base_channels
        for multiplier, stride in ((2, 2), (4, 2), (8, 1)):
            next_channels = base_channels * multiplier
            layers.extend(
                [
                    conv(channels, next_channels, 4, stride, 1, bias=False),
                    nn.InstanceNorm2d(next_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels = next_channels
        layers.append(conv(channels, 1, 4, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 64, scales: int = 2) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PatchDiscriminator(base_channels, use_spectral_norm=True) for _ in range(scales)]
        )
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for discriminator in self.discriminators:
            outputs.append(discriminator(x))
            x = self.downsample(x)
        return outputs


class GlobalDiscriminator(nn.Module):
    """Whole-frame critic for coherent illumination, complementing PatchGAN."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        channels = [3, base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        layers: list[nn.Module] = []
        for index, (input_channels, output_channels) in enumerate(pairwise(channels)):
            layers.append(
                spectral_norm(
                    nn.Conv2d(input_channels, output_channels, 4, stride=2, padding=1)
                )
            )
            if index:
                layers.append(nn.InstanceNorm2d(output_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.features = nn.Sequential(*layers)
        self.head = spectral_norm(nn.Conv2d(channels[-1], 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(nn.functional.adaptive_avg_pool2d(self.features(x), 1))


class LocalGlobalDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 64, scales: int = 2) -> None:
        super().__init__()
        self.local = MultiScaleDiscriminator(base_channels, scales)
        self.global_critic = GlobalDiscriminator(base_channels)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [*self.local(x), self.global_critic(x)]


def haar_high_frequency(image: torch.Tensor) -> torch.Tensor:
    """Return differentiable LH/HL/HH Haar bands concatenated by channel."""
    if image.shape[-2] % 2 or image.shape[-1] % 2:
        image = F.pad(image, (0, image.shape[-1] % 2, 0, image.shape[-2] % 2), mode="reflect")
    top_left = image[:, :, 0::2, 0::2]
    top_right = image[:, :, 0::2, 1::2]
    bottom_left = image[:, :, 1::2, 0::2]
    bottom_right = image[:, :, 1::2, 1::2]
    horizontal = (-top_left + top_right - bottom_left + bottom_right) * 0.5
    vertical = (-top_left - top_right + bottom_left + bottom_right) * 0.5
    diagonal = (top_left - top_right - bottom_left + bottom_right) * 0.5
    return torch.cat((horizontal, vertical, diagonal), dim=1)


class HaarWaveletDiscriminator(nn.Module):
    """Small critic that sees only target-domain high-frequency bands."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.critic = PatchDiscriminator(
            base_channels=base_channels,
            use_spectral_norm=True,
            input_channels=9,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.critic(haar_high_frequency(image))


class DetailResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class LaplacianDetailRefiner(nn.Module):
    def __init__(self, channels: int = 32, blocks: int = 3) -> None:
        super().__init__()
        # source band + coarse band + translated low context + source Sobel x/y
        self.stem = nn.Sequential(nn.Conv2d(11, channels, 3, padding=1), nn.SiLU(inplace=True))
        self.blocks = nn.Sequential(*[DetailResidualBlock(channels) for _ in range(blocks)])
        self.head = nn.Conv2d(channels, 4, 3, padding=1)

    def reset_identity(self) -> None:
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        source_band: torch.Tensor,
        coarse_band: torch.Tensor,
        translated_low: torch.Tensor,
        source_gradients: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (source_band, coarse_band, translated_low, source_gradients), dim=1
        )
        prediction = self.head(self.blocks(self.stem(features)))
        transfer = torch.tanh(prediction[:, :1])
        residual = 0.1 * torch.tanh(prediction[:, 1:])
        return coarse_band + transfer * (source_band - coarse_band) + residual


class LaplacianRefinementGenerator(nn.Module):
    """V2-compatible coarse generator plus zero-initialized detail refinement."""

    def __init__(
        self,
        base_channels: int = 64,
        blocks: int = 9,
        attention: bool = True,
        detail_channels: int = 32,
        detail_blocks: int = 3,
        pyramid_levels: int = 2,
    ) -> None:
        super().__init__()
        self.base = ResnetGenerator(base_channels, blocks, attention)
        self.pyramid_levels = pyramid_levels
        kernel_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        kernel = torch.outer(kernel_1d, kernel_1d)
        self.register_buffer("gaussian_kernel", (kernel / kernel.sum()).view(1, 1, 5, 5))
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(
                1, 1, 3, 3
            ),
        )
        self.register_buffer("sobel_y", self.sobel_x.transpose(2, 3).contiguous())
        self.refiners = nn.ModuleList(
            [LaplacianDetailRefiner(detail_channels, detail_blocks) for _ in range(pyramid_levels)]
        )

    def reset_refinement(self) -> None:
        for refiner in self.refiners:
            refiner.reset_identity()

    def _blur_down(self, image: torch.Tensor) -> torch.Tensor:
        kernel = self.gaussian_kernel.expand(image.shape[1], 1, 5, 5).to(image.dtype)
        return F.conv2d(F.pad(image, (2, 2, 2, 2), mode="reflect"), kernel, stride=2, groups=3)

    def _decompose(
        self, image: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        bands: list[torch.Tensor] = []
        levels: list[torch.Tensor] = []
        current = image
        for _ in range(self.pyramid_levels):
            low = self._blur_down(current)
            upsampled = F.interpolate(low, size=current.shape[-2:], mode="bilinear", align_corners=False)
            bands.append(current - upsampled)
            levels.append(current)
            current = low
        return bands, current, levels

    def _gradients(self, image: torch.Tensor) -> torch.Tensor:
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        return torch.cat(
            (
                F.conv2d(gray, self.sobel_x.to(gray.dtype), padding=1),
                F.conv2d(gray, self.sobel_y.to(gray.dtype), padding=1),
            ),
            dim=1,
        )

    def encode_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return self.base.encode_features(x)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_details: bool = False,
    ):
        coarse_result = self.base(x, return_features=return_features)
        if return_features:
            coarse, features = coarse_result
        else:
            coarse = coarse_result
            features = []
        source_bands, _source_low, source_levels = self._decompose(x)
        coarse_bands, translated_low, _coarse_levels = self._decompose(coarse)
        current = translated_low
        for level in reversed(range(self.pyramid_levels)):
            context = F.interpolate(
                current, size=coarse_bands[level].shape[-2:], mode="bilinear", align_corners=False
            )
            gradients = self._gradients(source_levels[level])
            refined_band = self.refiners[level](
                source_bands[level], coarse_bands[level], context, gradients
            )
            current = context + refined_band
        output = current.clamp(-1, 1)
        if return_features and return_details:
            return output, features, {"coarse": coarse}
        if return_features:
            return output, features
        if return_details:
            return output, {"coarse": coarse}
        return output


def init_weights(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if hasattr(module, "weight") and ("Conv" in classname or "Linear" in classname):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        bias = getattr(module, "bias", None)
        if bias is not None:
            nn.init.constant_(bias.data, 0.0)


def build_models(config: dict) -> dict[str, nn.Module]:
    model_config = config["model"]
    base = int(model_config.get("base_channels", 64))
    blocks = int(model_config.get("generator_blocks", 9))
    attention = bool(model_config.get("attention", False))
    multiscale = bool(model_config.get("multiscale_discriminator", False))
    global_discriminator = bool(model_config.get("global_discriminator", False))
    detail_refinement = bool(model_config.get("detail_refinement", False))
    frequency_discriminator = bool(model_config.get("frequency_discriminator", False))

    generator_class = LaplacianRefinementGenerator if detail_refinement else ResnetGenerator
    generator_kwargs = (
        {
            "base_channels": base,
            "blocks": blocks,
            "attention": attention,
            "detail_channels": int(model_config.get("detail_channels", 32)),
            "detail_blocks": int(model_config.get("detail_blocks", 3)),
            "pyramid_levels": int(model_config.get("pyramid_levels", 2)),
        }
        if detail_refinement
        else {"base_channels": base, "blocks": blocks, "attention": attention}
    )
    generator_day_to_night = generator_class(**generator_kwargs)
    generator_night_to_day = generator_class(**generator_kwargs)
    if global_discriminator:
        discriminator_class = LocalGlobalDiscriminator
    else:
        discriminator_class = MultiScaleDiscriminator if multiscale else PatchDiscriminator
    discriminator_day = discriminator_class(base)
    discriminator_night = discriminator_class(base)
    models = {
        "G_day_night": generator_day_to_night,
        "G_night_day": generator_night_to_day,
        "D_day": discriminator_day,
        "D_night": discriminator_night,
    }
    if frequency_discriminator:
        frequency_base = int(model_config.get("frequency_base_channels", 32))
        models["D_day_hf"] = HaarWaveletDiscriminator(frequency_base)
        models["D_night_hf"] = HaarWaveletDiscriminator(frequency_base)
    for model in models.values():
        model.apply(init_weights)
    if detail_refinement:
        generator_day_to_night.reset_refinement()
        generator_night_to_day.reset_refinement()
    return models


def discriminator_outputs(output: torch.Tensor | Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    return output if isinstance(output, (list, tuple)) else [output]
