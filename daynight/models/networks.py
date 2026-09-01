from __future__ import annotations

from collections.abc import Sequence

import torch
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
    def __init__(self, base_channels: int = 64, use_spectral_norm: bool = False) -> None:
        super().__init__()

        def conv(*args: int, **kwargs: int | bool) -> nn.Module:
            layer = nn.Conv2d(*args, **kwargs)
            return spectral_norm(layer) if use_spectral_norm else layer

        layers: list[nn.Module] = [
            conv(3, base_channels, 4, 2, 1),
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

    generator_day_to_night = ResnetGenerator(base, blocks, attention)
    generator_night_to_day = ResnetGenerator(base, blocks, attention)
    discriminator_class = MultiScaleDiscriminator if multiscale else PatchDiscriminator
    discriminator_day = discriminator_class(base)
    discriminator_night = discriminator_class(base)
    models = {
        "G_day_night": generator_day_to_night,
        "G_night_day": generator_night_to_day,
        "D_day": discriminator_day,
        "D_night": discriminator_night,
    }
    for model in models.values():
        model.apply(init_weights)
    return models


def discriminator_outputs(output: torch.Tensor | Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    return output if isinstance(output, (list, tuple)) else [output]
