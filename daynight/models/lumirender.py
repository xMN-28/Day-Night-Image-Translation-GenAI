from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    image = image.clamp(0, 1)
    return torch.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055).pow(2.4),
    )


def linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    image = image.clamp_min(0)
    return torch.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * image.pow(1 / 2.4) - 0.055,
    ).clamp(0, 1)


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, math.ceil(3 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel_1d = torch.exp(-coordinates.square() / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    channels = image.shape[1]
    horizontal = kernel_1d.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel_1d.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    mode = "reflect" if min(image.shape[-2:]) > radius else "replicate"
    image = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode=mode), horizontal, groups=channels)
    return F.conv2d(F.pad(image, (0, 0, radius, radius), mode=mode), vertical, groups=channels)


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(min(8, output_channels), output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(min(8, output_channels), output_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = (
            nn.Conv2d(input_channels, output_channels, 1, stride=stride)
            if input_channels != output_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image) + self.skip(image)


class SceneFactorizer(nn.Module):
    """Predict intrinsic, geometric, material and semantic scene factors."""

    semantic_names = ("sky", "road", "vehicle", "glass", "building", "emitter")

    def __init__(self, base_channels: int = 32, activation_checkpointing: bool = True) -> None:
        super().__init__()
        base = base_channels
        self.encoder1 = ConvBlock(3, base)
        self.encoder2 = ConvBlock(base, base * 2, stride=2)
        self.encoder3 = ConvBlock(base * 2, base * 4, stride=2)
        self.encoder4 = ConvBlock(base * 4, base * 8, stride=2)
        self.bottleneck = ConvBlock(base * 8, base * 8)
        self.decoder3 = ConvBlock(base * 8 + base * 4, base * 4)
        self.decoder2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.decoder1 = ConvBlock(base * 2 + base, base)
        # reflectance 3, day illumination 3, depth 1, normals 3,
        # roughness 1, wetness 1, semantic masks 6
        self.head = nn.Conv2d(base, 18, 1)
        self.activation_checkpointing = activation_checkpointing

    def _run(self, module: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        if self.training and self.activation_checkpointing and inputs.requires_grad:
            return checkpoint(module, inputs, use_reentrant=False)
        return module(inputs)

    def forward(self, linear_image: torch.Tensor) -> dict[str, torch.Tensor]:
        level1 = self._run(self.encoder1, linear_image)
        level2 = self._run(self.encoder2, level1)
        level3 = self._run(self.encoder3, level2)
        level4 = self._run(self.encoder4, level3)
        bottleneck = self._run(self.bottleneck, level4)
        decoded = F.interpolate(bottleneck, level3.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self._run(self.decoder3, torch.cat((decoded, level3), dim=1))
        decoded = F.interpolate(decoded, level2.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self._run(self.decoder2, torch.cat((decoded, level2), dim=1))
        decoded = F.interpolate(decoded, level1.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self._run(self.decoder1, torch.cat((decoded, level1), dim=1))
        prediction = self.head(decoded)
        reflectance = torch.sigmoid(prediction[:, 0:3])
        day_illumination = F.softplus(prediction[:, 3:6]) + 0.05
        depth = torch.sigmoid(prediction[:, 6:7])
        normals = F.normalize(prediction[:, 7:10], dim=1, eps=1e-6)
        roughness = torch.sigmoid(prediction[:, 10:11])
        wetness = torch.sigmoid(prediction[:, 11:12])
        semantic = torch.sigmoid(prediction[:, 12:18])
        return {
            "reflectance": reflectance,
            "day_illumination": day_illumination,
            "depth": depth,
            "normals": normals,
            "roughness": roughness,
            "wetness": wetness,
            "semantic": semantic,
            "features": decoded,
            "bottleneck": bottleneck,
        }


@dataclass
class GaussianLights:
    centers: torch.Tensor
    scales: torch.Tensor
    angles: torch.Tensor
    colors: torch.Tensor
    intensities: torch.Tensor
    ambient: torch.Tensor
    horizon: torch.Tensor
    exposure: torch.Tensor
    white_balance: torch.Tensor
    noise: torch.Tensor


class NightLightComposer(nn.Module):
    def __init__(self, feature_channels: int, lights: int = 8, latent_dim: int = 16) -> None:
        super().__init__()
        self.lights = lights
        self.latent_dim = latent_dim
        # Per light: center 2, scale 2, angle 1, RGB 3, intensity 1.
        light_parameters = lights * 9
        # Ambient RGB, horizon RGB, exposure, WB RGB, shot/read noise.
        global_parameters = 3 + 3 + 1 + 3 + 2
        self.project = nn.Sequential(
            nn.Linear(feature_channels + latent_dim, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, light_parameters + global_parameters),
        )

    def forward(self, features: torch.Tensor, latent: torch.Tensor) -> GaussianLights:
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        raw = self.project(torch.cat((pooled, latent), dim=1))
        light_raw = raw[:, : self.lights * 9].view(-1, self.lights, 9)
        global_raw = raw[:, self.lights * 9 :]
        centers = torch.sigmoid(light_raw[..., 0:2]) * 1.3 - 0.15
        scales = 0.06 + 0.44 * torch.sigmoid(light_raw[..., 2:4])
        angles = math.pi * torch.tanh(light_raw[..., 4:5])
        colors = 0.25 + 1.75 * torch.sigmoid(light_raw[..., 5:8])
        intensities = 0.02 + 0.45 * torch.sigmoid(light_raw[..., 8:9])
        ambient = 0.015 + 0.16 * torch.sigmoid(global_raw[:, 0:3])
        horizon = 0.01 + 0.12 * torch.sigmoid(global_raw[:, 3:6])
        exposure = 0.65 + 0.7 * torch.sigmoid(global_raw[:, 6:7])
        white_balance = 0.75 + 0.5 * torch.sigmoid(global_raw[:, 7:10])
        noise = torch.cat(
            (
                0.001 + 0.012 * torch.sigmoid(global_raw[:, 10:11]),
                0.0005 + 0.006 * torch.sigmoid(global_raw[:, 11:12]),
            ),
            dim=1,
        )
        return GaussianLights(
            centers, scales, angles, colors, intensities, ambient, horizon,
            exposure, white_balance, noise
        )


class BoundedCorrection(nn.Module):
    def __init__(self, feature_channels: int, maximum: float = 0.03) -> None:
        super().__init__()
        self.maximum = maximum
        self.net = nn.Sequential(
            nn.Conv2d(feature_channels + 6, 32, 3, padding=1),
            nn.SiLU(inplace=True),
            ConvBlock(32, 32),
            nn.Conv2d(32, 3, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, source_linear: torch.Tensor, rendered_linear: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        if features.shape[-2:] != source_linear.shape[-2:]:
            features = F.interpolate(features, source_linear.shape[-2:], mode="bilinear", align_corners=False)
        raw = torch.tanh(self.net(torch.cat((source_linear, rendered_linear, features), dim=1)))
        high_frequency = raw - gaussian_blur(raw, sigma=4.0)
        return high_frequency.clamp(-1, 1) * self.maximum


class LumiRender(nn.Module):
    """Physics-guided single-image daytime-to-nighttime renderer."""

    def __init__(
        self,
        base_channels: int = 32,
        lights: int = 8,
        latent_dim: int = 16,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.factorizer = SceneFactorizer(base_channels, activation_checkpointing)
        self.light_composer = NightLightComposer(base_channels * 8, lights, latent_dim)
        self.correction = BoundedCorrection(base_channels)

    @staticmethod
    def _meshgrid(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = image.shape[-2:]
        y = torch.linspace(0, 1, height, device=image.device, dtype=image.dtype)
        x = torch.linspace(0, 1, width, device=image.device, dtype=image.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return grid_x.view(1, 1, height, width), grid_y.view(1, 1, height, width)

    def _render_lights(
        self, image: torch.Tensor, lights: GaussianLights
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid_x, grid_y = self._meshgrid(image)
        x = grid_x - lights.centers[..., 0:1, None]
        y = grid_y - lights.centers[..., 1:2, None]
        cosine = torch.cos(lights.angles).unsqueeze(-1)
        sine = torch.sin(lights.angles).unsqueeze(-1)
        rotated_x = cosine * x + sine * y
        rotated_y = -sine * x + cosine * y
        scale_x = lights.scales[..., 0:1, None]
        scale_y = lights.scales[..., 1:2, None]
        gaussian = torch.exp(-0.5 * ((rotated_x / scale_x) ** 2 + (rotated_y / scale_y) ** 2))
        gaussian = gaussian.squeeze(2)
        colored = gaussian.unsqueeze(2) * lights.colors[..., None, None]
        colored = colored * lights.intensities[..., None, None]
        return colored.sum(dim=1), gaussian

    def _restrict_light_centers(
        self, lights: GaussianLights, emitter_mask: torch.Tensor
    ) -> GaussianLights:
        """Anchor on-frame lights to likely emitters while permitting off-frame illumination."""
        batch, count = lights.centers.shape[:2]
        coarse = F.adaptive_avg_pool2d(emitter_mask, (8, 8)).flatten(1)
        indices = coarse.topk(k=min(count, coarse.shape[1]), dim=1).indices
        if indices.shape[1] < count:
            indices = indices.repeat(1, math.ceil(count / indices.shape[1]))[:, :count]
        candidate_y = (indices // 8).to(lights.centers.dtype).add(0.5).div(8)
        candidate_x = (indices % 8).to(lights.centers.dtype).add(0.5).div(8)
        candidates = torch.stack((candidate_x, candidate_y), dim=-1)
        outside = ((lights.centers < 0) | (lights.centers > 1)).any(dim=-1, keepdim=True)
        offset = 0.08 * torch.tanh((lights.centers - 0.5) * 4)
        centers = torch.where(outside, lights.centers, (candidates + offset).clamp(0, 1))
        assert centers.shape == (batch, count, 2)
        return GaussianLights(
            centers,
            lights.scales,
            lights.angles,
            lights.colors,
            lights.intensities,
            lights.ambient,
            lights.horizon,
            lights.exposure,
            lights.white_balance,
            lights.noise,
        )

    def _render_reflections(
        self,
        emitter: torch.Tensor,
        road: torch.Tensor,
        glass: torch.Tensor,
        wetness: torch.Tensor,
        roughness: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        reflected = torch.flip(emitter, dims=(-2,))
        reflected = gaussian_blur(reflected, sigma=3.0)
        vertical_streak = F.avg_pool2d(reflected, (15, 3), stride=1, padding=(7, 1))
        road_reflection = (0.55 * reflected + 0.45 * vertical_streak)
        road_reflection = road_reflection * road * (0.15 + 0.85 * wetness) * (0.5 + 0.5 * depth)
        glass_reflection = gaussian_blur(emitter, sigma=1.5) * glass * (1 - roughness)
        return 0.45 * road_reflection + 0.2 * glass_reflection

    def _camera(
        self,
        radiance: torch.Tensor,
        lights: GaussianLights,
        intensity: torch.Tensor,
        noise_sample: torch.Tensor,
    ) -> torch.Tensor:
        radiance = radiance * lights.exposure[..., None, None] * intensity
        radiance = radiance * lights.white_balance[..., None, None]
        grid_x, grid_y = self._meshgrid(radiance)
        radius = ((grid_x - 0.5).square() + (grid_y - 0.5).square()).clamp(0, 0.5)
        radiance = radiance * (1 - 0.28 * radius)
        shot = lights.noise[:, 0:1, None, None] * radiance.clamp_min(0).sqrt()
        read = lights.noise[:, 1:2, None, None]
        radiance = radiance + noise_sample * (shot + read)
        tone_mapped = radiance.clamp_min(0) / (1 + radiance.clamp_min(0))
        return linear_to_srgb(tone_mapped)

    def _latent(
        self, image: torch.Tensor, seed: int | None, latent: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent is not None:
            noise = torch.zeros_like(image[:, :1])
            return latent.to(device=image.device, dtype=image.dtype), noise
        generator = torch.Generator(device=image.device)
        generator.manual_seed(int(seed if seed is not None else 0))
        code = torch.randn(
            image.shape[0], self.latent_dim, generator=generator, device=image.device, dtype=image.dtype
        )
        noise = torch.randn(
            image.shape[0], 1, *image.shape[-2:], generator=generator,
            device=image.device, dtype=image.dtype
        )
        return code, noise

    def forward(
        self,
        image: torch.Tensor,
        *,
        night_intensity: float | torch.Tensor = 1.0,
        seed: int | None = 0,
        surface_wetness: float | torch.Tensor | None = None,
        latent: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source_srgb = ((image + 1) / 2).clamp(0, 1)
        source_linear = srgb_to_linear(source_srgb)
        factors = self.factorizer(source_linear)
        latent_code, noise_sample = self._latent(source_linear, seed, latent)
        lights = self.light_composer(factors["bottleneck"], latent_code)

        semantic = factors["semantic"]
        sky, road, _vehicle, glass, _building, emitter_mask = semantic.split(1, dim=1)
        lights = self._restrict_light_centers(lights, emitter_mask)
        gaussian_light, gaussian_masks = self._render_lights(source_linear, lights)
        _grid_x, grid_y = self._meshgrid(source_linear)
        ambient = lights.ambient[..., None, None]
        horizon = lights.horizon[..., None, None] * (1 - grid_y)
        sky_field = ambient + horizon * (0.35 + 0.65 * sky)
        emitter = gaussian_light * (0.2 + 0.8 * emitter_mask)
        normal_facing = factors["normals"][:, 2:3].abs().clamp(0.15, 1)
        diffuse = factors["reflectance"] * (sky_field + 0.4 * gaussian_light) * normal_facing
        wetness = factors["wetness"]
        if surface_wetness is not None:
            wetness = torch.as_tensor(
                surface_wetness, device=image.device, dtype=image.dtype
            ).view(-1, 1, 1, 1).expand_as(wetness)
        reflections = self._render_reflections(
            emitter, road, glass, wetness, factors["roughness"], factors["depth"]
        )
        specular = gaussian_light * (1 - factors["roughness"]) * 0.12
        bloom = 0.16 * gaussian_blur(emitter + specular, 2.0)
        bloom = bloom + 0.08 * gaussian_blur(emitter, 5.0)
        radiance = diffuse + emitter + reflections + specular + bloom
        intensity = torch.as_tensor(
            night_intensity, device=image.device, dtype=image.dtype
        ).view(-1, 1, 1, 1)
        rendered_srgb = self._camera(radiance, lights, intensity, noise_sample)
        rendered_linear = srgb_to_linear(rendered_srgb)
        correction = self.correction(source_linear, rendered_linear, factors["features"])
        output_srgb = linear_to_srgb((rendered_linear + correction).clamp_min(0))
        output = output_srgb * 2 - 1

        if not return_details:
            return output
        day_reconstruction = (factors["reflectance"] * factors["day_illumination"]).clamp(0, 1)
        details = {
            **{key: value for key, value in factors.items() if key not in {"features", "bottleneck"}},
            "day_reconstruction": day_reconstruction,
            "gaussian_light": gaussian_light,
            "gaussian_masks": gaussian_masks,
            "light_centers": lights.centers,
            "emitter": emitter,
            "reflections": reflections,
            "bloom": bloom,
            "rendered_srgb": rendered_srgb,
            "correction": correction,
            "source_linear": source_linear,
            "output_linear": srgb_to_linear(output_srgb),
        }
        return output, details
