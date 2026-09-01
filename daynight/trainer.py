from __future__ import annotations

import math
import hashlib
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .checkpoint import CheckpointManager, build_checkpoint, restore_rng_state
from .config import config_hash
from .data import UnpairedDayNightDataset
from .losses import (
    DinoSemanticLoss,
    SobelEdgeLoss,
    color_statistics_loss,
    diff_augment,
    discriminator_accuracy,
    lsgan_discriminator_loss,
    lsgan_generator_loss,
    output_variance,
    patch_nce_loss,
    refinement_residual_loss,
    regional_illumination_loss,
    spatial_self_similarity_loss,
    wavelet_structure_loss,
)
from .models import build_models
from .replay import ImageReplayBuffer
from .utils import atomic_json_dump, seed_everything


class ExponentialMovingAverage:
    def __init__(self, module: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {name: value.detach().clone() for name, value in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for name, value in module.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].lerp_(value.detach(), 1 - self.decay)
            else:
                self.shadow[name].copy_(value)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = state["shadow"]

    def copy_to(self, module: nn.Module) -> dict[str, torch.Tensor]:
        original = {name: value.detach().clone() for name, value in module.state_dict().items()}
        module.load_state_dict(self.shadow)
        return original


def set_requires_grad(modules: list[nn.Module], enabled: bool) -> None:
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)


def make_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]):
    warmup = int(config["train"].get("warmup_steps", 0))
    decay_start = int(config["train"].get("decay_start_step", config["train"]["max_steps"] // 2))
    maximum = int(config["train"]["max_steps"])

    def schedule(step: int) -> float:
        if warmup and step < warmup:
            return max((step + 1) / warmup, 1e-3)
        if step <= decay_start:
            return 1.0
        return max(0.0, 1.0 - (step - decay_start) / max(1, maximum - decay_start))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


class Trainer:
    def __init__(
        self,
        config: dict[str, Any],
        resume: str | None = "auto",
        init_from: str | Path | None = None,
    ) -> None:
        self.config = config
        seed_everything(int(config["experiment"].get("seed", 42)))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(config["experiment"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(config, self.output_dir / "resolved_config.json")
        self.writer = SummaryWriter(self.output_dir / "tensorboard")

        self.models = build_models(config)
        for model in self.models.values():
            model.to(self.device)
        self.generators = [self.models["G_day_night"], self.models["G_night_day"]]
        self.spatial_discriminators = [self.models["D_day"], self.models["D_night"]]
        self.frequency_discriminators = [
            self.models[name] for name in ("D_day_hf", "D_night_hf") if name in self.models
        ]
        self.discriminators = [*self.spatial_discriminators, *self.frequency_discriminators]

        train_config = config["train"]
        generator_groups = self._generator_parameter_groups(train_config)
        discriminator_groups = self._discriminator_parameter_groups(train_config)
        self.optimizers = {
            "G": torch.optim.Adam(
                generator_groups,
                lr=float(train_config["generator_lr"]),
                betas=(float(train_config["beta1"]), float(train_config["beta2"])),
            ),
            "D": torch.optim.Adam(
                discriminator_groups,
                lr=float(train_config["discriminator_lr"]),
                betas=(float(train_config["beta1"]), float(train_config["beta2"])),
            ),
        }
        self.schedulers = {
            name: make_scheduler(optimizer, config) for name, optimizer in self.optimizers.items()
        }
        self.ema = {
            "G_day_night": ExponentialMovingAverage(
                self.models["G_day_night"], float(train_config["ema_decay"])
            ),
            "G_night_day": ExponentialMovingAverage(
                self.models["G_night_day"], float(train_config["ema_decay"])
            ),
        }
        replay_size = int(train_config.get("replay_size", 50))
        self.replay = {
            "day": ImageReplayBuffer(replay_size),
            "night": ImageReplayBuffer(replay_size),
        }
        self.edge_loss = SobelEdgeLoss().to(self.device)
        self.semantic_loss = (
            DinoSemanticLoss() if float(config["loss"].get("semantic", 0)) > 0 else None
        )

        data_config = config["data"]
        self.train_dataset = UnpairedDayNightDataset(
            data_config["root"],
            "train",
            int(data_config["image_size"]),
            int(data_config["resize_size"]),
        )
        self.val_dataset = UnpairedDayNightDataset(
            data_config["root"],
            "val",
            int(data_config["image_size"]),
            int(data_config["resize_size"]),
        )
        data_limit = data_config.get("limit")
        if data_limit is not None:
            limit = int(data_limit)
            self.train_dataset.day = self.train_dataset.day[:limit]
            self.train_dataset.night = self.train_dataset.night[:limit]
        workers = int(data_config.get("num_workers", 4))
        self.workers = workers
        self.hard_detail_weights: list[float] | None = None
        hard_fraction = float(data_config.get("hard_detail_fraction", 0.0))
        if hard_fraction > 0:
            self.hard_detail_weights = self.train_dataset.configure_hard_detail_sampling(
                hard_fraction, Path(data_config["root"]) / "detail_scores.json"
            )
        self.train_loader = self._make_train_loader()
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0)
        self.checkpoints = CheckpointManager(
            self.output_dir, int(train_config.get("keep_checkpoints", 4))
        )
        self.step = 0
        self.training_state = {
            "best_score": math.inf,
            "bad_validations": 0,
            "plateau_reductions": 0,
            "recoveries": 0,
        }
        if init_from is not None:
            self.initialize_weights(init_from)
        else:
            checkpoint_path = self.checkpoints.resolve(resume)
            if checkpoint_path is not None:
                self.load_checkpoint(checkpoint_path)
        self._configure_stage(self.step, force=True)

    def _generator_parameter_groups(self, train_config: dict[str, Any]) -> list[dict[str, Any]]:
        if not bool(self.config["model"].get("detail_refinement", False)):
            return [{
                "params": [p for model in self.generators for p in model.parameters()],
                "name": "generator",
            }]
        frozen, trainable, refiners = [], [], []
        for generator in self.generators:
            frozen.extend(generator.base.stem.parameters())
            frozen.extend(generator.base.down1.parameters())
            frozen.extend(generator.base.down2.parameters())
            trainable.extend(generator.base.residuals.parameters())
            trainable.extend(generator.base.up1.parameters())
            trainable.extend(generator.base.up2.parameters())
            trainable.extend(generator.base.head.parameters())
            refiners.extend(generator.refiners.parameters())
        for parameter in frozen:
            parameter.requires_grad_(False)
        return [
            {"params": trainable, "name": "base", "lr": float(train_config["generator_lr"])},
            {"params": refiners, "name": "refiner", "lr": float(train_config["generator_lr"])},
        ]

    def _discriminator_parameter_groups(
        self, train_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        groups = [{
            "params": [p for model in self.spatial_discriminators for p in model.parameters()],
            "name": "spatial_d",
            "lr": float(train_config["discriminator_lr"]),
        }]
        if self.frequency_discriminators:
            groups.append({
                "params": [p for model in self.frequency_discriminators for p in model.parameters()],
                "name": "hf_d",
                "lr": float(train_config["discriminator_lr"]),
            })
        return groups

    def _make_train_loader(self) -> DataLoader:
        data_config = self.config["data"]
        sampler = None
        if self.hard_detail_weights is not None:
            sampler = WeightedRandomSampler(
                self.hard_detail_weights, len(self.train_dataset), replacement=True
            )
        return DataLoader(
            self.train_dataset,
            batch_size=int(data_config.get("batch_size", 1)),
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=False,
        )

    def _configure_stage(self, step: int, force: bool = False) -> bool:
        stages = self.config["train"].get("stages", [])
        if not stages:
            return False
        index = next(
            (i for i, stage in enumerate(stages) if step < int(stage["until_step"])),
            len(stages) - 1,
        )
        if not force and self.training_state.get("current_stage") == index:
            return False
        stage = stages[index]
        rates = stage["learning_rates"]
        for optimizer, prefix in ((self.optimizers["G"], "G"), (self.optimizers["D"], "D")):
            scheduler = self.schedulers[prefix]
            for group_index, group in enumerate(optimizer.param_groups):
                rate = float(rates.get(group.get("name"), group["lr"]))
                group["lr"] = rate
                scheduler.base_lrs[group_index] = rate
                enabled = rate > 0
                for parameter in group["params"]:
                    parameter.requires_grad_(enabled)
        image_size = int(stage["image_size"])
        resize_size = int(stage["resize_size"])
        self.train_dataset.image_size = image_size
        self.train_dataset.resize_size = resize_size
        self.val_dataset.image_size = image_size
        self.val_dataset.resize_size = resize_size
        self.train_loader = self._make_train_loader()
        self.training_state["current_stage"] = index
        self.training_state["stage_name"] = str(stage["name"])
        print(f"Training stage: {stage['name']} ({image_size}px, step {step})")
        return True

    @property
    def autocast_context(self):
        amp = str(self.config["train"].get("amp", "none")).lower()
        if self.device.type != "cuda" or amp == "none":
            return nullcontext()
        dtype = torch.bfloat16 if amp == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("config_hash") != config_hash(self.config):
            print(
                "Warning: checkpoint configuration differs from the current config; continuing intentionally."
            )
        for name, state in checkpoint["models"].items():
            self.models[name].load_state_dict(state)
        for name, state in checkpoint["ema"].items():
            self.ema[name].load_state_dict(state)
        for name, state in checkpoint["optimizers"].items():
            self.optimizers[name].load_state_dict(state)
        for name, state in checkpoint["schedulers"].items():
            self.schedulers[name].load_state_dict(state)
        self.replay["day"].load_state_dict(checkpoint["replay"]["day"])
        self.replay["night"].load_state_dict(checkpoint["replay"]["night"])
        self.training_state.update(checkpoint.get("training_state", {}))
        self.step = int(checkpoint["step"])
        restore_rng_state(checkpoint["rng"])
        print(f"Resumed from {path} at step {self.step}")

    def initialize_weights(self, path: str | Path) -> None:
        """Load learned weights while keeping fresh optimizers and counters."""
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        scope = str(self.config["train"].get("init_scope", "all"))
        names = (
            {"G_day_night", "G_night_day"}
            if scope == "generators"
            else set(checkpoint["models"])
        )
        for name, state in checkpoint["models"].items():
            if name not in names:
                continue
            target = self.models.get(name)
            if target is None:
                continue
            if hasattr(target, "base") and not any(key.startswith("base.") for key in state):
                target.base.load_state_dict(state)
            else:
                target.load_state_dict(state)
        for name, state in checkpoint.get("ema", {}).items():
            if name in self.ema:
                source_shadow = state["shadow"]
                target_shadow = self.ema[name].shadow
                if hasattr(self.models[name], "base") and not any(
                    key.startswith("base.") for key in source_shadow
                ):
                    for key, value in source_shadow.items():
                        target_shadow[f"base.{key}"] = value.detach().clone()
                    self.ema[name].decay = float(state["decay"])
                else:
                    self.ema[name].load_state_dict(state)
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        self.training_state["parent_checkpoint"] = {
            "path": str(checkpoint_path.resolve()),
            "sha256": digest.hexdigest(),
            "source_step": int(checkpoint.get("step", 0)),
        }
        print(
            f"Initialized model weights from {checkpoint_path} "
            f"(source step {int(checkpoint.get('step', 0))}, scope={scope}); "
            "optimizers start fresh."
        )

    def checkpoint_payload(self) -> dict[str, Any]:
        return build_checkpoint(
            step=self.step,
            config=self.config,
            models=self.models,
            ema={name: item.state_dict() for name, item in self.ema.items()},
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            replay={name: item.state_dict() for name, item in self.replay.items()},
            training_state=self.training_state,
        )

    def save_checkpoint(self, best: bool = False) -> Path:
        return self.checkpoints.save(self.step, self.checkpoint_payload(), best=best)

    def _generator_pass(
        self, day: torch.Tensor, night: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        weights = self.config["loss"]
        if bool(self.config["model"].get("detail_refinement", False)):
            fake_night, day_features, night_details = self.models["G_day_night"](
                day, return_features=True, return_details=True
            )
            fake_day, night_features, day_details = self.models["G_night_day"](
                night, return_features=True, return_details=True
            )
        else:
            fake_night, day_features = self.models["G_day_night"](day, return_features=True)
            fake_day, night_features = self.models["G_night_day"](night, return_features=True)
            night_details = {"coarse": fake_night}
            day_details = {"coarse": fake_day}
        reconstructed_day = self.models["G_night_day"](fake_night)
        reconstructed_night = self.models["G_day_night"](fake_day)
        identity_day = self.models["G_night_day"](day)
        identity_night = self.models["G_day_night"](night)

        components = {
            "gan": lsgan_generator_loss(self.models["D_night"](diff_augment(fake_night)))
            + lsgan_generator_loss(self.models["D_day"](diff_augment(fake_day))),
            "cycle": F.l1_loss(reconstructed_day, day) + F.l1_loss(reconstructed_night, night),
            "identity": F.l1_loss(identity_day, day) + F.l1_loss(identity_night, night),
            "edge": self.edge_loss(day, fake_night) + self.edge_loss(night, fake_day),
            "color": color_statistics_loss(fake_night, night)
            + color_statistics_loss(fake_day, day),
            "illumination": regional_illumination_loss(
                day, fake_night, night, "day_to_night"
            )
            + regional_illumination_loss(night, fake_day, day, "night_to_day"),
            "wavelet": wavelet_structure_loss(day, fake_night)
            + wavelet_structure_loss(night, fake_day),
            "self_similarity": spatial_self_similarity_loss(day, fake_night)
            + spatial_self_similarity_loss(night, fake_day),
            "residual": refinement_residual_loss(night_details["coarse"], fake_night)
            + refinement_residual_loss(day_details["coarse"], fake_day),
        }
        if self.frequency_discriminators:
            components["frequency_gan"] = lsgan_generator_loss(
                self.models["D_night_hf"](fake_night)
            ) + lsgan_generator_loss(self.models["D_day_hf"](fake_day))
        else:
            components["frequency_gan"] = day.new_zeros(())
        if float(weights.get("nce", 0)) > 0:
            _, translated_night_features = self.models["G_day_night"].encode_features(fake_night)
            _, translated_day_features = self.models["G_night_day"].encode_features(fake_day)
            components["nce"] = patch_nce_loss(
                day_features, translated_night_features
            ) + patch_nce_loss(night_features, translated_day_features)
        else:
            components["nce"] = day.new_zeros(())
        if self.semantic_loss is not None:
            components["semantic"] = self.semantic_loss(day, fake_night) + self.semantic_loss(
                night, fake_day
            )
        else:
            components["semantic"] = day.new_zeros(())
        total = sum(float(weights.get(name, 0)) * value for name, value in components.items())
        components["total"] = total
        images = {
            "day": day,
            "fake_night": fake_night,
            "reconstructed_day": reconstructed_day,
            "night": night,
            "fake_day": fake_day,
            "reconstructed_night": reconstructed_night,
            "coarse_night": night_details["coarse"],
            "coarse_day": day_details["coarse"],
        }
        return components, images

    def _discriminator_pass(self, images: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        fake_day = self.replay["day"].query(images["fake_day"])
        fake_night = self.replay["night"].query(images["fake_night"])
        real_day_output = self.models["D_day"](diff_augment(images["day"]))
        fake_day_output = self.models["D_day"](diff_augment(fake_day))
        real_night_output = self.models["D_night"](diff_augment(images["night"]))
        fake_night_output = self.models["D_night"](diff_augment(fake_night))
        day_loss = lsgan_discriminator_loss(real_day_output, fake_day_output)
        night_loss = lsgan_discriminator_loss(real_night_output, fake_night_output)
        frequency_loss = images["day"].new_zeros(())
        frequency_accuracy = images["day"].new_zeros(())
        if self.frequency_discriminators:
            real_day_hf = self.models["D_day_hf"](images["day"])
            fake_day_hf = self.models["D_day_hf"](fake_day)
            real_night_hf = self.models["D_night_hf"](images["night"])
            fake_night_hf = self.models["D_night_hf"](fake_night)
            frequency_loss = lsgan_discriminator_loss(
                real_day_hf, fake_day_hf
            ) + lsgan_discriminator_loss(real_night_hf, fake_night_hf)
            frequency_accuracy = 0.5 * (
                discriminator_accuracy(real_day_hf, fake_day_hf)
                + discriminator_accuracy(real_night_hf, fake_night_hf)
            )
        frequency_weight = float(self.config["loss"].get("frequency_discriminator", 1.0))
        spatial_accuracy = 0.5 * (
            discriminator_accuracy(real_day_output, fake_day_output)
            + discriminator_accuracy(real_night_output, fake_night_output)
        )
        return {
            "total": day_loss + night_loss + frequency_weight * frequency_loss,
            "day": day_loss,
            "night": night_loss,
            "frequency": frequency_loss,
            "accuracy": spatial_accuracy,
            "frequency_accuracy": frequency_accuracy,
        }

    def _grad_norm(self, modules: list[nn.Module]) -> float:
        norms = [
            parameter.grad.detach().float().norm()
            for module in modules
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        return float(torch.stack(norms).norm().item()) if norms else 0.0

    def _enable_discriminator_update_groups(self) -> None:
        for group in self.optimizers["D"].param_groups:
            enabled = float(group["lr"]) > 0
            for parameter in group["params"]:
                parameter.requires_grad_(enabled)

    def train_microstep(
        self, batch: dict[str, Any], should_step: bool
    ) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        day = batch["day"].to(self.device, non_blocking=True)
        night = batch["night"].to(self.device, non_blocking=True)
        accumulation = int(self.config["train"].get("gradient_accumulation", 1))
        discriminator_every = int(self.config["train"].get("discriminator_every", 1))
        update_discriminator = (self.step + 1) % max(1, discriminator_every) == 0

        set_requires_grad(self.discriminators, False)
        with self.autocast_context:
            generator_losses, images = self._generator_pass(day, night)
            (generator_losses["total"] / accumulation).backward()

        self._enable_discriminator_update_groups()
        with self.autocast_context:
            if update_discriminator:
                discriminator_losses = self._discriminator_pass(images)
                (discriminator_losses["total"] / accumulation).backward()
            else:
                with torch.no_grad():
                    discriminator_losses = self._discriminator_pass(images)

        metrics = {
            f"G/{name}": float(value.detach().float()) for name, value in generator_losses.items()
        }
        metrics.update(
            {
                f"D/{name}": float(value.detach().float())
                for name, value in discriminator_losses.items()
            }
        )
        metrics["health/output_variance"] = float(
            0.5 * (output_variance(images["fake_day"]) + output_variance(images["fake_night"]))
        )
        if should_step:
            max_norm = float(self.config["train"].get("grad_clip", 5.0))
            metrics["health/G_grad_norm"] = self._grad_norm(self.generators)
            metrics["health/D_grad_norm"] = self._grad_norm(self.discriminators)
            torch.nn.utils.clip_grad_norm_(
                [parameter for module in self.generators for parameter in module.parameters()],
                max_norm,
            )
            torch.nn.utils.clip_grad_norm_(
                [parameter for module in self.discriminators for parameter in module.parameters()],
                max_norm,
            )
            self.optimizers["G"].step()
            self.optimizers["G"].zero_grad(set_to_none=True)
            self.schedulers["G"].step()
            if update_discriminator:
                self.optimizers["D"].step()
                self.schedulers["D"].step()
            self.optimizers["D"].zero_grad(set_to_none=True)
            for name, ema in self.ema.items():
                ema.update(self.models[name])
        return metrics, images

    @torch.no_grad()
    def validate(self, batches: int = 32) -> dict[str, float]:
        originals = {}
        for name, ema in self.ema.items():
            originals[name] = ema.copy_to(self.models[name])
        for model in self.models.values():
            model.eval()
        totals = {
            "cycle": 0.0,
            "edge": 0.0,
            "color": 0.0,
            "illumination": 0.0,
            "variance": 0.0,
            "wavelet": 0.0,
            "self_similarity": 0.0,
        }
        count = 0
        try:
            for batch in self.val_loader:
                day = batch["day"].to(self.device)
                night = batch["night"].to(self.device)
                with self.autocast_context:
                    fake_night = self.models["G_day_night"](day)
                    fake_day = self.models["G_night_day"](night)
                    cycle = F.l1_loss(self.models["G_night_day"](fake_night), day) + F.l1_loss(
                        self.models["G_day_night"](fake_day), night
                    )
                    edge = self.edge_loss(day, fake_night) + self.edge_loss(night, fake_day)
                    color = color_statistics_loss(fake_night, night) + color_statistics_loss(
                        fake_day, day
                    )
                    illumination = regional_illumination_loss(
                        day, fake_night, night, "day_to_night"
                    ) + regional_illumination_loss(night, fake_day, day, "night_to_day")
                    wavelet = wavelet_structure_loss(day, fake_night) + wavelet_structure_loss(
                        night, fake_day
                    )
                    similarity = spatial_self_similarity_loss(
                        day, fake_night
                    ) + spatial_self_similarity_loss(night, fake_day)
                totals["cycle"] += float(cycle.float())
                totals["edge"] += float(edge.float())
                totals["color"] += float(color.float())
                totals["illumination"] += float(illumination.float())
                totals["wavelet"] += float(wavelet.float())
                totals["self_similarity"] += float(similarity.float())
                totals["variance"] += float(
                    0.5 * (output_variance(fake_night) + output_variance(fake_day))
                )
                count += 1
                if count >= batches:
                    break
        finally:
            for name, state in originals.items():
                self.models[name].load_state_dict(state)
            for model in self.models.values():
                model.train()
        metrics = {name: value / max(1, count) for name, value in totals.items()}
        color_weight = float(self.config["train"].get("validation_color_weight", 0.0))
        illumination_weight = float(
            self.config["train"].get("validation_illumination_weight", 0.0)
        )
        metrics["score"] = (
            metrics["cycle"]
            + 0.25 * metrics["edge"]
            + color_weight * metrics["color"]
            + illumination_weight * metrics["illumination"]
            + float(self.config["train"].get("validation_wavelet_weight", 0.1))
            * metrics["wavelet"]
            + float(self.config["train"].get("validation_similarity_weight", 0.25))
            * metrics["self_similarity"]
        )
        return metrics

    def _log_images(self, images: dict[str, torch.Tensor]) -> None:
        grid = torch.cat(
            [
                images["day"],
                images["fake_night"],
                images["reconstructed_day"],
                images["night"],
                images["fake_day"],
                images["reconstructed_night"],
            ],
            dim=0,
        )
        self.writer.add_images(
            "translations/day_fake_cycle_night_fake_cycle", (grid + 1) / 2, self.step
        )
        if "coarse_night" in images:
            detail_grid = torch.cat(
                [images["day"], images["coarse_night"], images["fake_night"]], dim=0
            )
            self.writer.add_images(
                "translations/detail_input_coarse_refined", (detail_grid + 1) / 2, self.step
            )

    def _reduce_for_plateau(self) -> bool:
        maximum = int(self.config["train"].get("max_plateau_reductions", 1))
        if int(self.training_state["plateau_reductions"]) >= maximum:
            print("Validation plateau persists; the configured LR-reduction limit is reached.")
            return False
        for optimizer in self.optimizers.values():
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
        for scheduler in self.schedulers.values():
            scheduler.base_lrs = [value * 0.5 for value in scheduler.base_lrs]
        self.training_state["plateau_reductions"] += 1
        self.training_state["bad_validations"] = 0
        print("Validation plateau detected: learning rates reduced by 50%.")
        return True

    def _restore_best_weights(self) -> None:
        best_path = self.checkpoints.resolve_best()
        if best_path is None:
            return
        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        for name, state in checkpoint["models"].items():
            self.models[name].load_state_dict(state)
        for name, state in checkpoint.get("ema", {}).items():
            self.ema[name].load_state_dict(state)
        for name, state in checkpoint.get("optimizers", {}).items():
            current_lrs = [group["lr"] for group in self.optimizers[name].param_groups]
            self.optimizers[name].load_state_dict(state)
            for group, learning_rate in zip(
                self.optimizers[name].param_groups, current_lrs, strict=True
            ):
                group["lr"] = learning_rate
        print(f"Restored best model weights from {best_path.name} before plateau recovery.")

    def run(self, max_hours: float | None = None) -> None:
        maximum = int(self.config["train"]["max_steps"])
        accumulation = int(self.config["train"].get("gradient_accumulation", 1))
        log_every = int(self.config["train"].get("log_every", 25))
        image_every = int(self.config["train"].get("image_every", 500))
        validate_every = int(self.config["train"].get("validate_every", 1000))
        save_every = int(self.config["train"].get("save_every", 1000))
        start_time = time.monotonic()
        iterator = iter(self.train_loader)
        microstep = 0
        last_images: dict[str, torch.Tensor] | None = None
        progress = tqdm(total=maximum, initial=self.step, desc="Training")
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        try:
            while self.step < maximum:
                if self._configure_stage(self.step):
                    iterator = iter(self.train_loader)
                    microstep = 0
                if max_hours and (time.monotonic() - start_time) >= max_hours * 3600:
                    print("Overnight time budget reached; saving a resumable checkpoint.")
                    break
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(self.train_loader)
                    batch = next(iterator)
                microstep += 1
                should_step = microstep % accumulation == 0
                try:
                    metrics, last_images = self.train_microstep(batch, should_step)
                    if not all(math.isfinite(value) for value in metrics.values()):
                        raise FloatingPointError("Non-finite training metric")
                except (FloatingPointError, RuntimeError) as error:
                    if (
                        isinstance(error, RuntimeError)
                        and "out of memory" not in str(error).lower()
                    ):
                        raise
                    self.training_state["recoveries"] += 1
                    atomic_json_dump(
                        {
                            "step": self.step,
                            "error": str(error),
                            "recoveries": self.training_state["recoveries"],
                        },
                        self.output_dir / "diagnostics" / f"failure_{self.step:08d}.json",
                    )
                    for optimizer in self.optimizers.values():
                        optimizer.zero_grad(set_to_none=True)
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    checkpoint = self.checkpoints.resolve("auto")
                    if checkpoint is None or self.training_state["recoveries"] > 3:
                        raise
                    self.load_checkpoint(checkpoint)
                    self._reduce_for_plateau()
                    continue
                if not should_step:
                    continue
                self.step += 1
                progress.update(1)
                if self.step % log_every == 0:
                    for name, value in metrics.items():
                        self.writer.add_scalar(name, value, self.step)
                    for name, optimizer in self.optimizers.items():
                        for group in optimizer.param_groups:
                            self.writer.add_scalar(
                                f"lr/{group.get('name', name)}", group["lr"], self.step
                            )
                    if self.device.type == "cuda":
                        self.writer.add_scalar(
                            "health/peak_vram_gb",
                            torch.cuda.max_memory_allocated() / 1024**3,
                            self.step,
                        )
                if last_images is not None and self.step % image_every == 0:
                    self._log_images(last_images)
                if self.step % validate_every == 0:
                    validation = self.validate()
                    for name, value in validation.items():
                        self.writer.add_scalar(f"validation/{name}", value, self.step)
                    improved = validation["score"] < float(self.training_state["best_score"])
                    if improved:
                        self.training_state["best_score"] = validation["score"]
                        self.training_state["bad_validations"] = 0
                    else:
                        self.training_state["bad_validations"] += 1
                    collapsed = validation["variance"] < 1e-4
                    saturated = metrics.get("D/accuracy", 0.0) > 0.985
                    patience = int(self.config["train"].get("plateau_patience", 4))
                    needs_recovery = (
                        collapsed
                        or saturated
                        or self.training_state["bad_validations"] >= patience
                    )
                    reductions_available = int(
                        self.training_state["plateau_reductions"]
                    ) < int(self.config["train"].get("max_plateau_reductions", 1))
                    if needs_recovery and reductions_available:
                        atomic_json_dump(
                            {
                                "step": self.step,
                                "collapsed": collapsed,
                                "discriminator_saturated": saturated,
                                "bad_validations": self.training_state["bad_validations"],
                                "validation": validation,
                            },
                            self.output_dir / "diagnostics" / f"plateau_{self.step:08d}.json",
                        )
                        self._restore_best_weights()
                        self._reduce_for_plateau()
                    if improved:
                        self.save_checkpoint(best=True)
                if self.step % save_every == 0:
                    self.save_checkpoint()
                self.writer.flush()
        except KeyboardInterrupt:
            print("Interrupted by user; saving before exit.")
        finally:
            self.save_checkpoint()
            self.writer.flush()
            self.writer.close()
            progress.close()
