from __future__ import annotations

import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
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
    regional_illumination_loss,
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
        self.discriminators = [self.models["D_day"], self.models["D_night"]]

        train_config = config["train"]
        self.optimizers = {
            "G": torch.optim.Adam(
                [parameter for model in self.generators for parameter in model.parameters()],
                lr=float(train_config["generator_lr"]),
                betas=(float(train_config["beta1"]), float(train_config["beta2"])),
            ),
            "D": torch.optim.Adam(
                [parameter for model in self.discriminators for parameter in model.parameters()],
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
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=int(data_config.get("batch_size", 1)),
            shuffle=True,
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=workers > 0,
        )
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
            self.models[name].load_state_dict(state)
        for name, state in checkpoint.get("ema", {}).items():
            if name in self.ema:
                self.ema[name].load_state_dict(state)
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
        fake_night, day_features = self.models["G_day_night"](day, return_features=True)
        fake_day, night_features = self.models["G_night_day"](night, return_features=True)
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
        }
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
        return {
            "total": day_loss + night_loss,
            "day": day_loss,
            "night": night_loss,
            "accuracy": 0.5
            * (
                discriminator_accuracy(real_day_output, fake_day_output)
                + discriminator_accuracy(real_night_output, fake_night_output)
            ),
        }

    def _grad_norm(self, modules: list[nn.Module]) -> float:
        norms = [
            parameter.grad.detach().float().norm()
            for module in modules
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        return float(torch.stack(norms).norm().item()) if norms else 0.0

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

        set_requires_grad(self.discriminators, True)
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
                totals["cycle"] += float(cycle.float())
                totals["edge"] += float(edge.float())
                totals["color"] += float(color.float())
                totals["illumination"] += float(illumination.float())
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
                        self.writer.add_scalar(
                            f"lr/{name}", optimizer.param_groups[0]["lr"], self.step
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
