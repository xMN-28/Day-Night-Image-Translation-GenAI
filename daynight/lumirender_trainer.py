from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .checkpoint import CheckpointManager, build_checkpoint, restore_rng_state
from .config import config_hash
from .losses import (
    DinoSemanticLoss,
    SobelEdgeLoss,
    diff_augment,
    lsgan_discriminator_loss,
    lsgan_generator_loss,
    output_variance,
)
from .lumirender_data import LumiRenderDataset
from .lumirender_losses import (
    bounded_residual_loss,
    confidence_photometric_loss,
    depth_normal_consistency_loss,
    emitter_validity_loss,
    linear_reconstruction_loss,
    luminance,
    paired_perceptual_loss,
    physics_prior_loss,
    reflectance_consistency_loss,
    teacher_factorization_loss,
)
from .models import build_models
from .trainer import ExponentialMovingAverage, set_requires_grad
from .utils import atomic_json_dump, seed_everything


class LumiRenderTrainer:
    def __init__(self, config: dict[str, Any], resume: str = "auto") -> None:
        self.config = config
        seed_everything(int(config["experiment"].get("seed", 46)))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(config["experiment"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(config, self.output_dir / "resolved_config.json")
        self.writer = SummaryWriter(self.output_dir / "tensorboard")
        self.models = build_models(config)
        for model in self.models.values():
            model.to(self.device)
        self.generator = self.models["G_day_night"]
        self.discriminator = self.models["D_night"]
        train = config["train"]
        self.optimizers = {
            "G": torch.optim.AdamW(
                self.generator.parameters(), lr=float(train["generator_lr"]),
                betas=(0.5, 0.999), weight_decay=1e-4
            ),
            "D": torch.optim.Adam(
                self.discriminator.parameters(), lr=float(train["discriminator_lr"]),
                betas=(0.5, 0.999)
            ),
        }
        maximum = int(train["max_steps"])
        self.schedulers = {
            name: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, maximum, eta_min=1e-6)
            for name, optimizer in self.optimizers.items()
        }
        self.ema = {
            "G_day_night": ExponentialMovingAverage(
                self.generator, float(train.get("ema_decay", 0.999))
            )
        }
        data = config["data"]
        manifests = list(data.get("source_manifests", []))
        self.train_dataset = LumiRenderDataset(
            data["root"], "train", int(data["image_size"]), int(data["resize_size"]),
            manifests, float(data.get("paired_fraction", 0.5)), data.get("pseudo_manifest")
        )
        self.val_dataset = LumiRenderDataset(
            data["root"], "val", int(data["image_size"]), int(data["resize_size"]),
            manifests, 1.0, data.get("pseudo_manifest")
        )
        self.workers = int(data.get("num_workers", 4))
        self.train_loader = self._loader(self.train_dataset, True)
        self.val_loader = self._loader(self.val_dataset, False)
        self.edge = SobelEdgeLoss().to(self.device)
        self.semantic = DinoSemanticLoss()
        self.checkpoints = CheckpointManager(
            self.output_dir, int(train.get("keep_checkpoints", 6))
        )
        self.step = 0
        self.microstep = 0
        self.training_state: dict[str, Any] = {
            "best_score": math.inf,
            "current_stage": -1,
            "recoveries": 0,
            "stagnant_validations": 0,
            "learning_rate_reduced": False,
            "collapse_count": 0,
        }
        checkpoint = self.checkpoints.resolve(resume)
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        self._configure_stage(force=True)

    def _loader(self, dataset: LumiRenderDataset, training: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=int(self.config["data"].get("batch_size", 1)),
            shuffle=training,
            num_workers=self.workers if training else 0,
            pin_memory=self.device.type == "cuda",
            persistent_workers=False,
        )

    @property
    def autocast(self):
        if self.device.type != "cuda":
            return nullcontext()
        return torch.autocast("cuda", dtype=torch.bfloat16)

    def _stage_index(self) -> int:
        stages = self.config["train"]["stages"]
        return next(
            (index for index, stage in enumerate(stages) if self.step < int(stage["until_step"])),
            len(stages) - 1,
        )

    def _configure_stage(self, force: bool = False) -> bool:
        index = self._stage_index()
        if not force and index == self.training_state.get("current_stage"):
            return False
        stage = self.config["train"]["stages"][index]
        if stage.get("use_teacher_labels") and not self.train_dataset.pseudo:
            raise RuntimeError(
                "Factorization stage requires cached teacher targets. Run "
                "`python -m daynight.prepare_teacher_targets` first."
            )
        if float(stage.get("paired_fraction", 0)) > 0 and not self.train_dataset.pairs:
            raise RuntimeError(
                f"Stage {stage['name']} requires registered licensed pairs. Run "
                "`python -m daynight.prepare_lumirender_data` first."
            )
        self.train_dataset.image_size = int(stage["image_size"])
        self.train_dataset.resize_size = int(stage["resize_size"])
        self.val_dataset.image_size = int(stage["image_size"])
        self.val_dataset.resize_size = int(stage["resize_size"])
        self.train_dataset.paired_fraction = float(stage.get("paired_fraction", 0.5))
        self.train_dataset.deterministic_day = bool(stage.get("use_teacher_labels", False))
        self.val_dataset.deterministic_day = bool(stage.get("use_teacher_labels", False))
        factorizer_enabled = not bool(stage.get("freeze_factorizer", False))
        for parameter in self.generator.factorizer.parameters():
            parameter.requires_grad_(factorizer_enabled)
        self.train_loader = self._loader(self.train_dataset, True)
        self.val_loader = self._loader(self.val_dataset, False)
        self.training_state["current_stage"] = index
        self.training_state["stage_name"] = stage["name"]
        print(f"LumiRender stage: {stage['name']} ({stage['image_size']}px)")
        return True

    def _active_weights(self) -> dict[str, float]:
        weights = dict(self.config["loss"])
        stage = self.config["train"]["stages"][self._stage_index()]
        weights.update(stage.get("loss_overrides", {}))
        return {name: float(value) for name, value in weights.items()}

    def _generator_losses(
        self, batch: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        day = batch["day"].to(self.device, non_blocking=True)
        night = batch["night"].to(self.device, non_blocking=True)
        confidence = batch["pair_confidence"].to(self.device, non_blocking=True)
        aligned = batch["aligned"].to(self.device, non_blocking=True)
        pseudo_depth = batch["pseudo_depth"].to(self.device, non_blocking=True)
        pseudo_semantic = batch["pseudo_semantic"].to(self.device, non_blocking=True)
        pseudo_valid = batch["pseudo_valid"].to(self.device, non_blocking=True)
        generated, details = self.generator(
            day, seed=self.step, night_intensity=1.0, return_details=True
        )
        weights = self._active_weights()
        components = {
            "linear_reconstruction": linear_reconstruction_loss(details),
            "reflectance": reflectance_consistency_loss(details),
            "aligned_photometric": (
                confidence_photometric_loss(generated, night, confidence, aligned)
                if weights.get("aligned_photometric", 0) > 0
                else day.new_zeros(())
            ),
            "perceptual": (
                paired_perceptual_loss(generated, night, confidence, aligned)
                if weights.get("perceptual", 0) > 0
                else day.new_zeros(())
            ),
            "gan": (
                lsgan_generator_loss(self.discriminator(diff_augment(generated)))
                if weights.get("gan", 0) > 0
                else day.new_zeros(())
            ),
            "semantic": (
                self.semantic(day, generated)
                if weights.get("semantic", 0) > 0
                else day.new_zeros(())
            ),
            "depth_normal": depth_normal_consistency_loss(details),
            "teacher_factorization": teacher_factorization_loss(
                details, pseudo_depth, pseudo_semantic, pseudo_valid
            ),
            "emitter": emitter_validity_loss(details),
            "physics": physics_prior_loss(details),
            "residual": bounded_residual_loss(details),
        }
        components["total"] = sum(
            weights.get(name, 0.0) * value for name, value in components.items()
        )
        return components, generated, details

    def train_step(self, batch: dict[str, Any]) -> tuple[dict[str, float], bool]:
        accumulation = int(self.config["train"].get("gradient_accumulation", 2))
        set_requires_grad([self.discriminator], False)
        with self.autocast:
            generator_losses, generated, details = self._generator_losses(batch)
            (generator_losses["total"] / accumulation).backward()
        gan_enabled = self._active_weights().get("gan", 0) > 0
        discriminator_loss = generated.new_zeros(())
        if gan_enabled:
            set_requires_grad([self.discriminator], True)
            night = batch["night"].to(self.device, non_blocking=True)
            with self.autocast:
                discriminator_loss = lsgan_discriminator_loss(
                    self.discriminator(diff_augment(night)),
                    self.discriminator(diff_augment(generated.detach())),
                )
                (discriminator_loss / accumulation).backward()
        self.microstep += 1
        should_step = self.microstep % accumulation == 0
        generator_norm = generated.new_zeros(())
        discriminator_norm = generated.new_zeros(())
        if should_step:
            maximum = float(self.config["train"].get("grad_clip", 5.0))
            generator_norm = torch.nn.utils.clip_grad_norm_(self.generator.parameters(), maximum)
            discriminator_norm = torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), maximum
            )
            for name, optimizer in self.optimizers.items():
                if name == "D" and not gan_enabled:
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                self.schedulers[name].step()
            self.ema["G_day_night"].update(self.generator)
        metrics = {
            f"G/{name}": float(value.detach().float()) for name, value in generator_losses.items()
        }
        metrics.update(
            {
                "D/total": float(discriminator_loss.detach().float()),
                "health/output_variance": float(output_variance(generated).detach()),
                "health/mean_luminance": float(luminance(generated).mean().detach()),
                "health/max_correction": float(details["correction"].abs().max().detach()),
                "health/generator_grad_norm": float(generator_norm.detach()),
                "health/discriminator_grad_norm": float(discriminator_norm.detach())
                if gan_enabled
                else 0.0,
                "lr/generator": self.optimizers["G"].param_groups[0]["lr"],
                "lr/discriminator": self.optimizers["D"].param_groups[0]["lr"],
            }
        )
        variance = metrics["health/output_variance"]
        self.training_state["collapse_count"] = (
            int(self.training_state.get("collapse_count", 0)) + 1 if variance < 1e-5 else 0
        )
        if self.training_state["collapse_count"] >= 20:
            raise FloatingPointError("Generator collapse: output variance stayed near zero")
        return metrics, should_step

    @torch.no_grad()
    def validate(self, batches: int = 24) -> dict[str, float]:
        original = self.ema["G_day_night"].copy_to(self.generator)
        self.generator.eval()
        totals = {"edge": 0.0, "luminance": 0.0, "reconstruction": 0.0, "variance": 0.0}
        count = 0
        try:
            for batch in self.val_loader:
                day = batch["day"].to(self.device)
                with self.autocast:
                    generated, details = self.generator(day, seed=0, return_details=True)
                totals["edge"] += float(self.edge(day, generated))
                totals["luminance"] += float(luminance(generated).mean())
                totals["reconstruction"] += float(linear_reconstruction_loss(details))
                totals["variance"] += float(output_variance(generated))
                if count == 0:
                    self.writer.add_images(
                        "validation/fixed_day_generated",
                        torch.cat((day[:4], generated[:4])).add(1).div(2).clamp(0, 1),
                        self.step,
                    )
                count += 1
                if count >= batches:
                    break
        finally:
            self.generator.load_state_dict(original)
            self.generator.train()
        result = {name: value / max(1, count) for name, value in totals.items()}
        target_luminance = float(self.config["train"].get("target_night_luminance", 0.18))
        result["score"] = (
            result["edge"] + result["reconstruction"]
            + 2 * abs(result["luminance"] - target_luminance)
        )
        return result

    def checkpoint_payload(self) -> dict[str, Any]:
        payload = build_checkpoint(
            step=self.step,
            config=self.config,
            models=self.models,
            ema={name: value.state_dict() for name, value in self.ema.items()},
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            replay={},
            training_state=self.training_state,
        )
        payload["amp_state"] = None
        return payload

    def save_checkpoint(self, best: bool = False) -> Path:
        return self.checkpoints.save(self.step, self.checkpoint_payload(), best=best)

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("config_hash") != config_hash(self.config):
            print("Warning: LumiRender checkpoint configuration differs from current config.")
        for name, state in payload["models"].items():
            self.models[name].load_state_dict(state)
        for name, state in payload["ema"].items():
            self.ema[name].load_state_dict(state)
        for name, state in payload["optimizers"].items():
            self.optimizers[name].load_state_dict(state)
        for name, state in payload["schedulers"].items():
            self.schedulers[name].load_state_dict(state)
        self.training_state.update(payload.get("training_state", {}))
        self.step = int(payload["step"])
        restore_rng_state(payload["rng"])
        print(f"Resumed LumiRender from {path} at step {self.step}")

    def run(self, max_hours: float | None = None) -> None:
        train = self.config["train"]
        maximum = int(train["max_steps"])
        save_every = int(train.get("save_every", 500))
        validate_every = int(train.get("validate_every", 500))
        log_every = int(train.get("log_every", 10))
        vram_limit = float(train.get("max_vram_gb", 11.5))
        started = time.monotonic()
        iterator = iter(self.train_loader)
        progress = tqdm(total=maximum, initial=self.step, desc="LumiRender")
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        try:
            while self.step < maximum:
                if max_hours and time.monotonic() - started >= max_hours * 3600:
                    print("Time budget reached; saving LumiRender checkpoint.")
                    break
                if self._configure_stage():
                    iterator = iter(self.train_loader)
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(self.train_loader)
                    batch = next(iterator)
                metrics, should_step = self.train_step(batch)
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("Non-finite LumiRender metric")
                if not should_step:
                    continue
                self.step += 1
                progress.update(1)
                if self.device.type == "cuda":
                    peak = torch.cuda.max_memory_allocated() / 1024**3
                    if peak > vram_limit:
                        self.save_checkpoint()
                        raise RuntimeError(
                            f"Peak VRAM {peak:.2f} GB exceeded safety limit {vram_limit:.2f} GB"
                        )
                    metrics["health/peak_vram_gb"] = peak
                if self.step % log_every == 0:
                    for name, value in metrics.items():
                        self.writer.add_scalar(name, value, self.step)
                if self.step % validate_every == 0:
                    validation = self.validate()
                    for name, value in validation.items():
                        self.writer.add_scalar(f"validation/{name}", value, self.step)
                    improved = validation["score"] < float(self.training_state["best_score"])
                    if improved:
                        self.training_state["best_score"] = validation["score"]
                        self.training_state["stagnant_validations"] = 0
                        self.save_checkpoint(best=True)
                    else:
                        self.training_state["stagnant_validations"] = int(
                            self.training_state.get("stagnant_validations", 0)
                        ) + 1
                    if (
                        self.training_state["stagnant_validations"] >= 4
                        and not self.training_state.get("learning_rate_reduced", False)
                    ):
                        best = self.checkpoints.resolve_best()
                        if best is not None:
                            previous_step = self.step
                            self.load_checkpoint(best)
                            self.step = previous_step
                        for optimizer in self.optimizers.values():
                            for group in optimizer.param_groups:
                                group["lr"] *= 0.5
                        self.training_state["learning_rate_reduced"] = True
                        self.training_state["stagnant_validations"] = 0
                        print("Plateau: restored best weights and halved learning rates once.")
                if self.step % save_every == 0:
                    self.save_checkpoint()
                self.writer.flush()
        except KeyboardInterrupt:
            print("Interrupted; saving LumiRender checkpoint.")
        except Exception as error:
            self.training_state["recoveries"] += 1
            atomic_json_dump(
                {"step": self.step, "error": str(error)},
                self.output_dir / "diagnostics" / f"failure_{self.step:08d}.json",
            )
            latest = self.checkpoints.resolve("auto")
            if latest is not None:
                self.load_checkpoint(latest)
                print(f"Rolled back to last finite checkpoint: {latest}")
            raise
        finally:
            self.save_checkpoint()
            self.writer.flush()
            self.writer.close()
            progress.close()


def status_summary(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    latest = run_dir / "checkpoints" / "latest.json"
    return json.loads(latest.read_text(encoding="utf-8")) if latest.exists() else {}
