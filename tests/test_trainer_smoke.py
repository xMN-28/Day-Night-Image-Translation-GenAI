from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from daynight.trainer import Trainer


def test_one_step_training_and_resume(tiny_dataset: Path, tmp_path: Path) -> None:
    output = tmp_path / "run"
    config = {
        "experiment": {"name": "smoke", "seed": 3, "output_dir": str(output)},
        "data": {
            "root": str(tiny_dataset),
            "image_size": 64,
            "resize_size": 72,
            "num_workers": 0,
            "batch_size": 1,
        },
        "model": {
            "kind": "cyclegan",
            "generator_blocks": 1,
            "base_channels": 8,
            "attention": False,
            "multiscale_discriminator": False,
        },
        "loss": {
            "gan": 1.0,
            "cycle": 10.0,
            "identity": 5.0,
            "nce": 0.0,
            "semantic": 0.0,
            "edge": 1.0,
        },
        "train": {
            "max_steps": 1,
            "gradient_accumulation": 1,
            "generator_lr": 0.0002,
            "discriminator_lr": 0.0001,
            "beta1": 0.5,
            "beta2": 0.999,
            "warmup_steps": 0,
            "decay_start_step": 1,
            "grad_clip": 5.0,
            "amp": "none",
            "replay_size": 2,
            "ema_decay": 0.99,
            "log_every": 1,
            "image_every": 10,
            "validate_every": 10,
            "save_every": 1,
            "keep_checkpoints": 2,
            "plateau_patience": 4,
        },
    }
    trainer = Trainer(config, resume="none")
    trainer.run()
    assert trainer.step == 1
    expected = next(trainer.models["G_day_night"].parameters()).detach().cpu().clone()
    resumed = Trainer(config, resume="auto")
    assert resumed.step == 1
    actual = next(resumed.models["G_day_night"].parameters()).detach().cpu()
    assert actual.equal(expected)
    resumed.writer.close()

    fine_tune_config = deepcopy(config)
    fine_tune_config["experiment"]["output_dir"] = str(tmp_path / "fine-tune")
    initialized = Trainer(
        fine_tune_config,
        resume="none",
        init_from=output / "checkpoints" / "step_00000001.pt",
    )
    assert initialized.step == 0
    initialized_weight = (
        next(initialized.models["G_day_night"].parameters()).detach().cpu()
    )
    assert initialized_weight.equal(expected)
    initialized.writer.close()

    v21_config = deepcopy(config)
    v21_config["experiment"]["output_dir"] = str(tmp_path / "v2-1")
    v21_config["model"].update(
        {
            "detail_refinement": True,
            "detail_channels": 8,
            "detail_blocks": 1,
            "pyramid_levels": 2,
            "frequency_discriminator": True,
            "frequency_base_channels": 8,
        }
    )
    v21_config["loss"].update(
        {"wavelet": 1.0, "self_similarity": 1.0, "residual": 0.25, "frequency_gan": 0.2}
    )
    v21_config["train"].update(
        {
            "init_scope": "all",
            "stages": [
                {
                    "name": "detail_warmup",
                    "until_step": 1,
                    "image_size": 64,
                    "resize_size": 72,
                    "learning_rates": {
                        "base": 0.0,
                        "refiner": 0.0001,
                        "spatial_d": 0.0,
                        "hf_d": 0.0001,
                    },
                }
            ],
        }
    )
    migrated = Trainer(
        v21_config,
        resume="none",
        init_from=output / "checkpoints" / "step_00000001.pt",
    )
    migrated_weight = next(migrated.models["G_day_night"].base.parameters()).detach().cpu()
    assert migrated_weight.equal(expected)
    assert migrated.training_state["parent_checkpoint"]["source_step"] == 1
    assert migrated.optimizers["G"].param_groups[0]["lr"] == 0.0
    assert migrated.optimizers["G"].param_groups[1]["lr"] == 0.0001
    migrated.writer.close()
