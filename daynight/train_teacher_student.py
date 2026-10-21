"""Warm-start LumiCycle from its trained weights; external teacher training-only.

Compact day-to-night-only checkpoints intentionally omit unchanged discriminators
and inverse generator. They load via SkyPipeline/load_base, not the legacy all-
model manager. No externally pretrained diffusion backbone is used at inference.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .compare_sky_gan import file_digest
from .losses import DinoSemanticLoss
from .models.networks import PatchDiscriminator, ResnetGenerator
from .sky_pilot import load_base
from .teacher_objectives import (
    conditioned_pair,
    highlight_fidelity,
    least_squares_critic,
    least_squares_generator,
    light_footprint_fidelity,
    masked_target_fidelity,
)
from .train_resize_decoder import decoder_loss, state_digest
from .utils import atomic_json_dump, atomic_torch_save, seed_everything, tensor_to_pil


def configure_execution(seed, deterministic=False):
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed_everything(seed)
    # The shared seed helper enables cuDNN benchmarking. Override it AFTER
    # seeding: timing-selected algorithms may differ across restarted processes.
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)


def load_pair(path, source=None):
    path = Path(path)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        if source is None:
            raise ValueError("Target-only cache requires its original source image")
        from .turbo_pilot import fit_input

        image, _, _ = fit_input(Image.open(source), 512)
        target = Image.open(path).convert("RGB")
        return tuple(
            torch.from_numpy(np.asarray(im.resize(target.size, Image.Resampling.LANCZOS)).copy())
            .permute(2, 0, 1)
            .float()
            / 255
            for im in (image, target)
        )
    with np.load(path) as data:
        return tuple(
            torch.from_numpy(data[k].copy()).permute(2, 0, 1).float() / 255
            for k in ("image", "target")
        )


def validate_resolution_curriculum(previous_path, current, expected_hash):
    if file_digest(previous_path) != expected_hash:
        raise ValueError("Previous target manifest does not match the parent checkpoint")
    previous = json.loads(Path(previous_path).read_text())
    if previous["test_images_used"] or current["test_images_used"]:
        raise ValueError("Test images are forbidden in a training curriculum")

    def identities(manifest):
        return sorted(
            (row["split"], row["index"], row["source"], row["source_sha256"], row["seed"])
            for row in manifest["records"]
        )

    if identities(previous) != identities(current):
        raise ValueError(
            "Resolution curriculum must keep source identities, splits and teacher seeds unchanged"
        )
    old_recipe, new_recipe = dict(previous["recipe"]), dict(current["recipe"])
    old_size, new_size = old_recipe.pop("student_size"), new_recipe.pop("student_size")
    if old_recipe != new_recipe:
        raise ValueError(
            "Resolution curriculum cannot silently change the teacher or target recipe"
        )
    if new_size[0] <= old_size[0] or new_size[0] > new_recipe["teacher_maximum_edge"]:
        raise ValueError(
            "Curriculum must increase resolution without exceeding the teacher's native maximum edge"
        )
    if old_size[0] * new_size[1] != old_size[1] * new_size[0]:
        raise ValueError("Resolution curriculum must preserve aspect ratio")


def select_training_subset(records, count):
    if count < 0:
        raise ValueError("Diagnostic subset count must be nonnegative")
    if not count:
        return records
    indices = np.linspace(0, len(records) - 1, min(count, len(records)), dtype=int)
    return [records[int(index)] for index in indices]


def expand_trained_detail_skip(parent, options):
    if not isinstance(parent, ResnetGenerator) or parent.detail_skip is not None:
        raise ValueError("Detail expansion requires an existing ResNet without a detail adapter")
    adapted = ResnetGenerator(
        base_channels=options["base_channels"],
        blocks=options["generator_blocks"],
        attention=options["attention"],
        upsampling=options.get("day_upsampling", "transpose"),
        residual_norm=options.get("day_residual_norm", "instance"),
        detail_skip=True,
    ).to(next(parent.parameters()).device)
    previous = parent.state_dict()
    missing, unexpected = adapted.load_state_dict(previous, strict=False)
    if unexpected or not missing or any(not name.startswith("detail_skip.") for name in missing):
        raise ValueError("Unexpected parameter mismatch in trained-generator expansion")
    if any(not torch.equal(adapted.state_dict()[name], value) for name, value in previous.items()):
        raise ValueError("Trained parent weights changed during expansion")
    return adapted


def frozen_base_digest(model):
    return state_digest(
        {
            name: value
            for name, value in model.state_dict().items()
            if not name.startswith("detail_skip.")
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=Path("runs/resize_decoder_pilot/best.pt"))
    parser.add_argument(
        "--targets", type=Path, default=Path("E:/LumiCycleCache/teacher_targets/turbo_v1")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("E:/LumiCycleCache/experiments/turbo_distilled_v1")
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--max-minutes", type=float, default=25)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--highlight-weight", type=float, default=0)
    parser.add_argument("--adversarial-weight", type=float, default=0)
    parser.add_argument("--footprint-weight", type=float, default=0)
    parser.add_argument("--continue-state", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--previous-targets", type=Path)
    parser.add_argument("--overfit-images", type=int, default=0)
    parser.add_argument("--semantic-weight", type=float, default=0.25)
    parser.add_argument("--keep-every", type=int, default=0)
    parser.add_argument("--add-detail-skip", action="store_true")
    parser.add_argument("--train-scope", choices=("all", "detail-skip"), default="all")
    parser.add_argument("--sky-target-weight", type=float, default=0)
    parser.add_argument("--perceptual-target-weight", type=float, default=0)
    parser.add_argument("--vgg-cache", type=Path, default=Path("E:/LumiCycleCache/torchvision"))
    parser.add_argument(
        "--sky-matte", type=Path, default=Path("runs/sky_student_regions_v2/best.pt")
    )
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or min(
            args.highlight_weight,
            args.adversarial_weight,
            args.footprint_weight,
            args.semantic_weight,
            args.sky_target_weight,
            args.perceptual_target_weight,
        )
        < 0
    ):
        parser.error("Batch size must be positive and loss weights nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.keep_every < 0:
        parser.error("--keep-every must be nonnegative")
    if args.add_detail_skip and args.continue_state:
        parser.error(
            "Architecture expansion uses a new adapter optimizer; do not request unchanged optimizer continuation"
        )
    source_identity = {
        str(path): file_digest(path)
        for path in (
            Path(__file__),
            Path(__file__).with_name("teacher_objectives.py"),
            Path(__file__).parent / "models/networks.py",
            Path(__file__).with_name("perceptual_targets.py"),
        )
    }
    if args.previous_targets and not args.continue_state:
        parser.error("--previous-targets requires --continue-state")
    torch.set_num_threads(6)
    configure_execution(2030, args.deterministic)
    manifest = json.loads((args.targets / "manifest.json").read_text())
    if manifest["test_images_used"]:
        raise ValueError("Test targets cannot enter distillation")
    train = [r for r in manifest["records"] if r["split"] == "train"]
    val = [r for r in manifest["records"] if r["split"] == "val"]
    if {r["source"] for r in train} & {r["source"] for r in val}:
        raise ValueError("Training/validation leakage")
    train = select_training_subset(train, args.overfit_images)
    if args.overfit_images:
        atomic_json_dump(
            {"scope": "TRAINING memorization diagnostic, not generalization", "records": train},
            args.output / "training_selection.json",
        )
    provenance = {
        "parent_sha256": file_digest(args.parent),
        "targets_manifest_sha256": file_digest(args.targets / "manifest.json"),
        "learning_rate": args.lr,
        "semantic_weight": args.semantic_weight,
        "teacher_external_pretrained": True,
        "training": "all warm-started generator parameters"
        if args.train_scope == "all"
        else "detail adapter only; trained base frozen",
        "architecture_changed": args.add_detail_skip,
        "add_detail_skip": args.add_detail_skip,
        "train_scope": args.train_scope,
        "frozen_base_sha256": None,
        "seed": 2030,
        "batch_size": args.batch_size,
        "highlight_weight": args.highlight_weight,
        "adversarial_weight": args.adversarial_weight,
        "footprint_weight": args.footprint_weight,
        "continue_state": args.continue_state,
        "deterministic": args.deterministic,
        "overfit_images": args.overfit_images,
        "sky_target_weight": args.sky_target_weight,
        "sky_matte_sha256": file_digest(args.sky_matte) if args.sky_target_weight else None,
        "perceptual_target_weight": args.perceptual_target_weight,
        "vgg_target_sha256": file_digest(args.vgg_cache / "vgg16-397923af.pth")
        if args.perceptual_target_weight
        else None,
        "resolution_curriculum_from_sha256": file_digest(args.previous_targets / "manifest.json")
        if args.previous_targets
        else None,
    }
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    config = copy.deepcopy(parent["config"])
    config["experiment"]["name"] = args.output.name
    config["experiment"]["output_dir"] = str(args.output)
    config["train"]["max_steps"] = parent["step"] + args.steps
    model = load_base(args.parent)
    if args.add_detail_skip:
        model = expand_trained_detail_skip(model, config["model"])
        config["model"]["day_detail_skip"] = True
    model.train()
    if args.train_scope == "detail-skip" and model.detail_skip is None:
        raise ValueError("Detail-only training requires an expanded generator")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(args.train_scope == "all" or name.startswith("detail_skip."))
    if args.train_scope == "detail-skip":
        provenance["frozen_base_sha256"] = frozen_base_digest(model)
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr, betas=(0.5, 0.999)
    )
    critic = critic_optimizer = None
    if args.adversarial_weight:
        critic = PatchDiscriminator(
            base_channels=32, use_spectral_norm=True, input_channels=6
        ).cuda()
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.999))
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    semantic = None
    if args.semantic_weight:
        hub = Path("artifacts/torch_cache/hub").resolve()
        if not (hub / "checkpoints/dinov2_vits14_pretrain.pth").is_file():
            raise FileNotFoundError(
                "Local DINO weights missing; no automatic network download in this trainer"
            )
        torch.hub.set_dir(str(hub))
        semantic = DinoSemanticLoss()
        semantic.model = (
            torch.hub.load(
                str(hub / "facebookresearch_dinov2_main"), "dinov2_vits14", source="local"
            )
            .cuda()
            .eval()
            .requires_grad_(False)
        )
    sky_matte = None
    if args.sky_target_weight:
        from .sky_refinement import SkyStudent

        matte_payload = torch.load(args.sky_matte, map_location="cpu", weights_only=False)
        sky_matte = (
            SkyStudent(outputs=matte_payload["model"]["head.weight"].shape[0])
            .cuda()
            .eval()
            .requires_grad_(False)
        )
        sky_matte.load_state_dict(matte_payload["model"])
    perceptual_target = None
    if args.perceptual_target_weight:
        from .perceptual_targets import FrozenVggTargetLoss

        perceptual_target = FrozenVggTargetLoss(args.vgg_cache).cuda().eval()
    step, best = 0, float("inf")

    def restore_state(saved):
        model.load_state_dict(saved["models"]["G_day_night"])
        shadow = {k: v.cuda() for k, v in saved["ema"]["G_day_night"]["shadow"].items()}
        optimizer.load_state_dict(saved["optimizer"])
        if critic is not None:
            if saved.get("conditional_critic") is None:
                raise ValueError("Cannot continue a critic that was absent in the parent")
            critic.load_state_dict(saved["conditional_critic"])
            critic_optimizer.load_state_dict(saved["critic_optimizer"])
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        return shadow

    if args.continue_state:
        if parent["distillation"].get("train_scope", "all") != args.train_scope:
            raise ValueError("Optimizer continuation requires the same trainable parameter scope")
        if (
            parent["distillation"]["targets_manifest_sha256"]
            != provenance["targets_manifest_sha256"]
        ):
            if not args.previous_targets:
                raise ValueError(
                    "State continuation requires the same targets or a verified resolution curriculum"
                )
            validate_resolution_curriculum(
                args.previous_targets / "manifest.json",
                manifest,
                parent["distillation"]["targets_manifest_sha256"],
            )
        if parent["distillation"]["learning_rate"] != args.lr:
            raise ValueError("State continuation requires the saved learning rate")
        ema = restore_state(parent)
    latest = args.output / "latest.pt"
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        saved_provenance = dict(saved["distillation"])
        saved_provenance.setdefault("batch_size", 1)
        saved_provenance.setdefault("highlight_weight", 0)
        saved_provenance.setdefault("adversarial_weight", 0)
        saved_provenance.setdefault("footprint_weight", 0)
        saved_provenance.setdefault("continue_state", False)
        saved_provenance.setdefault("deterministic", False)
        saved_provenance.setdefault("resolution_curriculum_from_sha256", None)
        saved_provenance.setdefault("overfit_images", 0)
        saved_provenance.setdefault("sky_target_weight", 0)
        saved_provenance.setdefault("sky_matte_sha256", None)
        saved_provenance.setdefault("perceptual_target_weight", 0)
        saved_provenance.setdefault("vgg_target_sha256", None)
        saved_provenance.setdefault("add_detail_skip", False)
        saved_provenance.setdefault("train_scope", "all")
        saved_provenance.setdefault("frozen_base_sha256", None)
        if saved_provenance != provenance:
            raise ValueError("Resume provenance changed; use a new run folder")
        ema = restore_state(saved)
        step, best = saved["adaptation_step"], saved["best_fidelity"]
    atomic_json_dump(provenance, args.output / "experiment.json")

    def save(name):
        if args.train_scope == "detail-skip":
            if frozen_base_digest(model) != provenance["frozen_base_sha256"]:
                raise RuntimeError("Frozen trained base changed; checkpoint publication aborted")
            if (
                state_digest(
                    {key: value for key, value in ema.items() if not key.startswith("detail_skip.")}
                )
                != provenance["frozen_base_sha256"]
            ):
                raise RuntimeError("Frozen EMA base changed; checkpoint publication aborted")
        atomic_torch_save(
            {
                "step": parent["step"] + step,
                "adaptation_step": step,
                "best_fidelity": best,
                "config": config,
                "models": {"G_day_night": model.state_dict()},
                "ema": {"G_day_night": {"decay": 0.99, "shadow": ema}},
                "optimizer": optimizer.state_dict(),
                "conditional_critic": critic.state_dict() if critic is not None else None,
                "critic_optimizer": critic_optimizer.state_dict()
                if critic_optimizer is not None
                else None,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(),
                "distillation": provenance,
                "parent_checkpoint": str(args.parent),
                "checkpoint_scope": "day-to-night student only; not a full CycleGAN training resume",
                "source_code_at_process_start": source_identity,
            },
            args.output / name,
        )

    def validate():
        probe = copy.deepcopy(model).eval().requires_grad_(False)
        probe.load_state_dict(ema)
        scores = []
        folder = args.output / "previews" / f"step_{step:05d}"
        folder.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i, row in enumerate(val):
                x, target = load_pair(row["cache"], row["source"])
                x, target = x[None].cuda(), target[None].cuda()
                pred = probe(x * 2 - 1)
                scores.append(float(decoder_loss(pred.float(), target * 2 - 1)))
                if i in (0, 6, 13, 20, 27, 34, 40, 47):
                    old = reference(x * 2 - 1)
                    height, width = x.shape[-2:]
                    canvas = Image.new("RGB", (width * 4, height + 24), "#101010")
                    draw = ImageDraw.Draw(canvas)
                    for j, (im, label) in enumerate(
                        zip(
                            (x * 2 - 1, old, pred, target * 2 - 1),
                            (
                                "Input",
                                "Trained parent",
                                "Warm-started student",
                                "Official external teacher",
                            ),
                            strict=True,
                        )
                    ):
                        draw.text((j * width + 4, 4), label, fill="white")
                        canvas.paste(tensor_to_pil(im), (j * width, 24))
                    canvas.save(folder / f"val_{i:02d}.jpg", quality=95)
        return sum(scores) / len(scores)

    started, completed = time.monotonic(), step
    try:
        for step in range(completed + 1, args.steps + 1):
            inputs, targets = [], []
            for _ in range(args.batch_size):
                selected = random.choice(train)
                x, target = load_pair(selected["cache"], selected["source"])
                if random.random() < 0.5:
                    x, target = x.flip(-1), target.flip(-1)
                inputs.append(x)
                targets.append(target)
            x, target = torch.stack(inputs).cuda(), torch.stack(targets).cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x * 2 - 1)
                fidelity = decoder_loss(pred.float(), target * 2 - 1)
                structure = (
                    semantic(x * 2 - 1, pred) if semantic is not None else pred.new_zeros(())
                )
                highlight = highlight_fidelity(pred, target * 2 - 1)
                footprint = light_footprint_fidelity(pred, target * 2 - 1)
                sky_fidelity = pred.new_zeros(())
                if sky_matte is not None:
                    from .sky_refinement import predict_matte

                    alpha, _ = predict_matte(sky_matte, x)
                    sky_fidelity = masked_target_fidelity(pred, target * 2 - 1, alpha)
                perceptual_fidelity = (
                    perceptual_target(pred, target * 2 - 1)
                    if perceptual_target is not None
                    else pred.new_zeros(())
                )
            adversarial = pred.new_zeros(())
            discriminator_loss = pred.new_zeros(())
            if critic is not None:
                critic.train().requires_grad_(True)
                critic_optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    real_logits = critic(conditioned_pair(x * 2 - 1, target * 2 - 1))
                    fake_logits = critic(conditioned_pair(x * 2 - 1, pred.detach()))
                    discriminator_loss = least_squares_critic(
                        real_logits.float(), fake_logits.float()
                    )
                if not torch.isfinite(discriminator_loss):
                    raise FloatingPointError("Nonfinite conditional discriminator loss")
                discriminator_loss.backward()
                critic_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 3)
                if not torch.isfinite(critic_norm):
                    raise FloatingPointError("Nonfinite conditional discriminator gradients")
                critic_optimizer.step()
                # Freeze spectral-normalization power iteration as well as weights
                # while propagating the adversarial gradient into the generator.
                critic.eval().requires_grad_(False)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    adversarial = least_squares_generator(
                        critic(conditioned_pair(x * 2 - 1, pred)).float()
                    )
            loss = (
                fidelity
                + structure * args.semantic_weight
                + args.highlight_weight * highlight
                + args.adversarial_weight * adversarial
                + args.footprint_weight * footprint
                + args.sky_target_weight * sky_fidelity
                + args.perceptual_target_weight * perceptual_fidelity
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Nonfinite distillation loss; previous checkpoint retained"
                )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite student gradients")
            optimizer.step()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema[k].lerp_(v, 0.01)
            completed = step
            if step % 25 == 0 or step == 1:
                status = {
                    "phase": "training",
                    "adaptation_step": step,
                    "global_step": parent["step"] + step,
                    "fidelity": float(fidelity.detach()),
                    "semantic": float(structure.detach()),
                    "highlight": float(highlight.detach()),
                    "adversarial": float(adversarial.detach()),
                    "discriminator": float(discriminator_loss.detach()),
                    "footprint": float(footprint.detach()),
                    "sky_target": float(sky_fidelity.detach()),
                    "perceptual_target": float(perceptual_fidelity.detach()),
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "batch_size": args.batch_size,
                }
                atomic_json_dump(status, args.output / "status.json")
                with (args.output / "training.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(status) + "\n")
                print(json.dumps(status), flush=True)
            if step % 250 == 0:
                score = validate()
                if score < best:
                    best = score
                    save("best.pt")
                save("latest.pt")
                record = {
                    "phase": "validation",
                    "adaptation_step": step,
                    "global_step": parent["step"] + step,
                    "fidelity": score,
                    "best": best,
                }
                with (args.output / "validation.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                atomic_json_dump(record, args.output / "last_validation.json")
                print(json.dumps(record), flush=True)
            if args.keep_every and step % args.keep_every == 0:
                save(f"step_{step:05d}.pt")
            if (
                args.output / "STOP"
            ).exists() or time.monotonic() - started > args.max_minutes * 60:
                break
    except KeyboardInterrupt:
        step = completed
    save("latest.pt")
    atomic_json_dump(
        {
            "phase": "finished" if step == args.steps else "stopped",
            "adaptation_step": step,
            "global_step": parent["step"] + step,
            "best_fidelity": best if math.isfinite(best) else None,
            "teacher_external_pretrained": True,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "batch_size": args.batch_size,
        },
        args.output / "status.json",
    )


if __name__ == "__main__":
    main()
