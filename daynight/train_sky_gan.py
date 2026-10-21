"""V2 warm-start fine tuning with offline semantic sky and boundary supervision.

The segmentation teacher is training-only. Inference uses a normal V2 generator.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .data import read_manifest
from .losses import lsgan_discriminator_loss, lsgan_generator_loss
from .models import build_models
from .models.networks import InstanceLayerNorm
from .sky_refinement import compose_night
from .train_sky_student import CachedSkyDataset
from .utils import atomic_json_dump, atomic_torch_save, seed_everything, tensor_to_pil

PARENT = Path("runs/lumicycle_v2_bdd100k/checkpoints/step_00004500.pt")


def masked_l1(prediction, target, weight):
    return ((prediction - target).abs() * weight).sum() / (weight.sum() * 3).clamp_min(1)


def load_lighting_pair(path):
    with np.load(path) as data:
        tensors = [
            torch.from_numpy(data[key].copy()).float() / 255
            for key in ("image", "target", "static_mask")
        ]
    return tensors[0].permute(2, 0, 1), tensors[1].permute(2, 0, 1), tensors[2][None]


def lighting_pyramid_loss(prediction, target, mask):
    # Smooth correspondence tolerates small fixed-camera registration errors.
    # Never claim these are perfectly aligned pixel targets.
    losses = []
    for scale in (4, 16):
        mass = F.avg_pool2d(mask, scale)
        pred = F.avg_pool2d(prediction * mask, scale) / mass.clamp_min(1e-4)
        truth = F.avg_pool2d(target * mask, scale) / mass.clamp_min(1e-4)
        losses.append(masked_l1(pred, truth, mass))
    return sum(losses) / len(losses)


def critic_real_example(unpaired_night, paired_target, mode):
    """Paired targets use [0,1], critic images [-1,1]. No random source pairing."""
    if mode == "paired" and paired_target is not None:
        return paired_target.detach() * 2 - 1
    return unpaired_night


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--output", type=Path, default=Path("runs/lumicycle_v2_skygan_pilot"))
    parser.add_argument("--max-minutes", type=float, default=90)
    parser.add_argument("--init-from", type=Path, default=PARENT)
    parser.add_argument("--foreground-weight", type=float, default=4)
    parser.add_argument("--paired-cache", type=Path)
    parser.add_argument("--paired-probability", type=float, default=0.5)
    parser.add_argument("--decoder-only", action="store_true")
    parser.add_argument("--late-blocks", type=int, default=0)
    parser.add_argument("--generator-lr", type=float, default=1e-5)
    parser.add_argument("--adapt-norm", action="store_true")
    parser.add_argument("--norm-lr", type=float, default=0.001)
    parser.add_argument("--paired-critic", choices=("bdd", "paired"), default="bdd")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(2027)
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = True
    parent = torch.load(args.init_from, map_location="cpu", weights_only=False)
    config = copy.deepcopy(parent["config"])
    config["experiment"]["output_dir"] = str(args.output)
    config["experiment"]["name"] = "lumicycle_v2_skygan_pilot"
    config["train"]["max_steps"] = parent["step"] + args.steps
    if args.adapt_norm:
        config["model"]["day_residual_norm"] = "iln"
    models = build_models(config)
    for name, model in models.items():
        state = parent["ema"].get(name, {}).get("shadow", parent["models"][name])
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed = (
            args.adapt_norm
            and name == "G_day_night"
            and all(k.endswith((".rho", ".gamma", ".beta")) for k in missing)
        )
        if unexpected or (missing and not allowed):
            raise ValueError(f"Invalid warm start for {name}: {missing}, {unexpected}")
    generator = models["G_day_night"].cuda()
    reference = copy.deepcopy(generator).eval().requires_grad_(False)
    if args.decoder_only:
        for name, parameter in generator.named_parameters():
            late_block = (
                name.startswith("residuals.")
                and int(name.split(".")[1])
                >= config["model"]["generator_blocks"] - args.late_blocks
            )
            parameter.requires_grad_(name.startswith(("up1.", "up2.", "head.")) or late_block)
    if args.adapt_norm:
        for module in generator.modules():
            if isinstance(module, InstanceLayerNorm):
                module.requires_grad_(True)
    discriminator = models["D_night"].cuda()
    norm_ids = (
        {
            id(p)
            for m in generator.modules()
            if isinstance(m, InstanceLayerNorm)
            for p in m.parameters()
        }
        if args.adapt_norm
        else set()
    )
    groups = [
        {
            "params": [p for p in generator.parameters() if id(p) not in norm_ids],
            "lr": args.generator_lr,
        }
    ]
    if norm_ids:
        groups.append(
            {"params": [p for p in generator.parameters() if id(p) in norm_ids], "lr": args.norm_lr}
        )
    optim_g = torch.optim.Adam(groups, betas=(0.5, 0.999))
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=5e-6, betas=(0.5, 0.999))
    ema = {k: v.detach().clone() for k, v in generator.state_dict().items()}
    dataset = CachedSkyDataset(Path("data/sky_student_fullframe_v1"), "train", augment=False)
    validation = CachedSkyDataset(Path("data/sky_student_fullframe_v1"), "val")
    nights = read_manifest("data/bdd100k_daynight", "train", "night")
    pairs, pair_validation = [], []
    if args.paired_cache:
        pair_manifest = json.loads((args.paired_cache / "cache_manifest.json").read_text())
        pairs = [r for r in pair_manifest["pairs"] if r["split"] == "train"]
        pair_validation = [r for r in pair_manifest["pairs"] if r["split"] == "val"]
        if not pairs or not pair_validation:
            raise ValueError("Paired pilot requires separate nonempty train/validation cameras")
        if {r["camera"] for r in pairs} & {r["camera"] for r in pair_validation}:
            raise ValueError("Camera leakage in paired cache")
    pilot_options = {
        "paired_cache": str(args.paired_cache),
        "paired_probability": args.paired_probability,
        "decoder_only": args.decoder_only,
        "generator_lr": args.generator_lr,
        "late_blocks": args.late_blocks,
        "adapt_norm": args.adapt_norm,
        "norm_lr": args.norm_lr,
        "paired_critic": args.paired_critic,
    }
    start_step = 0
    latest = args.output / "latest.pt"
    if latest.exists():
        previous = torch.load(latest, map_location="cpu", weights_only=False)
        previous_options = previous.get("pilot_options", dict(pilot_options))
        previous_options.setdefault("late_blocks", 0)
        previous_options.setdefault("adapt_norm", False)
        previous_options.setdefault("norm_lr", 0.001)
        previous_options.setdefault("paired_critic", "bdd")
        if previous_options != pilot_options:
            raise ValueError("Resume options changed; use a distinct pilot output")
        generator.load_state_dict(previous["models"]["G_day_night"])
        discriminator.load_state_dict(previous["models"]["D_night"])
        ema = {k: v.cuda() for k, v in previous["ema"]["G_day_night"]["shadow"].items()}
        optim_g.load_state_dict(previous["optimizers"]["G"])
        optim_d.load_state_dict(previous["optimizers"]["D"])
        start_step = previous["fine_step"]
        torch.set_rng_state(previous["rng"])
        torch.cuda.set_rng_state_all(previous["cuda_rng"])
        random.setstate(previous["python_rng"])
        np.random.set_state(previous["numpy_rng"])
    atomic_json_dump(
        {
            "parent": str(args.init_from),
            "parent_step": parent["step"],
            "fine_steps": args.steps,
            "generator_lr": args.generator_lr,
            "discriminator_lr": 5e-6,
            "resolution": [384, 216],
            "loss_weights": {
                "sky": 10,
                "foreground_anchor": args.foreground_weight,
                "boundary": 2,
                "night_gan": 0.5,
                "identity_night": 0.5,
            },
            "inference": "generator only; semantic masks training-only",
            "pilot_options": pilot_options,
            "paired_loss_weights": {
                "pyramid": 5,
                "RGB": 0.5,
                "dynamic_anchor": 2,
                "night_gan": 0.25,
                "identity": 0.5,
            }
            if pairs
            else None,
        },
        args.output / "experiment.json",
    )
    started = time.monotonic()
    step = start_step

    def save():
        states = {name: model.state_dict() for name, model in models.items()}
        payload = {
            "step": parent["step"] + step,
            "fine_step": step,
            "parent": str(args.init_from),
            "config": config,
            "models": states,
            "ema": {
                "G_day_night": {"decay": 0.99, "shadow": ema},
                "G_night_day": parent["ema"]["G_night_day"],
            },
            "optimizers": {"G": optim_g.state_dict(), "D": optim_d.state_dict()},
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "pilot_options": pilot_options,
        }
        atomic_torch_save(payload, latest)
        atomic_json_dump(
            {"filename": "latest.pt", "step": payload["step"]}, args.output / "latest.json"
        )

    def previews():
        model = copy.deepcopy(generator).eval()
        model.load_state_dict(ema)
        folder = args.output / "previews" / f"step_{parent['step'] + step:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in (1, 3, 6, 10, 12, 14, 15, 27):
                x, a = validation[i]
                x, a = x[None].cuda(), a[None].cuda()
                old = reference(x * 2 - 1)
                new = model(x * 2 - 1)
                canvas = Image.new("RGB", (1152, 240), "#101010")
                draw = ImageDraw.Draw(canvas)
                for j, (im, label) in enumerate(
                    zip(
                        (x * 2 - 1, old, new),
                        ("Input", "V2 4.5k", f"SkyGAN global {parent['step'] + step}"),
                    )
                ):
                    draw.text((j * 384 + 5, 5), label, fill="white")
                    canvas.paste(tensor_to_pil(im), (j * 384, 24))
                canvas.save(folder / f"val_{i:02d}.jpg", quality=95)
            seen_cameras = set()
            pair_scores = []
            for row in pair_validation:
                if row["camera"] in seen_cameras:
                    continue
                seen_cameras.add(row["camera"])
                x, target, mask = load_lighting_pair(row["cache"])
                x, target, mask = x[None].cuda(), target[None].cuda(), mask[None].cuda()
                pred = model(x * 2 - 1)
                old = reference(x * 2 - 1)
                pair_scores.append(
                    {
                        "camera": row["camera"],
                        "parent_pyramid": float(
                            lighting_pyramid_loss((old.float() + 1) / 2, target, mask)
                        ),
                        "new_pyramid": float(
                            lighting_pyramid_loss((pred.float() + 1) / 2, target, mask)
                        ),
                    }
                )
                canvas = Image.new("RGB", (1536, 240), "#101010")
                draw = ImageDraw.Draw(canvas)
                for j, (im, label) in enumerate(
                    zip(
                        (x * 2 - 1, old, pred, target * 2 - 1),
                        (
                            "Unseen camera input",
                            "Frozen parent",
                            "Paired pilot",
                            "Real-night median",
                        ),
                        strict=True,
                    )
                ):
                    draw.text((j * 384 + 4, 4), label, fill="white")
                    canvas.paste(tensor_to_pil(im), (j * 384, 24))
                canvas.save(folder / f"paired_val_{row['camera']}.jpg", quality=95)
            if pair_scores:
                atomic_json_dump(pair_scores, folder / "paired_scores.json")
        del model

    try:
        for step in range(start_step + 1, args.steps + 1):
            generator.train()
            discriminator.train()
            x, alpha = dataset[random.randrange(len(dataset))]
            is_paired = bool(pairs) and random.random() < args.paired_probability
            if is_paired:
                pair_x, pair_target, pair_mask = load_lighting_pair(random.choice(pairs)["cache"])
                x, alpha = pair_x, torch.zeros_like(pair_mask)
            if random.random() < 0.5:
                x, alpha = x.flip(-1), alpha.flip(-1)
                if is_paired:
                    pair_target, pair_mask = pair_target.flip(-1), pair_mask.flip(-1)
            x, alpha = x[None].cuda(), alpha[None].cuda()
            night_image = (
                Image.open(nights[random.randrange(len(nights))].path)
                .convert("RGB")
                .resize((384, 216), Image.Resampling.LANCZOS)
            )
            night = (
                torch.from_numpy(np.asarray(night_image).copy())
                .permute(2, 0, 1)[None]
                .cuda()
                .float()
                / 127.5
                - 1
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                old = (reference(x * 2 - 1).float() + 1) / 2
                target = compose_night(x, old, alpha)
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            optim_g.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fake = generator(x * 2 - 1)
                y = (fake.float() + 1) / 2
                sky = masked_l1(y, target, alpha)
                fg = masked_l1(y, old, (1 - F.max_pool2d(alpha, 9, 1, 4)).clamp(0, 1))
                boundary = (F.max_pool2d(alpha, 9, 1, 4) + F.max_pool2d(-alpha, 9, 1, 4)).clamp(
                    0, 1
                )
                detail = masked_l1(y, target, boundary)
                gan = lsgan_generator_loss(discriminator(fake))
                identity = F.l1_loss(generator(night), night)
                loss = (
                    sky * 10 + fg * args.foreground_weight + detail * 2 + gan * 0.5 + identity * 0.5
                )
                if is_paired:
                    actual_target = pair_target[None].cuda()
                    static_mask = pair_mask[None].cuda()
                    sky = masked_l1(y, actual_target, static_mask)
                    detail = lighting_pyramid_loss(y, actual_target, static_mask)
                    fg = masked_l1(y, old, 1 - static_mask)
                    loss = detail * 5 + sky * 0.5 + fg * 2 + gan * 0.25 + identity * 0.5
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite GAN loss; previous checkpoint retained")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), 3)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite GAN gradients")
            optim_g.step()
            with torch.no_grad():
                for module in generator.modules():
                    if isinstance(module, InstanceLayerNorm):
                        module.rho.clamp_(0, 1)
                for key, value in generator.state_dict().items():
                    if value.is_floating_point():
                        ema[key].lerp_(value, 0.01)
                    else:
                        ema[key].copy_(value)
            if step % 2 == 0:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
                optim_d.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    real = critic_real_example(
                        night, actual_target if is_paired else None, args.paired_critic
                    )
                    d_loss = lsgan_discriminator_loss(
                        discriminator(real), discriminator(fake.detach())
                    )
                if not torch.isfinite(d_loss):
                    raise FloatingPointError(
                        "Nonfinite discriminator loss; previous checkpoint retained"
                    )
                d_loss.backward()
                d_norm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 3)
                if not torch.isfinite(d_norm):
                    raise FloatingPointError("Nonfinite discriminator gradient")
                optim_d.step()
            if step % 25 == 0:
                row = {
                    "phase": "training",
                    "global_step": parent["step"] + step,
                    "fine_step": step,
                    "loss": float(loss.detach()),
                    "sky": float(sky.detach()),
                    "foreground": float(fg.detach()),
                    "boundary": float(detail.detach()),
                    "gan": float(gan.detach()),
                    "paired": is_paired,
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
                atomic_json_dump(row, args.output / "status.json")
                with (args.output / "training.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if step % 200 == 0:
                save()
                previews()
            if (
                args.output / "STOP"
            ).exists() or time.monotonic() - started > args.max_minutes * 60:
                break
    except KeyboardInterrupt:
        pass
    save()
    previews()
    atomic_json_dump(
        {
            "phase": "finished" if step == args.steps else "stopped",
            "global_step": parent["step"] + step,
            "fine_step": step,
            "checkpoint": str(latest),
        },
        args.output / "status.json",
    )


if __name__ == "__main__":
    main()
