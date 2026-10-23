"""Isolated, bounded inverse continuation on audited EXTERNAL pseudo-targets.

Never registers with ModelManager or alters production checkpoints. Parent is an
existing trained inverse generator. Teacher agreement is not a quality metric.
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
from .experiment_night_to_day import inverse_from_payload
from .inverse_reliability import glare_reliability, reliability_weighted_loss
from .inverse_scene_context import add_context_preserving_weights
from .models.networks import LocalGlobalDiscriminator
from .turbo_pilot import digest
from .utils import atomic_json_dump, atomic_torch_save, pil_to_tensor, tensor_to_pil


def pilot_loss(prediction, target):
    rgb = F.l1_loss(prediction.float(), target.float())
    coarse = sum(F.l1_loss(F.avg_pool2d(prediction.float(), k), F.avg_pool2d(target.float(), k)) for k in (4, 16)) / 2
    gradient = sum(F.l1_loss(torch.diff(prediction.float(), dim=d), torch.diff(target.float(), dim=d)) for d in (-1, -2)) / 2
    return 10 * rgb + 2 * coarse + gradient


def perturb_training_input(source, sigma01, generator):
    """Seeded synthetic sRGB perturbation; source tensors use[-1,1]."""
    if sigma01 < 0:
        raise ValueError("Nonnegative synthetic noise required")
    if sigma01 == 0:
        return source
    return (source + 2 * sigma01 * torch.randn(source.shape,
        generator=generator, device=source.device, dtype=source.dtype)).clamp(-1, 1)


def noise_robust_loss(clean, noisy, target, consistency_weight):
    """Keep total target-supervision weight fixed; clean prediction is anchor."""
    consistency = F.l1_loss(noisy.float(), clean.detach().float())
    return .5 * (pilot_loss(clean, target) + pilot_loss(noisy, target)) + consistency_weight * consistency, consistency


def read_pair(row):
    source = Image.open(row["input"]).convert("RGB")
    target = Image.open(row["target"]).convert("RGB")
    if source.size != target.size:
        raise ValueError("Pseudo-pair dimensions differ")
    return pil_to_tensor(source), pil_to_tensor(target)


def select_training_rows(rows, kind="all", limit=0):
    """Deterministic training-only subset; never filters/replaces validation."""
    if kind not in {"all", "synthetic"} or limit < 0:
        raise ValueError("Invalid training subset")
    rows = [r for r in rows if r["split"] == "train"]
    if kind == "synthetic":
        rows = [r for r in rows if "original_day_train_index" in r]
    if limit and len(rows) > limit:
        indices = np.linspace(0, len(rows) - 1, limit, dtype=int)
        rows = [rows[int(i)] for i in indices]
    if not rows:
        raise ValueError("Empty training subset")
    return rows


def adversarial_lsgan(outputs, real):
    return sum(F.mse_loss(output.float(), torch.ones_like(output).float() if real
        else torch.zeros_like(output).float()) for output in outputs) / len(outputs)


def static_region_rgb_loss(prediction, target, confidence):
    if confidence.shape != (prediction.shape[0], 1, *prediction.shape[-2:]):
        raise ValueError("Require matching N1HW static confidence")
    denominator = confidence.float().sum((1, 2, 3)) * prediction.shape[1]
    if (denominator <= 0).any():
        raise ValueError("Empty static-region supervision")
    numerator = ((prediction.float() - target.float()).abs() * confidence).sum((1, 2, 3))
    return (numerator / denominator).mean()


def static_region_coarse_loss(prediction, target, confidence):
    return sum(static_region_rgb_loss(F.avg_pool2d(prediction.float(), k),
        F.avg_pool2d(target.float(), k), F.avg_pool2d(confidence.float(), k)) for k in (4, 8)) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parent", type=Path, default=Path("runs/lumicycle_bdd100k/checkpoints/step_00013000.pt"))
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-minutes", type=float, default=10)
    parser.add_argument("--resume", choices=("never", "auto"), default="never")
    parser.add_argument("--decoder", choices=("parent", "transpose", "resize"), default="parent")
    parser.add_argument("--train-kind", choices=("all", "synthetic"), default="all")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--diagnostic-train-count", type=int, default=0)
    parser.add_argument("--gan-weight", type=float, default=0)
    parser.add_argument("--real-day-count", type=int, default=256)
    parser.add_argument("--target-reliability", choices=("uniform", "retained_glare"), default="uniform")
    parser.add_argument("--real-regions", type=Path)
    parser.add_argument("--real-region-weight", type=float, default=3)
    parser.add_argument("--real-region-loss", choices=("rgb", "coarse"), default="rgb")
    parser.add_argument("--noise-training", action="store_true")
    parser.add_argument("--noise-sigma", type=float, default=2 / 255)
    parser.add_argument("--noise-consistency-weight", type=float, default=2)
    parser.add_argument("--scene-context", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 2000 or args.batch_size < 1:
        raise ValueError("Bounded pilot requires1..2000updates and positive batch")
    if args.gan_weight < 0 or args.real_day_count < 1:
        raise ValueError("Require nonnegative GAN weight and positive real-day count")
    if args.real_region_weight < 0:
        raise ValueError("Nonnegative real-region weight required")
    if args.noise_sigma < 0 or args.noise_consistency_weight < 0:
        raise ValueError("Nonnegative noise settings required")
    if args.noise_training and args.target_reliability != "uniform":
        raise ValueError("Do not conflate noise isolation with glare reweighting")
    manifest = json.loads((args.targets / "manifest.json").read_text())
    selection = json.loads(args.selection.read_text())
    if manifest["direction"] != "night_to_day" or manifest["test_used"]:
        raise ValueError("Require night→day training-only/validation pseudo-pairs")
    if selection["manifest_sha256"] != digest(args.targets / "manifest.json"):
        raise ValueError("Visual selection belongs to a different cache")
    selected = set(selection["accepted_train_indices"])
    train = [r for r in manifest["records"] if r["split"] == "train" and r["index"] in selected]
    val = [r for r in manifest["records"] if r["split"] == "val"]
    if not train or not val or len(train) != len(selected):
        raise ValueError("Empty/invalid audited selection")
    train = select_training_rows(train, args.train_kind, args.train_limit)
    if args.diagnostic_train_count < 0:
        raise ValueError("Diagnostic count cannot be negative")
    if {r["source"] for r in train} & {r["source"] for r in val}:
        raise ValueError("Training/validation overlap")
    for row in train + val:
        if digest(row["target"]) != row["target_sha256"]:
            raise ValueError("Audited target bytes changed")
    args.output.mkdir(parents=True, exist_ok=True)
    latest = args.output / "latest.pt"
    if latest.exists() and args.resume == "never":
        raise FileExistsError("Do not overwrite an earlier pilot")
    torch.set_num_threads(6)
    random.seed(2030); torch.manual_seed(2030); torch.cuda.manual_seed_all(2030)
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    model, architecture = inverse_from_payload(parent, args.decoder)
    if args.scene_context:
        if "scene_context" in architecture:
            raise ValueError("Parent already contains context; do not add twice")
        model = add_context_preserving_weights(model)
        architecture["scene_context"] = {"width": 128, "grid": 8, "heads": 4,
            "version": 1, "initialization": "zero output projection; all backbone weights exact"}
    parent_step = parent["step"]
    model.cuda().requires_grad_(True).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    discriminator = None
    discriminator_optimizer = None
    gan_random = random.Random(2031)
    region_random = random.Random(2032)
    noise_random = torch.Generator(device="cuda").manual_seed(2033)
    real_regions = []
    region_labels = {}
    if args.real_regions:
        regions = json.loads(args.real_regions.read_text())
        if not regions["approved_after_visual_review"] or regions["test_used"] or regions["lockbox_used"]:
            raise ValueError("Require visually approved training-only real regions")
        for row in regions["records"]:
            if row["split"] != "train":
                raise ValueError("No held-out real regions")
            for key in ("input", "target", "confidence"):
                if digest(Path(row[key])) != row[f"{key}_sha256"]:
                    raise ValueError("Audited real region bytes changed")
            x, y = read_pair(row)
            confidence = torch.from_numpy(np.asarray(Image.open(row["confidence"]).convert("L"), dtype=np.float32).copy())[None] / 255
            if x.shape != y.shape or confidence.shape != (1, *x.shape[-2:]) or confidence.sum() <= 0:
                raise ValueError("Invalid static-region dimensions or support")
            real_regions.append((row["index"], x, y, confidence))
            region_labels[row["index"]] = (row.get("source_label", "REAL night TRAIN"),
                row.get("target_label", "REAL day reference, ROI only"))
        if not real_regions:
            raise ValueError("Empty region cache")
    real_days = []
    real_day_paths = []
    if args.gan_weight:
        discriminator = LocalGlobalDiscriminator(base_channels=32, scales=1).cuda().train()
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=5e-5, betas=(.5, .999))
        day_records = read_manifest("data/bdd100k_daynight", "train", "day")
        for i in np.linspace(0, len(day_records) - 1, min(args.real_day_count, len(day_records)), dtype=int):
            path = day_records[int(i)].path
            with Image.open(path) as original:
                real_days.append(pil_to_tensor(original.convert("RGB").resize((256, 144), Image.Resampling.BICUBIC)))
            real_day_paths.append({"path": str(path), "sha256": digest(path)})
    config = copy.deepcopy(parent["config"])
    del parent
    recipe = {"direction": "night_to_day", "parent": str(args.parent), "parent_sha256": digest(args.parent),
        "parent_step": parent_step, "initial_weights": "EMA G_night_day",
        "architecture_changed": architecture["converted"] or args.scene_context,
        "manifest_sha256": digest(args.targets / "manifest.json"), "selection_sha256": digest(args.selection),
        "lr": args.lr, "batch_size": args.batch_size, "ema_decay": 0.99,
        "optimizer": "Adam(.5,.999), fresh due to new single-inverse supervised objective",
        "loss": "10 RGB-L1 +2 mean(pool4,pool16)L1 +1 mean(horizontal,vertical) target-gradient L1",
        "source_structure_loss": 0, "gan": args.gan_weight, "cycle": 0, "precision": "BF16", "clip": 3,
        "teacher_external": True, "train_count": len(train), "val_count": len(val), "seed": 2030}
    recipe.update({"train_kind": args.train_kind, "train_limit": args.train_limit,
        "train_indices": [r["index"] for r in train],
        "diagnostic_train_count": args.diagnostic_train_count,
        "training_target_origin": "REAL day photographs; synthetic night inputs from existing day-to-night model"
        if args.train_kind == "synthetic" else "See each manifest row; external pseudo-targets may be present"})
    recipe.update({"gan_weight": args.gan_weight, "gan_real_day_sources": real_day_paths,
        "gan_recipe": "LSGAN local+global32/scales1;256x144;D Adam5e-5;250-step linear G-adversarial ramp;no external generator at inference"
        if discriminator is not None else None})
    recipe.update({"target_reliability": args.target_reliability,
        "target_reliability_code_sha256": digest(Path(__file__).with_name("inverse_reliability.py"))
        if args.target_reliability != "uniform" else None,
        "validation_loss": "Uniform original pilot_loss; NOT region-weighted, for comparable reporting"})
    recipe.update({"real_regions_sha256": digest(args.real_regions) if args.real_regions else None,
        "real_region_weight": args.real_region_weight if real_regions else 0,
        "real_region_count": len(real_regions),
        "auxiliary_input_origin": regions.get("input_origin", "real night") if real_regions else None,
        "real_region_loss_mode": args.real_region_loss,
        "real_region_loss": "RGB-L1 normalized by confidence mass; zero outside manually reviewed static support" if real_regions else None})
    recipe.update({"noise_training": args.noise_training,
        "noise_sigma_srgb01": args.noise_sigma if args.noise_training else 0,
        "noise_consistency_weight": args.noise_consistency_weight if args.noise_training else 0,
        "noise_recipe": ".5clean+.5noisy target losses + weight L1(noisy,stopgrad(clean)); independentCUDA RNG2033"
        if args.noise_training else None})
    recipe["scene_context"] = architecture.get("scene_context")
    if args.decoder != "parent":
        recipe["inverse_architecture"] = architecture
    start_step = 0
    if args.resume == "auto" and latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["recipe"] != recipe:
            raise ValueError("Resume recipe mismatch")
        model.load_state_dict(saved["models"]["G_night_day"])
        optimizer.load_state_dict(saved["optimizer"])
        if discriminator is not None:
            discriminator.load_state_dict(saved["discriminator"])
            discriminator_optimizer.load_state_dict(saved["discriminator_optimizer"])
            gan_random.setstate(saved["gan_python_rng"])
        shadow = {k: v.cuda() for k, v in saved["ema"]["G_night_day"]["shadow"].items()}
        torch.set_rng_state(saved["rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        if real_regions:
            region_random.setstate(saved["region_python_rng"])
        if args.noise_training:
            noise_random.set_state(saved["noise_cuda_rng"].cpu())
        start_step = saved["adaptation_step"]
    atomic_json_dump(recipe, args.output / "recipe.json")
    cached = {r["input"]: read_pair(r) for r in train + val}
    probe = copy.deepcopy(model).eval().requires_grad_(False)

    def save(step):
        payload = {"step": parent_step + step, "adaptation_step": step, "config": config,
            "direction": "night_to_day", "recipe": recipe, "models": {"G_night_day": model.state_dict()},
            "inverse_architecture": architecture,
            "ema": {"G_night_day": {"decay": 0.99, "shadow": shadow}}, "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate(),
            "source_code_sha256_at_save": digest(Path(__file__))}
        if discriminator is not None:
            payload.update({"discriminator": discriminator.state_dict(),
                "discriminator_optimizer": discriminator_optimizer.state_dict(),
                "gan_python_rng": gan_random.getstate()})
        if real_regions:
            payload["region_python_rng"] = region_random.getstate()
        if args.noise_training:
            payload["noise_cuda_rng"] = noise_random.get_state()
        atomic_torch_save(payload, latest)
        if step > 0 and step % 250 == 0:
            destination = args.output / f"adaptation_{step:05d}.pt"
            if not destination.exists():
                atomic_torch_save(payload, destination)

    @torch.inference_mode()
    def validate(step):
        probe.load_state_dict(shadow)
        folder = args.output / f"validation_{step:05d}"; folder.mkdir(exist_ok=True)
        values = []; panels = []
        for row in val:
            x, y = cached[row["input"]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = probe(x[None].cuda()).float()
            values.append(float(pilot_loss(prediction, y[None].cuda())))
            result = tensor_to_pil(prediction)
            result.save(folder / f"val_{row['index']:04d}.png")
            panel = Image.new("RGB", (1536, 312), "#101010")
            for j, (label, im) in enumerate([("Night source", tensor_to_pil(x)), ("EXTERNAL teacher", tensor_to_pil(y)), (f"LOCAL inverse EMA +{step}", result)]):
                panel.paste(im, (512 * j, 24)); ImageDraw.Draw(panel).text((512 * j + 5, 5), f"VAL{row['index']} {label}", fill="white")
            panels.append(panel)
        for start in range(0, len(panels), 4):
            page = panels[start:start + 4]
            sheet = Image.new("RGB", (1536, 312 * len(page)), "#101010")
            for j, panel in enumerate(page):sheet.paste(panel, (0, 312 * j))
            sheet.save(folder / f"comparison_{start:02d}.jpg", quality=95)
        metrics = {"adaptation_step": step, "global_step": parent_step + step,
            "mean_teacher_fidelity": float(np.mean(values)), "per_image": values,
            "warning": "Pseudo-target agreement, NOT visual daylight success; open the saved images."}
        atomic_json_dump(metrics, folder / "metrics.json")
        if real_regions:
            diagnostic = folder / "REAL_STATIC_TRAINING_ONLY"
            diagnostic.mkdir(exist_ok=True)
            region_values = []
            for index, x, y, confidence in real_regions:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = probe(x[None].cuda()).float()
                region_values.append(float(static_region_rgb_loss(prediction, y[None].cuda(), confidence[None].cuda())))
                panel = Image.new("RGB", (2048, x.shape[-2] + 24), "#101010")
                mask = Image.fromarray(np.rint(confidence[0].numpy() * 255).astype(np.uint8)).convert("RGB")
                source_label, target_label = region_labels[index]
                for j, (label, image) in enumerate(((source_label, tensor_to_pil(x)),
                    (target_label, tensor_to_pil(y)),
                    (f"LOCAL inverse global{parent_step + step}", tensor_to_pil(prediction)), ("Static supervision ONLY", mask))):
                    panel.paste(image, (512 * j, 24))
                    ImageDraw.Draw(panel).text((512 * j + 5, 5), f"TRAIN PAIR{index} {label}", fill="white")
                panel.save(diagnostic / f"pair_{index:04d}.jpg", quality=96)
            atomic_json_dump({"split": "TRAINING ONLY; not generalization", "mean_static_rgb_l1": float(np.mean(region_values)),
                "per_image": region_values}, diagnostic / "metrics.json")
        if args.diagnostic_train_count:
            diagnostic = folder / "TRAINING_ONLY_diagnostic"
            diagnostic.mkdir(exist_ok=True)
            diagnostic_values = []
            for number, row in enumerate(train[:args.diagnostic_train_count]):
                x, y = cached[row["input"]]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = probe(x[None].cuda()).float()
                diagnostic_values.append(float(pilot_loss(prediction, y[None].cuda())))
                panel = Image.new("RGB", (1536, 312), "#101010")
                for j, (label, im) in enumerate([
                    ("TRAIN input", tensor_to_pil(x)),
                    ("TRAIN target (origin in manifest)", tensor_to_pil(y)),
                    (f"TRAIN local inverse global{parent_step + step}", tensor_to_pil(prediction)),
                ]):
                    panel.paste(im, (512 * j, 24))
                    ImageDraw.Draw(panel).text((512 * j + 5, 5), label, fill="white")
                panel.save(diagnostic / f"train_{number:02d}.jpg", quality=96)
            atomic_json_dump({"split": "TRAIN diagnostic, NOT validation",
                "mean_loss": float(np.mean(diagnostic_values)), "per_image": diagnostic_values,
                "indices": [r["index"] for r in train[:args.diagnostic_train_count]]},
                diagnostic / "metrics.json")
        return metrics["mean_teacher_fidelity"]

    validate(start_step)
    started = time.monotonic(); completed = start_step
    try:
        for step in range(start_step + 1, args.steps + 1):
            pairs = [cached[random.choice(train)["input"]] for _ in range(args.batch_size)]
            x = torch.stack([p[0] for p in pairs]).cuda(); y = torch.stack([p[1] for p in pairs]).cuda()
            if random.random() < 0.5:x, y = x.flip(-1), y.flip(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):prediction = model(x)
            loss = pilot_loss(prediction, y)
            noise_consistency = torch.zeros((), device="cuda")
            if args.noise_training:
                noisy_x = perturb_training_input(x, args.noise_sigma, noise_random)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    noisy_prediction = model(noisy_x)
                loss, noise_consistency = noise_robust_loss(prediction, noisy_prediction, y,
                    args.noise_consistency_weight)
            if args.target_reliability == "retained_glare":
                loss = reliability_weighted_loss(prediction, y, glare_reliability(x, y))
            region_loss = torch.zeros((), device="cuda")
            if real_regions:
                _, rx, ry, rw = region_random.choice(real_regions)
                rx, ry, rw = rx[None].cuda(), ry[None].cuda(), rw[None].cuda()
                if region_random.random() < .5:
                    rx, ry, rw = rx.flip(-1), ry.flip(-1), rw.flip(-1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    region_prediction = model(rx)
                region_loss = (static_region_coarse_loss if args.real_region_loss == "coarse"
                    else static_region_rgb_loss)(region_prediction, ry, rw)
                loss = loss + args.real_region_weight * region_loss
            gan_loss = torch.zeros((), device="cuda")
            d_loss = torch.zeros((), device="cuda")
            if discriminator is not None:
                discriminator.requires_grad_(False).eval()
                fake_small = F.interpolate(prediction, (144, 256), mode="area")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    gan_loss = adversarial_lsgan(discriminator(fake_small), True)
                loss = loss + args.gan_weight * min(step / 250, 1) * gan_loss
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite pilot loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            if not torch.isfinite(norm):raise FloatingPointError("Nonfinite pilot gradients")
            optimizer.step()
            if discriminator is not None:
                discriminator.requires_grad_(True).train()
                discriminator_optimizer.zero_grad(set_to_none=True)
                real = torch.stack(gan_random.choices(real_days, k=args.batch_size)).cuda()
                if gan_random.random() < .5:
                    real = real.flip(-1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    d_loss = .5 * (adversarial_lsgan(discriminator(real), True)
                        + adversarial_lsgan(discriminator(fake_small.detach()), False))
                if not torch.isfinite(d_loss):
                    raise FloatingPointError("Nonfinite discriminator loss")
                d_loss.backward()
                d_norm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 3)
                if not torch.isfinite(d_norm):
                    raise FloatingPointError("Nonfinite discriminator gradients")
                discriminator_optimizer.step()
            with torch.no_grad():
                for key, value in model.state_dict().items():shadow[key].lerp_(value, 0.01)
            completed = step
            if step % 25 == 0:
                record = {"adaptation_step": step, "global_step": parent_step + step, "loss": float(loss.detach()),
                    "gan_loss": float(gan_loss.detach()), "discriminator_loss": float(d_loss.detach()),
                    "real_static_region_loss": float(region_loss.detach()),
                    "synthetic_noise_consistency": float(noise_consistency.detach()),
                    "elapsed_seconds": time.monotonic() - started, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
                print(json.dumps(record), flush=True)
                with (args.output / "training.jsonl").open("a", encoding="utf-8") as handle:handle.write(json.dumps(record) + "\n")
            if step % 50 == 0:save(step)
            if step % 250 == 0:validate(step)
            if (time.monotonic() - started) / 60 >= args.max_minutes:break
    finally:
        save(completed)
    validate(completed)
    atomic_json_dump({"state": "stopped", "global_step": parent_step + completed, "adaptation_step": completed,
        "checkpoint": str(latest), "vision_review": "pending", "public_demo_unchanged": True}, args.output / "status.json")


if __name__ == "__main__":
    main()
