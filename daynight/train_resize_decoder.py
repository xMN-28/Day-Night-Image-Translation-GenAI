"""Isolated warm-start decoder distillation from a trained V2 generator.

Keep encoder/attention exactly frozen. Distill phase-averaged teacher RGB, not
source edges, to avoid teaching the student to keep daytime lighting.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .models.networks import ResnetGenerator
from .sky_refinement import phase_average
from .train_sky_student import CachedSkyDataset
from .utils import atomic_json_dump, atomic_torch_save, seed_everything, tensor_to_pil


def state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def adapt_decoder(state):
    """Explicit approximate kernel initialization, NOT functional equivalence."""
    result = {key: value.clone() for key, value in state.items()}
    for key in ("up1.0.weight", "up2.0.weight"):
        result[key] = state[key].transpose(0, 1).flip(-1, -2).contiguous() * 0.25
    return result


def decoder_loss(prediction, target):
    rgb = F.l1_loss(prediction, target)
    coarse = (
        sum(F.l1_loss(F.avg_pool2d(prediction, s), F.avg_pool2d(target, s)) for s in (4, 16)) / 2
    )
    gradient = (
        sum(F.l1_loss(torch.diff(prediction, dim=d), torch.diff(target, dim=d)) for d in (-1, -2))
        / 2
    )
    return 10 * rgb + 2 * coarse + gradient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent", type=Path, default=Path("runs/lumicycle_v2_skygan_pilot/latest.pt")
    )
    parser.add_argument("--output", type=Path, default=Path("runs/resize_decoder_pilot"))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--train-images", type=int, default=300)
    parser.add_argument("--max-minutes", type=float, default=35)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(2028)
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = True
    parent_bytes = args.parent.read_bytes()
    import io

    parent = torch.load(io.BytesIO(parent_bytes), map_location="cpu", weights_only=False)
    if parent["config"]["model"].get("day_upsampling", "transpose") != "transpose":
        raise ValueError("This experiment expects an original transpose decoder parent")
    parent_hash = hashlib.sha256(parent_bytes).hexdigest()
    del parent_bytes
    cfg = copy.deepcopy(parent["config"])
    cfg["model"]["day_upsampling"] = "resize"
    cfg["experiment"]["output_dir"] = str(args.output)
    cfg["experiment"]["name"] = "resize_decoder_distillation"
    cfg["train"]["max_steps"] = parent["step"] + args.steps
    options = cfg["model"]
    kwargs = {
        "base_channels": options["base_channels"],
        "blocks": options["generator_blocks"],
        "attention": options["attention"],
    }
    old_state = parent["ema"]["G_day_night"]["shadow"]
    teacher = ResnetGenerator(**kwargs).cuda().eval().requires_grad_(False)
    teacher.load_state_dict(old_state)
    model = ResnetGenerator(**kwargs, upsampling="resize").cuda()
    model.load_state_dict(adapt_decoder(old_state))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("up1.", "up2.", "head.")))
    frozen_keys = [k for k in old_state if not k.startswith(("up1.", "up2.", "head."))]
    frozen_hash = state_digest({k: old_state[k] for k in frozen_keys})
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, betas=(0.5, 0.999)
    )
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train = CachedSkyDataset(Path("data/sky_student_fullframe_v1"), "train")
    val = CachedSkyDataset(Path("data/sky_student_fullframe_v1"), "val")
    ids = np.linspace(0, len(train) - 1, min(args.train_images, len(train)), dtype=int).tolist()
    settings = {
        "parent_sha256": parent_hash,
        "train_indices": ids,
        "target": "16-phase-averaged trained generator; no test images",
        "architecture": "nearest resize + reflection pad + conv; frozen encoder",
        "frozen_sha256": frozen_hash,
        "resolution": [384, 216],
        "seed": 2028,
    }
    cache = args.output / "targets"
    cache.mkdir(exist_ok=True)
    manifest = cache / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError("Target cache configuration changed; choose a new output directory")
    atomic_json_dump(settings, manifest)
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for split, dataset, indices in (("train", train, ids), ("val", val, list(range(len(val))))):
            for j, index in enumerate(indices):
                path = cache / f"{split}_{index:05d}.pt"
                if not path.exists():
                    x, _ = dataset[index]
                    target = phase_average(teacher, x[None].cuda() * 2 - 1)
                    atomic_torch_save(target[0].half().cpu(), path)
                if j % 25 == 0:
                    print(
                        json.dumps(
                            {"phase": "cache", "split": split, "images": j, "total": len(indices)}
                        ),
                        flush=True,
                    )
                if (args.output / "STOP").exists():
                    print(
                        "Stopped safely during target preparation; rerun to continue cache",
                        flush=True,
                    )
                    return
    del teacher
    torch.cuda.empty_cache()
    step, best = 0, float("inf")
    latest = args.output / "latest.pt"
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["distillation"] != settings:
            raise ValueError("Resume provenance does not match")
        model.load_state_dict(saved["models"]["G_day_night"])
        ema = {k: v.cuda() for k, v in saved["ema"]["G_day_night"]["shadow"].items()}
        optimizer.load_state_dict(saved["optimizer"])
        step, best = saved["adaptation_step"], saved["best_loss"]
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])

    def save(name):
        current_frozen = {k: model.state_dict()[k] for k in frozen_keys}
        if state_digest(current_frozen) != frozen_hash:
            raise RuntimeError("Frozen pretrained encoder changed")
        models = dict(parent["models"])
        models["G_day_night"] = model.state_dict()
        ema_states = dict(parent["ema"])
        ema_states["G_day_night"] = {"decay": 0.99, "shadow": ema}
        atomic_torch_save(
            {
                "step": parent["step"] + step,
                "adaptation_step": step,
                "best_loss": best,
                "parent": str(args.parent),
                "config": cfg,
                "models": models,
                "ema": ema_states,
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(),
                "distillation": settings,
            },
            args.output / name,
        )

    def validate():
        probe = ResnetGenerator(**kwargs, upsampling="resize").cuda().eval()
        probe.load_state_dict(ema)
        losses = []
        folder = args.output / "previews" / f"step_{step:05d}"
        folder.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(len(val)):
                x, _ = val[i]
                target = torch.load(cache / f"val_{i:05d}.pt", weights_only=True).cuda()[None]
                pred = probe(x[None].cuda() * 2 - 1).float()
                losses.append(float(decoder_loss(pred, target)))
                if i in (0, 6, 13, 20, 27, 34, 40, 47):
                    sheet = Image.new("RGB", (1152, 240), "#101010")
                    draw = ImageDraw.Draw(sheet)
                    for j, (im, label) in enumerate(
                        zip(
                            (x * 2 - 1, target, pred),
                            ("Input", "Phase-averaged teacher", f"Resize decoder {step}"),
                            strict=True,
                        )
                    ):
                        draw.text((j * 384 + 4, 4), label, fill="white")
                        sheet.paste(tensor_to_pil(im), (j * 384, 24))
                    sheet.save(folder / f"val_{i:02d}.jpg", quality=95)
        del probe
        return sum(losses) / len(losses)

    start_step = step
    completed_step = step
    try:
        for step in range(start_step + 1, args.steps + 1):
            index = random.choice(ids)
            x, _ = train[index]
            target = torch.load(cache / f"train_{index:05d}.pt", weights_only=True)
            if random.random() < 0.5:
                x, target = x.flip(-1), target.flip(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(x[None].cuda() * 2 - 1).float()
                loss = decoder_loss(prediction, target[None].cuda().float())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite decoder loss; prior checkpoint retained")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite decoder gradient")
            optimizer.step()
            with torch.no_grad():
                for k, value in model.state_dict().items():
                    if k in frozen_keys:
                        ema[k].copy_(value)
                    else:
                        ema[k].lerp_(value, 0.01)
            completed_step = step
            if step % 25 == 0:
                status = {
                    "phase": "training",
                    "adaptation_step": step,
                    "global_step": parent["step"] + step,
                    "loss": float(loss.detach()),
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
                atomic_json_dump(status, args.output / "status.json")
                print(json.dumps(status), flush=True)
            if step % 200 == 0:
                value = validate()
                if value < best:
                    best = value
                    save("best.pt")
                save("latest.pt")
                print(
                    json.dumps(
                        {
                            "phase": "validation",
                            "adaptation_step": step,
                            "loss": value,
                            "best": best,
                        }
                    ),
                    flush=True,
                )
            if (
                args.output / "STOP"
            ).exists() or time.monotonic() - started > args.max_minutes * 60:
                break
    except KeyboardInterrupt:
        step = completed_step
    save("latest.pt")
    atomic_json_dump(
        {
            "phase": "finished" if step == args.steps else "stopped",
            "adaptation_step": step,
            "global_step": parent["step"] + step,
            "best_validation": best,
        },
        args.output / "status.json",
    )


if __name__ == "__main__":
    main()
