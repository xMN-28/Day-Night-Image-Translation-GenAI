"""Private experimental night→day benchmark; never changes public app scope.

Use validation only for iterative review. External output is always labeled.
No training is performed by this command. All variants use identical source IDs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from .data import read_manifest
from .inference import _fit_and_pad
from .inverse_scene_context import SceneContextInverse
from .models import ResnetGenerator, build_models
from .train_resize_decoder import adapt_decoder
from .turbo_pilot import digest, fit_input, load_official
from .utils import atomic_json_dump, pil_to_tensor, tensor_to_pil

INDICES = (0, 71, 142, 213, 284, 355, 426, 499)
BASELINES = {
    "v1_13k_inverse": Path("runs/lumicycle_bdd100k/checkpoints/step_00013000.pt"),
    "v2_4500_inverse": Path("runs/lumicycle_v2_bdd100k/checkpoints/step_00004500.pt"),
}


def freeze_suite(destination: Path):
    """Freeze development/lockbox IDs before candidate adaptation; no test access."""
    path = destination / "suite.json"
    if path.exists():
        raise FileExistsError("Frozen suite already exists; retain membership")
    records = read_manifest("data/bdd100k_daynight", "val", "night")
    remaining = [i for i in range(len(records)) if i not in INDICES]
    random.Random(2030).shuffle(remaining)
    selections = {"development": [*INDICES, *remaining[:16]], "lockbox": remaining[16:28]}
    destination.mkdir(parents=True, exist_ok=True)
    result = {"seed": 2030, "split": "validation night", "test_used": False, "groups": {}}
    for group, indices in selections.items():
        result["groups"][group] = [
            {"index": i, "path": str(records[i].path), "sha256": digest(records[i].path),
             "difficulty": "pending source-only visual tagging"} for i in indices
        ]
    atomic_json_dump(result, path)
    sheet = Image.new("RGB", (4 * 320, 6 * 204), "#101010")
    draw = ImageDraw.Draw(sheet)
    for j, i in enumerate(selections["development"]):
        image = Image.open(records[i].path).convert("RGB")
        image.thumbnail((320, 180), Image.Resampling.LANCZOS)
        x, y = (j % 4) * 320, (j // 4) * 204
        sheet.paste(image, (x, y + 24))
        draw.text((x + 5, y + 5), f"VAL night {i}", fill="white")
    sheet.save(destination / "development_sources.jpg", quality=95)
    print(f"Frozen24 development /12 lockbox at {path}; lockbox not rendered")


def visibility_control(image: Image.Image) -> Image.Image:
    """Gamma .45 visibility control: NOT learned day translation or a proposed fix."""
    x = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    return Image.fromarray(np.rint(np.clip(x, 0, 1) ** 0.45 * 255).astype(np.uint8))


def inverse_from_payload(payload, decoder=None):
    """Experimental-only decoder option; production factory/defaults unchanged."""
    options = payload["config"]["model"]
    original_mode = payload.get("inverse_architecture", {}).get("upsampling", "transpose")
    mode = original_mode if decoder in {None, "parent"} else decoder
    if mode not in {"transpose", "resize"}:
        raise ValueError("Unknown inverse decoder")
    weights = payload["ema"]["G_night_day"]["shadow"]
    if options.get("detail_refinement"):
        if mode != "transpose":
            raise ValueError("Resize conversion is not implemented for old refinement ablations")
        model = build_models(payload["config"])["G_night_day"]
    else:
        context = payload.get("inverse_architecture", {}).get("scene_context")
        factory = SceneContextInverse if context else ResnetGenerator
        context_args = ({"context_width": context["width"], "context_grid": context["grid"],
            "context_heads": context["heads"]} if context else {})
        model = factory(base_channels=options.get("base_channels", 64),
            blocks=options.get("generator_blocks", 9), attention=options.get("attention", False),
            upsampling=mode, **context_args)
    if mode != original_mode:
        if (original_mode, mode) != ("transpose", "resize"):
            raise ValueError("Only explicit transpose→resize warm-start conversion supported")
        weights = adapt_decoder(weights)
    model.load_state_dict(weights)
    architecture = {"upsampling": mode, "parent_upsampling": original_mode,
        "converted": mode != original_mode,
        "conversion": "transpose+spatial flip kernels ×.25; approximate, NOT function-equivalent" if mode != original_mode else "none"}
    if payload.get("inverse_architecture", {}).get("scene_context"):
        architecture["scene_context"] = payload["inverse_architecture"]["scene_context"]
    return model, architecture


def load_inverse(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, _ = inverse_from_payload(payload)
    return model.cuda().eval().requires_grad_(False), payload["step"]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-turbo", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=Path("E:/LumiCycleCache"))
    parser.add_argument("--edge", type=int, default=512)
    parser.add_argument("--indices", default=",".join(map(str, INDICES)))
    parser.add_argument("--freeze-suite", action="store_true")
    args = parser.parse_args()
    if args.freeze_suite:
        freeze_suite(args.output)
        return
    if (args.output / "evidence.json").exists():
        raise FileExistsError("Use a new output folder; do not overwrite previous reviewed evidence")
    if not torch.cuda.is_available():
        raise RuntimeError("This bounded experimental pilot requires CUDA")
    torch.set_num_threads(6)
    args.output.mkdir(parents=True, exist_ok=True)
    records = read_manifest("data/bdd100k_daynight", "val", "night")
    indices = tuple(int(value) for value in args.indices.split(","))
    if not indices or len(indices) != len(set(indices)) or any(i < 0 or i >= len(records) for i in indices):
        raise ValueError("Provide unique, in-range validation indices")
    evidence = {"created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "split": "validation night", "indices": list(indices), "test_used": False, "edge": args.edge,
        "public_demo_unchanged": True, "variants": {}, "sources": [], "vision_review_status": "pending"}
    images = []
    for index in indices:
        record = records[index]
        source = ImageOps.exif_transpose(Image.open(record.path)).convert("RGB")
        source.thumbnail((args.edge, args.edge), Image.Resampling.LANCZOS)
        folder = args.output / f"val_{index:04d}"
        folder.mkdir(exist_ok=True)
        source.save(folder / "source.png")
        visibility_control(source).save(folder / "gamma_visibility_NOT_day_translation.png")
        images.append((index, source, folder))
        evidence["sources"].append({"index": index, "path": str(record.path), "sha256": digest(record.path)})
    for name, path in BASELINES.items():
        model, step = load_inverse(path)
        evidence["variants"][name] = {"checkpoint": str(path), "sha256": digest(path),
            "step": step, "weights": "EMA G_night_day", "precision": "BF16", "external_pretrained": False}
        for index, source, folder in images:
            padded, crop = _fit_and_pad(source, args.edge)
            x = pil_to_tensor(padded)[None].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x)
            if not torch.isfinite(pred).all():
                raise FloatingPointError(f"Nonfinite inverse output at {index}")
            tensor_to_pil(pred).crop(crop).save(folder / f"{name}.png")
        del model
        torch.cuda.empty_cache()
    if args.external_turbo:
        model, checkpoint, commit = load_official(args.cache_root, direction="night_to_day")
        evidence["variants"]["EXTERNAL_Turbo_night2day"] = {"checkpoint": str(checkpoint),
            "sha256": digest(checkpoint), "source_commit": commit, "precision": "FP16", "seed_base": 4400,
            "external_pretrained": True, "teacher_pretraining_overlap": "unknown"}
        for index, source, folder in images:
            _, x, crop = fit_input(source, args.edge)
            torch.manual_seed(4400 + index)
            torch.cuda.manual_seed_all(4400 + index)
            pred = model(x.cuda().half())
            if not torch.isfinite(pred).all():
                raise FloatingPointError(f"Nonfinite external output at {index}")
            tensor_to_pil(pred).crop(crop).save(folder / "EXTERNAL_Turbo_night2day.png")
    labels = [("Source night", "source"), ("Gamma only — NOT day", "gamma_visibility_NOT_day_translation"),
              ("Our V1 inverse EMA13k", "v1_13k_inverse"), ("Our V2 inverse EMA4.5k", "v2_4500_inverse")]
    if args.external_turbo:
        labels.append(("EXTERNAL Turbo night2day", "EXTERNAL_Turbo_night2day"))
    for start in range(0, len(images), 4):
        batch = images[start:start + 4]
        canvas = Image.new("RGB", (512 * len(labels), 320 * len(batch)), "#101010")
        draw = ImageDraw.Draw(canvas)
        for row, (index, source, folder) in enumerate(batch):
            for column, (label, filename) in enumerate(labels):
                im = Image.open(folder / f"{filename}.png").convert("RGB")
                im.thumbnail((512, 288))
                canvas.paste(im, (column * 512, row * 320 + 32))
                draw.text((column * 512 + 6, row * 320 + 8), f"VAL {index}: {label}", fill="white")
        canvas.save(args.output / f"comparison_{start:02d}.jpg", quality=95)
    atomic_json_dump(evidence, args.output / "evidence.json")
    print(json.dumps({"output": str(args.output), "sources": len(images), "variants": list(evidence["variants"])}, indent=2))


if __name__ == "__main__":
    main()
