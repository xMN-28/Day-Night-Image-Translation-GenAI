"""Extract genuine current-release tensors for the publication architecture figure.

Run from the repository root with the project PyTorch environment. No training,
external teacher download, generative illustration, or model modification occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from daynight.data import read_manifest
from daynight.inference import ModelManager, _fit_and_pad
from daynight.sky_refinement import box, gray, predict_matte
from daynight.utils import pil_to_tensor, tensor_to_pil

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "output/pdf/lumicycle_architecture"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_rgb(tensor: torch.Tensor, path: Path, crop: tuple[int, ...]) -> None:
    tensor_to_pil(tensor.float().clamp(0, 1) * 2 - 1).crop(crop).save(path)


def save_mask(tensor: torch.Tensor, path: Path, crop: tuple[int, ...]) -> None:
    array = (tensor[0, 0].float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).crop(crop).save(path)


def compositor_components(source, base, alpha, vegetation):
    """Expose the exact intermediates in compose_night(recover_thin_edges=True).

    Formula duplication is checked against the production output below. This
    function only inspects and visualizes the current computation.
    """
    a = alpha.clamp(0, 1)
    ratio = (box(gray(base), 15) / box(gray(source), 15).clamp_min(0.04)).clamp(0.12, 0.85)
    foreground = source * ratio.clamp_max(0.12)
    near_sky = box(F.max_pool2d(a, 25, 1, 12), 5).clamp(0, 1)
    support = near_sky if vegetation is None else torch.maximum(a, near_sky * vegetation)
    boundary = box(F.max_pool2d(a, 17, 1, 8), 1)
    support = torch.maximum(support, (boundary * 10).clamp(0, 1))
    y = torch.linspace(0, 1, source.shape[-2], device=source.device)[None, None, :, None]
    sky = source.new_tensor([0.026, 0.035, 0.055])[None, :, None, None]
    sky = sky + y * source.new_tensor([0.035, 0.037, 0.038])[None, :, None, None]
    texture = (gray(source) - box(gray(source), 15)).clamp(-0.2, 0.2)
    sky = (sky + texture * 0.06).clamp(0, 0.15)
    known = (a > 0.9).float()
    global_bg = (source * known).sum((2, 3), True) / known.sum((2, 3), True).clamp_min(1)
    daylight_bg = (box(source * known, 63) + 0.003 * global_bg) / (box(known, 63) + 0.003)
    old_background = daylight_bg * ratio.clamp_max(0.12)
    unmixed = (foreground - a * old_background).clamp_min(0)
    sky_contribution = a * sky
    result = (base * (1 - support) + (unmixed + sky_contribution) * support).clamp(0, 1)
    return {
        "repair_support": support,
        "unmixed_foreground": unmixed,
        "sky_field": sky,
        "sky_contribution": sky_contribution,
        "reconstructed": result,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--indices", type=int, nargs="+", default=[12, 142, 23])
    args = parser.parse_args()
    output = args.output.resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    torch.manual_seed(42)
    records = read_manifest(ROOT / "data/bdd100k_daynight", "val", "day")
    manager = ModelManager()
    model_name = "LumiCycle distilled student"
    rows = []
    for index in args.indices:
        record = records[index]
        with Image.open(record.path) as opened:
            image = opened.copy()
        # Compare against both actual public-manager paths, not a hand-painted output.
        raw_pil, raw_meta = manager.translate_custom(
            image, "day_to_night", model_name, 512, sky_repair=False
        )
        repaired_pil, metadata = manager.translate_custom(image, "day_to_night", model_name, 512)
        padded, crop = _fit_and_pad(image, 512)
        normalized = pil_to_tensor(padded).unsqueeze(0).to(manager.device)
        assert manager.models is not None and manager.sky_repair is not None
        generator = manager.models["G_day_night"]
        with torch.autocast(
            manager.device.type, dtype=torch.bfloat16, enabled=manager.device.type == "cuda"
        ):
            predicted = generator(normalized)
            source = (normalized.float() + 1) / 2
            alpha, vegetation = predict_matte(
                manager.sky_repair.student, source, recover_color_holes=True
            )
            low = F.interpolate(source, (216, 384), mode="bilinear", align_corners=False)
            probabilities = manager.sky_repair.student(low).float().sigmoid()
            probabilities = F.interpolate(
                probabilities, source.shape[-2:], mode="bilinear", align_corners=False
            )
        base = (predicted.float() + 1) / 2
        components = compositor_components(source, base, alpha.float(), vegetation)
        production, _ = manager.sky_repair.apply(normalized, predicted)
        # Mirror the production normalization round trip before PNG quantization.
        reproduced = components["reconstructed"] * 2 - 1
        tensor_error = float((reproduced - production).abs().max())
        reproduced_pil = tensor_to_pil(reproduced).crop(crop)
        pixel_error = int(
            np.abs(
                np.asarray(reproduced_pil).astype(int) - np.asarray(repaired_pil).astype(int)
            ).max()
        )
        assert tensor_error <= 1e-6 and pixel_error == 0, (tensor_error, pixel_error)
        prefix = f"val_{index:04d}"
        padded.crop(crop).save(assets / f"{prefix}_source.png")
        raw_pil.save(assets / f"{prefix}_raw.png")
        repaired_pil.save(assets / f"{prefix}_repaired.png")
        save_mask(probabilities[:, :1], assets / f"{prefix}_sky_probability.png", crop)
        save_mask(probabilities[:, 1:2], assets / f"{prefix}_vegetation_probability.png", crop)
        save_mask(alpha, assets / f"{prefix}_alpha.png", crop)
        for name, tensor in components.items():
            if name == "repair_support":
                save_mask(tensor, assets / f"{prefix}_{name}.png", crop)
            else:
                save_rgb(tensor, assets / f"{prefix}_{name}.png", crop)
        rows.append(
            {
                "validation_index": index,
                "source": str(record.path),
                "source_sha256": sha256(record.path),
                "public_metadata": metadata,
                "raw_metadata": raw_meta,
                "intermediate_max_tensor_error": tensor_error,
                "intermediate_max_uint8_error": pixel_error,
            }
        )
    # Same region in all three images: no inpainting, recoloring, or exposure boost.
    crop_box = (215, 0, 335, 112)
    for kind in ("source", "raw", "repaired"):
        with Image.open(assets / f"val_0012_{kind}.png") as image:
            image.crop(crop_box).save(assets / f"wire_crop_{kind}.png")
    contact = Image.new("RGB", (1536, len(rows) * 318), "white")
    draw = ImageDraw.Draw(contact)
    for row_index, record in enumerate(rows):
        for column, kind in enumerate(("source", "raw", "repaired")):
            with Image.open(assets / f"val_{record['validation_index']:04d}_{kind}.png") as picture:
                contact.paste(picture.convert("RGB"), (column * 512, row_index * 318 + 30))
            draw.text(
                (column * 512 + 6, row_index * 318 + 8),
                f"Validation {record['validation_index']:04d} / {kind}",
                fill="black",
            )
    contact.save(output / "asset_review.png")
    source_files = [
        "daynight/inference.py",
        "daynight/sky_repair.py",
        "daynight/sky_refinement.py",
        "daynight/models/networks.py",
        "daynight/train_teacher_student.py",
        "daynight/train_sky_student.py",
    ]
    provenance = {
        "scope": "Current day-to-night hybrid; validation illustrations, not a benchmark",
        "generated_illustrations_used": False,
        "test_images_used": False,
        "image_processing": "Only real tensors, PNG encoding and identical-coordinate crops",
        "device": str(manager.device),
        "torch": torch.__version__,
        "generator_parameters": sum(p.numel() for p in manager.models["G_day_night"].parameters()),
        "matte_parameters": sum(p.numel() for p in manager.sky_repair.student.parameters()),
        "generator_sha256": sha256(manager.checkpoint_path),
        "matte_sha256": sha256(manager.sky_repair.checkpoint),
        "source_code_hashes": {name: sha256(ROOT / name) for name in source_files},
        "detail_crop_xyxy": crop_box,
        "records": rows,
        "assets": {p.name: sha256(p) for p in sorted(assets.glob("*.png"))},
        "license_notice": "Contains BDD100K imagery. Local use; verify dataset terms before redistribution.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "examples": len(rows),
                "exact_public_path_match": True,
                "generator_step": rows[0]["public_metadata"]["global_step"],
                "matte_step": rows[0]["public_metadata"]["matte_step"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
