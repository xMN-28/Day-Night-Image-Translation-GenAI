"""Bounded EXTERNAL masked SD-Turbo restoration of existing local inverse outputs.

Automatic identical-rule glare masks; no manual per-example masks, training,
downloads, or production routing changes. Not an inpainting-trained checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .turbo_pilot import REVISION, digest
from .utils import atomic_json_dump


def glare_mask(source: Image.Image) -> Image.Image:
    """Conservative image-space glare support from original night pixels only."""
    array = np.asarray(source.convert("RGB"), dtype=np.float32) / 255
    luminance = array @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    core = (array.max(2) > 0.90) & (luminance > 0.48)
    dilated = (
        np.asarray(Image.fromarray(core.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(41)))
        > 0
    )
    candidate = dilated & (luminance > 0.12)
    expanded = Image.fromarray(candidate.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(13))
    soft = np.asarray(expanded.filter(ImageFilter.GaussianBlur(3))).copy()
    # Bound blur support and eliminate tiny nonzero tails for exact outside checks.
    soft[soft < 8] = 0
    return Image.fromarray(soft)


def preserve_outside(base: Image.Image, proposal: Image.Image, mask: Image.Image):
    if proposal.size != base.size or mask.size != base.size:
        raise ValueError("Matched image/mask sizes required")
    b, p = np.asarray(base.convert("RGB")), np.asarray(proposal.convert("RGB"))
    weights = np.asarray(mask.convert("L"), dtype=np.float32)[..., None] / 255
    result = np.rint(b.astype(np.float32) * (1 - weights) + p * weights).astype(np.uint8)
    outside = weights[..., 0] == 0
    assert np.array_equal(result[outside], b[outside])
    return Image.fromarray(result)


def main():
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("E:/LumiCycleCache/teacher_targets/night2day_pilot128_v2"),
    )
    parser.add_argument(
        "--local-outputs",
        type=Path,
        default=Path(
            "E:/LumiCycleCache/experiments/night2day_diverse111_transpose1000/validation_02000"
        ),
    )
    parser.add_argument("--indices", default="0,12,105,142,355,441,78,283")
    parser.add_argument("--cache-root", type=Path, default=Path("E:/LumiCycleCache"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior experiments; use a new output directory")
    torch.set_num_threads(6)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifest_path = args.targets / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["direction"] != "night_to_day" or manifest["test_used"]:
        raise ValueError("Only experimental validation-night cache allowed")
    records = {row["index"]: row for row in manifest["records"] if row["split"] == "val"}
    indices = [int(value) for value in args.indices.split(",")]
    if not indices or len(set(indices)) != len(indices) or any(i not in records for i in indices):
        raise ValueError("Unique development validation indices required; lockbox excluded")
    snapshot = (
        args.cache_root / f"huggingface/hub/models--stabilityai--sd-turbo/snapshots/{REVISION}"
    )
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        str(snapshot),
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to("cuda")
    if pipeline.unet.config.in_channels != 4:
        raise ValueError("This pilot explicitly audits the cached four-channel base UNet")
    pipeline.enable_attention_slicing()
    pipeline.set_progress_bar_config(disable=True)
    args.output.mkdir(parents=True)
    prompt = (
        "A realistic daytime dashcam photograph of the same street, natural daylight, "
        "neutral colors, clear visibility, unlit streetlights, ordinary buildings and road"
    )
    report = {
        "method": "EXTERNAL SD-Turbo four-channel masked latent restoration over local inverse16000",
        "external_pretrained": True,
        "not_inpainting_trained_weights": True,
        "base_snapshot_revision": REVISION,
        "test_used": False,
        "lockbox_used": False,
        "targets_manifest_sha256": digest(manifest_path),
        "prompt": prompt,
        "guidance_scale": 0.0,
        "steps": 4,
        "strengths": [0.5, 0.85],
        "seed_base": 22000,
        "public_demo_unchanged": True,
        "mask_recipe": "source maxRGB>.9 and luma>.48; dilate41, luma>.12; dilate13; blur3; floor8",
        "records": [],
    }
    for index in indices:
        row = records[index]
        source = Image.open(row["input"]).convert("RGB")
        base_path = args.local_outputs / f"val_{index:04d}.png"
        base = Image.open(base_path).convert("RGB")
        mask = glare_mask(source)
        folder = args.output / f"val_{index:04d}"
        folder.mkdir()
        source.save(folder / "source.png")
        base.save(folder / "local16000.png")
        mask.save(folder / "automatic_glare_mask.png")
        columns = [
            ("Night source", source),
            ("LOCAL inverse16000", base),
            ("Automatic allowed-edit mask", mask.convert("RGB")),
        ]
        item = {
            "index": index,
            "source_sha256": digest(Path(row["input"])),
            "local_output_sha256": digest(base_path),
            "mask_support_fraction": float((np.asarray(mask) > 0).mean()),
            "variants": [],
        }
        for strength in (0.5, 0.85):
            generator = torch.Generator(device="cuda").manual_seed(22000 + index)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                generated = pipeline(
                    prompt=prompt,
                    image=base,
                    mask_image=mask,
                    width=base.width,
                    height=base.height,
                    strength=strength,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    generator=generator,
                ).images[0]
            result = preserve_outside(base, generated, mask)
            name = f"EXTERNAL_masked_strength{strength}"
            generated.save(folder / f"{name}_uncomposited.png")
            result.save(folder / f"{name}.png")
            outside = np.asarray(mask) == 0
            error = np.abs(np.asarray(result).astype(int) - np.asarray(base).astype(int))
            outside_error = int(error[outside].max()) if outside.any() else 0
            peak = torch.cuda.max_memory_allocated()
            if peak > 11.5 * 1024**3:
                raise RuntimeError("Pilot exceeded 11.5 GiB allocated VRAM ceiling")
            item["variants"].append(
                {
                    "strength": strength,
                    "outside_max_uint8_error": outside_error,
                    "peak_allocated_bytes": peak,
                }
            )
            columns.append((f"EXTERNAL masked {strength}", result))
        panel = Image.new("RGB", (512 * len(columns), 318), "#101010")
        draw = ImageDraw.Draw(panel)
        for column, (label, image) in enumerate(columns):
            panel.paste(image, (column * 512, 30))
            draw.text((column * 512 + 5, 7), f"VAL{index} {label}", fill="white")
        panel.save(args.output / f"comparison_{index:04d}.jpg", quality=96)
        report["records"].append(item)
        atomic_json_dump(report, args.output / "evidence.json")
        print(
            f"Completed {index}: support {item['mask_support_fraction']:.3f}; outside exact",
            flush=True,
        )
    print(
        f"Review all {len(indices)} panels; no quality acceptance inferred from masking.",
        flush=True,
    )


if __name__ == "__main__":
    main()
