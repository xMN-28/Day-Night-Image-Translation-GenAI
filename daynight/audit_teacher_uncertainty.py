"""Validation-only external-teacher sampling audit; no target caches overwritten."""

import argparse
import json
from pathlib import Path
from types import MethodType

import torch
from PIL import Image, ImageDraw

from .turbo_pilot import REVISION, SOURCE_COMMIT, digest, fit_input, load_official
from .utils import atomic_json_dump, tensor_to_pil


def variation_metrics(samples):
    """Samples are NCHW RGB in [0,1]. Never call this a physical realism metric."""
    if samples.ndim != 4 or samples.shape[0] < 2 or samples.shape[1] != 3:
        raise ValueError("Expected at least two NCHW RGB samples")
    differences, overlaps = [], []
    masks = samples.amax(1) > 0.7
    for i in range(len(samples)):
        for j in range(i):
            differences.append(float((samples[i] - samples[j]).abs().mean()))
            union = (masks[i] | masks[j]).sum()
            overlaps.append(float((masks[i] & masks[j]).sum() / union) if union else 1.0)
    return {
        "pairwise_rgb_l1_0to1": sum(differences) / len(differences),
        "pairwise_bright_mask_iou": sum(overlaps) / len(overlaps),
        "mean_rgb_standard_deviation": float(samples.std(0).mean()),
        "maximum_rgb_standard_deviation": float(samples.std(0).max()),
    }


def posterior_mode(self, x, direction):
    if direction not in ("a2b", "b2a"):
        raise ValueError("Unsupported teacher direction")
    vae = self.vae if direction == "a2b" else self.vae_b2a
    return vae.encode(x).latent_dist.mode() * vae.config.scaling_factor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("E:/LumiCycleCache"))
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("E:/LumiCycleCache/teacher_targets/turbo_road2048_native512"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/turbo_teacher_uncertainty"))
    args = parser.parse_args()
    torch.set_num_threads(6)
    manifest = json.loads((args.targets / "manifest.json").read_text())
    if manifest["test_images_used"]:
        raise ValueError("This diagnostic excludes project test images")
    rows = [r for r in manifest["records"] if r["split"] == "val"]
    model, checkpoint, _ = load_official(args.cache_root)
    original_forward = model.vae_enc.forward
    records, panels = [], []
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index in (0, 6, 13, 20, 27, 34, 40, 47):
            row = rows[index]
            source, x, crop = fit_input(Image.open(row["source"]), 512)
            x = x.cuda().half()
            outputs = []
            model.vae_enc.forward = original_forward
            for offset in range(4):
                torch.manual_seed(row["seed"] + offset)
                torch.cuda.manual_seed_all(row["seed"] + offset)
                y = model(x).float()[0]
                if not torch.isfinite(y).all():
                    raise FloatingPointError("Nonfinite teacher sample")
                outputs.append(((y + 1) / 2).clamp(0, 1))
            model.vae_enc.forward = MethodType(posterior_mode, model.vae_enc)
            modes = []
            for seed in (77, 989):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                y = model(x).float()[0]
                if not torch.isfinite(y).all():
                    raise FloatingPointError("Nonfinite teacher posterior-mode output")
                modes.append(((y + 1) / 2).clamp(0, 1))
            metrics = variation_metrics(torch.stack(outputs))
            metrics.update(
                {
                    "validation_position": index,
                    "source": row["source"],
                    "seeds": [row["seed"] + o for o in range(4)],
                    "mode_repeat_maximum_difference": float((modes[0] - modes[1]).abs().max()),
                    "mode_to_first_sample_rgb_l1": float((modes[0] - outputs[0]).abs().mean()),
                }
            )
            records.append(metrics)
            panel = Image.new("RGB", (512 * 6, 312), "#101010")
            draw = ImageDraw.Draw(panel)
            panel.paste(source, (0, 24))
            draw.text((4, 4), f"VAL {index}: Input", fill="white")
            for j, (label, value) in enumerate(
                [(f"EXTERNAL seed +{o}", output) for o, output in enumerate(outputs)]
                + [("EXTERNAL posterior mode ABLATION", modes[0])],
                start=1,
            ):
                draw.text((j * 512 + 4, 4), label, fill="white")
                panel.paste(tensor_to_pil(value * 2 - 1).crop(crop), (j * 512, 24))
            panel.save(args.output / f"val_{index:02d}.jpg", quality=95)
            panels.append(panel)
            print(json.dumps(metrics), flush=True)
    model.vae_enc.forward = original_forward
    gallery = Image.new("RGB", (3072, 312 * len(panels)))
    for i, panel in enumerate(panels):
        gallery.paste(panel, (0, i * 312))
    gallery.save(args.output / "gallery.jpg", quality=95)
    summary = {
        k: sum(row[k] for row in records) / len(records)
        for k in records[0]
        if k not in ("validation_position", "source", "seeds")
    }
    atomic_json_dump(
        {
            "teacher_external": True,
            "source_commit": SOURCE_COMMIT,
            "base_revision": REVISION,
            "checkpoint_sha256": digest(checkpoint),
            "split": "val",
            "test_images_used": False,
            "mode_variant": "Temporary in-process latent posterior-mode ablation, NOT official default forward. Pinned source and target caches unchanged.",
            "summary": summary,
            "records": records,
        },
        args.output / "metrics.json",
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
