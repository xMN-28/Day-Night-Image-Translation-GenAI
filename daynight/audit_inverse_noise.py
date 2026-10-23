"""Frozen-checkpoint noise-sensitivity diagnostic; never trains or deploys."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .experiment_night_to_day import load_inverse
from .turbo_pilot import digest
from .utils import atomic_json_dump, pil_to_tensor, tensor_to_pil


def bilateral(image, spatial_sigma=1.0, range_sigma=0.04):
    """Deterministic 5x5 RGB joint-range filter for CHW sRGB in [0,1]."""
    if image.ndim != 3 or image.shape[0] != 3 or min(image.shape[-2:]) < 3:
        raise ValueError("Expected RGB CHW with sides >=3")
    if spatial_sigma <= 0 or range_sigma <= 0:
        raise ValueError("Positive filter sigmas required")
    h, w = image.shape[-2:]
    patches = F.unfold(F.pad(image[None].float(), (2, 2, 2, 2), mode="reflect"), 5)
    patches = patches.reshape(3, 25, h, w)
    axis = torch.arange(-2, 3, device=image.device).float()
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    spatial = torch.exp(-(xx.square() + yy.square()).reshape(25, 1, 1) / (2 * spatial_sigma ** 2))
    difference = (patches - image[:, None]).square().mean(0)
    weights = spatial * torch.exp(-difference / (2 * range_sigma ** 2))
    return (patches * weights[None]).sum(1) / weights.sum(0)[None]


def gaussian(image, sigma=0.6):
    axis = torch.arange(-2, 3, device=image.device).float()
    kernel = torch.exp(-axis.square() / (2 * sigma ** 2))
    kernel = kernel[:, None] * kernel[None, :]
    kernel /= kernel.sum()
    return F.conv2d(F.pad(image[None], (2, 2, 2, 2), mode="reflect"),
        kernel.expand(3, 1, 5, 5), groups=3)[0]


def validate_sources(manifest, suite):
    if manifest.get("test_used") or manifest.get("lockbox_used") or suite.get("test_used"):
        raise ValueError("Test and lockbox are forbidden")
    if manifest["direction"] != "night_to_day":
        raise ValueError("Wrong direction")
    rows = [r for r in manifest["records"] if r["split"] == "val"]
    expected = {r["index"]: r["sha256"] for r in suite["groups"]["development"]}
    if len(rows) != 24 or len(expected) != 24 or {r["index"] for r in rows} != set(expected):
        raise ValueError("Require exact frozen24 development suite")
    if any(r["source_sha256"] != expected[r["index"]] for r in rows):
        raise ValueError("Source provenance differs")
    return rows


def rms(x):
    return float(x.float().square().mean().sqrt())


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=Path("E:/LumiCycleCache/teacher_targets/night2day_pilot128_v2"))
    parser.add_argument("--suite", type=Path, default=Path("outputs/night_to_day_experimental/fixed_suite_v1/suite.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("E:/LumiCycleCache/experiments/night2day_diverse111_transpose1000/adaptation_02000.pt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep prior evidence")
    manifest = json.loads((args.targets / "manifest.json").read_text())
    rows = validate_sources(manifest, json.loads(args.suite.read_text()))
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = False
    model, step = load_inverse(args.checkpoint)
    args.output.mkdir(parents=True)
    panels, evidence = [], []

    def predict(image01, full_precision=False):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not full_precision):
            result = model(image01[None].cuda() * 2 - 1).float().cpu()[0]
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite result")
        return (result + 1).clamp(0, 2) / 2

    def pil(image01):
        return tensor_to_pil(image01 * 2 - 1)

    for row in rows:
        source = Image.open(row["input"]).convert("RGB")
        if source.size != (512, 288):
            raise ValueError("Fixed diagnostic requires512x288 cached images")
        x = (pil_to_tensor(source) + 1) / 2
        pre_bilateral, pre_gaussian = bilateral(x), gaussian(x)
        baseline = predict(x)
        repeated = predict(x)
        random = torch.Generator().manual_seed(2030 + row["index"])
        perturbed = (x + torch.randn(x.shape, generator=random) / 255).clamp(0, 1)
        perturbed_output = predict(perturbed)
        fp32_baseline = predict(x, full_precision=True)
        fp32_perturbed = predict(perturbed, full_precision=True)
        outputs = {"baseline": baseline, "pre_bilateral": predict(pre_bilateral),
            "pre_gaussian": predict(pre_gaussian), "post_bilateral_CONTROL": bilateral(baseline)}
        folder = args.output / f"val_{row['index']:04d}"
        folder.mkdir()
        source.save(folder / "source.png")
        pil(fp32_baseline).save(folder / "FP32_baseline.png")
        pil(pre_bilateral).save(folder / "INPUT_bilateral.png")
        pil(pre_gaussian).save(folder / "INPUT_gaussian.png")
        for label, array in outputs.items():
            pil(array).save(folder / f"{label}.png")
        delta = (perturbed_output - baseline).abs().mean(0)
        # Fixed visualization scale, NOT a learned confidence or semantic map.
        Image.fromarray((delta.clamp(0, .05) / .05 * 255).byte().numpy()).save(folder / "perturbation_delta_0to005.png")
        columns = [("Night source", source)] + [(key, pil(value)) for key, value in outputs.items()]
        panel = Image.new("RGB", (512 * len(columns), 312), "#101010")
        for j, (label, im) in enumerate(columns):
            panel.paste(im, (512 * j, 24))
            ImageDraw.Draw(panel).text((512 * j + 5, 5), f"VAL{row['index']} {label}", fill="white")
        panels.append(panel)
        crops = Image.new("RGB", (256 * len(columns), 216 * 3), "#101010")
        for k, box in enumerate(((192, 0, 320, 96), (0, 96, 128, 192), (192, 192, 320, 288))):
            for j, (label, im) in enumerate(columns):
                crops.paste(im.crop(box).resize((256, 192), Image.Resampling.NEAREST), (j * 256, k * 216 + 24))
                ImageDraw.Draw(crops).text((j * 256 + 4, k * 216 + 4), f"{label}: crop{k}", fill="white")
        crops.save(folder / "fixed_coordinate_crops.png")
        input_rms = rms(perturbed - x)
        evidence.append({"index": row["index"], "cached_source_sha256": digest(Path(row["input"])),
            "repeat_max_abs": float((repeated - baseline).abs().max()),
            "perturbation_input_rms": input_rms, "perturbation_output_rms": rms(perturbed_output - baseline),
            "perturbation_rms_ratio": rms(perturbed_output - baseline) / max(input_rms, 1e-12),
            "fp32_perturbation_rms_ratio": rms(fp32_perturbed - fp32_baseline) / max(input_rms, 1e-12),
            "bf16_vs_fp32_baseline_rms": rms(baseline - fp32_baseline),
            "input_bilateral_change_rms": rms(pre_bilateral - x),
            "output_pre_bilateral_change_rms": rms(outputs["pre_bilateral"] - baseline),
            "output_post_bilateral_change_rms": rms(outputs["post_bilateral_CONTROL"] - baseline)})
        print(f"VAL{row['index']} repeat={evidence[-1]['repeat_max_abs']:.6f} sensitivity_BF16={evidence[-1]['perturbation_rms_ratio']:.3f} FP32={evidence[-1]['fp32_perturbation_rms_ratio']:.3f}", flush=True)
    for start in range(0, len(panels), 4):
        sheet = Image.new("RGB", (2560, 312 * len(panels[start:start + 4])), "#101010")
        for j, panel in enumerate(panels[start:start + 4]):
            sheet.paste(panel, (0, 312 * j))
        sheet.save(args.output / f"comparison_{start:02d}.jpg", quality=96)
    atomic_json_dump({"checkpoint": str(args.checkpoint), "checkpoint_sha256": digest(args.checkpoint),
        "global_step": step, "suite_sha256": digest(args.suite), "records": evidence,
        "filters": {"bilateral": {"kernel": 5, "spatial_sigma": 1, "rgb_mean_range_sigma": .04},
            "gaussian": {"kernel": 5, "sigma": .6}},
        "precision": "BF16", "test_used": False, "lockbox_used": False,
        "training_performed": False, "external_rgb_generator_used": False, "public_demo_unchanged": True,
        "interpretation": "Sensitivity/blur controls, NOT sensor-noise estimation or daytime acceptance.",
        "vision_review": "pending"}, args.output / "audit.json")


if __name__ == "__main__":
    main()
