"""Generate a reproducible validation comparison with frozen semantic teacher masks."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .data import read_manifest
from .inference import _fit_and_pad
from .models import build_models
from .sky_refinement import SkyStudent, compose_night, predict_matte, refine_sky_alpha
from .utils import atomic_json_dump, pil_to_tensor, tensor_to_pil


def load_base(path: Path, device: str = "cuda"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_models(payload["config"])["G_day_night"]
    model.load_state_dict(payload["ema"]["G_day_night"]["shadow"])
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--indices", default="0,71,142,213,284,355,426,499")
    parser.add_argument("--output", type=Path, default=Path("outputs/sky_pilot/teacher_val"))
    parser.add_argument("--student", type=Path)
    parser.add_argument("--edge", type=int, default=768)
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = True
    args.output.mkdir(parents=True, exist_ok=True)
    base = load_base(Path("runs/lumicycle_v2_bdd100k/checkpoints/step_00004500.pt"))
    student = None
    student_record = None
    if args.student:
        checkpoint_bytes = args.student.read_bytes()
        state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
        student = SkyStudent(outputs=state["model"]["head.weight"].shape[0]).cuda().eval()
        student.load_state_dict(state["model"])
        student_record = {
            "path": str(args.student),
            "step": state["step"],
            "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        }
    else:
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        name = "facebook/mask2former-swin-small-cityscapes-semantic"
        processor = AutoImageProcessor.from_pretrained(name, local_files_only=True)
        teacher = (
            Mask2FormerForUniversalSegmentation.from_pretrained(name, local_files_only=True)
            .cuda()
            .eval()
        )
    records = read_manifest("data/bdd100k_daynight", args.split, "day")
    rows = []
    with torch.inference_mode():
        for index in map(int, args.indices.split(",")):
            image = Image.open(records[index].path).convert("RGB")
            image, _crop = _fit_and_pad(image, args.edge)
            source = (pil_to_tensor(image).unsqueeze(0).cuda() + 1) / 2
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = (base(source * 2 - 1).float() + 1) / 2
            if student:
                alpha, mixture = predict_matte(student, source)
            else:
                inputs = processor(image, return_tensors="pt")
                prediction = teacher(**{k: v.cuda() for k, v in inputs.items()})
                classes = prediction.class_queries_logits.softmax(-1)[..., :-1]
                masks = prediction.masks_queries_logits.sigmoid()
                semantic = torch.einsum("bqc,bqhw->bchw", classes, masks)
                semantic = F.interpolate(
                    semantic, source.shape[-2:], mode="bilinear", align_corners=False
                )
                labels = semantic.argmax(1, keepdim=True)
                p = (labels == 10).float()
                mixture = (labels == 8).float()
                alpha = refine_sky_alpha(source, p, mixture)
            corrected = compose_night(
                source, output, alpha, vegetation=mixture if student else None
            )
            images = [
                image,
                tensor_to_pil(output * 2 - 1),
                tensor_to_pil(corrected * 2 - 1),
                tensor_to_pil(alpha.repeat(1, 3, 1, 1) * 2 - 1),
            ]
            canvas = Image.new("RGB", (image.width * 2, image.height * 2 + 48), "#101010")
            draw = ImageDraw.Draw(canvas)
            for j, (im, label) in enumerate(
                zip(images, ["Input", "V2 4.5k", "Sky refinement", "Sky alpha"])
            ):
                left, top = (j % 2) * image.width, (j // 2) * (image.height + 24)
                draw.text((left + 8, top + 5), label, fill="white")
                canvas.paste(im, (left, top + 24))
            canvas.save(args.output / f"{index:04d}_{records[index].path.stem}.jpg", quality=94)
            images[2].save(args.output / f"{index:04d}_output.png")
            Image.fromarray((alpha[0, 0].cpu().numpy() * 255).astype(np.uint8)).save(
                args.output / f"{index:04d}_alpha.png"
            )
            rows.append(
                {
                    "index": index,
                    "input": str(records[index].path),
                    "sky_fraction": float(alpha.mean()),
                    "base_sky_luma": float(
                        (output.mean(1, True) * alpha).sum() / alpha.sum().clamp_min(1)
                    ),
                    "refined_sky_luma": float(
                        (corrected.mean(1, True) * alpha).sum() / alpha.sum().clamp_min(1)
                    ),
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    atomic_json_dump(
        {"split": args.split, "student": student_record, "records": rows},
        args.output / "evidence.json",
    )


if __name__ == "__main__":
    main()
