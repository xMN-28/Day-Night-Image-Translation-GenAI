"""Uncomposited V2 vs warm-started generator comparisons at demo resolution."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .data import read_manifest
from .inference import _fit_and_pad
from .sky_pilot import load_base
from .sky_refinement import SkyStudent, compose_night, phase_average, predict_matte
from .utils import atomic_json_dump, pil_to_tensor, tensor_to_pil


def file_digest(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/lumicycle_v2_skygan_pilot/latest.pt")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/sky_pilot/skygan_val"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--indices", default="0,71,142,213,284,355,426,499")
    parser.add_argument("--edge", type=int, default=768)
    parser.add_argument("--student", type=Path)
    parser.add_argument("--phase-average", action="store_true")
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=Path("runs/lumicycle_v2_bdd100k/checkpoints/step_00004500.pt"),
    )
    parser.add_argument("--reference-phase-average", action="store_true")
    parser.add_argument("--reference-hybrid", action="store_true")
    parser.add_argument("--reference-label", default="V2 4.5k")
    parser.add_argument("--label", default="Experimental trained sky candidate")
    parser.add_argument("--raw-candidate", action="store_true")
    args = parser.parse_args()
    if args.reference_hybrid and not args.student:
        parser.error("--reference-hybrid requires --student")
    torch.set_num_threads(6)
    reference = load_base(args.reference_checkpoint)
    refined = load_base(args.checkpoint)
    student = None
    if args.student:
        student_state = torch.load(args.student, map_location="cpu", weights_only=False)
        student = SkyStudent(outputs=student_state["model"]["head.weight"].shape[0]).cuda().eval()
        student.load_state_dict(student_state["model"])
    rows = []
    gallery = []
    args.output.mkdir(parents=True, exist_ok=True)
    records = read_manifest("data/bdd100k_daynight", args.split, "day")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in map(int, args.indices.split(",")):
            image, crop = _fit_and_pad(Image.open(records[i].path).convert("RGB"), args.edge)
            source = pil_to_tensor(image)[None].cuda()
            old_raw = (
                phase_average(reference, source)
                if args.reference_phase_average
                else reference(source)
            )
            old = tensor_to_pil(old_raw).crop(crop)
            raw = phase_average(refined, source) if args.phase_average else refined(source)
            new = tensor_to_pil(raw).crop(crop)
            width, height = new.size
            images = [image.crop(crop), old, new]
            labels = [
                "Input",
                args.reference_label,
                args.label if args.raw_candidate else "Candidate (generator only)",
            ]
            if student:
                rgb = (source.float() + 1) / 2
                a, vegetation = predict_matte(student, rgb)
                if args.reference_hybrid:
                    old_combined = compose_night(
                        rgb, (old_raw.float() + 1) / 2, a, vegetation=vegetation
                    )
                    old = tensor_to_pil(old_combined * 2 - 1).crop(crop)
                    images[1] = old
                if not args.raw_candidate:
                    combined = compose_night(rgb, (raw.float() + 1) / 2, a, vegetation=vegetation)
                    images.append(tensor_to_pil(combined * 2 - 1).crop(crop))
                    labels.append(args.label)
                    images[-1].save(args.output / f"{i:04d}_hybrid.png")
            canvas = Image.new("RGB", (width * len(images), height + 24), "#101010")
            draw = ImageDraw.Draw(canvas)
            for j, (im, label) in enumerate(zip(images, labels)):
                draw.text((j * width + 8, 5), label, fill="white")
                canvas.paste(im, (j * width, 24))
            canvas.save(args.output / f"{i:04d}_comparison.jpg", quality=95)
            new.save(args.output / f"{i:04d}_output.png")
            rows.append({"index": i, "source": str(records[i].path)})
            gallery.append((i, old.copy(), images[-1].copy()))
    if gallery:
        sheet = Image.new("RGB", (1024, 312 * len(gallery)), "#101010")
        draw = ImageDraw.Draw(sheet)
        for row, (index, old, new) in enumerate(gallery):
            draw.text(
                (8, row * 312 + 5), f"{args.split} {index}: {args.reference_label}", fill="white"
            )
            draw.text((520, row * 312 + 5), args.label, fill="white")
            sheet.paste(old.resize((512, 288), Image.Resampling.LANCZOS), (0, row * 312 + 24))
            sheet.paste(new.resize((512, 288), Image.Resampling.LANCZOS), (512, row * 312 + 24))
        sheet.save(args.output / "comparison_gallery.jpg", quality=95)
        sheet.crop((0, 0, 1024, 312)).save(args.output / "preview_before_after.jpg", quality=95)
    atomic_json_dump(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_digest(args.checkpoint),
            "split": args.split,
            "rows": rows,
            "student": str(args.student),
            "student_sha256": file_digest(args.student) if args.student else None,
            "student_step": student_state["step"] if args.student else None,
            "phase_average": args.phase_average,
            "reference_checkpoint": str(args.reference_checkpoint),
            "reference_sha256": file_digest(args.reference_checkpoint),
            "reference_phase_average": args.reference_phase_average,
            "reference_hybrid": args.reference_hybrid,
            "candidate_hybrid": bool(student) and not args.raw_candidate,
            "labels": [args.reference_label, args.label],
            "renderer_sha256": file_digest(Path(__file__).with_name("sky_refinement.py")),
            "inference": "raw generator plus optional separately labeled student/compositor; no teacher",
            "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        args.output / "evidence.json",
    )
    print(str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
