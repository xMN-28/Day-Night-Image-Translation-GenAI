"""Locally cache explicitly attributed official-teacher targets; no test images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import read_manifest
from .turbo_pilot import REVISION, SOURCE_COMMIT, digest, fit_input, load_official
from .utils import atomic_json_dump, tensor_to_pil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("E:/LumiCycleCache"))
    parser.add_argument(
        "--output", type=Path, default=Path("E:/LumiCycleCache/teacher_targets/turbo_v1")
    )
    parser.add_argument("--train-images", type=int, default=128)
    parser.add_argument("--val-images", type=int, default=48)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--storage", choices=("npz", "jpeg-targets"), default="npz")
    parser.add_argument("--student-width", type=int, choices=(384, 512), default=384)
    args = parser.parse_args()
    torch.set_num_threads(6)
    checkpoint = args.cache_root / "official_models/day2night.pkl"
    recipe = {
        "teacher": "Official CycleGAN-Turbo day_to_night, externally pretrained",
        "source_commit": SOURCE_COMMIT,
        "base_revision": REVISION,
        "teacher_checkpoint_sha256": digest(checkpoint),
        "seed_base": 10101,
        "teacher_maximum_edge": 512,
        "student_size": [args.student_width, args.student_width * 9 // 16],
        "train_count": args.train_images,
        "val_count": args.val_images,
        "warning": "Pseudo-targets may hallucinate lights or alter text. Not true paired nighttime ground truth.",
    }
    if args.storage == "jpeg-targets":
        recipe["storage"] = (
            "target JPEG quality95/subsampling0; source images referenced, not duplicated"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "recipe.json"
    if path.exists() and json.loads(path.read_text()) != recipe:
        raise ValueError("Teacher cache recipe changed; choose a new output folder")
    atomic_json_dump(recipe, path)
    records = {s: read_manifest("data/bdd100k_daynight", s, "day") for s in ("train", "val")}
    if {r.path for r in records["train"]} & {r.path for r in records["val"]}:
        raise ValueError("Source path leakage between train and validation")
    model, _, _ = load_official(args.cache_root)
    rows = []
    verified = []
    with torch.inference_mode():
        for split, count in (("train", args.train_images), ("val", args.val_images)):
            selected = np.linspace(
                0, len(records[split]) - 1, min(count, len(records[split])), dtype=int
            )
            folder = args.output / split
            folder.mkdir(exist_ok=True)
            for i, index in enumerate(selected):
                record = records[split][index]
                extension = "npz" if args.storage == "npz" else "jpg"
                destination = folder / f"{int(index):05d}.{extension}"
                seed = 10101 + int(index) + (100000 if split == "val" else 0)
                if not destination.exists() or args.verify_existing:
                    source, x, crop = fit_input(Image.open(record.path), 512)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    raw = model(x.cuda().half())
                    if not torch.isfinite(raw).all():
                        raise FloatingPointError("Nonfinite teacher target")
                    result = tensor_to_pil(raw).crop(crop)
                    size = tuple(recipe["student_size"])
                    image = np.asarray(source.resize(size, Image.Resampling.LANCZOS)).copy()
                    target = np.asarray(result.resize(size, Image.Resampling.LANCZOS)).copy()
                    if args.verify_existing and destination.exists():
                        if args.storage == "npz":
                            with np.load(destination) as cached:
                                difference = int(
                                    np.abs(
                                        cached["target"].astype(np.int16) - target.astype(np.int16)
                                    ).max()
                                )
                                source_equal = np.array_equal(cached["image"], image)
                        else:
                            # JPEG is lossy: compare a freshly encoded/decoded target,
                            # not the pre-compression RGB values.
                            import io

                            buffer = io.BytesIO()
                            Image.fromarray(target).save(
                                buffer, format="JPEG", quality=95, subsampling=0
                            )
                            buffer.seek(0)
                            fresh = np.asarray(Image.open(buffer).convert("RGB"))
                            cached = np.asarray(Image.open(destination).convert("RGB"))
                            difference = int(
                                np.abs(cached.astype(np.int16) - fresh.astype(np.int16)).max()
                            )
                            source_equal = True
                        if not source_equal or difference > 1:
                            raise RuntimeError(
                                "Teacher cache reproduction differs; existing cache not overwritten"
                            )
                        verified.append(
                            {
                                "cache": str(destination),
                                "finite_raw_output": True,
                                "maximum_byte_difference": difference,
                                "cache_sha256": digest(destination),
                            }
                        )
                    else:
                        # Atomic publication: interrupted files are not reused.
                        partial = destination.with_suffix(".partial")
                        with partial.open("wb") as handle:
                            if args.storage == "npz":
                                np.savez_compressed(handle, image=image, target=target)
                            else:
                                Image.fromarray(target).save(
                                    handle, format="JPEG", quality=95, subsampling=0
                                )
                        partial.replace(destination)
                rows.append(
                    {
                        "split": split,
                        "index": int(index),
                        "source": str(record.path),
                        "source_sha256": digest(record.path),
                        "seed": seed,
                        "cache": str(destination),
                    }
                )
                if i % 16 == 0:
                    print(
                        json.dumps({"split": split, "completed": i + 1, "total": len(selected)}),
                        flush=True,
                    )
    atomic_json_dump(
        {"recipe": recipe, "records": rows, "test_images_used": False},
        args.output / "manifest.json",
    )
    if verified:
        atomic_json_dump(
            {"checked": len(verified), "records": verified},
            args.output / "finite_reproduction_audit.json",
        )


if __name__ == "__main__":
    main()
