"""Generate local validation-only showcase candidates; selection is explicitly curated."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from .data import read_manifest
from .inference import MODEL_PATHS, ModelManager


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo_screening"))
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--select", help="Comma-separated validation indices, after visual review")
    parser.add_argument("--showcase", type=Path, default=Path("outputs/demo_showcase"))
    args = parser.parse_args()
    if args.select:
        selection = [int(i) for i in args.select.split(",")]
        if len(set(selection)) != len(selection):
            raise ValueError("Selection must not contain duplicate images")
        manifest = json.loads((args.output / "manifest.json").read_text())
        by_index = {r["validation_index"]: r for r in manifest["records"]}
        if any(i not in by_index for i in selection):
            raise ValueError("Only screened validation images may be selected")
        if args.showcase.exists() and any(args.showcase.iterdir()):
            raise FileExistsError(
                "Showcase folder is not empty. Archive the old selection first to avoid stale samples."
            )
        checkpoint = MODEL_PATHS["LumiCycle distilled student"]
        if digest(checkpoint) != manifest["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed since screening; regenerate and review candidates first")
        manager = ModelManager()
        for folder in ("daytime_inputs", "generated_nights", "comparisons"):
            (args.showcase / folder).mkdir(parents=True, exist_ok=True)
        selected = []
        sheet = Image.new("RGB", (1024, 172 * ((len(selection) + 1) // 2)), "#111111")
        for number, index in enumerate(selection, 1):
            row = by_index[index]
            stem = f"{number:02d}_val_{index:04d}.jpg"
            shutil.copy2(args.output / row["input"], args.showcase / "daytime_inputs" / stem)
            # Run the actual JPEG that will be uploaded, not its pre-encoding source.
            with Image.open(args.showcase / "daytime_inputs" / stem) as uploaded:
                source = uploaded.convert("RGB")
            result, metadata = manager.translate_custom(
                source, "day_to_night", "LumiCycle distilled student", manifest["edge"]
            )
            result.save(args.showcase / "generated_nights" / Path(stem).with_suffix(".png"))
            width, height = source.size
            panel = Image.new("RGB", (width * 2, height + 24), "#111111")
            draw = ImageDraw.Draw(panel)
            draw.text((5, 5), f"VAL {index:04d} - EXACT UPLOAD", fill="white")
            draw.text((width + 5, 5), "LOCAL STUDENT - 512 PX", fill="white")
            panel.paste(source, (0, 24))
            panel.paste(result, (width, 24))
            panel.save(args.showcase / "comparisons" / stem, quality=95)
            panel.thumbnail((510, 170))
            sheet.paste(panel, (((number - 1) % 2) * 512, ((number - 1) // 2) * 172))
            selected.append({**row, "showcase_filename": stem,
                             "generated_filename": str(Path(stem).with_suffix(".png")),
                             "upload_sha256": digest(args.showcase / "daytime_inputs" / stem),
                             "metadata": metadata})
        sheet.save(args.showcase / "SHOWCASE_CONTACT_SHEET.jpg", quality=95)
        manifest["records"] = selected
        manifest["selection"] = "Manually curated best-case validation demonstrations; not a benchmark"
        manifest["exact_uploads_rerun"] = True
        (args.showcase / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Selected {len(selected)} examples in {args.showcase.resolve()}")
        return

    if args.count < 1:
        raise ValueError("count must be positive")
    torch.set_num_threads(6)
    records = read_manifest("data/bdd100k_daynight", "val", "day")
    manager = ModelManager()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    indices = np.linspace(0, len(records) - 1, min(args.count, len(records)), dtype=int)
    for index in indices:
        index = int(index)
        with Image.open(records[index].path) as original:
            source = ImageOps.exif_transpose(original).convert("RGB")
            source.thumbnail((512, 512), Image.Resampling.LANCZOS)
        output, metadata = manager.translate_custom(
            source, "day_to_night", "LumiCycle distilled student", 512
        )
        source = source.resize(output.size, Image.Resampling.LANCZOS)
        stem = f"val_{index:04d}"
        source.save(args.output / f"{stem}_day.jpg", quality=95, subsampling=0)
        output.save(args.output / f"{stem}_night.jpg", quality=95, subsampling=0)
        width, height = source.size
        panel = Image.new("RGB", (width * 2, height + 24), "#111111")
        draw = ImageDraw.Draw(panel)
        draw.text((5, 5), f"VAL {index:04d} - DAY INPUT", fill="white")
        draw.text((width + 5, 5), "LOCAL DISTILLED STUDENT - NIGHT", fill="white")
        panel.paste(source, (0, 24))
        panel.paste(output, (width, 24))
        panel.save(args.output / f"{stem}_comparison.jpg", quality=94)
        rgb = np.asarray(source, dtype=np.float32) / 255
        night = np.asarray(output, dtype=np.float32) / 255
        # A screening hint, not a learned quality metric or an acceptance criterion.
        top = max(1, height // 2)
        glow = float((night[:top].mean(-1) > 0.65).mean())
        sun = float((rgb[:top].min(-1) > 0.95).mean())
        rows.append({"validation_index": index, "source": str(records[index].path),
                     "input": f"{stem}_day.jpg", "output": f"{stem}_night.jpg",
                     "comparison": f"{stem}_comparison.jpg", "screening_hint": glow + sun,
                     "metadata": metadata})
    for page, start in enumerate(range(0, len(rows), 24)):
        page_rows = rows[start:start + 24]
        sheet = Image.new("RGB", (1280, 198 * ((len(page_rows) + 2) // 3)), "#111111")
        for k, row in enumerate(page_rows):
            with Image.open(args.output / row["comparison"]) as panel:
                panel.thumbnail((424, 194))
                sheet.paste(panel, ((k % 3) * 426, (k // 3) * 198))
        sheet.save(args.output / f"contact_sheet_{page + 1:02d}.jpg", quality=94)
    checkpoint = MODEL_PATHS["LumiCycle distilled student"]
    (args.output / "manifest.json").write_text(json.dumps({
        "split": "validation", "test_images_used": False,
        "model": "Local student trained using external CycleGAN-Turbo pseudo-targets",
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
        "edge": 512, "records": rows,
        "redistribution": "Local BDD100K examples only; do not commit or publicly redistribute images",
    }, indent=2), encoding="utf-8")
    print(f"Screened {len(rows)} validation examples at {args.output.resolve()}")


if __name__ == "__main__":
    main()
