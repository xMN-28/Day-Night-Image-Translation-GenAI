"""Curated validation showcase screening, never a representative benchmark."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from daynight.data import read_manifest
from daynight.inference import ModelManager
from daynight.remote_worker import MODEL_CATALOG

DEST = Path("E:/LumiCycleCache/experiments/night_day_demo_screen_20261024")
DEST.mkdir(parents=True, exist_ok=True)
for name in ("inputs", "outputs", "review"):
    (DEST / name).mkdir(exist_ok=True)
suite = json.loads((ROOT / "outputs/night_to_day_experimental/fixed_suite_v1/suite.json").read_text())
lockbox = {r["index"] for r in suite["groups"]["lockbox"]}
records = read_manifest(ROOT / "data/bdd100k_daynight", "val", "night")
spec = MODEL_CATALOG["night_day_experimental"]
torch.set_num_threads(6)
manager = ModelManager()
rows = []
for index, record in enumerate(records):
    if index in lockbox or index == 142:  # prior user-rejected underpass
        continue
    upload = DEST / "inputs" / f"val_{index:04d}.jpg"
    result_path = DEST / "outputs" / f"val_{index:04d}.png"
    if not upload.exists():
        with Image.open(record.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            image.save(upload, quality=95, subsampling=0)
    with Image.open(upload) as image:
        image = image.convert("RGB")
        if not result_path.exists():
            prediction, metadata = manager.translate_custom(image, "night_to_day", spec.engine_name,
                512, checkpoint_override=spec.checkpoint, display_name=spec.label)
            prediction.save(result_path)
        source = np.asarray(image, dtype=np.float32) / 255
    with Image.open(result_path) as image:
        output = np.asarray(image, dtype=np.float32) / 255
    luminance = output @ np.array([.2126, .7152, .0722])
    upper = luminance[:round(output.shape[0] * .35)]
    noise = np.abs(upper[1:-1, 1:-1] * 4 - upper[:-2, 1:-1] - upper[2:, 1:-1]
                   - upper[1:-1, :-2] - upper[1:-1, 2:]).mean()
    # Presentation-review ordering only; not a validated quality metric.
    order = float(luminance.mean() + .45 * upper.mean() - .8 * noise)
    source_mean = float((source @ np.array([.2126, .7152, .0722])).mean())
    rows.append({"index": index, "source": str(record.path), "input": str(upload),
                 "output": str(result_path), "review_order_only": order,
                 "source_luminance": source_mean, "output_luminance": float(luminance.mean())})
    if len(rows) % 40 == 0:
        print(f"Generated {len(rows)} validation previews", flush=True)

ranked = sorted(rows, key=lambda r: r["review_order_only"], reverse=True)
# Review top192 and retain all other generated predictions for traceability.
review = ranked[:192]
font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
for start in range(0, len(review), 16):
    sheet = Image.new("RGB", (2048, 4 * 168), "#15171b")
    draw = ImageDraw.Draw(sheet)
    for cell, row in enumerate(review[start:start + 16]):
        x, y = cell % 4 * 512, cell // 4 * 168
        draw.text((x + 5, y + 2), f"Rank {start+cell+1} | VAL {row['index']:04d} | night → model", font=font, fill="white")
        for column, name in enumerate(("input", "output")):
            with Image.open(row[name]) as image:
                image = image.convert("RGB")
                image.thumbnail((256, 144), Image.Resampling.LANCZOS)
                sheet.paste(image, (x + column * 256, y + 24))
    sheet.save(DEST / "review" / f"review_{start:03d}.jpg", quality=95)
with spec.checkpoint.open("rb") as stream:
    checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
(DEST / "manifest.json").write_text(json.dumps({"split": "validation night", "test_used": False,
    "lockbox_excluded": sorted(lockbox), "fixed_gate_unchanged": True,
    "checkpoint": str(spec.checkpoint), "checkpoint_sha256": checkpoint_hash,
    "model": spec.label, "step": 16500, "edge": 512,
    "inference": "Same ModelManager path as web worker; exact saved upload JPEG reopened before inference",
    "editing": "No output retouching, brightness changes, denoising or external model substitution",
    "warning": "Explicitly cherry-picked showcase; not an unbiased quality evaluation",
    "records": ranked}, indent=2))
print(json.dumps({"screened": len(rows), "ranked_review_pairs": len(review), "root": str(DEST)}), flush=True)
