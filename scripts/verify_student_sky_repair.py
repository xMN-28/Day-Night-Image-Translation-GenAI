"""Local, validation-only visual smoke test of the public integrated inference path."""

import json
from pathlib import Path
from statistics import median

import torch
from PIL import Image, ImageDraw

from daynight.data import read_manifest
from daynight.inference import ModelManager


def main():
    output = Path("outputs/student_sky_integrated_verified")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    manager = ModelManager()
    records = read_manifest("data/bdd100k_daynight", "val", "day")
    results = []
    for index in (12, 142, 209, 37, 161, 407, 71, 213, 284, 355, 426, 499):
        with Image.open(records[index].path) as original:
            image = original.convert("RGB")
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        raw, raw_metadata = manager.translate_custom(
            image, "day_to_night", "LumiCycle distilled student", 512, sky_repair=False
        )
        repaired, metadata = manager.translate_custom(
            image, "day_to_night", "LumiCycle distilled student", 512
        )
        assert metadata["sky_repair"] and metadata["global_step"] == raw_metadata["global_step"]
        assert repaired.size == raw.size == image.size
        width, height = image.size
        panel = Image.new("RGB", (width * 3, height + 26), "#111111")
        draw = ImageDraw.Draw(panel)
        for column, (label, picture) in enumerate((
            (f"VAL {index:04d}: day input", image), ("Raw student", raw),
            ("Student + integrated guided sky repair", repaired),
        )):
            panel.paste(picture, (column * width, 26))
            draw.text((column * width + 5, 6), label, fill="white")
        panel.save(output / f"val_{index:04d}_comparison.jpg", quality=96)
        repaired.save(output / f"val_{index:04d}_repaired.png")
        results.append({"validation_index": index, "source": str(records[index].path),
                        "raw": raw_metadata, "repaired": metadata})
    report = {"test_images_used": False, "records": results,
              "median_warm_repair_seconds": median(r["repaired"]["seconds"] for r in results[1:]),
              "warning": "Visual smoke test, not proof of general quality or benchmark acceptance"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Verified {len(results)} validation sources; warm median "
          f"{report['median_warm_repair_seconds']:.3f}s; outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
