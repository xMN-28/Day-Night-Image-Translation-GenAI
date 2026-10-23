"""Arrange genuine saved BDD validation inputs/predictions; no image synthesis."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from daynight.turbo_pilot import digest


def main():
    root = Path(__file__).resolve().parents[2]
    manifest_path = Path("E:/LumiCycleCache/teacher_targets/night2day_pilot128_v2/manifest.json")
    comparison = root / "outputs/night_to_day_experimental/20261023_CARLA12_matched500"
    evidence = json.loads((comparison / "evidence.json").read_text())
    model = evidence["results"]["CARLA_aux10"]
    if digest(Path(model["checkpoint"])) != model["sha256"]:
        raise ValueError("Checkpoint changed since predictions were generated")
    manifest = json.loads(manifest_path.read_text())
    if evidence["targets_manifest_sha256"] != digest(manifest_path) or manifest["test_used"]:
        raise ValueError("Comparison provenance mismatch")
    rows = {r["index"]: r for r in manifest["records"] if r["split"] == "val"}
    indices = [71, 243, 358, 426]
    margin, gap, width, height = 24, 16, 512, 288
    canvas = Image.new("RGB", (2 * margin + 2 * width + gap, 1364), "#15171b")
    draw = ImageDraw.Draw(canvas)
    font_path = "C:/Windows/Fonts/segoeui.ttf"
    title_font = ImageFont.truetype(font_path, 26)
    font = ImageFont.truetype(font_path, 18)
    small = ImageFont.truetype(font_path, 15)
    draw.text((margin, 16), "LumiCycle | Experimental night-to-day", font=title_font, fill="#f2f4f8")
    draw.text((margin, 52), f"Latest trained candidate: {model['global_step']:,} steps | BDD validation images", font=small, fill="#aeb8c8")
    draw.text((margin, 83), "NIGHT INPUT", font=font, fill="#a9c9f5")
    draw.text((margin + width + gap, 83), "OUR MODEL'S TRANSLATION", font=font, fill="#f0c585")
    records = []
    for number, index in enumerate(indices):
        source_path = Path(rows[index]["input"])
        output_path = comparison / "CARLA_aux10" / f"val_{index:04d}.png"
        top = 112 + number * 304
        for column, path in enumerate((source_path, output_path)):
            with Image.open(path) as image:
                if image.size != (width, height):
                    raise ValueError("Keep native512x288samples; no stretching or retouching")
                canvas.paste(image.convert("RGB"), (margin + column * (width + gap), top))
        draw.text((margin + 7, top + 7), f"BDD val {index:04d}", font=small,
            fill="white", stroke_width=2, stroke_fill="#15171b")
        records.append({"index": index, "input": str(source_path), "input_sha256": digest(source_path),
            "prediction": str(output_path), "prediction_sha256": digest(output_path)})
    draw.text((margin, 1330), "Unretouched outputs. Experimental: noise, glare and lighting artifacts remain.",
        font=small, fill="#aeb8c8")
    destination = root / "outputs/night_to_day_experimental/showcase_collages"
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "bdd_four_night_to_day_latest_16500.png"
    canvas.save(output)
    (destination / "bdd_four_night_to_day_latest_16500.json").write_text(json.dumps({
        "model": model, "samples": records, "test_used": False,
        "selection": "Four existing development examples; bridge case142replaced by open-street243at user request. Curated display, not a success-rate evaluation",
        "editing": "Layout/text only; original512x288saved pixels pasted without modification"}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
