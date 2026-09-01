from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .data import read_manifest

CATEGORIES = (
    "branches_wires",
    "open_skies",
    "glare",
    "wet_roads",
    "vehicles",
    "ordinary_roads",
    "landscapes",
    "buildings",
    "indoor_windows",
)


def _features(path: Path, label_root: Path) -> dict[str, float | str]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB").resize((192, 108))
    array = np.asarray(image, dtype=np.float32) / 255
    gray = array.mean(axis=2)
    upper = gray[:65]
    edge = np.abs(np.diff(upper, axis=0)).mean() + np.abs(np.diff(upper, axis=1)).mean()
    label_path = label_root / path.parent.name / f"{path.stem}.json"
    label = json.loads(label_path.read_text(encoding="utf-8"))
    objects = label.get("frames", [{}])[0].get("objects", [])
    categories = [item.get("category", "") for item in objects]
    attributes = label.get("attributes", {})
    return {
        "upper_edge": float(edge),
        "upper_brightness": float(upper.mean()),
        "saturation": float((array.max(axis=2) > 0.96).mean()),
        "lower_dark": float((gray[80:] < 0.12).mean()),
        "vehicles": float(sum(name in {"car", "truck", "bus"} for name in categories)),
        "objects": float(len(objects)),
        "weather": str(attributes.get("weather", "undefined")),
        "scene": str(attributes.get("scene", "undefined")),
    }


def _rank(records: list[dict[str, object]], category: str) -> list[dict[str, object]]:
    def score(record: dict[str, object]) -> float:
        feature = record["features"]
        assert isinstance(feature, dict)
        if category == "branches_wires":
            return float(feature["upper_edge"]) - 0.01 * float(feature["objects"])
        if category == "open_skies":
            return float(feature["upper_brightness"]) - 5 * float(feature["upper_edge"])
        if category == "glare":
            return float(feature["saturation"]) + 0.2 * float(feature["upper_brightness"])
        if category == "wet_roads":
            return (2.0 if feature["weather"] in {"rainy", "snowy"} else 0.0) + float(
                feature["saturation"]
            )
        if category == "vehicles":
            return float(feature["vehicles"])
        if category == "ordinary_roads":
            return (1.0 if feature["weather"] in {"clear", "undefined"} else 0.0) - abs(
                float(feature["vehicles"]) - 4
            ) * 0.05
        if category == "landscapes":
            return (1.0 if feature["scene"] == "highway" else 0.0) - 0.03 * float(
                feature["objects"]
            )
        if category == "buildings":
            return (1.0 if feature["scene"] in {"city street", "residential"} else 0.0) + 0.01 * float(
                feature["objects"]
            )
        return float(feature["lower_dark"])

    return sorted(records, key=score, reverse=True)


def suggest(data_root: Path, output: Path, candidates: int) -> Path:
    dataset = json.loads((data_root / "dataset.json").read_text(encoding="utf-8"))
    label_root = Path(dataset["source_root"]) / "labels"
    records = []
    for record in read_manifest(data_root, "test", "day"):
        records.append({"path": record.path, "features": _features(record.path, label_root)})
    chosen: list[tuple[str, Path]] = []
    used: set[Path] = set()
    for category in CATEGORIES:
        category_count = 0
        for record in _rank(records, category):
            path = record["path"]
            assert isinstance(path, Path)
            if path not in used:
                chosen.append((category, path))
                used.add(path)
                category_count += 1
                if category_count >= candidates:
                    break
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "suggested_suite.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "image_path"])
        writer.writeheader()
        writer.writerows(
            {"category": category, "image_path": str(path.resolve())} for category, path in chosen
        )
    tile_width, tile_height = 300, 195
    sheet = Image.new(
        "RGB", (tile_width * candidates, tile_height * len(CATEGORIES)), "#111827"
    )
    draw = ImageDraw.Draw(sheet)
    for index, (category, path) in enumerate(chosen):
        with Image.open(path) as source:
            image = ImageOps.fit(source.convert("RGB"), (tile_width, tile_height - 34))
        x, y = (index % candidates) * tile_width, (index // candidates) * tile_height
        sheet.paste(image, (x, y + 34))
        draw.text((x + 8, y + 9), category, fill="white")
    sheet.save(output / "suggested_suite.jpg", quality=92)
    print(f"Suggested nine-category suite: {csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest a diverse fixed suite from BDD100K test")
    parser.add_argument("--data-root", type=Path, default=Path("data/bdd100k_daynight"))
    parser.add_argument("--output", type=Path, default=Path("outputs/lumirender/suite_candidates"))
    parser.add_argument("--candidates", type=int, default=4)
    args = parser.parse_args()
    suggest(args.data_root, args.output, args.candidates)


if __name__ == "__main__":
    main()
