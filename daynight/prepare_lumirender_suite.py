from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CATEGORIES = {
    "branches_wires",
    "open_skies",
    "glare",
    "wet_roads",
    "vehicles",
    "ordinary_roads",
    "landscapes",
    "buildings",
    "indoor_windows",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Register the fixed LumiRender acceptance suite")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/lumirender_suite.jsonl"))
    args = parser.parse_args()
    rows = []
    with args.csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            category = row["category"].strip().lower().replace("/", "_").replace(" ", "_")
            if category not in CATEGORIES:
                raise ValueError(f"Unknown category {category!r}; expected one of {sorted(CATEGORIES)}")
            path = Path(row["image_path"])
            if not path.is_absolute():
                path = args.csv.parent / path
            if not path.exists():
                raise FileNotFoundError(path)
            rows.append({"category": category, "image_path": str(path.resolve())})
    missing = CATEGORIES - {row["category"] for row in rows}
    if missing:
        raise ValueError(f"Fixed suite is missing categories: {sorted(missing)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(f"Registered {len(rows)} fixed-suite images without copying the source files.")


if __name__ == "__main__":
    main()
