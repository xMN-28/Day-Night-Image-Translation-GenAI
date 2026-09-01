from __future__ import annotations

import argparse
import json
from pathlib import Path

from .utils import atomic_json_dump


def write_acdc_nights(root: Path, output: Path, split: str = "train") -> int:
    """Register night-only images from the cleaned Kaggle ACDC package."""
    night_root = root / "acdc" / "images" / "night" / split
    images = sorted(path.resolve() for path in night_root.rglob("*.png"))
    if not images:
        raise FileNotFoundError(f"No ACDC night PNGs found below {night_root}")
    rows = [
        {
            "source": "acdc",
            "split": "train",
            "paired": False,
            "night_path": str(path),
        }
        for path in images
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    size_gb = sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) / 1024**3
    atomic_json_dump(
        {
            "source": "acdc",
            "pairs": 0,
            "unpaired_night": len(rows),
            "source_gb": round(size_gb, 3),
            "note": "Kaggle cleaned copy omits ACDC normal-condition references",
        },
        output.with_suffix(".summary.json"),
    )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register night-only images from Kaggle's cleaned ACDC package"
    )
    parser.add_argument("--acdc-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/lumirender_sources/acdc.jsonl"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    args = parser.parse_args()
    count = write_acdc_nights(args.acdc_root.resolve(), args.output, args.split)
    print(f"Registered {count} unpaired ACDC night images in {args.output}")


if __name__ == "__main__":
    main()
