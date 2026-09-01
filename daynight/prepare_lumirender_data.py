from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from .utils import atomic_json_dump


def register_pair_csv(
    csv_path: Path, source: str, output: Path, split: str = "train"
) -> int:
    rows = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            day = Path(row["day_path"])
            night = Path(row["night_path"])
            if not day.is_absolute():
                day = csv_path.parent / day
            if not night.is_absolute():
                night = csv_path.parent / night
            if not day.exists() or not night.exists():
                raise FileNotFoundError(f"Missing pair: {day} / {night}")
            confidence_path = Path(row["confidence_path"]) if row.get("confidence_path") else None
            if confidence_path is not None and not confidence_path.is_absolute():
                confidence_path = csv_path.parent / confidence_path
            if confidence_path is not None and not confidence_path.exists():
                raise FileNotFoundError(f"Missing confidence mask: {confidence_path}")
            rows.append(
                {
                    "source": source,
                    "split": row.get("split", split),
                    "day_path": str(day.resolve()),
                    "night_path": str(night.resolve()),
                    "confidence": float(row.get("confidence", 1.0)),
                    **(
                        {"confidence_path": str(confidence_path.resolve())}
                        if confidence_path is not None
                        else {}
                    ),
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate licensed day/night pair lists and register LumiRender manifests"
    )
    parser.add_argument("--pairs-csv", required=True, type=Path)
    parser.add_argument("--source", required=True, choices=["acdc", "dark_zurich", "samsung", "timelapse"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-added-gb", type=float, default=55.0)
    args = parser.parse_args()
    output = args.output or Path("data/lumirender_sources") / f"{args.source}.jsonl"
    size = sum(path.stat().st_size for path in args.pairs_csv.parent.rglob("*") if path.is_file())
    size_gb = size / 1024**3
    summaries = output.parent.glob("*.summary.json") if output.parent.exists() else []
    existing_gb = sum(
        float(json.loads(path.read_text(encoding="utf-8")).get("source_gb", 0))
        for path in summaries
        if path.stem != output.with_suffix(".summary.json").stem
    )
    if existing_gb + size_gb > args.max_added_gb:
        raise SystemExit(
            f"Registered sources would total {existing_gb + size_gb:.1f} GB, above the "
            f"configured {args.max_added_gb:.1f} GB cap."
        )
    free_gb = shutil.disk_usage(output.resolve().anchor).free / 1024**3
    if free_gb < 20:
        raise SystemExit(f"Only {free_gb:.1f} GB free; LumiRender requires a 20 GB reserve.")
    count = register_pair_csv(args.pairs_csv, args.source, output)
    atomic_json_dump(
        {"source": args.source, "pairs": count, "source_gb": round(size_gb, 3)},
        output.with_suffix(".summary.json"),
    )
    print(f"Registered {count} licensed {args.source} pairs in {output}")


if __name__ == "__main__":
    main()
