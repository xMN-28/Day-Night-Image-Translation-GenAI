from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
from pathlib import Path


def _records(image_root: Path, timestamp_path: Path) -> list[tuple[datetime, Path]]:
    records: list[tuple[datetime, Path]] = []
    for line in timestamp_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        filename = parts[0]
        path = image_root / timestamp_path.stem / filename
        if path.exists():
            # Some AMOS metadata files use a day counter rather than YYYYMMDD.
            # The capture filename itself is consistent across this Kaggle subset.
            stamp = datetime.strptime(Path(filename).stem, "%Y%m%d_%H%M%S").replace(tzinfo=UTC)
            records.append((stamp, path.resolve()))
    return records


def write_amos_pairs(root: Path, output: Path, max_pairs: int = 10_000) -> int:
    """Create same-camera coarse day/night pairs from the Kaggle AMOS subset."""
    image_root = root / "DNIM" / "Image"
    timestamp_root = root / "DNIM" / "time_stamp"
    pairs: list[tuple[Path, Path]] = []
    for timestamp_path in sorted(timestamp_root.glob("*.txt")):
        records = _records(image_root, timestamp_path)
        days = [(stamp, path) for stamp, path in records if 9 <= stamp.hour <= 16]
        nights = [(stamp, path) for stamp, path in records if stamp.hour <= 5 or stamp.hour >= 21]
        if not days:
            continue
        for night_stamp, night_path in nights:
            _, day_path = min(days, key=lambda item: abs((item[0] - night_stamp).total_seconds()))
            pairs.append((day_path, night_path))
    pairs = pairs[:max_pairs]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["day_path", "night_path", "split", "confidence"]
        )
        writer.writeheader()
        for day_path, night_path in pairs:
            writer.writerow(
                {
                    "day_path": str(day_path),
                    "night_path": str(night_path),
                    "split": "train",
                    # Same static camera, but illumination/season can be far apart.
                    "confidence": "0.5",
                }
            )
    return len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare coarse same-camera pairs from the Kaggle AMOS day/night subset"
    )
    parser.add_argument("--amos-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-pairs", type=int, default=10_000)
    args = parser.parse_args()
    count = write_amos_pairs(args.amos_root.resolve(), args.output, args.max_pairs)
    print(f"Prepared {count} AMOS timelapse correspondences in {args.output}")


if __name__ == "__main__":
    main()
