from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _rgb_path(root: Path, minimal_name: str) -> Path:
    """Expand a Dark Zurich list/correspondence key into an anonymized RGB path."""
    key = minimal_name.strip().replace("\\", "/")
    path = root / "rgb_anon" / f"{key}_rgb_anon.png"
    if not path.exists():
        raise FileNotFoundError(f"Dark Zurich image is missing: {path}")
    return path.resolve()


def _validation_pairs(root: Path) -> list[tuple[Path, Path]]:
    lists = root / "lists_file_names"
    nights = (lists / "val_filenames.txt").read_text(encoding="utf-8").splitlines()
    days = (lists / "val_ref_filenames.txt").read_text(encoding="utf-8").splitlines()
    if len(nights) != len(days):
        raise ValueError(f"Validation lists differ in length: {len(days)} day / {len(nights)} night")
    return [(_rgb_path(root, day), _rgb_path(root, night)) for day, night in zip(days, nights)]


def _training_pairs(root: Path) -> list[tuple[Path, Path]]:
    correspondence_root = root / "corresp" / "train" / "night"
    files = sorted(correspondence_root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No night-to-day correspondence CSVs found below {correspondence_root}")
    pairs: list[tuple[Path, Path]] = []
    for path in files:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle):
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 2:
                    raise ValueError(f"Malformed correspondence in {path}: {row}")
                night_key, day_key = row[0].strip(), row[1].strip()
                pairs.append((_rgb_path(root, day_key), _rgb_path(root, night_key)))
    return pairs


def write_pairs_csv(root: Path, output: Path, split: str) -> int:
    pairs = _training_pairs(root) if split == "train" else _validation_pairs(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["day_path", "night_path", "split", "confidence"],
        )
        writer.writeheader()
        for day, night in pairs:
            # These are GPS/coarse correspondences. RAFT creates the dense confidence mask later.
            writer.writerow(
                {
                    "day_path": str(day),
                    "night_path": str(night),
                    "split": split,
                    "confidence": "1.0",
                }
            )
    return len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an official Dark Zurich archive into LumiRender's pair CSV format"
    )
    parser.add_argument("--dark-zurich-root", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    count = write_pairs_csv(args.dark_zurich_root.resolve(), args.output, args.split)
    print(f"Prepared {count} Dark Zurich {args.split} correspondences in {args.output}")


if __name__ == "__main__":
    main()
