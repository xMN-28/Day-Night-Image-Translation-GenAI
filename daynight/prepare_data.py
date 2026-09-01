from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

from .utils import atomic_json_dump


@dataclass
class Candidate:
    name: str
    relative_path: str
    domain: str
    source_split: str
    dhash: int
    quality_flags: list[str]


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def find_label_files(root: Path) -> list[Path]:
    json_files = list(root.rglob("*.json"))
    files = []
    for path in json_files:
        name = path.name.lower()
        if "label" in name and ("train" in name or "val" in name):
            files.append(path)
    # Some legal mirrors preserve BDD100K's annotations as one JSON file per
    # image under labels/{train,val}/ instead of the official combined files.
    if not files:
        for path in json_files:
            parts = {part.lower() for part in path.parts}
            if "labels" in parts and ("train" in parts or "val" in parts):
                files.append(path)
    if not files:
        raise FileNotFoundError(
            "Could not find BDD100K train/val label JSON files under the supplied root. "
            "Download the 100K Images and detection labels, then extract both first."
        )
    return sorted(files)


def index_images(root: Path) -> dict[str, Path]:
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    indexed: dict[str, Path] = {}
    for path in tqdm(root.rglob("*"), desc="Indexing BDD images"):
        if path.is_file() and path.suffix.lower() in extensions:
            indexed.setdefault(path.name, path)
    if not indexed:
        raise FileNotFoundError("No images were found under the BDD100K root")
    return indexed


def dhash_and_quality(path: Path) -> tuple[int, list[str]]:
    flags: list[str] = []
    with Image.open(path) as source:
        source.verify()
    with Image.open(path) as source:
        gray = ImageOps.exif_transpose(source).convert("L")
        tiny = gray.resize((9, 8), Image.Resampling.LANCZOS)
        values = np.asarray(tiny, dtype=np.int16)
        bits = values[:, 1:] > values[:, :-1]
        hash_value = 0
        for bit in bits.flatten():
            hash_value = (hash_value << 1) | int(bit)

        sample = np.asarray(gray.resize((256, 144), Image.Resampling.BILINEAR), dtype=np.float32)
        laplacian = (
            -4 * sample[1:-1, 1:-1]
            + sample[:-2, 1:-1]
            + sample[2:, 1:-1]
            + sample[1:-1, :-2]
            + sample[1:-1, 2:]
        )
        if float(laplacian.var()) < 18.0:
            flags.append("very_blurry")
        dark_fraction = float((sample < 10).mean())
        bright_fraction = float((sample > 250).mean())
        if dark_fraction > 0.65:
            flags.append("severely_underexposed")
        if bright_fraction > 0.40:
            flags.append("severely_overexposed")
    return hash_value, flags


def _load_annotation_file(path: Path) -> list[tuple[str, str, str]]:
    records: list[tuple[str, str, str]] = []
    lower_parts = {part.lower() for part in path.parts}
    source_split = "val" if "val" in lower_parts or "val" in path.name.lower() else "train"
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else [payload]
    for item in items:
        if not isinstance(item, dict):
            continue
        attributes = item.get("attributes") or {}
        timeofday = str(attributes.get("timeofday", "")).lower()
        raw_name = str(item.get("name", ""))
        if not raw_name or timeofday not in {"daytime", "night"}:
            continue
        name = raw_name if Path(raw_name).suffix else f"{raw_name}.jpg"
        records.append((name, timeofday, source_split))
    return records


def load_annotations(label_files: list[Path]) -> dict[str, tuple[str, str]]:
    annotations: dict[str, tuple[str, str]] = {}
    # Per-image mirrors contain 100k tiny files. Parallel reads avoid turning
    # dataset preparation into a long single-threaded filesystem crawl.
    workers = min(16, len(label_files))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        batches = executor.map(_load_annotation_file, label_files)
        for records in tqdm(batches, total=len(label_files), desc="Reading labels"):
            for name, timeofday, source_split in records:
                domain = "day" if timeofday == "daytime" else "night"
                annotations[name] = (domain, source_split)
    if not annotations:
        raise ValueError("No strict daytime/night annotations were found in the label files")
    return annotations


def duplicate_groups(candidates: list[Candidate], max_distance: int = 4) -> list[list[int]]:
    """LSH-assisted grouping of exact and near-identical 64-bit dHashes."""
    union_find = UnionFind(len(candidates))
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        for band in range(4):
            value = (candidate.dhash >> (band * 16)) & 0xFFFF
            buckets[(band, value)].append(index)
    checked: set[tuple[int, int]] = set()
    for members in buckets.values():
        for offset, left in enumerate(members):
            for right in members[offset + 1 :]:
                pair = (min(left, right), max(left, right))
                if pair in checked:
                    continue
                checked.add(pair)
                if (candidates[left].dhash ^ candidates[right].dhash).bit_count() <= max_distance:
                    union_find.union(left, right)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(candidates)):
        grouped[union_find.find(index)].append(index)
    return list(grouped.values())


def choose_splits(
    candidates: list[Candidate], groups: list[list[int]], counts: tuple[int, int, int], seed: int
) -> dict[str, list[Candidate]]:
    targets = dict(zip(("train", "val", "test"), counts, strict=True))
    result: dict[str, list[Candidate]] = {key: [] for key in targets}
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    for group in groups:
        group_items = [candidates[index] for index in group]
        available = [key for key in targets if targets[key] - len(result[key]) >= len(group_items)]
        if not available:
            continue
        split = max(available, key=lambda key: (targets[key] - len(result[key])) / targets[key])
        result[split].extend(group_items)
    for split, target in targets.items():
        if len(result[split]) < target:
            raise RuntimeError(
                f"Not enough duplicate-safe images for {split}: needed {target}, got {len(result[split])}"
            )
    return result


def write_contact_sheet(items: list[Candidate], source_root: Path, output: Path) -> None:
    chosen = items[:64]
    tile_w, tile_h = 192, 120
    sheet = Image.new("RGB", (tile_w * 8, tile_h * 8), "#111827")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(chosen):
        with Image.open(source_root / item.relative_path) as image:
            thumb = ImageOps.fit(
                ImageOps.exif_transpose(image).convert("RGB"), (tile_w, tile_h - 20)
            )
        x, y = (index % 8) * tile_w, (index // 8) * tile_h
        sheet.paste(thumb, (x, y))
        draw.text((x + 5, y + tile_h - 18), item.name[:20], fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def prepare(args: argparse.Namespace) -> None:
    source_root = Path(args.bdd_root).resolve()
    output_root = Path(args.output).resolve()
    image_index = index_images(source_root)
    annotations = load_annotations(find_label_files(source_root))
    candidates_by_domain: dict[str, list[Candidate]] = {"day": [], "night": []}
    missing = 0
    corrupt = 0

    def inspect_image(
        entry: tuple[str, tuple[str, str]],
    ) -> tuple[str, Candidate | None]:
        name, (domain, source_split) = entry
        image_path = image_index.get(name)
        if image_path is None:
            return "missing", None
        try:
            hash_value, flags = dhash_and_quality(image_path)
        except (OSError, ValueError):
            return "corrupt", None
        return (
            "ok",
            Candidate(
                name=name,
                relative_path=image_path.relative_to(source_root).as_posix(),
                domain=domain,
                source_split=source_split,
                dhash=hash_value,
                quality_flags=flags,
            ),
        )

    with ThreadPoolExecutor(max_workers=12) as executor:
        checked = executor.map(inspect_image, annotations.items())
        for status, candidate in tqdm(checked, total=len(annotations), desc="Checking images"):
            if status == "missing":
                missing += 1
            elif status == "corrupt":
                corrupt += 1
            elif candidate is not None:
                candidates_by_domain[candidate.domain].append(candidate)

    output_root.joinpath("manifests").mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "missing_images": missing,
        "corrupt_images": corrupt,
        "domains": {},
    }
    for domain_index, domain in enumerate(("day", "night")):
        candidates = candidates_by_domain[domain]
        groups = duplicate_groups(candidates, args.hash_distance)
        splits = choose_splits(
            candidates,
            groups,
            (args.train_count, args.val_count, args.test_count),
            args.seed + domain_index,
        )
        domain_summary: dict[str, object] = {
            "available": len(candidates),
            "duplicate_groups": len(groups),
            "flags": dict(Counter(flag for item in candidates for flag in item.quality_flags)),
        }
        for split, selected in splits.items():
            manifest = output_root / "manifests" / f"{split}_{domain}.jsonl"
            lines = [
                json.dumps(
                    {
                        "name": item.name,
                        "path": item.relative_path,
                        "source_split": item.source_split,
                        "quality_flags": item.quality_flags,
                        "dhash": f"{item.dhash:016x}",
                    }
                )
                for item in selected
            ]
            manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
            write_contact_sheet(
                selected, source_root, output_root / "review" / f"{split}_{domain}.jpg"
            )
            domain_summary[split] = len(selected)
        summary["domains"][domain] = domain_summary

    atomic_json_dump(
        {
            "format_version": 1,
            "source_root": str(source_root),
            "seed": args.seed,
            "counts": {
                "train": args.train_count,
                "val": args.val_count,
                "test": args.test_count,
            },
            "summary": summary,
        },
        output_root / "dataset.json",
    )
    print(f"Prepared dataset manifests at {output_root}")
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare strict, leakage-resistant BDD100K splits")
    parser.add_argument(
        "--bdd-root", required=True, help="Extracted BDD100K images and labels root"
    )
    parser.add_argument("--output", default="data/bdd100k_daynight")
    parser.add_argument("--train-count", type=int, default=5000)
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--hash-distance", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    prepare(build_parser().parse_args())


if __name__ == "__main__":
    main()
