from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .utils import pil_to_tensor


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    domain: str
    source_split: str = ""
    flagged: bool = False


def read_manifest(dataset_root: str | Path, split: str, domain: str) -> list[ImageRecord]:
    dataset_root = Path(dataset_root)
    metadata = json.loads((dataset_root / "dataset.json").read_text(encoding="utf-8"))
    source_root = Path(metadata["source_root"])
    manifest = dataset_root / "manifests" / f"{split}_{domain}.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Missing {manifest}. Run `python -m daynight.prepare_data --bdd-root <path>` first."
        )
    records: list[ImageRecord] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        records.append(
            ImageRecord(
                path=source_root / item["path"],
                domain=domain,
                source_split=item.get("source_split", ""),
                flagged=bool(item.get("quality_flags")),
            )
        )
    return records


def _resize_short_side(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    scale = size / min(width, height)
    return image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)


def train_transform(image: Image.Image, resize_size: int, crop_size: int) -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = _resize_short_side(image, resize_size)
    width, height = image.size
    left = random.randint(0, max(0, width - crop_size))
    top = random.randint(0, max(0, height - crop_size))
    image = image.crop((left, top, left + crop_size, top + crop_size))
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    return pil_to_tensor(image)


def eval_transform(image: Image.Image, crop_size: int) -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = _resize_short_side(image, crop_size)
    width, height = image.size
    left, top = (width - crop_size) // 2, (height - crop_size) // 2
    return pil_to_tensor(image.crop((left, top, left + crop_size, top + crop_size)))


class UnpairedDayNightDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Unpaired day/night images; B is sampled independently during training."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 256,
        resize_size: int = 286,
    ) -> None:
        self.day = read_manifest(root, split, "day")
        self.night = read_manifest(root, split, "night")
        self.training = split == "train"
        self.image_size = image_size
        self.resize_size = resize_size
        self.hard_detail_fraction = 0.0
        self.hard_night: list[int] = []
        self.normal_night: list[int] = []
        if not self.day or not self.night:
            raise ValueError(f"Split {split!r} must contain both day and night images")

    @staticmethod
    def _detail_score(path: Path) -> float:
        with Image.open(path) as image:
            gray = ImageOps.exif_transpose(image).convert("L")
            width = 96
            height = max(32, round(gray.height * width / gray.width))
            array = np.asarray(gray.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
        upper = array[: max(2, round(height * 0.65))] / 255.0
        horizontal = np.abs(np.diff(upper, axis=1)).mean()
        vertical = np.abs(np.diff(upper, axis=0)).mean()
        return float(horizontal + vertical)

    def configure_hard_detail_sampling(
        self, fraction: float, cache_path: str | Path
    ) -> list[float]:
        """Return day-index weights and configure matched hard sampling for night."""
        self.hard_detail_fraction = min(max(float(fraction), 0.0), 1.0)
        cache_path = Path(cache_path)
        scores: dict[str, float] = {}
        if cache_path.exists():
            scores = json.loads(cache_path.read_text(encoding="utf-8"))
        changed = False
        for record in [*self.day, *self.night]:
            key = str(record.path)
            if key not in scores:
                scores[key] = self._detail_score(record.path)
                changed = True
        if changed:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")

        def split_indices(records: list[ImageRecord]) -> tuple[list[int], list[int]]:
            ranked = sorted(range(len(records)), key=lambda i: scores[str(records[i].path)])
            hard_count = max(1, round(len(ranked) * 0.25))
            return ranked[-hard_count:], ranked[:-hard_count]

        hard_day, normal_day = split_indices(self.day)
        self.hard_night, self.normal_night = split_indices(self.night)
        hard_weight = self.hard_detail_fraction / max(1, len(hard_day))
        normal_weight = (1 - self.hard_detail_fraction) / max(1, len(normal_day))
        hard_set = set(hard_day)
        return [hard_weight if index in hard_set else normal_weight for index in range(len(self.day))]

    def __len__(self) -> int:
        return max(len(self.day), len(self.night))

    def _load(self, record: ImageRecord) -> torch.Tensor:
        with Image.open(record.path) as image:
            if self.training:
                return train_transform(image, self.resize_size, self.image_size)
            return eval_transform(image, self.image_size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        day_record = self.day[index % len(self.day)]
        if self.training and self.hard_night and random.random() < self.hard_detail_fraction:
            night_index = random.choice(self.hard_night)
        elif self.training and self.normal_night:
            night_index = random.choice(self.normal_night)
        else:
            night_index = random.randrange(len(self.night)) if self.training else index % len(self.night)
        night_record = self.night[night_index]
        return {
            "day": self._load(day_record),
            "night": self._load(night_record),
            "day_path": str(day_record.path),
            "night_path": str(night_record.path),
        }


def iter_image_paths(records: Iterable[ImageRecord]) -> Iterable[Path]:
    for record in records:
        yield record.path
