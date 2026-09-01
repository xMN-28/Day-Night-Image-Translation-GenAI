from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

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
        if not self.day or not self.night:
            raise ValueError(f"Split {split!r} must contain both day and night images")

    def __len__(self) -> int:
        return max(len(self.day), len(self.night))

    def _load(self, record: ImageRecord) -> torch.Tensor:
        with Image.open(record.path) as image:
            if self.training:
                return train_transform(image, self.resize_size, self.image_size)
            return eval_transform(image, self.image_size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        day_record = self.day[index % len(self.day)]
        night_index = (
            random.randrange(len(self.night)) if self.training else index % len(self.night)
        )
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
