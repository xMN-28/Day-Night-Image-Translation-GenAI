from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .data import UnpairedDayNightDataset, eval_transform, train_transform


class LumiRenderDataset(Dataset[dict[str, Any]]):
    """BDD100K plus optional aligned or coarse-aligned external day/night pairs."""

    def __init__(
        self,
        bdd_root: str | Path,
        split: str,
        image_size: int,
        resize_size: int,
        source_manifests: list[str] | None = None,
        paired_fraction: float = 0.5,
        pseudo_manifest: str | None = None,
    ) -> None:
        self.bdd = UnpairedDayNightDataset(bdd_root, split, image_size, resize_size)
        self.training = split == "train"
        self.paired_fraction = float(paired_fraction)
        self.deterministic_day = False
        self.pseudo: dict[str, Path] = {}
        if pseudo_manifest and Path(pseudo_manifest).exists():
            for line in Path(pseudo_manifest).read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                self.pseudo[str(Path(item["image_path"]).resolve())] = Path(item["target_path"])
        self.pairs: list[dict[str, Any]] = []
        for manifest_value in source_manifests or []:
            manifest = Path(manifest_value)
            if not manifest.exists():
                continue
            for line in manifest.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if item.get("split", "train") == split:
                    item["manifest_root"] = str(manifest.parent)
                    self.pairs.append(item)

    @property
    def image_size(self) -> int:
        return self.bdd.image_size

    @image_size.setter
    def image_size(self, value: int) -> None:
        self.bdd.image_size = value

    @property
    def resize_size(self) -> int:
        return self.bdd.resize_size

    @resize_size.setter
    def resize_size(self, value: int) -> None:
        self.bdd.resize_size = value

    def __len__(self) -> int:
        return max(len(self.bdd), len(self.pairs)) if self.pairs else len(self.bdd)

    def _transform(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            if self.training:
                return train_transform(image, self.resize_size, self.image_size)
            return eval_transform(image, self.image_size)

    def _transform_pair(
        self, day_path: Path, night_path: Path
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        if not self.training:
            return self._transform(day_path), self._transform(night_path), None
        state = random.getstate()
        day = self._transform(day_path)
        random.setstate(state)
        night = self._transform(night_path)
        return day, night, state

    def _confidence(self, item: dict[str, Any], state: object, fallback: float) -> torch.Tensor:
        if not item.get("confidence_path"):
            return torch.full((1, self.image_size, self.image_size), fallback)
        path = self._resolve(item, "confidence_path")
        if self.training and state is not None:
            random.setstate(state)
        confidence = self._transform(path)[:1].add(1).div(2)
        return confidence.clamp(0, 1) * fallback

    @staticmethod
    def _resolve(item: dict[str, Any], key: str) -> Path:
        path = Path(item[key])
        return path if path.is_absolute() else Path(item["manifest_root"]) / path

    def __getitem__(self, index: int) -> dict[str, Any]:
        use_pair = bool(self.pairs) and (
            not self.training or random.random() < self.paired_fraction
        )
        if not use_pair:
            sample = self.bdd[index % len(self.bdd)]
            if self.deterministic_day:
                with Image.open(sample["day_path"]) as image:
                    sample["day"] = eval_transform(image, self.image_size)
            sample.update(
                {
                    "aligned": torch.tensor(0.0),
                    "pair_confidence": torch.zeros(1, self.image_size, self.image_size),
                    "source_name": "bdd100k",
                }
            )
            return self._add_pseudo(sample)
        item = self.pairs[index % len(self.pairs)]
        day_path = self._resolve(item, "day_path")
        night_path = self._resolve(item, "night_path")
        confidence = float(item.get("confidence", 1.0))
        day, night, transform_state = self._transform_pair(day_path, night_path)
        return self._add_pseudo({
            "day": day,
            "night": night,
            "day_path": str(day_path),
            "night_path": str(night_path),
            "aligned": torch.tensor(1.0),
            "pair_confidence": self._confidence(item, transform_state, confidence),
            "source_name": str(item.get("source", "external")),
        })

    def _add_pseudo(self, sample: dict[str, Any]) -> dict[str, Any]:
        target = self.pseudo.get(str(Path(sample["day_path"]).resolve()))
        if not self.deterministic_day or target is None or not target.exists():
            sample.update(
                {
                    "pseudo_valid": torch.tensor(0.0),
                    "pseudo_depth": torch.zeros(1, self.image_size, self.image_size),
                    "pseudo_semantic": torch.zeros(6, self.image_size, self.image_size),
                }
            )
            return sample
        data = np.load(target)
        depth = torch.from_numpy(data["depth"].astype(np.float32) / 65535.0)[None, None]
        semantic = torch.from_numpy(data["semantic"].astype(np.float32) / 255.0)[None]
        depth = F.interpolate(depth, (self.image_size, self.image_size), mode="bilinear")[0]
        semantic = F.interpolate(semantic, (self.image_size, self.image_size), mode="nearest")[0]
        sample.update(
            {"pseudo_valid": torch.tensor(1.0), "pseudo_depth": depth, "pseudo_semantic": semantic}
        )
        return sample
