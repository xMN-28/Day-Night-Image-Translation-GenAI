from __future__ import annotations

import json
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def atomic_torch_save(payload: Any, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def atomic_json_dump(payload: Any, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        for attempt in range(12):
            try:
                os.replace(temporary, destination)
                return
            except PermissionError:
                if attempt == 11:
                    raise
                # Windows may briefly lock a JSON file while the UI reads it.
                time.sleep(0.01 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(-1, 1)
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = ((tensor.permute(1, 2, 0).numpy() + 1.0) * 127.5).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
