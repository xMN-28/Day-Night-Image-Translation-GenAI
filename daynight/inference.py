from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps

from .models import build_models
from .turbo_reference import TurboReference
from .utils import pil_to_tensor, tensor_to_pil

MODEL_PATHS = {
    "LumiCycle": Path("runs/lumicycle_bdd100k/checkpoints"),
    "CycleGAN": Path("runs/cyclegan_bdd100k/checkpoints"),
}


def _resolve_checkpoint(value: str | Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    pointer = path / "best.json"
    if not pointer.exists():
        pointer = path / "latest.json"
    if not pointer.exists():
        raise FileNotFoundError(
            f"No trained checkpoint found in {path}. Train the model first or set its checkpoint path."
        )
    data = json.loads(pointer.read_text(encoding="utf-8"))
    checkpoint = path / data["filename"]
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint pointer refers to missing file: {checkpoint}")
    return checkpoint


def _fit_and_pad(
    image: Image.Image, maximum_edge: int
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = min(1.0, maximum_edge / max(width, height))
    if scale < 1:
        image = image.resize(
            (round(width * scale), round(height * scale)), Image.Resampling.LANCZOS
        )
    width, height = image.size
    right = (-width) % 4
    bottom = (-height) % 4
    padded = ImageOps.expand(image, (0, 0, right, bottom), fill=None)
    if right or bottom:
        # Pillow cannot reflection-pad directly; mirror the last few pixels.
        if right:
            strip = image.crop((max(0, width - right), 0, width, height)).transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )
            padded.paste(strip.resize((right, height)), (width, 0))
        if bottom:
            strip = padded.crop((0, max(0, height - bottom), width + right, height)).transpose(
                Image.Transpose.FLIP_TOP_BOTTOM
            )
            padded.paste(strip.resize((width + right, bottom)), (0, height))
    return padded, (0, 0, width, height)


class ModelManager:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_name: str | None = None
        self.checkpoint_path: Path | None = None
        self.models: dict[str, torch.nn.Module] | None = None
        self.turbo: TurboReference | None = None
        self.turbo_direction: str | None = None

    def _load_custom(self, model_name: str) -> None:
        configured = os.getenv(f"{model_name.upper()}_CHECKPOINT")
        checkpoint = _resolve_checkpoint(configured or MODEL_PATHS[model_name])
        if (
            self.loaded_name == model_name
            and self.checkpoint_path == checkpoint
            and self.models is not None
        ):
            return
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        models = build_models(payload["config"])
        for name, model in models.items():
            state = payload["models"][name]
            if name in payload.get("ema", {}):
                state = payload["ema"][name].get("shadow", state)
            model.load_state_dict(state)
            model.to(self.device).eval()
        self.models = models
        self.loaded_name = model_name
        self.checkpoint_path = checkpoint
        self.turbo = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def translate_custom(
        self, image: Image.Image, direction: str, model_name: str, maximum_edge: int
    ) -> tuple[Image.Image, dict[str, Any]]:
        self._load_custom(model_name)
        assert self.models is not None
        padded, crop = _fit_and_pad(image, maximum_edge)
        tensor = pil_to_tensor(padded).unsqueeze(0).to(self.device)
        generator_name = "G_day_night" if direction == "day_to_night" else "G_night_day"
        started = time.perf_counter()
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else torch.no_grad()
        )
        with amp:
            output = self.models[generator_name](tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        result = tensor_to_pil(output).crop(crop)
        metadata = {
            "model": model_name,
            "direction": direction,
            "input_resolution": f"{image.width}x{image.height}",
            "inference_resolution": f"{result.width}x{result.height}",
            "seconds": round(elapsed, 3),
            "device": str(self.device),
            "checkpoint": str(self.checkpoint_path),
        }
        return result, metadata

    def translate_turbo(
        self, image: Image.Image, direction: str
    ) -> tuple[Image.Image, dict[str, Any]]:
        if self.turbo is None or self.turbo_direction != direction:
            self.models = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.turbo = TurboReference(direction)
            self.turbo_direction = direction
        started = time.perf_counter()
        result = self.turbo(image)
        elapsed = time.perf_counter() - started
        return result, {
            "model": "Turbo reference (external pretrained)",
            "direction": direction,
            "input_resolution": f"{image.width}x{image.height}",
            "inference_resolution": f"{result.width}x{result.height}",
            "seconds": round(elapsed, 3),
            "device": str(self.device),
            "checkpoint": "Official GaParmar/img2img-turbo weights",
        }


_MANAGER = ModelManager()


def translate(
    image: Image.Image,
    direction: str,
    model_name: str = "LumiCycle",
    maximum_edge: int = 768,
) -> tuple[Image.Image, dict[str, Any]]:
    if image is None:
        raise ValueError("Upload an image before translating")
    normalized = direction.lower().replace(" ", "_").replace("→", "to")
    if normalized in {"day_to_night", "day_to_to_night", "day2night"}:
        normalized = "day_to_night"
    elif normalized in {"night_to_day", "night_to_to_day", "night2day"}:
        normalized = "night_to_day"
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    maximum_edge = max(128, min(int(maximum_edge), 1536))
    try:
        if model_name == "Turbo reference":
            return _MANAGER.translate_turbo(image, normalized)
        if model_name not in MODEL_PATHS:
            raise ValueError(f"Unknown model: {model_name}")
        return _MANAGER.translate_custom(image, normalized, model_name, maximum_edge)
    except torch.OutOfMemoryError as error:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(
            "The GPU ran out of memory. Close GPU-heavy apps or retry with a 512 px maximum edge."
        ) from error
