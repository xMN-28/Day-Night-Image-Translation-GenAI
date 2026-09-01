from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

from .utils import pil_to_tensor, tensor_to_pil


class TurboReference:
    """Optional adapter around the official GaParmar/img2img-turbo implementation."""

    def __init__(
        self, direction: str, external_root: str | Path = "external/img2img-turbo"
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CycleGAN-Turbo reference requires an NVIDIA GPU")
        root = Path(external_root).resolve()
        source = root / "src"
        if not source.exists():
            raise FileNotFoundError(
                "Official Turbo reference is not installed. Run scripts/setup_turbo_reference.ps1 first."
            )
        sys.path.insert(0, str(source))
        try:
            from cyclegan_turbo import CycleGAN_Turbo  # type: ignore
        except Exception as error:
            raise RuntimeError(
                "Could not import the official Turbo reference. Install the optional `turbo` dependencies."
            ) from error
        model_name = "day_to_night" if direction == "day_to_night" else "night_to_day"
        self.model = CycleGAN_Turbo(
            pretrained_name=model_name, ckpt_folder=str(root / "checkpoints")
        )
        self.model.eval().half()
        self.direction = direction

    @torch.inference_mode()
    def __call__(self, image: Image.Image) -> Image.Image:
        resized = image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        tensor = pil_to_tensor(resized).unsqueeze(0).cuda().half()
        output = self.model(tensor)
        return tensor_to_pil(output).resize(image.size, Image.Resampling.LANCZOS)
