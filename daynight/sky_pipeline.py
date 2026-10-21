"""Offline experimental day-to-night pipeline, retaining trained V2 generators."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

from .models.networks import ResnetGenerator
from .sky_refinement import SkyStudent, compose_night, phase_average, predict_matte
from .utils import pil_to_tensor, tensor_to_pil

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "LumiCycle Sky candidate"


class SkyPipeline:
    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or ROOT / "configs/sky_candidate.yaml"
        self.lock = threading.RLock()
        self.generator = self.student = None

    def _load(self):
        if self.generator is not None:
            return
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.generator_path = ROOT / self.config["generator_checkpoint"]
        self.matte_path = ROOT / self.config["matte_checkpoint"]
        for path in (self.generator_path, self.matte_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Sky candidate weights are missing: {path}. See docs/SKY_REFINEMENT_RESEARCH_AND_EXPERIMENTS.md"
                )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(min(torch.get_num_threads(), 8))
        payload = torch.load(self.generator_path, map_location="cpu", weights_only=False)
        options = payload["config"]["model"]
        if options.get("detail_refinement"):
            raise ValueError("Sky candidate requires a V2 ResNet generator checkpoint")
        generator = ResnetGenerator(
            base_channels=int(options["base_channels"]),
            blocks=int(options["generator_blocks"]),
            attention=bool(options["attention"]),
            upsampling=options.get("day_upsampling", "transpose"),
            residual_norm=options.get("day_residual_norm", "instance"),
            detail_skip=bool(options.get("day_detail_skip", False)),
        )
        generator.load_state_dict(payload["ema"]["G_day_night"]["shadow"])
        self.global_step = payload["step"]
        matte = torch.load(self.matte_path, map_location="cpu", weights_only=False)
        student = SkyStudent(outputs=matte["model"]["head.weight"].shape[0])
        student.load_state_dict(matte["model"])
        self.matte_step = matte["step"]
        self.generator = generator.to(self.device).eval()
        self.student = student.to(self.device).eval()

    @torch.inference_mode()
    def translate(self, image: Image.Image, maximum_edge: int = 768):
        from .inference import _fit_and_pad

        with self.lock:
            self._load()
            limit = min(maximum_edge, int(self.config.get("maximum_edge", 768)))
            started = time.perf_counter()
            fitted, crop = _fit_and_pad(image, limit)
            x = pil_to_tensor(fitted)[None].to(self.device)
            with torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ):
                raw = (
                    phase_average(self.generator, x)
                    if self.config.get("phase_average", True)
                    else self.generator(x)
                )
                rgb = (x.float() + 1) / 2
                alpha, vegetation = predict_matte(self.student, rgb)
            output = compose_night(rgb, (raw.float() + 1) / 2, alpha, vegetation=vegetation)
            result = tensor_to_pil(output * 2 - 1).crop(crop)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            return result, {
                "model": MODEL_NAME,
                "status": "experimental",
                "direction": "day_to_night",
                "input_resolution": f"{image.width}x{image.height}",
                "inference_resolution": f"{result.width}x{result.height}",
                "seconds": round(time.perf_counter() - started, 3),
                "device": str(self.device),
                "generator_global_step": self.global_step,
                "matte_step": self.matte_step,
                "checkpoint": str(self.generator_path),
                "matte_checkpoint": str(self.matte_path),
                "method": "fine-tuned V2 + locally trained semantic matte + explicit sky compositing",
                "teacher_required_at_inference": False,
            }


_PIPELINE = SkyPipeline()


def translate_sky(image: Image.Image, maximum_edge: int = 768):
    return _PIPELINE.translate(image, maximum_edge)
