"""Training-independent, explicit sky-matte compositing for RGB student inference."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .sky_refinement import SkyStudent, compose_night, predict_matte

ROOT = Path(__file__).resolve().parents[1]


class SkyRepair:
    def __init__(self, checkpoint: Path | None = None):
        self.checkpoint = checkpoint or Path(os.getenv(
            "LUMICYCLE_SKY_MATTE_CHECKPOINT", str(ROOT / "runs/sky_student_regions_v2/best.pt")
        ))
        self.student = None
        self.step = None

    def _load(self, device: torch.device):
        if self.student is not None:
            self.student.to(device)
            return
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"Sky repair requires its locally trained matte weights: {self.checkpoint}. "
                "Restore the weights or explicitly request sky_repair=False for raw inference."
            )
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        state = payload["model"]
        student = SkyStudent(outputs=state["head.weight"].shape[0])
        student.load_state_dict(state)
        student.requires_grad_(False).to(device).eval()
        self.student = student
        self.step = payload.get("step")

    @torch.inference_mode()
    def apply(self, source: torch.Tensor, prediction: torch.Tensor):
        """Inputs/output [-1,1]; matte guided by original daytime pixels, not GAN output."""
        self._load(source.device)
        rgb = (source.float() + 1) / 2
        base = (prediction.float() + 1) / 2
        with torch.autocast(source.device.type, dtype=torch.bfloat16,
                            enabled=source.device.type == "cuda"):
            alpha, vegetation = predict_matte(self.student, rgb, recover_color_holes=True)
        # Color unmixing and explicit blending stay float32 to protect thin boundaries.
        result = compose_night(
            rgb, base, alpha.float(), vegetation=vegetation, recover_thin_edges=True
        )
        if not torch.isfinite(result).all():
            raise FloatingPointError("Sky repair produced nonfinite pixels")
        return result * 2 - 1, {
            "sky_repair": True,
            "sky_repair_version": "2026-10-22-v1",
            "sky_repair_method": "local semantic matte + guided color unmixing + explicit night-sky compositing",
            "matte_checkpoint": str(self.checkpoint),
            "matte_step": self.step,
            "sky_coverage": round(float(alpha.float().mean()), 5),
        }
