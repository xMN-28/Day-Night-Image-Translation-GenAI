from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .inference import _fit_and_pad
from .models import build_models
from .utils import pil_to_tensor, tensor_to_pil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = [
    ("V1 best · 13,000 steps", "runs/lumicycle_bdd100k/checkpoints/step_00013000.pt"),
    ("V1 final · 40,000 steps", "runs/lumicycle_bdd100k/checkpoints/step_00040000.pt"),
    ("V2 best · 4,500 fine-tune steps", "runs/lumicycle_v2_bdd100k/checkpoints/step_00004500.pt"),
    ("V2 final · 12,000 fine-tune steps", "runs/lumicycle_v2_bdd100k/checkpoints/step_00012000.pt"),
    ("V2.1 best · 5,500 steps (failed ablation)", "runs/lumicycle_v2_1_bdd100k/checkpoints/step_00005500.pt"),
    ("V2.1 final · 6,000 steps (failed ablation)", "runs/lumicycle_v2_1_bdd100k/checkpoints/step_00006000.pt"),
    ("LumiRender · 34,004 steps", "runs/lumirender_physics_bdd100k/checkpoints/step_00034004.pt"),
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/segoeuib.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


@torch.inference_mode()
def _generate(
    source: Image.Image, checkpoint_path: Path, maximum_edge: int, device: torch.device
) -> Image.Image:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    models = build_models(payload["config"])
    generator = models["G_day_night"]
    state = payload["models"]["G_day_night"]
    ema = payload.get("ema", {}).get("G_day_night")
    if isinstance(ema, dict):
        state = ema.get("shadow", state)
    generator.load_state_dict(state)
    generator.to(device).eval()
    padded, crop = _fit_and_pad(source, maximum_edge)
    tensor = pil_to_tensor(padded).unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        if str(payload["config"]["model"].get("kind", "")).lower() == "lumirender":
            output = generator(tensor, seed=0, night_intensity=1.0)
        else:
            output = generator(tensor)
    result = tensor_to_pil(output).crop(crop)
    generator.to("cpu")
    del generator, models, payload, tensor, output
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _panel(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    title_height = 54
    canvas = Image.new("RGB", (size[0], size[1] + title_height), "#0b111b")
    fitted = ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)
    canvas.paste(fitted, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    font = _font(23)
    box = draw.textbbox((0, 0), label, font=font)
    draw.text(((size[0] - (box[2] - box[0])) / 2, 12), label, fill="#f4f7fb", font=font)
    return canvas


def build_comparison(input_path: Path, output_path: Path, maximum_edge: int = 768) -> Path:
    source = ImageOps.exif_transpose(Image.open(input_path)).convert("RGB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = [("Untouched test input", source)]
    for label, relative_checkpoint in CHECKPOINTS:
        checkpoint = PROJECT_ROOT / relative_checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing comparison checkpoint: {checkpoint}")
        print(f"Generating {label}...", flush=True)
        entries.append((label, _generate(source, checkpoint, maximum_edge, device)))

    fitted, _ = _fit_and_pad(source, maximum_edge)
    panel_size = fitted.size
    columns = 2
    rows = math.ceil(len(entries) / columns)
    panels = [_panel(image, label, panel_size) for label, image in entries]
    canvas = Image.new(
        "RGB", (columns * panel_size[0], rows * panels[0].height), "#070b12"
    )
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % columns) * panel.width, (index // columns) * panel.height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the fixed LumiRender model comparison")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-edge", type=int, default=768)
    args = parser.parse_args()
    print(build_comparison(args.input, args.output, args.maximum_edge).resolve())


if __name__ == "__main__":
    main()
