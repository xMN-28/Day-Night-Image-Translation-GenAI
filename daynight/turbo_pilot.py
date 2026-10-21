"""Clearly labeled official CycleGAN-Turbo validation reference, isolated env.

No local LumiCycle weights are replaced or trained. Heavy files live in the
user-approved E: cache. The upstream source checkout is left unmodified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

from .data import read_manifest
from .download_reference import ranged_download
from .utils import atomic_json_dump, pil_to_tensor, tensor_to_pil

REVISION = "b261bac6fd2cf515557d5d0707481eafa0485ec2"
SOURCE_COMMIT = "86f54146590ffb4543c8cf85b5a36657da670924"
OFFICIAL_URL = "https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch_official_checkpoint(path):
    ranged_download(OFFICIAL_URL, path, workers=4)


def load_official(cache):
    # Local-only loading uses the pinned CLI snapshot; no hidden main-branch fetch.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_CACHE"] = str(cache / "huggingface/hub")
    os.environ["HF_HOME"] = str(cache / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(cache / "huggingface/hub")
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import AutoTokenizer, CLIPTextModel

    snapshot = cache / f"huggingface/hub/models--stabilityai--sd-turbo/snapshots/{REVISION}"
    if not (snapshot / "unet/config.json").is_file():
        raise FileNotFoundError(
            "Pinned SD-Turbo snapshot missing; run the documented hf download first"
        )
    source = cache / "img2img-turbo"
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"Reference source changed: {commit}")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("Official reference checkout is modified; inspect before using it")
    # Patch only repository resolution/dtype in this isolated process. The model
    # forward, trained weights, scheduler, LoRA, and VAE skip connections are upstream.
    for klass in (AutoTokenizer, CLIPTextModel, AutoencoderKL, UNet2DConditionModel, DDPMScheduler):
        original = klass.from_pretrained.__func__
        weighted = klass in (CLIPTextModel, AutoencoderKL, UNet2DConditionModel)

        def pinned(cls, name, *args, _original=original, _weighted=weighted, **kwargs):
            if str(name) == "stabilityai/sd-turbo":
                name = str(snapshot)
            kwargs["local_files_only"] = True
            if _weighted:
                kwargs["torch_dtype"] = torch.float16
            return _original(cls, name, *args, **kwargs)

        klass.from_pretrained = classmethod(pinned)
    sys.path.insert(0, str(source / "src"))
    from cyclegan_turbo import CycleGAN_Turbo

    checkpoint = cache / "official_models/day2night.pkl"
    fetch_official_checkpoint(checkpoint)

    class CheckedTurbo(CycleGAN_Turbo):
        def load_ckpt_from_url(self, url, ckpt_folder):
            if url != OFFICIAL_URL:
                raise ValueError(f"Unexpected official checkpoint request: {url}")
            # Upstream torch.load assumes older defaults. Explicit CPU/weights-only
            # loading avoids accidental CUDA allocations and unrestricted pickle.
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.load_ckpt_from_state_dict(state)

    print("Loading official external CycleGAN-Turbo day_to_night", flush=True)
    model = CheckedTurbo(pretrained_name="day_to_night", ckpt_folder=str(checkpoint.parent))
    return model.eval().half().requires_grad_(False), checkpoint, commit


def fit_input(image, edge):
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = min(1, edge / max(image.size))
    if scale < 1:
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
        )
    x = pil_to_tensor(image)[None]
    h, w = x.shape[-2:]
    x = F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode="reflect")
    return image, x, (0, 0, w, h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("E:/LumiCycleCache"))
    parser.add_argument("--output", type=Path, default=Path("outputs/turbo_reference_val"))
    parser.add_argument("--indices", default="0,71,142,213,284,355,426,499")
    parser.add_argument("--edge", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2029)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This official reference pilot requires CUDA")
    torch.set_num_threads(6)
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint, commit = load_official(args.cache_root)
    args.output.mkdir(parents=True, exist_ok=True)
    records = read_manifest("data/bdd100k_daynight", "val", "day")
    rows = []
    sheet = Image.new("RGB", (1024, 312 * len(args.indices.split(","))), "#101010")
    draw = ImageDraw.Draw(sheet)
    with torch.inference_mode():
        for row, index in enumerate(map(int, args.indices.split(","))):
            source, x, crop = fit_input(Image.open(records[index].path), args.edge)
            x = x.cuda().half()
            # VAE samples are stochastic; set the seed independently per input.
            torch.manual_seed(args.seed + index)
            torch.cuda.manual_seed_all(args.seed + index)
            torch.cuda.synchronize()
            started = time.perf_counter()
            y = model(x)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            if not torch.isfinite(y).all():
                raise FloatingPointError("Nonfinite official reference output")
            output = tensor_to_pil(y).crop(crop)
            output.save(args.output / f"{index:04d}_turbo_external.png")
            source.save(args.output / f"{index:04d}_input.png")
            draw.text((8, row * 312 + 4), f"val {index}: Input", fill="white")
            draw.text(
                (520, row * 312 + 4), "Official CycleGAN-Turbo (external pretrained)", fill="white"
            )
            sheet.paste(source.resize((512, 288)), (0, row * 312 + 24))
            sheet.paste(output.resize((512, 288)), (512, row * 312 + 24))
            record = {
                "index": index,
                "source": str(records[index].path),
                "seconds": seconds,
                "size": output.size,
                "seed": args.seed + index,
            }
            rows.append(record)
            print(json.dumps(record), flush=True)
    sheet.save(args.output / "comparison_gallery.jpg", quality=95)
    atomic_json_dump(
        {
            "model": "official CycleGAN-Turbo day_to_night, externally pretrained",
            "source_commit": commit,
            "base_model": "stabilityai/sd-turbo",
            "base_revision": REVISION,
            "official_checkpoint_url": OFFICIAL_URL,
            "checkpoint_sha256": digest(checkpoint),
            "split": "val",
            "precision": "float16",
            "preprocessing": "aspect-preserving resize, reflect pad to 8",
            "peak_torch_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "records": rows,
            "local_model_changed": False,
            "source_code_sha256": digest(Path(__file__)),
        },
        args.output / "evidence.json",
    )


if __name__ == "__main__":
    main()
