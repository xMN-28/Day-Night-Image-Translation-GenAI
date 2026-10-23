"""EXTERNAL Klein editing reference and refinement of the trained local inverse.

Two separate process stages prevent concurrent residency of two ~8 GB components
in system RAM. No public inference registrations or local weight updates.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .experiment_night_to_day import visibility_control
from .prepare_klein_reference import MODEL, REVISION, ROOT
from .turbo_pilot import digest
from .utils import atomic_json_dump, atomic_torch_save

PROMPT = (
    "Change this photograph to a realistic overcast daytime scene. Keep the exact camera "
    "position, composition, buildings, road markings, vehicles, people and signs. Replace "
    "night illumination with neutral natural daylight and a coherent pale daytime sky. "
    "Remove artificial light bloom and headlight pools without removing or moving light "
    "fixtures or objects. Preserve fine wires, branches, geometry and textures."
)
EMBEDDINGS = ROOT / "night_to_day_prompt128.pt"


def environment():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(6)


@torch.inference_mode()
def encode():
    from diffusers import Flux2KleinPipeline
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    if EMBEDDINGS.exists():
        saved = torch.load(EMBEDDINGS, map_location="cpu", weights_only=False)
        if saved["prompt"] != PROMPT or saved["revision"] != REVISION:
            raise ValueError("Existing prompt cache has a different recipe")
        print("Matching prompt embedding cache exists; no re-encoding needed", flush=True)
        return
    model = (
        Qwen3ForCausalLM.from_pretrained(
            ROOT / "text_encoder",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .eval()
        .requires_grad_(False)
    )
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "tokenizer", local_files_only=True)
    parameters_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    if torch.cuda.mem_get_info()[0] < parameters_bytes * 1.1 + 512 * 1024**2:
        raise RuntimeError(
            "Insufficient free CUDA memory for bounded text encoding; existing processes untouched"
        )
    model.cuda()
    started = time.monotonic()
    embeddings = Flux2KleinPipeline._get_qwen3_prompt_embeds(
        model,
        tokenizer,
        PROMPT,
        device=torch.device("cuda"),
        max_sequence_length=128,
    ).cpu()
    atomic_torch_save(
        {
            "prompt": PROMPT,
            "revision": REVISION,
            "prompt_embeds": embeddings,
            "max_sequence_length": 128,
            "text_layers": [9, 18, 27],
        },
        EMBEDDINGS,
    )
    print(
        f"Prompt encoded {list(embeddings.shape)} in {time.monotonic() - started:.1f}s; exit releases text weights",
        flush=True,
    )


@torch.inference_mode()
def infer(args):
    import diffusers
    import transformers
    from diffusers import Flux2KleinPipeline

    if args.output.exists():
        raise FileExistsError("Keep previous reference results; use a new output directory")
    manifest_path = args.targets / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["test_used"] or manifest["direction"] != "night_to_day":
        raise ValueError("Development validation only")
    rows = {row["index"]: row for row in manifest["records"] if row["split"] == "val"}
    local_metrics = json.loads((args.local_outputs / "metrics.json").read_text())
    if local_metrics["global_step"] != 16000:
        raise ValueError("This matched reference labels and compares the retained 16,000-step parent only")
    indices = [int(value) for value in args.indices.split(",")]
    modes = args.modes.split(",")
    if (
        not modes
        or len(modes) != len(set(modes))
        or any(mode not in {"source", "local", "lifted", "local_plus_lifted"} for mode in modes)
    ):
        raise ValueError("Unknown or duplicate editing conditioning mode")
    if not indices or len(set(indices)) != len(indices) or any(i not in rows for i in indices):
        raise ValueError("Only unique frozen development IDs; lockbox excluded")
    saved = torch.load(EMBEDDINGS, map_location="cpu", weights_only=False)
    if saved["prompt"] != PROMPT or saved["revision"] != REVISION:
        raise ValueError("Prompt embedding provenance mismatch")
    # No text model is loaded in this process, preventing ~16 GB host residency.
    pipeline = Flux2KleinPipeline.from_pretrained(
        ROOT,
        text_encoder=None,
        tokenizer=None,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model_bytes = sum(p.numel() * p.element_size() for p in pipeline.transformer.parameters())
    if torch.cuda.mem_get_info()[0] < model_bytes + 1024**3:
        raise RuntimeError(
            "Insufficient free GPU memory for the 512px reference; no existing worker stopped"
        )
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    args.output.mkdir(parents=True)
    report = {
        "model": MODEL,
        "revision": REVISION,
        "external_pretrained": True,
        "steps": 4,
        "guidance_scale": 1.0,
        "precision": "BF16",
        "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__,
        "transformers_version": transformers.__version__,
        "prompt": PROMPT,
        "prompt_cache_sha256": digest(EMBEDDINGS),
        "seed_base": 61000,
        "local_parent_global_step": 16000,
        "test_used": False,
        "lockbox_used": False,
        "public_demo_unchanged": True,
        "records": [],
        "conditioning_modes": modes,
        "warning": "Both new variants depend on EXTERNAL editing; not locally trained improvements",
    }
    for index in indices:
        row = rows[index]
        source = Image.open(row["input"]).convert("RGB")
        local = Image.open(args.local_outputs / f"val_{index:04d}.png").convert("RGB")
        folder = args.output / f"val_{index:04d}"
        folder.mkdir()
        source.save(folder / "source.png")
        local.save(folder / "local16000.png")
        columns = [("Night source", source), ("LOCAL inverse16000", local)]
        item = {
            "index": index,
            "source_sha256": digest(Path(row["input"])),
            "local_output_sha256": digest(args.local_outputs / f"val_{index:04d}.png"),
            "variants": [],
        }
        lifted = visibility_control(source)
        lifted.save(folder / "gamma045_conditioning.png")
        variants = {
            "source": ("EXTERNAL_direct", source),
            "local": ("LOCAL16000_plus_EXTERNAL", local),
            "lifted": ("EXTERNAL_lifted", lifted),
            "local_plus_lifted": ("LOCAL16000_plus_EXTERNAL_multiref", [local, lifted]),
        }
        for mode in modes:
            name, conditioning = variants[mode]
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            result = pipeline(
                image=conditioning,
                prompt_embeds=saved["prompt_embeds"].cuda(),
                height=source.height,
                width=source.width,
                num_inference_steps=4,
                guidance_scale=1.0,
                max_sequence_length=128,
                generator=torch.Generator(device="cuda").manual_seed(61000 + index),
            ).images[0]
            torch.cuda.synchronize()
            result.save(folder / f"{name}.png")
            columns.append((name, result))
            item["variants"].append(
                {
                    "name": name,
                    "seconds": time.monotonic() - started,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "output_size": list(result.size),
                }
            )
            if torch.cuda.max_memory_allocated() > 11.5 * 1024**3:
                raise RuntimeError("Reference exceeded 11.5 GiB process allocation ceiling")
            print(f"Finished {index}/{name}; {item['variants'][-1]['seconds']:.1f}s", flush=True)
        panel = Image.new("RGB", (512 * len(columns), 320), "#101010")
        draw = ImageDraw.Draw(panel)
        for column, (label, image) in enumerate(columns):
            panel.paste(image, (512 * column, 30))
            draw.text((512 * column + 5, 7), f"VAL{index} {label}", fill="white")
        panel.save(args.output / f"comparison_{index:04d}.jpg", quality=96)
        report["records"].append(item)
        atomic_json_dump(report, args.output / "evidence.json")
    print("Reference complete; open every image before drawing quality conclusions", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("encode", "infer", "sheets"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--indices", default="12,142,283,441")
    parser.add_argument("--modes", default="source,local")
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("E:/LumiCycleCache/teacher_targets/night2day_pilot128_v2"),
    )
    parser.add_argument(
        "--local-outputs",
        type=Path,
        default=Path(
            "E:/LumiCycleCache/experiments/night2day_diverse111_transpose1000/validation_02000"
        ),
    )
    args = parser.parse_args()
    environment()
    if args.stage == "encode":
        encode()
    elif args.stage == "infer":
        if args.output is None:
            parser.error("--output required for inference")
        infer(args)
    else:
        if args.output is None:
            parser.error("--output required for contact sheets")
        make_contact_sheets(args.output)


def make_contact_sheets(output):
    report = json.loads((output / "evidence.json").read_text())
    rows = report["records"]
    for start in range(0, len(rows), 4):
        page = rows[start : start + 4]
        first = Image.open(output / f"comparison_{page[0]['index']:04d}.jpg")
        sheet = Image.new("RGB", (first.width, 320 * len(page)), "#101010")
        for number, row in enumerate(page):
            panel = Image.open(output / f"comparison_{row['index']:04d}.jpg").convert("RGB")
            sheet.paste(panel, (0, number * 320))
        sheet.save(output / f"sheet_{start:02d}.jpg", quality=96)
    print(f"Prepared {len(rows)} development cases for full visual review")


if __name__ == "__main__":
    main()
