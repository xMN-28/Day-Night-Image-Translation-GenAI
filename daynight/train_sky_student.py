"""Train/resume a small sky matte predictor using frozen offline pseudo-labels.

No evaluation image enters gradient updates. Atomic checkpoints contain optimizer
and RNG state. A STOP file saves at the next completed step. V2 is untouched.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .data import read_manifest
from .sky_refinement import SkyStudent, refine_sky_alpha
from .utils import atomic_json_dump, atomic_torch_save, seed_everything


class CachedSkyDataset(Dataset):
    def __init__(self, root: Path, split: str, augment: bool = False, regions: bool = False):
        self.paths = sorted((root / split).glob("*.npz"))
        self.augment = augment
        self.regions = regions
        if not self.paths:
            raise FileNotFoundError(f"No cached labels in {root / split}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with np.load(self.paths[index]) as data:
            image = torch.from_numpy(data["image"].copy()).float().permute(2, 0, 1) / 255
            alpha = torch.from_numpy(data["alpha"].copy()).float()[None] / 255
            if self.regions:
                vegetation = torch.from_numpy(data["vegetation"].copy()).float()[None] / 255
                alpha = torch.cat((alpha, vegetation), 0)
        if self.augment:
            if random.random() < 0.5:
                image, alpha = image.flip(-1), alpha.flip(-1)
            # Match daylight variety while retaining the semantic target.
            image = (image * random.uniform(0.75, 1.15)).clamp(0, 1)
        return image, alpha


def prepare(args):
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    name = "facebook/mask2former-swin-small-cityscapes-semantic"
    processor = AutoImageProcessor.from_pretrained(name, local_files_only=True)
    teacher = (
        Mask2FormerForUniversalSegmentation.from_pretrained(name, local_files_only=True)
        .cuda()
        .eval()
    )
    provenance = {
        "teacher": name,
        "input_size": [384, 216],
        "matting": "RGB local color mixture",
        "splits": {},
        "training_only_teacher": True,
    }
    with torch.inference_mode():
        for split, count in (("train", args.train_images), ("val", args.val_images)):
            records = read_manifest("data/bdd100k_daynight", split, "day")
            # Even spacing is predetermined, independent of model output quality.
            ids = np.linspace(0, len(records) - 1, min(count, len(records)), dtype=int)
            (args.cache / split).mkdir(parents=True, exist_ok=True)
            provenance["splits"][split] = [
                {"index": int(i), "path": str(records[i].path)} for i in ids
            ]
            for j, i in enumerate(ids):
                path = args.cache / split / f"{int(i):05d}.npz"
                if path.exists():
                    continue
                image = (
                    Image.open(records[i].path)
                    .convert("RGB")
                    .resize((384, 216), Image.Resampling.LANCZOS)
                )
                x = (
                    torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].cuda().float()
                    / 255
                )
                inputs = processor(
                    image, return_tensors="pt", size={"shortest_edge": 384, "longest_edge": 768}
                )
                output = teacher(**{k: v.cuda() for k, v in inputs.items()})
                labels = processor.post_process_semantic_segmentation(
                    output, target_sizes=[(216, 384)]
                )[0][None, None]
                p, mixture = (labels == 10).float(), (labels == 8).float()
                alpha = refine_sky_alpha(x, p, mixture)
                np.savez_compressed(
                    path,
                    image=np.asarray(image),
                    alpha=(alpha[0, 0].cpu().numpy() * 255).round().astype(np.uint8),
                    vegetation=(mixture[0, 0].cpu().numpy() * 255).astype(np.uint8),
                )
                if j % 20 == 0:
                    print(f"Cached {split} {j + 1}/{len(ids)}", flush=True)
                    atomic_json_dump(
                        {
                            "phase": "pseudo_labels",
                            "split": split,
                            "completed": j + 1,
                            "total": len(ids),
                        },
                        args.output / "status.json",
                    )
    atomic_json_dump(provenance, args.cache / "provenance.json")


def objective(logits, target):
    p = logits.sigmoid()
    boundary = (F.max_pool2d(target, 5, 1, 2) + F.max_pool2d(-target, 5, 1, 2)).clamp(0, 1)
    bce = (
        F.binary_cross_entropy_with_logits(logits, target, reduction="none") * (1 + 3 * boundary)
    ).mean()
    dice = 1 - (2 * (p * target).sum((1, 2, 3)) + 1) / (
        p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1
    )
    gx = F.l1_loss(p[..., 1:] - p[..., :-1], target[..., 1:] - target[..., :-1])
    gy = F.l1_loss(p[..., 1:, :] - p[..., :-1, :], target[..., 1:, :] - target[..., :-1, :])
    return bce + dice.mean() * 0.5 + (gx + gy) * 0.5


def train(args):
    seed_everything(2026)
    model = SkyStudent(outputs=2 if args.regions else 1).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    start, best = 0, float("inf")
    latest = args.output / "latest.pt"
    if args.init_from and not latest.exists():
        initial = torch.load(args.init_from, map_location="cpu", weights_only=False)
        source_state = initial["model"]
        target_state = model.state_dict()
        for key, value in source_state.items():
            if key in ("head.weight", "head.bias"):
                target_state[key][: value.shape[0]].copy_(value)
            else:
                target_state[key].copy_(value)
        model.load_state_dict(target_state)
        start = initial["step"]
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best = state["step"], state["best"]
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
    dataset = CachedSkyDataset(args.cache, "train", True, args.regions)
    validation = DataLoader(
        CachedSkyDataset(args.cache, "val", regions=args.regions), batch_size=4, num_workers=0
    )
    started = time.monotonic()
    step = start

    def save():
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best": best,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "base_checkpoint": "runs/lumicycle_v2_bdd100k/checkpoints/step_00004500.pt",
                "architecture": "SkyStudent24-regions-v2" if args.regions else "SkyStudent24-v1",
                "outputs": 2 if args.regions else 1,
                "init_from": str(args.init_from),
                "cache": str(args.cache),
            },
            latest,
        )

    try:
        while step < args.steps:
            # Explicit index draw makes resume independent of a hidden shuffled
            # DataLoader iterator and preserves the random sampling trajectory.
            indices = torch.randint(len(dataset), (args.batch,)).tolist()
            batch = [dataset[i] for i in indices]
            x = torch.stack([b[0] for b in batch]).cuda()
            target = torch.stack([b[1] for b in batch]).cuda()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = objective(model(x).float(), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite student loss; prior checkpoint retained")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradients; prior checkpoint retained")
            optimizer.step()
            step += 1
            if step % 25 == 0:
                status = {
                    "phase": "training",
                    "step": step,
                    "max_steps": args.steps,
                    "loss": float(loss.detach()),
                    "best_val": best if np.isfinite(best) else None,
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
                atomic_json_dump(status, args.output / "status.json")
                print(json.dumps(status), flush=True)
            if step % 200 == 0:
                model.eval()
                values = []
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for vx, vt in validation:
                        values.append(float(objective(model(vx.cuda()).float(), vt.cuda())))
                score = float(np.mean(values))
                improved = score < best
                best = min(best, score)
                save()
                if improved:
                    atomic_torch_save(
                        torch.load(latest, map_location="cpu", weights_only=False),
                        args.output / "best.pt",
                    )
                with (args.output / "validation.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": step, "score": score, "best": best}) + "\n")
                print(f"Validation step={step} score={score:.5f} best={best:.5f}", flush=True)
            if (
                args.output / "STOP"
            ).exists() or time.monotonic() - started > args.max_minutes * 60:
                break
    except KeyboardInterrupt:
        pass
    save()
    atomic_json_dump(
        {
            "phase": "stopped" if step < args.steps else "finished",
            "step": step,
            "best_val": best,
            "checkpoint": str(latest),
        },
        args.output / "status.json",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--cache", type=Path, default=Path("data/sky_student_fullframe_v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/sky_student_v1"))
    parser.add_argument("--train-images", type=int, default=1200)
    parser.add_argument("--val-images", type=int, default=48)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--max-minutes", type=float, default=60)
    parser.add_argument("--regions", action="store_true")
    parser.add_argument("--init-from", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(6)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
