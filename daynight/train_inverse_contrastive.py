"""Private warm-start unpaired inverse pilot; no teacher RGB, no public routing.

CUT-inspired objective adaptation, not an exact CUT/N2D3 reproduction.
Bounded experiments refuse overwriting existing checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .audit_inverse_degradations import degradation_memberships
from .data import read_manifest, train_transform
from .experiment_night_to_day import inverse_from_payload
from .models.networks import LocalGlobalDiscriminator
from .train_night_to_day_pilot import adversarial_lsgan
from .turbo_pilot import digest
from .utils import atomic_json_dump, atomic_torch_save


class PatchProjection(nn.Module):
    """Training-only shared MLPs, one per selected encoder feature level."""

    def __init__(self, channels=(64, 256, 256), width=128):
        super().__init__()
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(c, width), nn.ReLU(),
            nn.Linear(width, width)) for c in channels])

    def forward(self, features, ids):
        return [F.normalize(head(f.flatten(2).transpose(1, 2)[:, index].float()), dim=-1)
            for head, f, index in zip(self.heads, features, ids, strict=True)]


def patch_contrastive(query, key, temperature=.07, memberships=None):
    """Same-position positives; within-image negatives; stop-gradient keys."""
    if temperature <= 0 or query.shape != key.shape or query.ndim != 3:
        raise ValueError("Require matching BPC and positive temperature")
    logits = torch.bmm(query.float(), key.detach().float().transpose(1, 2)) / temperature
    if memberships is not None:
        if memberships.shape != (*query.shape[:2], 4):
            raise ValueError("Require BP4 photometric memberships")
        group = memberships.detach().float()
        if not torch.isfinite(group).all() or (group < 0).any():
            raise ValueError("Invalid photometric memberships")
        if not torch.allclose(group.sum(-1), torch.ones_like(group[..., 0]), atol=1e-4):
            raise ValueError("Memberships must partition each patch")
        similarity = torch.bmm(group, group.transpose(1, 2))
        # Our soft adaptation: keep a small cross-group negative floor; never
        # delete positive pairs or directly mask generated pixels.
        weights = .02 + .98 * similarity
        diagonal = torch.eye(query.shape[1], device=query.device, dtype=torch.bool)[None]
        weights = weights.masked_fill(diagonal, 1)
        logits = logits + weights.log()
    labels = torch.arange(query.shape[1], device=query.device).expand(query.shape[0], -1)
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten())


def content_loss(model, projection, source, translated, rng, regions=None):
    # Early/middle/deep representation, not raw source RGB reconstruction.
    _, keys = model.encode_features(source)
    _, queries = model.encode_features(translated)
    keys = [keys[i] for i in (0, 2, 5)]
    queries = [queries[i] for i in (0, 2, 5)]
    ids = [torch.randperm(f.shape[-2] * f.shape[-1], generator=rng,
        device=f.device)[:128] for f in keys]
    q = projection(queries, ids)
    with torch.no_grad():
        k = projection(keys, ids)
    groups = [None] * len(q)
    if regions is not None:
        groups = [F.interpolate(regions, f.shape[-2:], mode="area").flatten(2).transpose(1, 2)[:, index]
            for f, index in zip(keys, ids, strict=True)]
    return sum(patch_contrastive(a, b, memberships=g)
        for a, b, g in zip(q, k, groups, strict=True)) / len(q)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--nce-weight", type=float, default=1)
    parser.add_argument("--photometric-groups", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 2000 or args.nce_weight < 0:
        raise ValueError("Bounded1..2000 updates and nonnegative NCE required")
    if args.output.exists():
        raise FileExistsError("Use new isolated experiment directory")
    torch.set_num_threads(6)
    random.seed(2035)
    torch.manual_seed(2035)
    torch.cuda.manual_seed_all(2035)
    payload = torch.load(args.parent, map_location="cpu", weights_only=False)
    model, architecture = inverse_from_payload(payload)
    parent_step = payload["step"]
    config = copy.deepcopy(payload["config"])
    del payload
    model = model.cuda().train().requires_grad_(True)
    projection = PatchProjection().cuda()
    discriminator = LocalGlobalDiscriminator(base_channels=32, scales=1).cuda()
    # Initialize spectral-normalization power vectors before its first eval pass.
    # Otherwise fresh random u/v can produce enormous first-step GAN logits.
    with torch.no_grad():
        for _ in range(20):
            discriminator(torch.zeros(1, 3, 256, 256, device="cuda"))
    optimizer = torch.optim.Adam([*model.parameters(), *projection.parameters()],
        lr=5e-5, betas=(.5, .999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=5e-5, betas=(.5, .999))
    shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    patch_rng = torch.Generator(device="cuda").manual_seed(2036)
    nights = read_manifest("data/bdd100k_daynight", "train", "night")
    days = read_manifest("data/bdd100k_daynight", "train", "day")
    if not nights or not days:
        raise ValueError("Empty training domain")
    args.output.mkdir(parents=True)
    recipe = {"method": "CUT-inspired unpaired warm-start; NOT exact CUT or N2D3",
        "parent": str(args.parent), "parent_sha256": digest(args.parent),
        "parent_step": parent_step, "generator_architecture_changed": False,
        "fresh_training_only": ["projection MLPs", "day discriminator", "Adam optimizers"],
        "nce_weight": args.nce_weight, "loss": "LSGAN + weight*.5*(night PatchNCE + day identity PatchNCE)",
        "teacher_rgb_loss": 0, "external_model_executed": False,
        "train_nights": len(nights), "train_days": len(days), "seed": 2035,
        "patch_seed": 2036, "precision": "BF16", "crop": 256, "resize_short": 286,
        "G_D_lr": 5e-5, "batch": 1, "EMA": .99, "clip": 3,
        "spectral_norm_initialization": "20 no-gradient train-mode passes before first eval; no optimizer updates",
        "patches": 128, "temperature": .07, "feature_indices": [0, 2, 5],
        "test_used": False, "lockbox_used": False, "public_demo_unchanged": True,
        "code_sha256": digest(Path(__file__))}
    recipe.update({"photometric_groups": args.photometric_groups,
        "group_recipe": "Night negatives weighted by .02+.98*membership dot; positive weight1. Day identity remains uniform. No OT, no anchor reweighting."
        if args.photometric_groups else None,
        "group_map_code_sha256": digest(Path(__file__).with_name("audit_inverse_degradations.py"))
        if args.photometric_groups else None})
    atomic_json_dump(recipe, args.output / "recipe.json")
    start = time.monotonic()
    completed = 0

    def save():
        state = {"step": parent_step + completed, "adaptation_step": completed,
            "config": config, "inverse_architecture": architecture,
            "ema": {"G_night_day": {"shadow": shadow}}, "generator": model.state_dict(),
            "projection": projection.state_dict(), "discriminator": discriminator.state_dict(),
            "optimizer": optimizer.state_dict(), "d_optimizer": d_optimizer.state_dict(),
            "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(), "patch": patch_rng.get_state()}, "recipe": recipe}
        atomic_torch_save(state, args.output / "latest.pt")
        if completed % 250 == 0:
            atomic_torch_save(state, args.output / f"adaptation_{completed:05d}.pt")

    try:
        for step in range(1, args.steps + 1):
            tensors = []
            for domain in (nights, days):
                with Image.open(random.choice(domain).path) as im:
                    tensors.append(train_transform(im, 286, 256)[None].cuda())
            night, day = tensors
            regions = None
            if args.photometric_groups:
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    regions = degradation_memberships((night.float() + 1) / 2)[0]
            optimizer.zero_grad(set_to_none=True)
            discriminator.requires_grad_(False).eval()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fake = model(night)
                identity = model(day)
                gan = adversarial_lsgan(discriminator(fake), True)
                nce = .5 * (content_loss(model, projection, night, fake, patch_rng, regions)
                    + content_loss(model, projection, day, identity, patch_rng))
                loss = gan + args.nce_weight * nce
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite generator loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_([*model.parameters(), *projection.parameters()], 3)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite generator gradients")
            optimizer.step()
            discriminator.requires_grad_(True).train()
            d_optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d_loss = .5 * (adversarial_lsgan(discriminator(day), True)
                    + adversarial_lsgan(discriminator(fake.detach()), False))
            if not torch.isfinite(d_loss):
                raise FloatingPointError("Nonfinite discriminator loss")
            d_loss.backward()
            d_norm = nn.utils.clip_grad_norm_(discriminator.parameters(), 3)
            if not torch.isfinite(d_norm):
                raise FloatingPointError("Nonfinite discriminator gradients")
            d_optimizer.step()
            with torch.no_grad():
                for key, value in model.state_dict().items():
                    shadow[key].lerp_(value, .01)
            completed = step
            peak = torch.cuda.max_memory_allocated() / 1024**3
            if peak > 11.5:
                raise RuntimeError("Exceeds11.5GiB memory gate")
            if step % 25 == 0 or step == 1:
                row = {"step": parent_step + step, "adaptation_step": step,
                    "G": float(loss.detach()), "GAN": float(gan.detach()), "NCE": float(nce.detach()),
                    "D": float(d_loss.detach()), "peak_gib": peak, "seconds": time.monotonic() - start}
                print(json.dumps(row), flush=True)
                with (args.output / "training.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
            if step % 250 == 0:
                save()
    finally:
        save()
    atomic_json_dump({"state": "completed", "global_step": parent_step + completed,
        "visual_acceptance": "pending", "public_demo_unchanged": True}, args.output / "status.json")


if __name__ == "__main__":
    main()
