"""Frozen external VGG features for NIGHT-target fidelity, training only.

No feature normalization to erase target contrast. ImageNet input normalization
is retained; aligned full images are not cropped to the classification input size.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .compare_sky_gan import file_digest
from .utils import atomic_json_dump

WEIGHTS_NAME = "vgg16-397923af.pth"
HASH_PREFIX = "397923af"
LAYERS = (3, 8, 15)  # relu1_2, relu2_2, relu3_3 in torchvision VGG16.


def prepare(root):
    from torchvision.models import VGG16_Weights

    root.mkdir(parents=True, exist_ok=True)
    path = root / WEIGHTS_NAME
    url = VGG16_Weights.IMAGENET1K_V1.url
    if url != f"https://download.pytorch.org/models/{WEIGHTS_NAME}":
        raise ValueError("Unexpected VGG weight source; inspect the installed torchvision version")
    if not path.exists():
        print(f"Downloading official external VGG16 weights into {root}", flush=True)
        torch.hub.download_url_to_file(url, str(path), hash_prefix=HASH_PREFIX, progress=False)
    digest = file_digest(path)
    if not digest.startswith(HASH_PREFIX):
        raise ValueError(
            "VGG file does not match the official hash prefix; file retained for inspection"
        )
    atomic_json_dump(
        {
            "url": url,
            "sha256": digest,
            "external_pretrained": True,
            "features": ["relu1_2", "relu2_2", "relu3_3"],
            "preprocessing": "RGB [0,1], ImageNet mean/std, full aligned extent; no classification crop",
        },
        root / "provenance.json",
    )
    print(json.dumps({"weights": str(path), "sha256": digest}), flush=True)


def feature_distance(prediction, target, features, layers=LAYERS):
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("Perceptual targets must be matching BCHW RGB tensors")
    if not layers or max(layers) >= len(features):
        raise ValueError("Invalid perceptual feature layers")
    working = prediction.float()
    mean = working.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = working.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    p = ((working + 1) / 2 - mean) / std
    with torch.no_grad():
        t = ((target.detach().float() + 1) / 2 - mean) / std
    loss = prediction.new_zeros((), dtype=torch.float32)
    for index, layer in enumerate(features):
        p = layer(p)
        with torch.no_grad():
            t = layer(t)
        if index in layers:
            loss = loss + F.l1_loss(p.float(), t.float())
    return loss / len(layers)


class FrozenVggTargetLoss(nn.Module):
    def __init__(self, root):
        super().__init__()
        from torchvision.models import vgg16

        root = Path(root)
        path = root / WEIGHTS_NAME
        if not path.is_file() or not (root / "provenance.json").is_file():
            raise FileNotFoundError(
                "VGG cache missing; run python -m daynight.perceptual_targets --prepare first"
            )
        metadata = json.loads((root / "provenance.json").read_text())
        self.weights_sha256 = file_digest(path)
        if self.weights_sha256 != metadata["sha256"] or not self.weights_sha256.startswith(
            HASH_PREFIX
        ):
            raise ValueError("Cached VGG weights changed")
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Public torchvision builder; the large classifier is never moved to GPU.
        model = vgg16(weights=None)
        self.features = model.features[:16]
        selected = {
            key.removeprefix("features."): value
            for key, value in state.items()
            if key.startswith("features.") and int(key.split(".")[1]) < 16
        }
        self.features.load_state_dict(selected, strict=True)
        self.features.eval().requires_grad_(False)

    def forward(self, prediction, target):
        return feature_distance(prediction, target, self.features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=Path("E:/LumiCycleCache/torchvision"))
    args = parser.parse_args()
    if args.prepare:
        prepare(args.cache_root)


if __name__ == "__main__":
    main()
