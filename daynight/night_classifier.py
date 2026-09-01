from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .data import UnpairedDayNightDataset
from .utils import atomic_torch_save, seed_everything


class NightClassifier(nn.Module):
    """Small frozen evaluator trained only to distinguish held-out day/night domains."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_channels = 3
        for multiplier in (1, 2, 4, 8):
            output_channels = channels * multiplier
            layers.extend(
                (
                    nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.SiLU(inplace=True),
                )
            )
            input_channels = output_channels
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(channels * 8, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = F.adaptive_avg_pool2d(self.features(image), 1).flatten(1)
        return self.head(features).flatten()


def train_classifier(
    data_root: Path, output: Path, epochs: int = 5, size: int = 224
) -> Path:
    seed_everything(73)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = UnpairedDayNightDataset(data_root, "train", size, round(size * 1.12))
    val_data = UnpairedDayNightDataset(data_root, "val", size, size)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=0)
    model = NightClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    best_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            day, night = batch["day"].to(device), batch["night"].to(device)
            images = torch.cat((day, night))
            labels = torch.cat((torch.zeros(day.shape[0]), torch.ones(night.shape[0]))).to(device)
            loss = F.binary_cross_entropy_with_logits(model(images), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for batch in val_loader:
                day, night = batch["day"].to(device), batch["night"].to(device)
                images = torch.cat((day, night))
                labels = torch.cat((torch.zeros(day.shape[0]), torch.ones(night.shape[0]))).to(device)
                predictions = model(images).sigmoid() >= 0.5
                correct += int((predictions == labels.bool()).sum())
                total += labels.numel()
        accuracy = correct / max(1, total)
        print(f"epoch={epoch + 1} validation_accuracy={accuracy:.4f}")
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            atomic_torch_save(
                {"model": model.state_dict(), "validation_accuracy": accuracy, "size": size}, output
            )
    return output


def load_night_classifier(path: str | Path, device: torch.device) -> NightClassifier:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = NightClassifier().to(device)
    model.load_state_dict(payload["model"])
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a held-out day/night evaluation classifier")
    parser.add_argument("--data-root", type=Path, default=Path("data/bdd100k_daynight"))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/evaluators/night_classifier.pt")
    )
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    train_classifier(args.data_root, args.output, args.epochs)


if __name__ == "__main__":
    main()
