from __future__ import annotations

import random

import torch


class ImageReplayBuffer:
    def __init__(self, capacity: int = 50) -> None:
        self.capacity = capacity
        self.images: list[torch.Tensor] = []

    def query(self, batch: torch.Tensor) -> torch.Tensor:
        if self.capacity <= 0:
            return batch.detach()
        returned = []
        for image in batch.detach():
            image = image.unsqueeze(0)
            if len(self.images) < self.capacity:
                self.images.append(image.cpu())
                returned.append(image)
            elif random.random() < 0.5:
                index = random.randrange(self.capacity)
                historical = self.images[index].to(image.device)
                self.images[index] = image.cpu()
                returned.append(historical)
            else:
                returned.append(image)
        return torch.cat(returned, dim=0)

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "images": self.images}

    def load_state_dict(self, state: dict) -> None:
        self.capacity = int(state["capacity"])
        self.images = list(state["images"])
