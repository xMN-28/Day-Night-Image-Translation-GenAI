from __future__ import annotations

import torch

from .utils import tensor_to_pil


def box_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = torch.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    left_area = (left[:, 2:] - left[:, :2]).clamp(min=0).prod(dim=-1)
    right_area = (right[:, 2:] - right[:, :2]).clamp(min=0).prod(dim=-1)
    union = left_area[:, None] + right_area[None, :] - intersection
    return intersection / union.clamp(min=1e-6)


class ObjectRetentionMetric:
    """Frozen COCO detector consistency for safety-relevant object retention."""

    def __init__(self, weights: str = "yolo11n.pt", confidence: float = 0.25) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Install the detector extra with `pip install -e .[detector]`"
            ) from error
        self.model = YOLO(weights)
        self.confidence = confidence
        self.matched = 0
        self.reference = 0
        self.confidence_ratio_sum = 0.0
        self.samples = 0

    def update(self, source: torch.Tensor, translated: torch.Tensor) -> None:
        source_images = [tensor_to_pil(item) for item in source]
        translated_images = [tensor_to_pil(item) for item in translated]
        predictions = self.model.predict(
            source=source_images + translated_images, conf=self.confidence, verbose=False
        )
        batch = len(source_images)
        for source_result, translated_result in zip(
            predictions[:batch], predictions[batch:], strict=True
        ):
            source_boxes = source_result.boxes.xyxy.detach().cpu()
            source_classes = source_result.boxes.cls.detach().cpu().long()
            source_conf = source_result.boxes.conf.detach().cpu()
            target_boxes = translated_result.boxes.xyxy.detach().cpu()
            target_classes = translated_result.boxes.cls.detach().cpu().long()
            target_conf = translated_result.boxes.conf.detach().cpu()
            self.reference += len(source_boxes)
            matched = 0
            matched_confidence = 0.0
            if len(source_boxes) and len(target_boxes):
                ious = box_iou(source_boxes, target_boxes)
                used: set[int] = set()
                for index in torch.argsort(source_conf, descending=True).tolist():
                    compatible = torch.where(source_classes[index] == target_classes)[0].tolist()
                    compatible = [item for item in compatible if item not in used]
                    if not compatible:
                        continue
                    best = max(compatible, key=lambda item: float(ious[index, item]))
                    if float(ious[index, best]) >= 0.5:
                        used.add(best)
                        matched += 1
                        matched_confidence += float(
                            target_conf[best] / source_conf[index].clamp(min=1e-6)
                        )
            self.matched += matched
            self.confidence_ratio_sum += matched_confidence / max(1, matched)
            self.samples += 1

    def compute(self) -> dict[str, float]:
        return {
            "retention_rate": self.matched / max(1, self.reference),
            "matched_confidence_ratio": self.confidence_ratio_sum / max(1, self.samples),
            "reference_objects": float(self.reference),
            "matched_objects": float(self.matched),
        }
