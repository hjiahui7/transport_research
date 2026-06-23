from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.nn.modules import Conv


@dataclass(frozen=True)
class DistanceHeadBatchMetrics:
    loss: float
    samples: int


class YoloGridDistanceHead(nn.Module):
    """Small YOLO-style per-grid distance branch copied from the Detect head pattern."""

    def __init__(self, channels: Sequence[int], *, init_distance_m: float = 3.0) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels must not be empty")
        hidden = max(16, int(channels[0]) // 4)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(int(ch), hidden, 3), Conv(hidden, hidden, 3), nn.Conv2d(hidden, 1, 1))
            for ch in channels
        )
        init_log_distance = math.log(max(0.05, init_distance_m))
        for branch in self.cv4:
            nn.init.constant_(branch[-1].bias, init_log_distance)

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.cv4):
            raise ValueError(f"Expected {len(self.cv4)} feature maps, got {len(features)}")
        return [branch(feature).squeeze(1) for branch, feature in zip(self.cv4, features)]


def load_frozen_yolo_feature_model(model_path: str | Path, *, device: str = "cuda:0"):
    model = YOLO(str(model_path)).model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    detect = model.model[-1]
    channels = _detect_input_channels(detect)
    strides = [float(value) for value in detect.stride.detach().cpu().tolist()]
    return model, channels, strides


@torch.no_grad()
def extract_yolo_detect_features(model, images: torch.Tensor) -> list[torch.Tensor]:
    """Run the YOLO backbone/neck and return the feature maps consumed by the Detect head."""
    y: list[torch.Tensor | None] = []
    x = images
    detect = model.model[-1]
    for module in model.model[:-1]:
        if module.f != -1:
            x = y[module.f] if isinstance(module.f, int) else [x if j == -1 else y[j] for j in module.f]
        x = module(x)
        y.append(x if module.i in model.save else None)

    features: list[torch.Tensor] = []
    for source in detect.f:
        if source == -1 or source == len(y) - 1:
            feature = x
        else:
            feature = y[source]
        if feature is None:
            raise RuntimeError(f"YOLO feature source {source} was not saved")
        features.append(feature)
    return features


def distance_head_loss(
    predictions: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    *,
    strides: Sequence[float],
    image_size: int,
    distance_column: int = 4,
) -> tuple[torch.Tensor, int]:
    """SmoothL1 loss on log(distance) at the grid cell containing each GT bbox center."""
    if len(predictions) != len(strides):
        raise ValueError(f"predictions/strides length mismatch: {len(predictions)} vs {len(strides)}")
    device = predictions[0].device
    losses: list[torch.Tensor] = []
    for batch_index, label_tensor in enumerate(labels):
        if label_tensor.numel() == 0:
            continue
        label_tensor = label_tensor.to(device)
        for target in label_tensor:
            cx, cy, width, height = target[:4]
            distance = torch.clamp(target[distance_column], min=0.05)
            level = select_distance_level(float(width), float(height), strides)
            stride = float(strides[level])
            grid_y = int(torch.clamp(torch.floor(cy / stride), 0, predictions[level].shape[1] - 1).item())
            grid_x = int(torch.clamp(torch.floor(cx / stride), 0, predictions[level].shape[2] - 1).item())
            pred_log_distance = predictions[level][batch_index, grid_y, grid_x]
            target_log_distance = torch.log(distance)
            losses.append(F.smooth_l1_loss(pred_log_distance, target_log_distance, reduction="none"))
    if not losses:
        return predictions[0].sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


def predict_distance_for_boxes(
    predictions: Sequence[torch.Tensor],
    boxes_xyxy: Sequence[Sequence[float]],
    *,
    strides: Sequence[float],
) -> list[float]:
    """Read distance predictions for image-0 boxes from the closest YOLO grid level."""
    distances: list[float] = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = [float(value) for value in box]
        width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        level = select_distance_level(width, height, strides)
        stride = float(strides[level])
        grid_y = int(max(0, min(predictions[level].shape[1] - 1, math.floor(cy / stride))))
        grid_x = int(max(0, min(predictions[level].shape[2] - 1, math.floor(cx / stride))))
        log_distance = predictions[level][0, grid_y, grid_x]
        distances.append(float(torch.exp(log_distance).detach().cpu()))
    return distances


def select_distance_level(width: float, height: float, strides: Sequence[float]) -> int:
    """Select a feature level by matching object size to roughly four grid cells."""
    object_size = max(float(width), float(height), 1.0)
    ideal_stride = max(1.0, object_size / 4.0)
    return min(range(len(strides)), key=lambda index: abs(math.log(max(strides[index], 1e-6) / ideal_stride)))


def _detect_input_channels(detect_module) -> list[int]:
    channels: list[int] = []
    for branch in detect_module.cv2:
        first = branch[0]
        channels.append(int(first.conv.in_channels))
    return channels
