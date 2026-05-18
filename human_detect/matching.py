from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .pm_hmcw import KittiObject


@dataclass(frozen=True)
class BBoxMatch:
    pred_index: int
    gt_index: int
    iou: float


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def greedy_bbox_matches(
    predictions: list[dict],
    gt_objects: list[KittiObject],
    *,
    iou_threshold: float = 0.3,
) -> list[BBoxMatch]:
    candidates: list[BBoxMatch] = []
    for pred_index, pred in enumerate(predictions):
        pred_bbox = pred.get("bbox_xyxy")
        if not pred_bbox:
            continue
        for gt_index, gt in enumerate(gt_objects):
            iou = bbox_iou(pred_bbox, gt.bbox_xyxy)
            if iou >= iou_threshold:
                candidates.append(BBoxMatch(pred_index=pred_index, gt_index=gt_index, iou=iou))

    candidates.sort(key=lambda item: item.iou, reverse=True)
    used_predictions: set[int] = set()
    used_gt: set[int] = set()
    matches: list[BBoxMatch] = []
    for match in candidates:
        if match.pred_index in used_predictions or match.gt_index in used_gt:
            continue
        used_predictions.add(match.pred_index)
        used_gt.add(match.gt_index)
        matches.append(match)
    return sorted(matches, key=lambda item: item.pred_index)

