from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PersonMask:
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    mask: np.ndarray
    mask_source: str = "segmentation"


class YoloPersonSegmenter:
    def __init__(
        self,
        model_name: str = "yolo11n-seg.pt",
        *,
        device: str = "cuda:0",
        imgsz: int = 640,
        half: bool = True,
        conf: float = 0.25,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed. Install it in the qwen conda environment first.") from exc

        self.model_name = model_name
        self.device = device
        self.imgsz = imgsz
        self.half = half
        self.conf = conf
        self.model = YOLO(model_name)

    def predict(self, image_path: str | Path) -> list[PersonMask]:
        image_path = Path(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]

        results = self.model.predict(
            source=str(image_path),
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            conf=self.conf,
            classes=[0],
            retina_masks=True,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None:
            return []

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        masks = result.masks.data.detach().cpu().numpy() if result.masks is not None else None

        persons: list[PersonMask] = []
        for idx, (bbox, score) in enumerate(zip(boxes, scores)):
            if masks is None:
                # Rawalk fine-tuning gives a detect-only YOLO model. In that case we keep the
                # distance pipeline running by using the detected box as a coarse person region.
                mask_bool = _bbox_mask(bbox, width, height)
                mask_source = "bbox"
            else:
                mask_bool = masks[idx] > 0.5
                if mask_bool.shape != (height, width):
                    mask_bool = cv2.resize(mask_bool.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                mask_source = "segmentation"
            persons.append(
                PersonMask(
                    score=float(score),
                    bbox_xyxy=tuple(float(v) for v in bbox),  # type: ignore[arg-type]
                    mask=mask_bool,
                    mask_source=mask_source,
                )
            )
        return persons


def _bbox_mask(bbox_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy[:4]]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    mask = np.zeros((height, width), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask
