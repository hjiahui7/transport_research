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


class YoloPersonSegmenter:
    def __init__(self, model_name: str = "yolo11n-seg.pt", *, device: str = "cuda:0", imgsz: int = 640, half: bool = True) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed. Install it in the qwen conda environment first.") from exc

        self.model_name = model_name
        self.device = device
        self.imgsz = imgsz
        self.half = half
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
            classes=[0],
            retina_masks=True,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None or result.masks is None:
            return []

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        masks = result.masks.data.detach().cpu().numpy()

        persons: list[PersonMask] = []
        for bbox, score, mask in zip(boxes, scores, masks):
            mask_bool = mask > 0.5
            if mask_bool.shape != (height, width):
                mask_bool = cv2.resize(mask_bool.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            persons.append(
                PersonMask(
                    score=float(score),
                    bbox_xyxy=tuple(float(v) for v in bbox),  # type: ignore[arg-type]
                    mask=mask_bool,
                )
            )
        return persons

