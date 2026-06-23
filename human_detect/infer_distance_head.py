from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from .vis import save_visualization
from .yolo_distance_head import (
    YoloGridDistanceHead,
    extract_yolo_detect_features,
    load_frozen_yolo_feature_model,
    predict_distance_for_boxes,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scheme-1 inference: YOLO boxes plus trained distance head.")
    parser.add_argument("--image", required=True, help="Path to one RGB image.")
    parser.add_argument("--checkpoint", required=True, help="Distance-head checkpoint from train_yolo_distance_head.")
    parser.add_argument("--base-model", default=None, help="YOLO feature model. Defaults to checkpoint['model'].")
    parser.add_argument("--detector", default=None, help="YOLO detector for person boxes. Defaults to --base-model.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--vis", default=None, help="Optional visualization output path.")
    parser.add_argument("--imgsz", type=int, default=None, help="Square feature image size. Defaults to checkpoint image_size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True, help="Use FP16 in the detector where supported.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result, masks = infer_distance_head(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        base_model=args.base_model,
        detector=args.detector,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        half=args.half,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.vis:
        save_visualization(args.image, result["persons"], masks, args.vis)
    return 0


def infer_distance_head(
    *,
    image_path: str | Path,
    checkpoint_path: str | Path,
    base_model: str | Path | None = None,
    detector: str | Path | None = None,
    imgsz: int | None = None,
    conf: float = 0.25,
    device: str = "cuda:0",
    half: bool = True,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    image_path = Path(image_path)
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    image_size = int(imgsz or checkpoint["image_size"])
    base_model_path = str(base_model or checkpoint["model"])
    detector_path = str(detector or base_model_path)
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image_height, image_width = image_bgr.shape[:2]

    detections = _detect_person_boxes(
        image_path,
        detector=detector_path,
        imgsz=image_size,
        conf=conf,
        device=str(torch_device),
        half=half,
    )

    yolo_model, _channels, strides = load_frozen_yolo_feature_model(base_model_path, device=str(torch_device))
    head = YoloGridDistanceHead(checkpoint["channels"], init_distance_m=3.0).to(torch_device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()

    image_tensor = _load_square_rgb_tensor(image_path, image_size).to(torch_device)
    with torch.no_grad():
        features = extract_yolo_detect_features(yolo_model, image_tensor)
        predictions = head(features)

    original_boxes = [det["bbox_xyxy"] for det in detections]
    scaled_boxes = _scale_boxes_to_square(original_boxes, image_width=image_width, image_height=image_height, image_size=image_size)
    distances = predict_distance_for_boxes(predictions, scaled_boxes, strides=strides)

    persons: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    for idx, (det, distance) in enumerate(zip(detections, distances)):
        mask = _bbox_mask(det["bbox_xyxy"], image_width, image_height)
        persons.append(
            {
                "id": idx,
                "score": det["score"],
                "bbox_xyxy": [float(value) for value in det["bbox_xyxy"]],
                "mask_area_px": int(mask.sum()),
                "mask_source": "bbox",
                "z_depth_m": None,
                "distance_m": float(distance),
                "bearing_yaw_deg": None,
                "elevation_pitch_deg": None,
                "facing_yaw_deg": None,
                "distance_source": "yolo_distance_head",
            }
        )
        masks.append(mask)

    result = {
        "image_path": str(image_path),
        "image_size": {"width": image_width, "height": image_height},
        "scheme": "yolo_distance_head",
        "models": {
            "checkpoint": str(checkpoint_path),
            "base_model": base_model_path,
            "detector": detector_path,
        },
        "persons": persons,
    }
    return result, masks


def _detect_person_boxes(
    image_path: Path,
    *,
    detector: str,
    imgsz: int,
    conf: float,
    device: str,
    half: bool,
) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    model = YOLO(detector)
    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        device=device,
        half=half,
        conf=conf,
        classes=[0],
        verbose=False,
    )
    if not results or results[0].boxes is None:
        return []
    boxes = results[0].boxes.xyxy.detach().cpu().numpy()
    scores = results[0].boxes.conf.detach().cpu().numpy()
    return [{"bbox_xyxy": tuple(float(v) for v in box), "score": float(score)} for box, score in zip(boxes, scores)]


def _load_square_rgb_tensor(image_path: Path, image_size: int) -> torch.Tensor:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def _scale_boxes_to_square(
    boxes_xyxy: list[tuple[float, float, float, float]],
    *,
    image_width: int,
    image_height: int,
    image_size: int,
) -> list[tuple[float, float, float, float]]:
    sx = image_size / max(float(image_width), 1.0)
    sy = image_size / max(float(image_height), 1.0)
    return [(x1 * sx, y1 * sy, x2 * sx, y2 * sy) for x1, y1, x2, y2 in boxes_xyxy]


def _bbox_mask(bbox_xyxy: tuple[float, float, float, float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox_xyxy]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    mask = np.zeros((height, width), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


if __name__ == "__main__":
    raise SystemExit(main())
