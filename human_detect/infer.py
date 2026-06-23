from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .calibration import load_calibrator
from .pipeline import DistanceEstimator
from .vis import save_visualization


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-image multi-person distance and viewing-angle inference.")
    parser.add_argument("--image", required=True, help="Path to one RGB image.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--vis", default=None, help="Optional visualization output path.")
    parser.add_argument("--calib", default=None, help="Optional KITTI-style calib txt; P2 overrides estimated intrinsics.")
    parser.add_argument("--detector", default="yolo11n-seg.pt", help="Ultralytics person model path/name; segmentation is preferred, detect-only is supported.")
    parser.add_argument("--geometry-model", default="Ruicheng/moge-2-vits-normal", help="MoGe Hugging Face model or local path.")
    parser.add_argument("--geometry", default="moge", choices=["moge"], help="Geometry backend.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--geom-size", type=int, default=768, help="Resize long edge before MoGe inference.")
    parser.add_argument("--num-tokens", type=int, default=1200, help="MoGe token budget when supported.")
    parser.add_argument("--device", default="cuda:0", help="Torch device.")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True, help="Use FP16 where supported.")
    parser.add_argument("--calibrator", default=None, help="Optional joblib calibrator from human_detect.fit_calibrator.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result, masks = infer_image(
        image_path=args.image,
        detector=args.detector,
        geometry_model=args.geometry_model,
        imgsz=args.imgsz,
        conf=args.conf,
        geom_size=args.geom_size,
        num_tokens=args.num_tokens,
        device=args.device,
        half=args.half,
        calib_path=args.calib,
        calibrator_path=args.calibrator,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.vis:
        save_visualization(args.image, result["persons"], masks, args.vis)
    return 0


def infer_image(
    *,
    image_path: str | Path,
    detector: str = "yolo11n-seg.pt",
    geometry_model: str = "Ruicheng/moge-2-vits-normal",
    imgsz: int = 640,
    conf: float = 0.25,
    geom_size: int = 768,
    num_tokens: int | None = 1200,
    device: str = "cuda:0",
    half: bool = True,
    calib_path: str | Path | None = None,
    calibrator_path: str | Path | None = None,
) -> tuple[dict[str, Any], list]:
    calibrator = load_calibrator(calibrator_path) if calibrator_path else None
    estimator = DistanceEstimator(
        detector=detector,
        geometry_model=geometry_model,
        imgsz=imgsz,
        conf=conf,
        geom_size=geom_size,
        num_tokens=num_tokens,
        device=device,
        half=half,
        calibrator=calibrator,
    )
    return estimator.infer(Path(image_path), calib_path=calib_path)


if __name__ == "__main__":
    raise SystemExit(main())
