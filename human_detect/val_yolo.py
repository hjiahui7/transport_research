from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an Ultralytics YOLO detector and save compact metrics.")
    parser.add_argument("--model", required=True, help="YOLO model name or checkpoint path.")
    parser.add_argument("--data", default=r"data\rawalk_yolo_person\rawalk_person.yaml", help="YOLO data.yaml path.")
    parser.add_argument("--out", required=True, help="Output metrics JSON path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", default="cuda:0", help="Torch/Ultralytics device.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--project", default=r"runs\yolo", help="Ultralytics output project directory.")
    parser.add_argument("--name", default="val", help="Ultralytics validation run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed in the current Python environment.") from exc

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found: {data_path}")

    model = YOLO(args.model)
    metrics = model.val(
        data=str(data_path.resolve()),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(Path(args.project).resolve()),
        name=args.name,
        single_cls=True,
        plots=True,
        exist_ok=args.exist_ok,
    )
    result = {
        "model": str(args.model),
        "data": str(data_path.resolve()),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "save_dir": str(getattr(metrics, "save_dir", "")),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
