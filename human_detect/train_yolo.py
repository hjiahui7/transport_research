from __future__ import annotations

import argparse
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a small Ultralytics YOLO person detector on a YOLO dataset.")
    parser.add_argument("--data", default=r"data\rawalk_yolo_person\rawalk_person.yaml", help="YOLO data.yaml path.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base YOLO detection model or local checkpoint.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size. Lower this first if VRAM is tight.")
    parser.add_argument("--device", default="cuda:0", help="Torch/Ultralytics device.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--project", default=r"runs\yolo", help="Ultralytics output project directory.")
    parser.add_argument("--name", default="rawalk_yolo11n_exo", help="Ultralytics run name.")
    parser.add_argument("--freeze", type=int, default=10, help="Freeze first N layers for a small, conservative first pass.")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience.")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False, help="Cache images during training.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use mixed precision training.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing Ultralytics run directory.")
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
    project_path = Path(args.project).resolve()

    # Rawalk has bbox labels, so this trains a detect model. Segmentation masks remain handled by
    # yolo11n-seg.pt unless you pass this detector into inference and accept bbox-as-mask pooling.
    model = YOLO(args.model)
    results = model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_path),
        name=args.name,
        freeze=args.freeze,
        patience=args.patience,
        cache=args.cache,
        amp=args.amp,
        single_cls=True,
        exist_ok=args.exist_ok,
        plots=True,
    )
    save_dir = Path(getattr(results, "save_dir", project_path / args.name))
    print(f"Run dir: {save_dir}")
    print(f"Best weights: {save_dir / 'weights' / 'best.pt'}")
    print(f"Last weights: {save_dir / 'weights' / 'last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
