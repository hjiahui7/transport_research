from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workzone import prepare_workzone_outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert the work-zone RGB-D safety dataset into YOLO and distance CSV files.")
    parser.add_argument("--dataset-root", default="work-zone-safety-rgbd-dataset")
    parser.add_argument("--yolo-out", default=r"data\workzone_yolo_person")
    parser.add_argument("--labels-out", default=r"runs\workzone")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = prepare_workzone_outputs(
        dataset_root=Path(args.dataset_root),
        yolo_out=Path(args.yolo_out),
        labels_out=Path(args.labels_out),
        val_fraction=args.val_fraction,
        seed=args.seed,
        link_mode=args.link_mode,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
