from __future__ import annotations

import argparse
from pathlib import Path

from .rawalk import LinkMode, StreamMode, prepare_rawalk_yolo_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Rawalk bbox annotations into a YOLO person-detection dataset.")
    parser.add_argument(
        "--rawalk-root",
        default=r"data\media\rawalk",
        help="Rawalk root. The script accepts either data/media/rawalk or the final 01_tagging folder.",
    )
    parser.add_argument("--out", default=r"data\rawalk_yolo_person", help="Output YOLO dataset directory.")
    parser.add_argument("--streams", choices=["exo", "ego", "all"], default="exo", help="Camera streams to convert.")
    parser.add_argument("--frame-step", type=int, default=10, help="Use every Nth frame per camera stream.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction by sequence, not by frame.")
    parser.add_argument("--min-score", type=float, default=0.2, help="Drop Rawalk boxes below this annotation score.")
    parser.add_argument("--min-box-px", type=float, default=4.0, help="Drop clipped boxes smaller than this in width or height.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick smoke datasets.")
    parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink", help="How to place images in the YOLO dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output images/labels.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    stats = prepare_rawalk_yolo_dataset(
        rawalk_root=Path(args.rawalk_root),
        out_root=Path(args.out),
        streams=args.streams,  # type: ignore[arg-type]
        frame_step=args.frame_step,
        val_fraction=args.val_fraction,
        min_score=args.min_score,
        min_box_px=args.min_box_px,
        max_images=args.max_images,
        link_mode=args.link_mode,  # type: ignore[arg-type]
        overwrite=args.overwrite,
    )
    print(f"Rawalk root: {stats.tagging_root}")
    print(f"YOLO data:   {stats.out_root}")
    print(f"YAML:        {stats.yaml_path}")
    print(f"Manifest:    {stats.manifest_path}")
    print(f"Images:      train={stats.train_images} val={stats.val_images}")
    print(f"Boxes:       {stats.boxes}")
    print(f"Skipped:     missing_images={stats.skipped_missing_images} empty_labels={stats.skipped_empty_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
