from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .calibration import FEATURE_COLUMNS, person_feature_vector
from .matching import greedy_bbox_matches
from .pipeline import DistanceEstimator
from .pm_hmcw import KittiObject, iter_samples


CSV_COLUMNS = [
    "image_path",
    "image_id",
    "split_dir",
    "pred_id",
    "gt_id",
    "iou",
    *FEATURE_COLUMNS,
    "z_gt",
    "distance_gt",
    "gt_x",
    "gt_y",
    "gt_z",
    "gt_rotation_y",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PM-HMCW inference and write matched prediction/GT rows.")
    parser.add_argument("--data", default="data/pm_hmcw/raw", help="Extracted PM-HMCW root.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--split-contains", default="real-world\\test", help="Only evaluate split dirs containing this text; empty for all.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum images to process; 0 means no limit.")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--detector", default="yolo11n-seg.pt")
    parser.add_argument("--geometry-model", default="Ruicheng/moge-2-vits-normal")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--geom-size", type=int, default=640)
    parser.add_argument("--num-tokens", type=int, default=1200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    samples = list(iter_samples(args.data, category="Pedestrian"))
    if args.split_contains:
        needle = args.split_contains.replace("/", "\\").lower()
        samples = [s for s in samples if needle in str(s.split_dir).replace("/", "\\").lower()]
    if args.limit > 0:
        samples = samples[: args.limit]

    estimator = DistanceEstimator(
        detector=args.detector,
        geometry_model=args.geometry_model,
        imgsz=args.imgsz,
        geom_size=args.geom_size,
        num_tokens=args.num_tokens,
        device=args.device,
        half=args.half,
    )

    rows: list[dict[str, Any]] = []
    total_gt = 0
    total_matches = 0
    for sample_index, sample in enumerate(samples, start=1):
        gt_objects = [obj for obj in sample.load_objects() if obj.category == "Pedestrian"]
        total_gt += len(gt_objects)
        result, _ = estimator.infer(sample.image_path, calib_path=sample.calib_path)
        matches = greedy_bbox_matches(result["persons"], gt_objects, iou_threshold=args.iou_threshold)
        total_matches += len(matches)
        for match in matches:
            person = result["persons"][match.pred_index]
            gt = gt_objects[match.gt_index]
            rows.append(_row_from_match(sample, result, person, gt, match.pred_index, match.gt_index, match.iou))
        print(f"[{sample_index}/{len(samples)}] {sample.image_id}: gt={len(gt_objects)} matched={len(matches)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    rate = 0.0 if total_gt == 0 else total_matches / total_gt
    print(f"wrote {len(rows)} rows to {out_path}; matched_rate={rate:.3f}")
    return 0


def _row_from_match(sample, result: dict[str, Any], person: dict[str, Any], gt: KittiObject, pred_id: int, gt_id: int, iou: float) -> dict[str, Any]:
    width = int(result["image_size"]["width"])
    height = int(result["image_size"]["height"])
    features = dict(zip(FEATURE_COLUMNS, person_feature_vector(person, result["camera"], width, height)))
    gt_x, gt_y, gt_z = gt.location_xyz
    return {
        "image_path": str(sample.image_path),
        "image_id": sample.image_id,
        "split_dir": str(sample.split_dir),
        "pred_id": pred_id,
        "gt_id": gt_id,
        "iou": iou,
        **features,
        "z_gt": gt.z_gt,
        "distance_gt": gt.distance_gt,
        "gt_x": gt_x,
        "gt_y": gt_y,
        "gt_z": gt_z,
        "gt_rotation_y": gt.rotation_y,
    }


if __name__ == "__main__":
    raise SystemExit(main())
