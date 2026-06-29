from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .distance_eval import distance_band, summarize_distance_errors
from .eval_rawalk_ego_depth import greedy_rawalk_matches, load_rawalk_gt_groups
from .infer_distance_head import DistanceHeadEstimator


CSV_COLUMNS = [
    "image_path",
    "image_id",
    "pred_id",
    "gt_id",
    "human_name",
    "iou",
    "score",
    "pred_distance_m",
    "gt_distance_m",
    "error_m",
    "abs_error_m",
    "pred_band",
    "gt_band",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate scheme-1 YOLO distance-head inference against bbox distance GT.")
    parser.add_argument("--labels", required=True, help="Distance GT CSV with image_path, bbox and distance_m columns.")
    parser.add_argument("--checkpoint", required=True, help="Distance-head checkpoint.")
    parser.add_argument("--out", required=True, help="Matched prediction CSV output.")
    parser.add_argument("--summary-out", required=True, help="Summary metrics JSON output.")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--detector", default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--min-distance", type=float, default=0.2)
    parser.add_argument("--max-distance", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    groups = load_rawalk_gt_groups(args.labels, min_distance=args.min_distance, max_distance=args.max_distance)
    image_paths = sorted(groups)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    estimator = DistanceHeadEstimator(
        checkpoint_path=args.checkpoint,
        base_model=args.base_model,
        detector=args.detector,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        half=args.half,
    )

    rows: list[dict[str, Any]] = []
    total_gt = 0
    for image_index, image_path in enumerate(image_paths, start=1):
        gt_rows = groups[image_path]
        total_gt += len(gt_rows)
        result, _ = estimator.infer(image_path)
        matches = greedy_rawalk_matches(result["persons"], gt_rows, iou_threshold=args.iou_threshold)
        for match in matches:
            pred = result["persons"][match.pred_index]
            gt = gt_rows[match.gt_index]
            pred_distance = float(pred["distance_m"])
            gt_distance = float(gt.distance_gt)
            x1, y1, x2, y2 = pred["bbox_xyxy"]
            rows.append(
                {
                    "image_path": image_path,
                    "image_id": Path(image_path).stem,
                    "pred_id": match.pred_index,
                    "gt_id": match.gt_index,
                    "human_name": gt.human_name,
                    "iou": f"{match.iou:.6f}",
                    "score": f"{float(pred.get('score') or 0.0):.6f}",
                    "pred_distance_m": f"{pred_distance:.6f}",
                    "gt_distance_m": f"{gt_distance:.6f}",
                    "error_m": f"{pred_distance - gt_distance:.6f}",
                    "abs_error_m": f"{abs(pred_distance - gt_distance):.6f}",
                    "pred_band": distance_band(pred_distance),
                    "gt_band": distance_band(gt_distance),
                    "bbox_x1": f"{float(x1):.6f}",
                    "bbox_y1": f"{float(y1):.6f}",
                    "bbox_x2": f"{float(x2):.6f}",
                    "bbox_y2": f"{float(y2):.6f}",
                }
            )
        print(f"[{image_index}/{len(image_paths)}] {Path(image_path).name}: gt={len(gt_rows)} matched={len(matches)}")

    y_true = [float(row["gt_distance_m"]) for row in rows]
    y_pred = [float(row["pred_distance_m"]) for row in rows]
    summary = summarize_distance_errors(y_true, y_pred, total_gt=total_gt)
    summary.update(
        {
            "scheme": "yolo_distance_head",
            "labels": str(Path(args.labels)),
            "checkpoint": str(Path(args.checkpoint)),
            "base_model": args.base_model,
            "detector": args.detector,
            "iou_threshold": args.iou_threshold,
            "images": len(image_paths),
        }
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
