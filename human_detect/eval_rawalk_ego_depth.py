from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .calibration import FEATURE_COLUMNS, person_feature_vector
from .matching import BBoxMatch, bbox_iou
from .pipeline import DistanceEstimator


CSV_COLUMNS = [
    "image_path",
    "image_id",
    "sequence",
    "frame_id",
    "viewer",
    "pred_id",
    "gt_id",
    "human_name",
    "iou",
    *FEATURE_COLUMNS,
    "z_gt",
    "distance_gt",
]


@dataclass(frozen=True)
class RawalkEgoGt:
    row_index: int
    sequence: str
    frame_id: str
    viewer: str
    human_name: str
    image_path: Path
    bbox_xyxy: tuple[float, float, float, float]
    z_gt: float
    distance_gt: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Rawalk ego inference and write matched rows for calibration.")
    parser.add_argument("--labels", required=True, help="CSV from human_detect.rawalk_ego_depth.")
    parser.add_argument("--out", required=True, help="Output matched prediction/GT CSV path.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum images to process; 0 means no limit.")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--min-distance", type=float, default=0.2, help="Drop GT rows below this distance in meters.")
    parser.add_argument("--max-distance", type=float, default=20.0, help="Drop GT rows above this distance in meters.")
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
    groups = load_rawalk_gt_groups(
        args.labels,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
    )
    image_paths = sorted(groups)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

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
    for image_index, image_path in enumerate(image_paths, start=1):
        gt_rows = groups[image_path]
        total_gt += len(gt_rows)
        result, _ = estimator.infer(image_path)
        matches = greedy_rawalk_matches(result["persons"], gt_rows, iou_threshold=args.iou_threshold)
        total_matches += len(matches)
        for match in matches:
            person = result["persons"][match.pred_index]
            gt = gt_rows[match.gt_index]
            rows.append(row_from_match(result, person, gt, match.pred_index, match.gt_index, match.iou))
        print(f"[{image_index}/{len(image_paths)}] {Path(image_path).name}: gt={len(gt_rows)} matched={len(matches)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    match_rate = 0.0 if total_gt == 0 else total_matches / total_gt
    print(f"wrote {len(rows)} rows to {out_path}; matched_rate={match_rate:.3f}")
    return 0


def load_rawalk_gt_groups(
    labels_path: str | Path,
    *,
    min_distance: float | None = 0.2,
    max_distance: float | None = 20.0,
) -> dict[str, list[RawalkEgoGt]]:
    groups: dict[str, list[RawalkEgoGt]] = defaultdict(list)
    with Path(labels_path).open(newline="", encoding="utf-8") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            try:
                z_gt = float(row["depth_m"])
                distance_gt = float(row["distance_m"])
            except (KeyError, TypeError, ValueError):
                continue
            if min_distance is not None and distance_gt < min_distance:
                continue
            if max_distance is not None and distance_gt > max_distance:
                continue
            image_path = Path(row["image_path"])
            if not image_path.exists():
                continue
            groups[image_path.as_posix()].append(
                RawalkEgoGt(
                    row_index=row_index,
                    sequence=row.get("sequence", ""),
                    frame_id=row.get("frame_id", ""),
                    viewer=row.get("viewer", ""),
                    human_name=row.get("human_name", ""),
                    image_path=image_path,
                    bbox_xyxy=(
                        float(row["bbox_x1"]),
                        float(row["bbox_y1"]),
                        float(row["bbox_x2"]),
                        float(row["bbox_y2"]),
                    ),
                    z_gt=z_gt,
                    distance_gt=distance_gt,
                )
            )
    return dict(groups)


def greedy_rawalk_matches(predictions: list[dict[str, Any]], gt_rows: list[RawalkEgoGt], *, iou_threshold: float = 0.3) -> list[BBoxMatch]:
    candidates: list[BBoxMatch] = []
    for pred_index, pred in enumerate(predictions):
        pred_bbox = pred.get("bbox_xyxy")
        if not pred_bbox:
            continue
        for gt_index, gt in enumerate(gt_rows):
            iou = bbox_iou(pred_bbox, gt.bbox_xyxy)
            if iou >= iou_threshold:
                candidates.append(BBoxMatch(pred_index=pred_index, gt_index=gt_index, iou=iou))

    candidates.sort(key=lambda item: item.iou, reverse=True)
    used_predictions: set[int] = set()
    used_gt: set[int] = set()
    matches: list[BBoxMatch] = []
    for match in candidates:
        if match.pred_index in used_predictions or match.gt_index in used_gt:
            continue
        used_predictions.add(match.pred_index)
        used_gt.add(match.gt_index)
        matches.append(match)
    return sorted(matches, key=lambda item: item.pred_index)


def row_from_match(result: dict[str, Any], person: dict[str, Any], gt: RawalkEgoGt, pred_id: int, gt_id: int, iou: float) -> dict[str, Any]:
    width = int(result["image_size"]["width"])
    height = int(result["image_size"]["height"])
    features = dict(zip(FEATURE_COLUMNS, person_feature_vector(person, result["camera"], width, height)))
    image_id = f"{gt.sequence}_{gt.viewer}_{gt.frame_id}"
    return {
        "image_path": str(gt.image_path),
        "image_id": image_id,
        "sequence": gt.sequence,
        "frame_id": gt.frame_id,
        "viewer": gt.viewer,
        "pred_id": pred_id,
        "gt_id": gt_id,
        "human_name": gt.human_name,
        "iou": iou,
        **features,
        "z_gt": gt.z_gt,
        "distance_gt": gt.distance_gt,
    }


if __name__ == "__main__":
    raise SystemExit(main())
