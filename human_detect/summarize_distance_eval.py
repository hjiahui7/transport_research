from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .calibration import load_calibrator, row_feature_vector
from .distance_eval import summarize_distance_errors
from .eval_rawalk_ego_depth import load_rawalk_gt_groups


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize matched distance predictions, optionally after applying a calibrator.")
    parser.add_argument("--preds", required=True, help="Matched prediction/GT CSV from eval_rawalk_ego_depth.")
    parser.add_argument("--labels", default=None, help="Optional GT label CSV used to compute all-GT coverage.")
    parser.add_argument("--out", required=True, help="Summary JSON output.")
    parser.add_argument("--calibrator", default=None, help="Optional calibrator joblib to apply to prediction rows.")
    parser.add_argument("--min-distance", type=float, default=0.2)
    parser.add_argument("--max-distance", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    with Path(args.preds).open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("distance_m") not in {"", None} and row.get("distance_gt") not in {"", None}]

    total_gt = None
    if args.labels:
        groups = load_rawalk_gt_groups(args.labels, min_distance=args.min_distance, max_distance=args.max_distance)
        total_gt = sum(len(group) for group in groups.values())

    y_true = [float(row["distance_gt"]) for row in rows]
    raw_pred = [float(row["distance_m"]) for row in rows]
    summary = {
        "preds": str(Path(args.preds)),
        "labels": None if args.labels is None else str(Path(args.labels)),
        "raw": summarize_distance_errors(y_true, raw_pred, total_gt=total_gt),
    }

    if args.calibrator:
        calibrator = load_calibrator(args.calibrator)
        calibrated_pred = [float(calibrator.distance_model.predict([row_feature_vector(row)])[0]) for row in rows]
        summary["calibrated"] = summarize_distance_errors(y_true, calibrated_pred, total_gt=total_gt)
        summary["calibrator"] = str(Path(args.calibrator))
        summary["calibrator_type"] = calibrator.model_type

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
