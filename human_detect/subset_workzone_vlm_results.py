from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .eval_workzone_report import aggregate_metrics, write_worker_csv
from .eval_workzone_vlm_distance import aggregate_direct_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute VLM metrics for a fixed image subset without API calls.")
    parser.add_argument("--labels", required=True, help="Subset distance CSV containing the selected image_id values.")
    parser.add_argument("--gt-csv", required=True, help="Wave 1 worker_gt_merged.csv.")
    parser.add_argument("--summary", required=True, help="Existing all-image VLM summary JSON.")
    parser.add_argument("--per-worker", required=True, help="Existing all-image per-worker CSV.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["attributes", "direct"], default="attributes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = subset_results(
        labels=Path(args.labels),
        gt_csv=Path(args.gt_csv),
        source_summary=Path(args.summary),
        source_per_worker=Path(args.per_worker),
        out_dir=Path(args.out_dir),
        mode=args.mode,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def subset_results(
    *,
    labels: Path,
    gt_csv: Path,
    source_summary: Path,
    source_per_worker: Path,
    out_dir: Path,
    mode: str,
) -> dict[str, Any]:
    selected = selected_image_ids(labels)
    source = json.loads(source_summary.read_text(encoding="utf-8"))
    source_reports = Path(source["reports_dir"])
    if not source_reports.is_absolute():
        source_reports = Path.cwd() / source_reports

    rows = read_csv(source_per_worker)
    subset_rows = [row for row in rows if normalize_image_id(row["image_id"]) in selected]
    matched_by_image = Counter(normalize_image_id(row["image_id"]) for row in subset_rows)
    gt_by_image = gt_worker_counts(gt_csv, selected)

    out_dir.mkdir(parents=True, exist_ok=True)
    reports_out = out_dir / "reports"
    reports_out.mkdir(parents=True, exist_ok=True)
    per_image: list[dict[str, int]] = []
    missing_reports: list[str] = []
    for image_id in sorted(selected):
        report_path = source_reports / f"{image_id}.json"
        if not report_path.exists():
            missing_reports.append(image_id)
            pred_workers = 0
        else:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            pred_workers = int(report.get("worker_count", len(report.get("workers", []))))
            place_file(report_path, reports_out / report_path.name)
        per_image.append(
            {
                "pred_workers": pred_workers,
                "gt_workers": gt_by_image.get(image_id, 0),
                "matched": matched_by_image.get(image_id, 0),
            }
        )

    aggregate = aggregate_metrics(per_image, subset_rows) if mode == "attributes" else aggregate_direct_metrics(per_image, subset_rows)
    output = dict(source)
    output.update(aggregate)
    output.update(
        {
            "labels": str(labels),
            "images": len(selected),
            "reports_dir": str(reports_out),
            "per_worker_csv": str(out_dir / "per_worker.csv"),
            "recomputed_from": str(source_summary),
            "api_calls": 0,
            "missing_reports": missing_reports,
        }
    )
    write_worker_csv(out_dir / "per_worker.csv", subset_rows)
    (out_dir / "summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def selected_image_ids(labels: Path) -> set[str]:
    return {normalize_image_id(row["image_id"]) for row in read_csv(labels)}


def gt_worker_counts(gt_csv: Path, selected: set[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in read_csv(gt_csv):
        image_id = normalize_image_id(row["image_id"])
        if image_id in selected:
            counts[image_id] += 1
    return counts


def normalize_image_id(value: str) -> str:
    return Path(value).stem


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def place_file(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


if __name__ == "__main__":
    raise SystemExit(main())
