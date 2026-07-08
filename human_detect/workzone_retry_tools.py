from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .eval_workzone_report import aggregate_metrics, write_worker_csv
from .eval_workzone_vlm_distance import aggregate_direct_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for retrying and merging work-zone VLM eval batches.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    labels = subparsers.add_parser("make-labels", help="Create a retry labels CSV from failed VLM batch logs.")
    labels.add_argument("--source-labels", required=True)
    labels.add_argument("--log", required=True)
    labels.add_argument("--batch-size", type=int, required=True)
    labels.add_argument("--out", required=True)

    merge = subparsers.add_parser("merge", help="Merge retry results into a full base run.")
    merge.add_argument("--base-dir", required=True)
    merge.add_argument("--retry-dir", required=True)
    merge.add_argument("--out-dir", required=True)
    merge.add_argument("--mode", choices=["attributes", "direct"], required=True)
    merge.add_argument("--api-timeout", type=float, default=None)
    merge.add_argument("--vlm-batch-size", type=int, default=None)
    merge.add_argument("--note", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "make-labels":
        make_retry_labels(
            source_labels=args.source_labels,
            log_path=args.log,
            batch_size=args.batch_size,
            out_path=args.out,
        )
        return 0
    if args.command == "merge":
        merge_runs(
            base_dir=args.base_dir,
            retry_dir=args.retry_dir,
            out_dir=args.out_dir,
            mode=args.mode,
            api_timeout=args.api_timeout,
            vlm_batch_size=args.vlm_batch_size,
            note=args.note,
        )
        return 0
    raise AssertionError(args.command)


def make_retry_labels(
    *,
    source_labels: str | Path,
    log_path: str | Path,
    batch_size: int,
    out_path: str | Path,
) -> None:
    failed_batches = parse_failed_batches(log_path)
    image_paths = select_unique_image_paths(source_labels)
    retry_paths: list[str] = []
    for batch_index in failed_batches:
        start = (batch_index - 1) * batch_size
        retry_paths.extend(image_paths[start : start + batch_size])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path"])
        writer.writeheader()
        for image_path in retry_paths:
            writer.writerow({"image_path": image_path})
    print(json.dumps({"failed_batches": failed_batches, "images": len(retry_paths), "out": str(out_path)}, indent=2))


def parse_failed_batches(log_path: str | Path) -> list[int]:
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    failed = []
    for match in re.finditer(r"\[(?:local )?vlm batch (\d+)/\d+\].*status=failed", text):
        failed.append(int(match.group(1)))
    return sorted(set(failed))


def select_unique_image_paths(labels_csv: str | Path) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    with Path(labels_csv).open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image_path = row["image_path"]
            if image_path in seen:
                continue
            seen.add(image_path)
            paths.append(image_path)
    return paths


def merge_runs(
    *,
    base_dir: str | Path,
    retry_dir: str | Path,
    out_dir: str | Path,
    mode: str,
    api_timeout: float | None,
    vlm_batch_size: int | None,
    note: str | None,
) -> None:
    base_dir = Path(base_dir)
    retry_dir = Path(retry_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports_out = out_dir / "reports"
    if reports_out.exists():
        shutil.rmtree(reports_out)
    shutil.copytree(base_dir / "reports", reports_out)
    for retry_report in (retry_dir / "reports").glob("*.json"):
        shutil.copy2(retry_report, reports_out / retry_report.name)

    base_rows = read_csv_rows(base_dir / "per_worker.csv")
    retry_rows = read_csv_rows(retry_dir / "per_worker.csv")
    retry_image_ids = {f"{path.stem}.png" for path in (retry_dir / "reports").glob("*.json")}
    retry_image_ids.update(row["image_id"] for row in retry_rows)
    merged_rows = [row for row in base_rows if row["image_id"] not in retry_image_ids]
    merged_rows.extend(retry_rows)
    merged_rows.sort(key=lambda row: (row["image_id"], int(row.get("pred_worker_index") or 0), int(row.get("gt_worker_index") or 0)))
    write_worker_csv(out_dir / "per_worker.csv", merged_rows)

    base_summary = json.loads((base_dir / "summary.json").read_text(encoding="utf-8"))
    retry_summary_path = retry_dir / "summary.json"
    retry_summary = json.loads(retry_summary_path.read_text(encoding="utf-8")) if retry_summary_path.exists() else {}
    fake_per_image = [
        {
            "pred_workers": base_summary["pred_workers"],
            "gt_workers": base_summary["gt_workers"],
            "matched": base_summary["matched"],
        }
    ]
    if mode == "attributes":
        merged_summary = aggregate_metrics(fake_per_image, merged_rows)
    else:
        merged_summary = aggregate_direct_metrics(fake_per_image, merged_rows)

    summary = dict(base_summary)
    summary.update(merged_summary)
    summary["reports_dir"] = str(reports_out)
    summary["per_worker_csv"] = str(out_dir / "per_worker.csv")
    summary["merged_from"] = {"base_dir": str(base_dir), "retry_dir": str(retry_dir), "retry_images": len(retry_image_ids)}
    if "vlm_failed_batches" in retry_summary:
        summary["vlm_failed_batches"] = retry_summary["vlm_failed_batches"]
    if api_timeout is not None:
        summary["retry_api_timeout_s"] = api_timeout
    if vlm_batch_size is not None:
        summary["retry_vlm_batch_size"] = vlm_batch_size
    if note:
        summary["note"] = note
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())
