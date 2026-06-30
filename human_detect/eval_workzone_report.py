from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .infer_distance_head import DistanceHeadEstimator
from .workzone_report import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_WORKZONE_CHECKPOINT,
    DEFAULT_WORKZONE_DETECTOR,
    build_report,
    call_qwen_visual_attributes_batch,
    evaluate_report,
    load_gt_for_image,
    merge_vlm_attributes,
    run_workzone_report_with_estimator,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the work-zone JSON pipeline on a small image set and evaluate against GT.")
    parser.add_argument("--labels", default=r"runs\workzone\workzone_depth.eval.csv", help="CSV used to choose images.")
    parser.add_argument("--gt-csv", default=r"work-zone-safety-rgbd-dataset\annotations\worker_gt_merged.csv")
    parser.add_argument("--out-dir", default=r"runs\workzone\qwen_eval20")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--checkpoint", default=DEFAULT_WORKZONE_CHECKPOINT)
    parser.add_argument("--base-model", default=DEFAULT_WORKZONE_DETECTOR)
    parser.add_argument("--detector", default=DEFAULT_WORKZONE_DETECTOR)
    parser.add_argument("--equipment-type", default="dump truck")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--api-base", default=os.environ.get("QWEN_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", DEFAULT_MODEL))
    parser.add_argument("--vlm-batch-size", type=int, default=1, help="Number of annotated images per VLM request.")
    parser.add_argument("--vlm-workers", type=int, default=1, help="Number of concurrent VLM requests.")
    parser.add_argument("--api-timeout", type=float, default=180.0, help="VLM request timeout in seconds.")
    parser.add_argument("--skip-vlm", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    image_paths = select_image_paths(args.labels, limit=args.limit)
    out_dir = Path(args.out_dir)
    reports_dir = out_dir / "reports"
    annotated_dir = out_dir / "annotated"
    reports_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    estimator = DistanceHeadEstimator(
        checkpoint_path=args.checkpoint,
        base_model=args.base_model,
        detector=args.detector,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
    )

    report_results: list[tuple[Path, dict[str, Any]]] = []
    for image_index, image_path in enumerate(image_paths, start=1):
        image_path = Path(image_path)
        report_result = run_workzone_report_with_estimator(
            image_path=image_path,
            estimator=estimator,
            equipment_type=args.equipment_type,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            annotated_image=annotated_dir / f"{image_path.stem}.jpg",
            skip_vlm=True,
        )
        report_results.append((image_path, report_result))
        print(f"[local {image_index}/{len(image_paths)}] {image_path.name}: workers={len(report_result['internal_workers'])}", flush=True)

    if not args.skip_vlm:
        if not args.api_key:
            raise RuntimeError("QWEN_API_KEY is required unless --skip-vlm is used.")
        batches = make_vlm_batches(report_results, batch_size=max(1, args.vlm_batch_size))
        worker_count = max(1, min(args.vlm_workers, len(batches)))
        if worker_count == 1:
            for batch in batches:
                batch_attrs, batch_status = request_vlm_batch(
                    batch["items"],
                    api_base=args.api_base,
                    api_key=args.api_key,
                    model=args.model,
                    timeout=args.api_timeout,
                )
                merge_batch_attrs(batch["chunk"], batch_attrs)
                print(f"[vlm batch {batch['index']}/{len(batches)}] images={len(batch['chunk'])} status={batch_status}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        request_vlm_batch,
                        batch["items"],
                        api_base=args.api_base,
                        api_key=args.api_key,
                        model=args.model,
                        timeout=args.api_timeout,
                    ): batch
                    for batch in batches
                }
                for future in as_completed(futures):
                    batch = futures[future]
                    batch_attrs, batch_status = future.result()
                    merge_batch_attrs(batch["chunk"], batch_attrs)
                    print(
                        f"[vlm batch {batch['index']}/{len(batches)}] images={len(batch['chunk'])} "
                        f"workers={len(batch['items'])} status={batch_status}",
                        flush=True,
                    )

    per_worker_rows: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    for image_index, (image_path, report_result) in enumerate(report_results, start=1):
        report = build_report(image_path, args.equipment_type, report_result["internal_workers"])
        report_result["report"] = report
        (reports_dir / f"{image_path.stem}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        gt_workers = load_gt_for_image(args.gt_csv, image_path.stem)
        eval_result = evaluate_report(report_result["internal_workers"], gt_workers)
        per_image.append(
            {
                "image_id": image_path.name,
                "pred_workers": eval_result["pred_workers"],
                "gt_workers": eval_result["gt_workers"],
                "matched": eval_result["matched"],
                "metrics": eval_result["metrics"],
            }
        )
        for row in eval_result["per_worker"]:
            per_worker_rows.append(flatten_worker_row(image_path.name, row))
        print(f"[eval {image_index}/{len(image_paths)}] {image_path.name}: matched={eval_result['matched']}/{eval_result['gt_workers']}", flush=True)

    summary = aggregate_metrics(per_image, per_worker_rows)
    summary.update(
        {
            "model": args.model,
            "labels": str(Path(args.labels)),
            "gt_csv": str(Path(args.gt_csv)),
            "images": len(image_paths),
            "vlm_batch_size": max(1, args.vlm_batch_size),
            "vlm_workers": max(1, args.vlm_workers),
            "api_timeout_s": args.api_timeout,
            "reports_dir": str(reports_dir),
            "annotated_dir": str(annotated_dir),
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_worker_csv(out_dir / "per_worker.csv", per_worker_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def make_vlm_batches(
    report_results: list[tuple[Path, dict[str, Any]]],
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(report_results), batch_size), start=1):
        chunk = report_results[start : start + batch_size]
        items = [
            {
                "image_id": image_path.name,
                "annotated_image": report_result["annotated_image"],
                "workers": report_result["internal_workers"],
            }
            for image_path, report_result in chunk
            if report_result["internal_workers"]
        ]
        batches.append({"index": batch_index, "chunk": chunk, "items": items})
    return batches


def request_vlm_batch(
    items: list[dict[str, Any]],
    *,
    api_base: str,
    api_key: str,
    model: str,
    timeout: float,
) -> tuple[dict[str, dict[str, Any]], str]:
    try:
        batch_attrs = call_qwen_visual_attributes_batch(
            items,
            api_base=api_base,
            api_key=api_key,
            model=model,
            timeout=timeout,
        )
        return batch_attrs, "ok"
    except Exception as exc:  # Keep full-run outputs even if one remote VLM call fails.
        return {}, f"failed: {type(exc).__name__}: {exc}"


def merge_batch_attrs(
    chunk: list[tuple[Path, dict[str, Any]]],
    batch_attrs: dict[str, dict[str, Any]],
) -> None:
    for image_path, report_result in chunk:
        attrs = batch_attrs.get(image_path.name, {"workers": []})
        merge_vlm_attributes(report_result["internal_workers"], attrs)


def select_image_paths(labels_csv: str | Path, *, limit: int) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    with Path(labels_csv).open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image_path = row["image_path"]
            if image_path in seen:
                continue
            seen.add(image_path)
            paths.append(image_path)
            if limit > 0 and len(paths) >= limit:
                break
    return paths


def flatten_worker_row(image_id: str, row: dict[str, Any]) -> dict[str, Any]:
    pred = row["pred"]
    gt = row["gt"]
    return {
        "image_id": image_id,
        "pred_worker_index": row["pred_worker_index"],
        "gt_worker_index": row["gt_worker_index"],
        "iou": row["iou"],
        "pred_distance_m": pred.get("distance_to_equipment_m"),
        "gt_distance_m": gt.get("distance_to_equipment_m"),
        "distance_error_m": row.get("distance_error_m"),
        "pred_distance_band": pred.get("distance_band"),
        "gt_distance_band": gt.get("distance_band"),
        "pred_high_visibility_vest": pred.get("high_visibility_vest"),
        "gt_high_visibility_vest": gt.get("high_visibility_vest"),
        "pred_helmet_status": pred.get("helmet_status"),
        "gt_helmet_status": gt.get("helmet_status"),
        "pred_orientation": pred.get("orientation"),
        "gt_orientation": gt.get("orientation"),
        "pred_occlusion_level": pred.get("occlusion_level"),
        "gt_occlusion_level": gt.get("occlusion_level"),
    }


def aggregate_metrics(per_image: list[dict[str, Any]], per_worker_rows: list[dict[str, Any]]) -> dict[str, Any]:
    attr_specs = [
        ("high_visibility_vest", "pred_high_visibility_vest", "gt_high_visibility_vest"),
        ("helmet_status", "pred_helmet_status", "gt_helmet_status"),
        ("orientation", "pred_orientation", "gt_orientation"),
        ("occlusion_level", "pred_occlusion_level", "gt_occlusion_level"),
        ("distance_band", "pred_distance_band", "gt_distance_band"),
    ]
    metrics: dict[str, Any] = {}
    for name, pred_key, gt_key in attr_specs:
        valid = [row for row in per_worker_rows if row.get(gt_key) not in {None, "", "uncertain"}]
        correct = sum(row.get(pred_key) == row.get(gt_key) for row in valid)
        metrics[name] = {"correct": int(correct), "total": len(valid), "accuracy": correct / len(valid) if valid else None}

    errors = [abs(float(row["distance_error_m"])) for row in per_worker_rows if row.get("distance_error_m") not in {None, ""}]
    metrics["distance_mae_m"] = float(np.mean(errors)) if errors else None
    pred_workers = sum(int(item["pred_workers"]) for item in per_image)
    gt_workers = sum(int(item["gt_workers"]) for item in per_image)
    matched = sum(int(item["matched"]) for item in per_image)
    return {
        "pred_workers": pred_workers,
        "gt_workers": gt_workers,
        "matched": matched,
        "match_rate": matched / gt_workers if gt_workers else 0.0,
        "metrics": metrics,
    }


def write_worker_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
