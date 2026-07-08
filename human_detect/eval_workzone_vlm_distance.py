from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
from openai import OpenAI

from .eval_workzone_report import select_image_paths, write_worker_csv
from .infer_distance_head import DistanceHeadEstimator
from .workzone_report import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_WORKZONE_CHECKPOINT,
    DEFAULT_WORKZONE_DETECTOR,
    distance_band_from_meters,
    greedy_matches,
    load_gt_for_image,
    normalize_bool_uncertain,
    normalize_choice,
    normalize_helmet,
    parse_json_object,
    run_workzone_report_with_estimator,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate VLM direct distance estimates on the work-zone dataset.")
    parser.add_argument("--labels", default=r"artifacts\workzone_v1\data_splits\workzone_depth.all500.csv")
    parser.add_argument("--gt-csv", default=r"work-zone-safety-rgbd-dataset\annotations\worker_gt_merged.csv")
    parser.add_argument("--out-dir", default=r"runs\workzone\vlm_direct_distance")
    parser.add_argument("--limit", type=int, default=0)
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
    parser.add_argument("--vlm-batch-size", type=int, default=10)
    parser.add_argument("--vlm-workers", type=int, default=5)
    parser.add_argument("--api-timeout", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.api_key:
        raise RuntimeError("QWEN_API_KEY is required.")

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
        result = run_workzone_report_with_estimator(
            image_path=image_path,
            estimator=estimator,
            equipment_type=args.equipment_type,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            annotated_image=annotated_dir / f"{image_path.stem}.jpg",
            skip_vlm=True,
        )
        for worker in result["internal_workers"]:
            worker["distance_to_equipment_m"] = None
            worker["distance_band"] = "unknown"
        report_results.append((image_path, result))
        print(f"[local {image_index}/{len(image_paths)}] {image_path.name}: workers={len(result['internal_workers'])}", flush=True)

    batches = make_vlm_batches(report_results, batch_size=max(1, args.vlm_batch_size))
    failed_batches = run_vlm_batches(
        batches,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        timeout=args.api_timeout,
        worker_count=max(1, min(args.vlm_workers, len(batches))),
    )

    per_image: list[dict[str, Any]] = []
    per_worker_rows: list[dict[str, Any]] = []
    for image_index, (image_path, result) in enumerate(report_results, start=1):
        report = build_direct_report(image_path, args.equipment_type, result["internal_workers"])
        (reports_dir / f"{image_path.stem}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        gt_workers = load_gt_for_image(args.gt_csv, image_path.stem)
        eval_result = evaluate_direct(result["internal_workers"], gt_workers)
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
            per_worker_rows.append(flatten_direct_row(image_path.name, row))
        print(f"[eval {image_index}/{len(image_paths)}] {image_path.name}: matched={eval_result['matched']}/{eval_result['gt_workers']}", flush=True)

    summary = aggregate_direct_metrics(per_image, per_worker_rows)
    summary.update(
        {
            "model": args.model,
            "vlm_mode": "direct_distance_and_attributes",
            "labels": str(Path(args.labels)),
            "gt_csv": str(Path(args.gt_csv)),
            "images": len(image_paths),
            "vlm_batch_size": max(1, args.vlm_batch_size),
            "vlm_workers": max(1, args.vlm_workers),
            "api_timeout_s": args.api_timeout,
            "vlm_failed_batches": failed_batches,
            "reports_dir": str(reports_dir),
            "annotated_dir": str(annotated_dir),
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_worker_csv(out_dir / "per_worker.csv", per_worker_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


def make_vlm_batches(report_results: list[tuple[Path, dict[str, Any]]], *, batch_size: int) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(report_results), batch_size), start=1):
        chunk = report_results[start : start + batch_size]
        items = [
            {
                "image_id": image_path.name,
                "annotated_image": result["annotated_image"],
                "workers": result["internal_workers"],
            }
            for image_path, result in chunk
            if result["internal_workers"]
        ]
        batches.append({"index": batch_index, "chunk": chunk, "items": items})
    return batches


def run_vlm_batches(
    batches: list[dict[str, Any]],
    *,
    api_base: str,
    api_key: str,
    model: str,
    timeout: float,
    worker_count: int,
) -> int:
    failed = 0
    if worker_count == 1:
        for batch in batches:
            attrs, status = request_vlm_direct_batch(batch["items"], api_base=api_base, api_key=api_key, model=model, timeout=timeout)
            failed += int(status != "ok")
            merge_direct_batch(batch["chunk"], attrs)
            print(f"[vlm batch {batch['index']}/{len(batches)}] images={len(batch['chunk'])} status={status}", flush=True)
        return failed

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(request_vlm_direct_batch, batch["items"], api_base=api_base, api_key=api_key, model=model, timeout=timeout): batch
            for batch in batches
        }
        for future in as_completed(futures):
            batch = futures[future]
            attrs, status = future.result()
            failed += int(status != "ok")
            merge_direct_batch(batch["chunk"], attrs)
            print(
                f"[vlm batch {batch['index']}/{len(batches)}] images={len(batch['chunk'])} "
                f"workers={len(batch['items'])} status={status}",
                flush=True,
            )
    return failed


def request_vlm_direct_batch(
    items: list[dict[str, Any]],
    *,
    api_base: str,
    api_key: str,
    model: str,
    timeout: float,
) -> tuple[dict[str, dict[str, Any]], str]:
    try:
        attrs = call_qwen_direct_distance_batch(items, api_base=api_base, api_key=api_key, model=model, timeout=timeout)
        return attrs, "ok"
    except Exception as exc:
        return {}, f"failed: {type(exc).__name__}: {exc}"


def call_qwen_direct_distance_batch(
    items: list[dict[str, Any]],
    *,
    api_base: str,
    api_key: str,
    model: str,
    timeout: float,
) -> dict[str, dict[str, Any]]:
    if not items:
        return {}

    content: list[dict[str, Any]] = [{"type": "text", "text": direct_distance_prompt(items)}]
    for item in items:
        image_b64 = base64.b64encode(Path(item["annotated_image"]).read_bytes()).decode("ascii")
        content.extend(
            [
                {"type": "text", "text": f"IMAGE_ID: {item['image_id']}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]
        )

    client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return strict JSON only. No markdown."},
            {"role": "user", "content": content},
        ],
        temperature=0,
    )
    parsed = parse_json_object(response.choices[0].message.content or "{}")
    by_image: dict[str, dict[str, Any]] = {}
    for image_row in parsed.get("images", []):
        image_id = image_row.get("image_id")
        if image_id:
            by_image[str(image_id)] = {"workers": image_row.get("workers", [])}
    return by_image


def direct_distance_prompt(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in items:
        lines.append(f"image_id={item['image_id']}")
        for worker in item["workers"]:
            lines.append(
                f"- worker_index={worker['worker_index']}, bbox_xyxy="
                f"{[round(float(v), 1) for v in worker['bbox_xyxy']]}"
            )
    worker_block = "\n".join(lines)
    return f"""
You are labeling construction worker safety attributes and estimating distance from annotated work-zone images.
Each image is from an equipment-mounted or equipment-reference camera. Each image has red boxes labeled W1, W2.
Use the red W number as worker_index for that image.

Images and workers:
{worker_block}

Return JSON only with this schema:
{{
  "images": [
    {{
      "image_id": "image file name from IMAGE_ID",
      "workers": [
        {{
          "worker_index": 1,
          "distance_to_equipment_m": 4.2,
          "high_visibility_vest": true | false | "uncertain",
          "helmet_status": "worn" | "absent" | "uncertain",
          "orientation": "Facing" | "Side" | "Back" | "uncertain",
          "occlusion_level": "none" | "partial" | "heavy" | "uncertain"
        }}
      ]
    }}
  ]
}}

Rules:
- distance_to_equipment_m must be your best numeric estimate in meters from the camera/equipment to the worker.
- Use image perspective, worker size, ground plane, and surrounding scene cues to estimate distance.
- Do not return distance bands; return only the numeric meter value.
- high_visibility_vest=true only when a high-visibility vest or jacket is clearly visible.
- helmet_status=worn only when a helmet is on the worker's head. Helmet in hand means absent.
- orientation is relative to camera view.
- Include every image_id and every worker_index listed above.
""".strip()


def merge_direct_batch(chunk: list[tuple[Path, dict[str, Any]]], attrs_by_image: dict[str, dict[str, Any]]) -> None:
    for image_path, result in chunk:
        attrs = attrs_by_image.get(image_path.name, {"workers": []})
        by_index = {
            worker_index: item
            for item in attrs.get("workers", [])
            if (worker_index := normalize_worker_index(item.get("worker_index"))) is not None
        }
        for worker in result["internal_workers"]:
            item = by_index.get(int(worker["worker_index"]), {})
            distance = normalize_distance(item.get("distance_to_equipment_m"))
            worker["distance_to_equipment_m"] = distance
            worker["distance_band"] = distance_band_from_meters(distance) if distance is not None else "unknown"
            worker["high_visibility_vest"] = normalize_bool_uncertain(item.get("high_visibility_vest"))
            worker["helmet_status"] = normalize_helmet(item.get("helmet_status"))
            worker["orientation"] = normalize_choice(item.get("orientation"), {"Facing", "Side", "Back", "uncertain"})
            worker["occlusion_level"] = normalize_choice(item.get("occlusion_level"), {"none", "partial", "heavy", "uncertain"})


def normalize_worker_index(value: Any) -> int | None:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else None


def normalize_distance(value: Any) -> float | None:
    if value in {None, "", "uncertain", "unknown"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        distance = float(match.group(0))
    except ValueError:
        return None
    if not math.isfinite(distance) or distance <= 0:
        return None
    return distance


def build_direct_report(image_path: str | Path, equipment_type: str, workers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image_id": Path(image_path).name,
        "equipment_type": equipment_type,
        "worker_count": len(workers),
        "workers": [
            {
                "worker_index": worker["worker_index"],
                "distance_to_equipment_m": round(worker["distance_to_equipment_m"], 3)
                if worker["distance_to_equipment_m"] is not None
                else None,
                "distance_band": worker["distance_band"],
                "high_visibility_vest": worker["high_visibility_vest"],
                "helmet_status": worker["helmet_status"],
                "orientation": worker["orientation"],
                "occlusion_level": worker["occlusion_level"],
            }
            for worker in workers
        ],
    }


def evaluate_direct(workers: list[dict[str, Any]], gt_workers: list[Any]) -> dict[str, Any]:
    matches = greedy_matches(workers, gt_workers)
    per_worker: list[dict[str, Any]] = []
    attr_names = ["high_visibility_vest", "helmet_status", "orientation", "occlusion_level"]
    correct = {name: 0 for name in attr_names}
    total = {name: 0 for name in attr_names}
    distance_errors: list[float] = []
    distance_band_correct = 0
    distance_band_total = 0

    for pred_index, gt_index, iou in matches:
        pred = workers[pred_index]
        gt = gt_workers[gt_index]
        pred_distance = pred.get("distance_to_equipment_m")
        row = {
            "pred_worker_index": pred["worker_index"],
            "gt_worker_index": gt.worker_index,
            "iou": iou,
            "pred": {
                "distance_to_equipment_m": pred_distance,
                "distance_band": pred.get("distance_band"),
                "high_visibility_vest": pred["high_visibility_vest"],
                "helmet_status": pred["helmet_status"],
                "orientation": pred["orientation"],
                "occlusion_level": pred["occlusion_level"],
            },
            "gt": {
                "distance_to_equipment_m": gt.distance_m,
                "high_visibility_vest": gt.high_visibility_vest,
                "helmet_status": gt.helmet_status,
                "orientation": gt.orientation,
                "occlusion_level": gt.occlusion_level,
            },
        }
        if gt.distance_m is not None and pred_distance is not None:
            error = float(pred_distance) - float(gt.distance_m)
            row["distance_error_m"] = error
            distance_errors.append(abs(error))
            gt_band = distance_band_from_meters(gt.distance_m)
            row["gt"]["distance_band"] = gt_band
            distance_band_total += 1
            distance_band_correct += int(pred.get("distance_band") == gt_band)

        for name in attr_names:
            pred_value = pred[name]
            gt_value = getattr(gt, name)
            if gt_value != "uncertain":
                total[name] += 1
                correct[name] += int(pred_value == gt_value)
        per_worker.append(row)

    metrics = {
        name: {"correct": correct[name], "total": total[name], "accuracy": correct[name] / total[name] if total[name] else None}
        for name in attr_names
    }
    metrics["distance_mae_m"] = float(np.mean(distance_errors)) if distance_errors else None
    metrics["distance_valid"] = len(distance_errors)
    metrics["distance_band"] = {
        "correct": distance_band_correct,
        "total": distance_band_total,
        "accuracy": distance_band_correct / distance_band_total if distance_band_total else None,
    }
    return {
        "matched": len(matches),
        "pred_workers": len(workers),
        "gt_workers": len(gt_workers),
        "metrics": metrics,
        "per_worker": per_worker,
    }


def flatten_direct_row(image_id: str, row: dict[str, Any]) -> dict[str, Any]:
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


def aggregate_direct_metrics(per_image: list[dict[str, Any]], per_worker_rows: list[dict[str, Any]]) -> dict[str, Any]:
    attr_specs = [
        ("high_visibility_vest", "pred_high_visibility_vest", "gt_high_visibility_vest"),
        ("helmet_status", "pred_helmet_status", "gt_helmet_status"),
        ("orientation", "pred_orientation", "gt_orientation"),
        ("occlusion_level", "pred_occlusion_level", "gt_occlusion_level"),
        ("distance_band", "pred_distance_band", "gt_distance_band"),
    ]
    metrics: dict[str, Any] = {}
    for name, pred_key, gt_key in attr_specs:
        valid = [row for row in per_worker_rows if row.get(gt_key) not in {None, "", "uncertain"} and row.get(pred_key) not in {None, "", "unknown"}]
        correct = sum(row.get(pred_key) == row.get(gt_key) for row in valid)
        metrics[name] = {"correct": int(correct), "total": len(valid), "accuracy": correct / len(valid) if valid else None}

    errors = [abs(float(row["distance_error_m"])) for row in per_worker_rows if row.get("distance_error_m") not in {None, ""}]
    metrics["distance_mae_m"] = float(np.mean(errors)) if errors else None
    metrics["distance_valid"] = len(errors)
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


if __name__ == "__main__":
    raise SystemExit(main())
