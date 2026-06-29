from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openai import OpenAI

from .infer_distance_head import DistanceHeadEstimator
from .matching import bbox_iou


DEFAULT_BASE_URL = "https://ws-2vah2d019k5467zo.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-omni-flash"


@dataclass(frozen=True)
class WorkzoneGtWorker:
    image_id: str
    worker_index: int
    bbox_xyxy: tuple[float, float, float, float]
    distance_m: float | None
    high_visibility_vest: str
    helmet_status: str
    orientation: str
    occlusion_level: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scheme-1 work-zone report pipeline: distance head + VLM visual attributes.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eval-out", default=None)
    parser.add_argument("--gt-csv", default=r"work-zone-safety-rgbd-dataset\annotations\worker_gt_merged.csv")
    parser.add_argument("--checkpoint", default=r"runs\workzone\workzone_yolo_distance_head.pt")
    parser.add_argument("--base-model", default=r"models\yolo11n.pt")
    parser.add_argument("--detector", default=r"models\yolo11n.pt")
    parser.add_argument("--equipment-type", default="dump truck")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--api-base", default=os.environ.get("QWEN_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", DEFAULT_MODEL))
    parser.add_argument("--annotated-image", default=None)
    parser.add_argument("--skip-vlm", action="store_true", help="Return uncertain visual attributes without calling the VLM.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_workzone_report(
        image_path=args.image,
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        detector=args.detector,
        equipment_type=args.equipment_type,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        annotated_image=args.annotated_image,
        skip_vlm=args.skip_vlm,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result["report"], indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["report"], indent=2, ensure_ascii=False))

    if args.eval_out:
        gt_workers = load_gt_for_image(args.gt_csv, Path(args.image).stem)
        eval_result = evaluate_report(result["internal_workers"], gt_workers)
        eval_path = Path(args.eval_out)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps(eval_result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(eval_result, indent=2, ensure_ascii=False))
    return 0


def run_workzone_report(
    *,
    image_path: str | Path,
    checkpoint: str | Path,
    base_model: str | Path,
    detector: str | Path,
    equipment_type: str,
    imgsz: int,
    conf: float,
    device: str,
    api_base: str,
    api_key: str | None,
    model: str,
    annotated_image: str | Path | None = None,
    skip_vlm: bool = False,
) -> dict[str, Any]:
    estimator = DistanceHeadEstimator(
        checkpoint_path=checkpoint,
        base_model=base_model,
        detector=detector,
        imgsz=imgsz,
        conf=conf,
        device=device,
    )
    return run_workzone_report_with_estimator(
        image_path=image_path,
        estimator=estimator,
        equipment_type=equipment_type,
        api_base=api_base,
        api_key=api_key,
        model=model,
        annotated_image=annotated_image,
        skip_vlm=skip_vlm,
    )


def run_workzone_report_with_estimator(
    *,
    image_path: str | Path,
    estimator: DistanceHeadEstimator,
    equipment_type: str,
    api_base: str,
    api_key: str | None,
    model: str,
    annotated_image: str | Path | None = None,
    skip_vlm: bool = False,
) -> dict[str, Any]:
    scheme1, _ = estimator.infer(image_path)
    persons = sorted(scheme1["persons"], key=lambda item: (item["bbox_xyxy"][0], item["bbox_xyxy"][1]))
    internal_workers = []
    for index, person in enumerate(persons, start=1):
        distance_m = float(person["distance_m"])
        internal_workers.append(
            {
                "worker_index": index,
                "bbox_xyxy": [float(value) for value in person["bbox_xyxy"]],
                "score": float(person["score"]),
                "distance_to_equipment_m": distance_m,
                "distance_band": distance_band_from_meters(distance_m),
                "high_visibility_vest": "uncertain",
                "helmet_status": "uncertain",
                "orientation": "uncertain",
                "occlusion_level": "uncertain",
            }
        )

    annotated_path = Path(annotated_image) if annotated_image else Path("runs") / "workzone" / f"{Path(image_path).stem}_vlm_annotated.jpg"
    make_annotated_image(image_path, internal_workers, annotated_path)

    if internal_workers and not skip_vlm:
        if not api_key:
            raise RuntimeError("QWEN_API_KEY is required unless --skip-vlm is used.")
        vlm_attrs = call_qwen_visual_attributes(
            annotated_path,
            workers=internal_workers,
            api_base=api_base,
            api_key=api_key,
            model=model,
        )
        merge_vlm_attributes(internal_workers, vlm_attrs)

    report = build_report(image_path, equipment_type, internal_workers)
    return {"report": report, "internal_workers": internal_workers, "annotated_image": str(annotated_path)}


def build_report(image_path: str | Path, equipment_type: str, internal_workers: list[dict[str, Any]]) -> dict[str, Any]:
    report = {
        "image_id": Path(image_path).name,
        "equipment_type": equipment_type,
        "worker_count": len(internal_workers),
        "workers": [
            {
                "worker_index": worker["worker_index"],
                "distance_to_equipment_m": round(float(worker["distance_to_equipment_m"]), 3),
                "distance_band": worker["distance_band"],
                "high_visibility_vest": worker["high_visibility_vest"],
                "helmet_status": worker["helmet_status"],
                "orientation": worker["orientation"],
                "occlusion_level": worker["occlusion_level"],
            }
            for worker in internal_workers
        ],
    }
    return report


def call_qwen_visual_attributes(
    image_path: str | Path,
    *,
    workers: list[dict[str, Any]],
    api_base: str,
    api_key: str,
    model: str,
) -> dict[str, Any]:
    image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    worker_lines = "\n".join(
        f"- worker_index={worker['worker_index']}, bbox_xyxy={[round(float(v), 1) for v in worker['bbox_xyxy']]}"
        for worker in workers
    )
    prompt = f"""
You are labeling construction worker visual attributes from one annotated image.
The image has red boxes with labels like W1, W2. Use those labels as worker_index.

Workers:
{worker_lines}

Return JSON only with this schema:
{{
  "workers": [
    {{
      "worker_index": 1,
      "high_visibility_vest": true | false | "uncertain",
      "helmet_status": "worn" | "absent" | "uncertain",
      "orientation": "Facing" | "Side" | "Back" | "uncertain",
      "occlusion_level": "none" | "partial" | "heavy" | "uncertain"
    }}
  ]
}}

Rules:
- high_visibility_vest=true only when a high-visibility vest or jacket is clearly visible.
- helmet_status=worn only when a helmet is on the worker's head. Helmet in hand means absent.
- orientation is relative to camera view.
- Do not estimate distance.
""".strip()
    client = OpenAI(api_key=api_key, base_url=api_base)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return strict JSON only. No markdown."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            },
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    return parse_json_object(content)


def call_qwen_visual_attributes_batch(
    items: list[dict[str, Any]],
    *,
    api_base: str,
    api_key: str,
    model: str,
) -> dict[str, dict[str, Any]]:
    if not items:
        return {}

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": _batch_prompt(
                [
                    {
                        "image_id": item["image_id"],
                        "workers": item["workers"],
                    }
                    for item in items
                ]
            ),
        }
    ]
    for item in items:
        image_b64 = base64.b64encode(Path(item["annotated_image"]).read_bytes()).decode("ascii")
        content.extend(
            [
                {"type": "text", "text": f"IMAGE_ID: {item['image_id']}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]
        )

    client = OpenAI(api_key=api_key, base_url=api_base)
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


def _batch_prompt(items: list[dict[str, Any]]) -> str:
    lines = []
    for item in items:
        lines.append(f"image_id={item['image_id']}")
        for worker in item["workers"]:
            lines.append(
                f"- worker_index={worker['worker_index']}, bbox_xyxy="
                f"{[round(float(v), 1) for v in worker['bbox_xyxy']]}"
            )
    worker_block = "\n".join(lines)
    return f"""
You are labeling construction worker visual attributes from multiple annotated images.
Each image is preceded by a text marker IMAGE_ID. The image itself has red boxes with labels like W1, W2.
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
- high_visibility_vest=true only when a high-visibility vest or jacket is clearly visible.
- helmet_status=worn only when a helmet is on the worker's head. Helmet in hand means absent.
- orientation is relative to camera view.
- Do not estimate distance.
- Include every image_id and every worker_index listed above.
""".strip()


def make_annotated_image(image_path: str | Path, workers: list[dict[str, Any]], out_path: str | Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    for worker in workers:
        x1, y1, x2, y2 = [int(round(float(value))) for value in worker["bbox_xyxy"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"W{worker['worker_index']}"
        cv2.putText(image, label, (max(0, x1), max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def merge_vlm_attributes(workers: list[dict[str, Any]], vlm_attrs: dict[str, Any]) -> None:
    by_index = {int(item.get("worker_index")): item for item in vlm_attrs.get("workers", []) if item.get("worker_index") is not None}
    for worker in workers:
        attrs = by_index.get(int(worker["worker_index"]), {})
        worker["high_visibility_vest"] = normalize_bool_uncertain(attrs.get("high_visibility_vest"))
        worker["helmet_status"] = normalize_helmet(attrs.get("helmet_status"))
        worker["orientation"] = normalize_choice(attrs.get("orientation"), {"Facing", "Side", "Back", "uncertain"})
        worker["occlusion_level"] = normalize_choice(attrs.get("occlusion_level"), {"none", "partial", "heavy", "uncertain"})


def load_gt_for_image(gt_csv: str | Path, image_id: str) -> list[WorkzoneGtWorker]:
    rows: list[WorkzoneGtWorker] = []
    with Path(gt_csv).open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["image_id"] != image_id:
                continue
            depth_raw = row.get("depth_z_m")
            try:
                distance_m = None if depth_raw in {"", None} else float(depth_raw)
            except ValueError:
                distance_m = None
            rows.append(
                WorkzoneGtWorker(
                    image_id=row["image_id"],
                    worker_index=int(row["worker_index"]),
                    bbox_xyxy=(float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"])),
                    distance_m=distance_m,
                    high_visibility_vest=normalize_bool_uncertain(row.get("high_visibility_vest")),
                    helmet_status=normalize_gt_helmet(row.get("helmet_status")),
                    orientation=normalize_choice(row.get("orientation"), {"Facing", "Side", "Back", "uncertain"}),
                    occlusion_level=normalize_choice(row.get("occlusion_level"), {"none", "partial", "heavy", "uncertain"}),
                )
            )
    return rows


def evaluate_report(workers: list[dict[str, Any]], gt_workers: list[WorkzoneGtWorker]) -> dict[str, Any]:
    matches = greedy_matches(workers, gt_workers)
    per_worker = []
    attr_names = ["high_visibility_vest", "helmet_status", "orientation", "occlusion_level"]
    correct = {name: 0 for name in attr_names}
    total = {name: 0 for name in attr_names}
    distance_band_correct = 0
    distance_band_total = 0
    distance_errors = []
    for pred_index, gt_index, iou in matches:
        pred = workers[pred_index]
        gt = gt_workers[gt_index]
        row = {
            "pred_worker_index": pred["worker_index"],
            "gt_worker_index": gt.worker_index,
            "iou": iou,
            "pred": {
                "distance_to_equipment_m": pred["distance_to_equipment_m"],
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
        if gt.distance_m is not None:
            error = float(pred["distance_to_equipment_m"]) - gt.distance_m
            row["distance_error_m"] = error
            distance_errors.append(abs(error))
            gt_band = distance_band_from_meters(gt.distance_m)
            row["gt"]["distance_band"] = gt_band
            row["pred"]["distance_band"] = pred["distance_band"]
            distance_band_total += 1
            distance_band_correct += int(pred["distance_band"] == gt_band)
        for name in attr_names:
            pred_value = pred[name]
            gt_value = getattr(gt, name)
            if gt_value != "uncertain":
                total[name] += 1
                correct[name] += int(pred_value == gt_value)
        per_worker.append(row)

    metrics = {
        name: {"correct": correct[name], "total": total[name], "accuracy": (correct[name] / total[name] if total[name] else None)}
        for name in attr_names
    }
    metrics["distance_mae_m"] = float(np.mean(distance_errors)) if distance_errors else None
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


def greedy_matches(workers: list[dict[str, Any]], gt_workers: list[WorkzoneGtWorker], *, iou_threshold: float = 0.3) -> list[tuple[int, int, float]]:
    candidates = []
    for pred_index, worker in enumerate(workers):
        for gt_index, gt in enumerate(gt_workers):
            iou = bbox_iou(worker["bbox_xyxy"], gt.bbox_xyxy)
            if iou >= iou_threshold:
                candidates.append((pred_index, gt_index, iou))
    candidates.sort(key=lambda item: item[2], reverse=True)
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches = []
    for pred_index, gt_index, iou in candidates:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        matches.append((pred_index, gt_index, iou))
    return sorted(matches)


def distance_band_from_meters(distance_m: float) -> str:
    if distance_m < 3.0:
        return "Close"
    if distance_m <= 5.0:
        return "Careful"
    return "Safe"


def normalize_gt_helmet(value: Any) -> str:
    value = str(value or "uncertain")
    if value in {"worn_secured", "worn_unsecured", "worn_unknown", "worn"}:
        return "worn"
    if value in {"absent", "in_hand"}:
        return "absent"
    return "uncertain"


def normalize_helmet(value: Any) -> str:
    value = str(value or "uncertain").strip().lower()
    if value in {"worn", "wearing", "yes", "true"}:
        return "worn"
    if value in {"absent", "no", "false", "none", "not_worn", "not worn", "in_hand"}:
        return "absent"
    return "uncertain"


def normalize_bool_uncertain(value: Any) -> bool | str:
    if isinstance(value, bool):
        return value
    text = str(value or "uncertain").strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return "uncertain"


def normalize_choice(value: Any, allowed: set[str]) -> str:
    text = str(value or "uncertain").strip()
    if text in allowed:
        return text
    lower_map = {item.lower(): item for item in allowed}
    return lower_map.get(text.lower(), "uncertain")


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


if __name__ == "__main__":
    raise SystemExit(main())
