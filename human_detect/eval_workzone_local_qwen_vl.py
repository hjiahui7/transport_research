from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .eval_workzone_report import (
    aggregate_metrics,
    flatten_worker_row,
    make_vlm_batches,
    select_image_paths,
    write_worker_csv,
)
from .infer_distance_head import DistanceHeadEstimator
from .workzone_report import (
    DEFAULT_WORKZONE_CHECKPOINT,
    DEFAULT_WORKZONE_DETECTOR,
    build_report,
    evaluate_report,
    load_gt_for_image,
    merge_vlm_attributes,
    parse_json_object,
    run_workzone_report_with_estimator,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate local Qwen2.5-VL attributes on the work-zone dataset.")
    parser.add_argument("--labels", default=r"artifacts\workzone_v1\data_splits\workzone_depth.all500.csv")
    parser.add_argument("--gt-csv", default=r"work-zone-safety-rgbd-dataset\annotations\worker_gt_merged.csv")
    parser.add_argument("--out-dir", default=r"runs\workzone\qwen2_5_vl_3b_local_all500_ft_head")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint", default=DEFAULT_WORKZONE_CHECKPOINT)
    parser.add_argument("--base-model", default=DEFAULT_WORKZONE_DETECTOR)
    parser.add_argument("--detector", default=DEFAULT_WORKZONE_DETECTOR)
    parser.add_argument("--equipment-type", default="dump truck")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face download if the model is not cached.")
    parser.add_argument("--vlm-batch-size", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-pixels", type=int, default=1048576, help="Maximum input pixels per image for the Qwen processor.")
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
            api_base="",
            api_key=None,
            model=args.local_model,
            annotated_image=annotated_dir / f"{image_path.stem}.jpg",
            skip_vlm=True,
        )
        report_results.append((image_path, report_result))
        print(f"[local {image_index}/{len(image_paths)}] {image_path.name}: workers={len(report_result['internal_workers'])}", flush=True)

    model, processor = load_local_qwen_vl(
        args.local_model,
        local_files_only=not args.allow_download,
        max_pixels=args.max_pixels,
    )

    failed_batches = 0
    batches = make_vlm_batches(report_results, batch_size=max(1, args.vlm_batch_size))
    for batch in batches:
        batch_attrs, batch_status = request_local_batch(
            batch["items"],
            model=model,
            processor=processor,
            max_new_tokens=args.max_new_tokens,
        )
        failed_batches += int(batch_status != "ok")
        merge_batch_attrs(batch["chunk"], batch_attrs)
        print(
            f"[local vlm batch {batch['index']}/{len(batches)}] "
            f"images={len(batch['chunk'])} workers={len(batch['items'])} status={batch_status}",
            flush=True,
        )

    per_worker_rows: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    for image_index, (image_path, report_result) in enumerate(report_results, start=1):
        report = build_report(image_path, args.equipment_type, report_result["internal_workers"])
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
            "model": args.local_model,
            "vlm_mode": "local_qwen_vl_attributes_only",
            "labels": str(Path(args.labels)),
            "gt_csv": str(Path(args.gt_csv)),
            "images": len(image_paths),
            "vlm_batch_size": max(1, args.vlm_batch_size),
            "vlm_failed_batches": failed_batches,
            "reports_dir": str(reports_dir),
            "annotated_dir": str(annotated_dir),
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_worker_csv(out_dir / "per_worker.csv", per_worker_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


def load_local_qwen_vl(
    model_id: str,
    *,
    local_files_only: bool,
    max_pixels: int,
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    processor_kwargs: dict[str, Any] = {}
    if max_pixels > 0:
        processor_kwargs["max_pixels"] = max_pixels
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=local_files_only,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only, **processor_kwargs)
    return model, processor


def request_local_batch(
    items: list[dict[str, Any]],
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    max_new_tokens: int,
) -> tuple[dict[str, dict[str, Any]], str]:
    try:
        attrs = call_local_qwen_visual_attributes_batch(
            items,
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
        )
        return attrs, "ok"
    except Exception as exc:
        return {}, f"failed: {type(exc).__name__}: {exc}"


def call_local_qwen_visual_attributes_batch(
    items: list[dict[str, Any]],
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    max_new_tokens: int,
) -> dict[str, dict[str, Any]]:
    if not items:
        return {}

    content: list[dict[str, Any]] = [{"type": "text", "text": local_batch_prompt(items)}]
    for item in items:
        content.extend(
            [
                {"type": "text", "text": f"IMAGE_ID: {item['image_id']}"},
                {"type": "image", "image": str(Path(item["annotated_image"]).resolve())},
            ]
        )
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    content_text = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    parsed = parse_json_object(content_text)
    by_image: dict[str, dict[str, Any]] = {}
    for image_row in parsed.get("images", []):
        image_id = image_row.get("image_id")
        if image_id:
            by_image[str(image_id)] = {"workers": image_row.get("workers", [])}
    return by_image


def local_batch_prompt(items: list[dict[str, Any]]) -> str:
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


def merge_batch_attrs(
    chunk: list[tuple[Path, dict[str, Any]]],
    batch_attrs: dict[str, dict[str, Any]],
) -> None:
    for image_path, report_result in chunk:
        attrs = batch_attrs.get(image_path.name, {"workers": []})
        merge_vlm_attributes(report_result["internal_workers"], attrs)


if __name__ == "__main__":
    raise SystemExit(main())
