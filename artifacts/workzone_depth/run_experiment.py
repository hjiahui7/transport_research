from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from human_detect.calibration import FEATURE_COLUMNS, load_calibrator, person_feature_vector, row_feature_vector
from human_detect.distance_eval import distance_band, summarize_distance_errors
from human_detect.fit_calibrator import main as fit_calibrator
from human_detect.infer_distance_head import _load_square_rgb_tensor, _scale_boxes_to_square
from human_detect.matching import greedy_bbox_matches
from human_detect.pipeline import DistanceEstimator
from human_detect.train_yolo_distance_head import (
    RawalkDistanceCsvDataset,
    batch_target_cell_mae,
    collate_distance_batch,
    load_distance_samples,
    mean_distance,
    save_checkpoint,
    split_samples,
)
from human_detect.yolo_distance_head import (
    YoloGridDistanceHead,
    distance_head_loss,
    extract_yolo_detect_features,
    load_frozen_yolo_feature_model,
    predict_distance_for_boxes,
)


IMGSZ = 960
CONFIDENCE = 0.25
NMS_IOU = 0.70
MATCH_IOU = 0.50
GEOM_SIZE = 640
TOKENS = 1200
BATCH_SIZE = 8
HEAD_SEED = 7
MLP_SEED = 13
INNER_VAL_FRACTION = 0.20
WEIGHT = REPO / "artifacts/rawalk_v1/models/rawalk_scheme1_yolo11n_base.pt"
SPLITS = REPO / "runs" / "workzone_depth" / "prepared" / "data_splits"
METHODS = ("yolo_distance_head", "moge_raw", "moge_calibrated")
ZONES = ("<3", "3-6", ">6")


@dataclass(frozen=True)
class Worker:
    key: str
    bbox_xyxy: tuple[float, float, float, float]
    distance_m: float | None
    z_m: float | None
    zone: str


@dataclass(frozen=True)
class Frame:
    image_id: str
    image_path: Path
    workers: list[Worker]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields if fields is not None else list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staging.replace(path)


def load_frames(rows: list[dict[str, str]]) -> list[Frame]:
    frames = {}
    worker_keys = set()
    for row in rows:
        image_id = row["image_id"]
        image_path = Path(row["image_path"])
        frame = frames.setdefault(image_id, Frame(image_id, image_path, []))
        distance = float(row["distance_m"]) if row["distance_m"] else None
        z = float(row["depth_m"]) if row["depth_m"] else None
        zone = row["distance_class_3"]
        key = row["worker_key"]
        if key in worker_keys or frame.image_path != image_path or zone not in ZONES:
            raise ValueError(f"Invalid or duplicate worker row: {key}")
        if distance is not None and (
            not math.isfinite(distance) or distance <= 0 or z is None or not math.isfinite(z)
            or z <= 0 or distance_band(distance) != zone
        ):
            raise ValueError(f"Invalid LiDAR distance GT: {key}")
        bbox = tuple(float(row[f"bbox_{axis}"]) for axis in ("x1", "y1", "x2", "y2"))
        frame.workers.append(Worker(key, bbox, distance, z, zone))
        worker_keys.add(key)
    ordered = sorted(frames.values(), key=lambda frame: frame.image_id)
    for frame in ordered:
        if not frame.image_path.is_file():
            raise FileNotFoundError(frame.image_path)
    return ordered


def smoke_subset(rows: list[dict[str, str]], n_images: int, seed: int) -> list[dict[str, str]]:
    numeric_ids = sorted({r["image_id"] for r in rows if r["distance_m"]})
    coarse_ids = sorted({r["image_id"] for r in rows if not r["distance_m"]})
    rng = random.Random(seed)
    selected = set(rng.sample(numeric_ids, n_images - 2))
    selected.update(rng.sample([image for image in coarse_ids if image not in selected], 2))
    return [row for row in rows if row["image_id"] in selected]


def collect_moge(estimator: DistanceEstimator, frames: list[Frame], out: Path) -> dict[str, dict]:
    if out.exists():
        with out.open(encoding="utf-8") as stream:
            cached = {record["image_id"]: record for record in map(json.loads, stream)}
        if set(cached) == {frame.image_id for frame in frames}:
            print(f"Reused {len(cached)} images from {out.name}", flush=True)
            return cached
        raise ValueError(f"Cached images do not match the current split: {out}")
    records = {}
    partial = out.with_suffix(".jsonl.tmp")
    with partial.open("w", encoding="utf-8") as stream:
        for index, frame in enumerate(frames, 1):
            result, _ = estimator.infer(frame.image_path)
            width, height = result["image_size"]["width"], result["image_size"]["height"]
            persons = []
            for person in result["persons"]:
                raw_distance = person["distance_m"]
                usable = raw_distance is not None and math.isfinite(raw_distance) and raw_distance > 0
                features = person_feature_vector(person, result["camera"], width, height) if usable else None
                persons.append({
                    "bbox_xyxy": person["bbox_xyxy"], "score": person["score"],
                    "moge_raw": raw_distance if usable else None, "features": features,
                })
            matches = greedy_bbox_matches(persons, frame.workers, iou_threshold=MATCH_IOU)
            record = {
                "image_id": frame.image_id, "image_path": str(frame.image_path),
                "width": width, "height": height, "persons": persons,
                "matches": [{"pred_index": m.pred_index, "gt_index": m.gt_index, "iou": m.iou} for m in matches],
            }
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            records[frame.image_id] = record
            if index % 100 == 0 or index == len(frames):
                print(f"{out.stem}: {index}/{len(frames)} images", flush=True)
    partial.replace(out)
    return records


def fit_moge(frames: list[Frame], records: dict[str, dict], out: Path):
    matched = []
    for frame in frames:
        record = records[frame.image_id]
        for match in record["matches"]:
            worker = frame.workers[match["gt_index"]]
            person = record["persons"][match["pred_index"]]
            if worker.distance_m is None or person["features"] is None:
                continue
            matched.append({
                "image_id": frame.image_id, "worker_key": worker.key, "iou": match["iou"],
                **dict(zip(FEATURE_COLUMNS, person["features"], strict=True)),
                "z_gt": worker.z_m, "distance_gt": worker.distance_m,
            })
    print(f"MoGe calibration: {len(matched)} matched numeric training workers", flush=True)
    samples = out / "moge_calibration_train.csv"
    model_path = out / "models" / "moge_calibrator.joblib"
    write_csv(samples, matched)
    fit_calibrator([
        "--preds", str(samples), "--out", str(model_path),
        "--metrics-out", str(out / "moge_calibration_fit.json"), "--model", "mlp",
        "--seed", str(MLP_SEED), "--val-frac", str(INNER_VAL_FRACTION), "--group-column", "image_id",
    ])
    return load_calibrator(model_path), len(matched)


def cache_frozen_features(samples, model, channels, strides, device):
    shapes = [(channel, IMGSZ // int(stride), IMGSZ // int(stride)) for channel, stride in zip(channels, strides)]
    bytes_needed = len(samples) * 4 * sum(math.prod(shape) for shape in shapes)
    cache_device = device if torch.cuda.mem_get_info(device)[0] > bytes_needed + 4 * 1024**3 else torch.device("cpu")
    buffers = [torch.empty((len(samples), *shape), dtype=torch.float32, device=cache_device) for shape in shapes]
    labels = []
    source = RawalkDistanceCsvDataset(samples, image_size=IMGSZ)
    print(f"Caching frozen FP32 feature maps for {len(samples)} images on {cache_device} ({bytes_needed / 1024**3:.2f} GiB)", flush=True)
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as loader, torch.no_grad():
        for start in range(0, len(samples), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(samples))
            images, batch_labels, _ = collate_distance_batch(loader.map(source.__getitem__, range(start, stop)))
            features = extract_yolo_detect_features(model, images.to(device))
            for buffer, feature in zip(buffers, features, strict=True):
                buffer[start:stop].copy_(feature)
            labels.extend(label.to(device) for label in batch_labels)
            if stop % 200 == 0 or stop == len(samples):
                print(f"YOLO frozen features: {stop}/{len(samples)}", flush=True)
    return buffers, labels


def head_epoch(head, buffers, labels, order, strides, device, optimizer=None):
    training = optimizer is not None
    head.train(training)
    loss_sum = mae_sum = count_sum = 0.0
    with torch.set_grad_enabled(training):
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            batch_index = torch.as_tensor(indices, device=buffers[0].device)
            features = [buffer.index_select(0, batch_index).to(device) for buffer in buffers]
            batch_labels = [labels[index] for index in indices]
            predictions = head(features)
            loss, count = distance_head_loss(predictions, batch_labels, strides=strides, image_size=IMGSZ)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.detach()) * count
            mae_sum += batch_target_cell_mae(predictions, batch_labels, strides=strides) * count
            count_sum += count
    return {"loss": loss_sum / count_sum, "mae_m": mae_sum / count_sum, "samples": int(count_sum)}


def fit_head(train_csv: Path, out: Path, epochs: int, device):
    samples = load_distance_samples(train_csv, distance_column="distance_m")
    inner_train, inner_val = split_samples(samples, val_fraction=INNER_VAL_FRACTION, seed=HEAD_SEED)
    indexes = {sample.image_path: i for i, sample in enumerate(samples)}
    fit_indices = [indexes[s.image_path] for s in inner_train]
    val_indices = [indexes[s.image_path] for s in inner_val]
    val_index_set = set(val_indices)
    write_csv(out / "yolo_head_inner_split.csv", [
        {"image_path": str(sample.image_path), "split": "inner_validation" if i in val_index_set else "inner_train"}
        for i, sample in enumerate(samples)
    ])
    model, channels, strides = load_frozen_yolo_feature_model(WEIGHT, device=str(device))
    buffers, labels = cache_frozen_features(samples, model, channels, strides, device)
    history = []
    head_path = out / "models" / "yolo11n_distance_head.pt"

    def init_head(rows):
        random.seed(HEAD_SEED)
        np.random.seed(HEAD_SEED)
        torch.manual_seed(HEAD_SEED)
        head = YoloGridDistanceHead(channels, init_distance_m=mean_distance(rows)).to(device)
        return head, torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

    def record(phase, epoch, fit, val=None):
        row = {"phase": phase, "epoch": epoch, "fit": fit, "inner_validation": val}
        history.append(row)
        write_json(out / "yolo_head_training_history.json", history)
        val_text = "" if val is None else f", inner val MAE {val['mae_m']:.4f} m"
        print(f"YOLO head {phase} epoch {epoch}: train MAE {fit['mae_m']:.4f} m{val_text}", flush=True)

    head, optimizer = init_head(inner_train)
    best = (float("inf"), 0)
    for epoch in range(1, epochs + 1):
        shuffled = torch.randperm(len(fit_indices)).tolist()
        fit = head_epoch(head, buffers, labels, [fit_indices[i] for i in shuffled], strides, device, optimizer)
        val = head_epoch(head, buffers, labels, val_indices, strides, device)
        record("select", epoch, fit, val)
        best = min(best, (val["mae_m"], epoch))

    head, optimizer = init_head(samples)
    for epoch in range(1, best[1] + 1):
        fit = head_epoch(head, buffers, labels, torch.randperm(len(samples)).tolist(), strides, device, optimizer)
        record("refit", epoch, fit)
    fit_info = {
        "selected_epoch": best[1], "inner_validation_mae_m": best[0],
        "inner_train_images": len(inner_train), "inner_validation_images": len(inner_val),
        "refit_images": len(samples), "refit_workers": sum(len(s.labels) for s in samples),
    }
    save_checkpoint(
        head_path, head=head, model=str(WEIGHT), image_size=IMGSZ, channels=channels,
        strides=strides, distance_column="distance_m", metrics=fit_info,
    )
    write_json(out / "yolo_head_fit.json", fit_info)
    return model, head.eval(), strides, fit_info


def evaluate(frames: list[Frame], records: dict[str, dict], model, head, strides, calibrator, out: Path, device):
    workers = []
    detections = []
    for frame in frames:
        record = records[frame.image_id]
        persons = record["persons"]
        matches_by_gt = {m["gt_index"]: m for m in record["matches"]}
        matches_by_pred = {m["pred_index"]: m for m in record["matches"]}
        distances = []
        if persons:
            image = _load_square_rgb_tensor(frame.image_path, IMGSZ).to(device)
            with torch.no_grad():
                features = extract_yolo_detect_features(model, image)
                outputs = head(features)
            scaled = _scale_boxes_to_square(
                [p["bbox_xyxy"] for p in persons], image_width=record["width"],
                image_height=record["height"], image_size=IMGSZ,
            )
            distances = predict_distance_for_boxes(outputs, scaled, strides=strides)
        per_pred = []
        for pred_index, (person, head_m) in enumerate(zip(persons, distances, strict=True)):
            calibrated = None
            if person["features"] is not None:
                feature_row = dict(zip(FEATURE_COLUMNS, person["features"], strict=True))
                calibrated = max(0.0, float(calibrator.distance_model.predict([row_feature_vector(feature_row)])[0]))
            prediction = dict(zip(METHODS, (head_m, person["moge_raw"], calibrated), strict=True))
            if any(v is not None and not math.isfinite(v) for v in prediction.values()):
                raise ValueError(f"Non-finite prediction for {frame.image_id} detection {pred_index}")
            per_pred.append(prediction)
            match = matches_by_pred.get(pred_index)
            detections.append({
                "image_id": frame.image_id, "image_path": str(frame.image_path),
                "pred_index": pred_index, "score": person["score"],
                **dict(zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), person["bbox_xyxy"], strict=True)),
                "worker_key": frame.workers[match["gt_index"]].key if match else None,
                "match_iou": match["iou"] if match else None, **prediction,
            })
        for gt_index, worker in enumerate(frame.workers):
            match = matches_by_gt.get(gt_index)
            values = per_pred[match["pred_index"]] if match else dict.fromkeys(METHODS)
            row = {
                "image_id": frame.image_id, "image_path": str(frame.image_path), "worker_key": worker.key,
                "gt_distance_m": worker.distance_m, "gt_zone": worker.zone,
                "pred_index": match["pred_index"] if match else None, "match_iou": match["iou"] if match else None,
            }
            for method, predicted in values.items():
                row[method] = predicted
                row[f"{method}_zone"] = distance_band(predicted) if predicted is not None else None
                row[f"{method}_error_m"] = predicted - worker.distance_m if predicted is not None and worker.distance_m is not None else None
            workers.append(row)
    write_csv(out / "validation_workers.csv", workers)
    write_csv(out / "validation_detections.csv", detections)
    return workers, detections


def summarize(workers, detections, out: Path):
    numeric = [row for row in workers if row["gt_distance_m"] is not None]
    shared = [row for row in numeric if all(row[method] is not None for method in METHODS)]
    tp = sum(row["pred_index"] is not None for row in workers)
    detector = {
        "true_positive": tp, "false_positive": len(detections) - tp, "false_negative": len(workers) - tp,
        "precision": tp / len(detections), "recall": tp / len(workers),
    }
    comparison = []
    confusion_rows = []
    zone_predictions = (*ZONES, "no_detection", "no_distance")
    for method in METHODS:
        valid = [row for row in numeric if row[method] is not None]
        scores = summarize_distance_errors([r["gt_distance_m"] for r in valid], [r[method] for r in valid], total_gt=len(numeric))
        common_scores = summarize_distance_errors([r["gt_distance_m"] for r in shared], [r[method] for r in shared])
        zone_key = f"{method}_zone"
        zone_available = [row for row in workers if row[zone_key] is not None]
        correct = sum(r[zone_key] == r["gt_zone"] for r in zone_available)
        matrix = {gt: dict.fromkeys(zone_predictions, 0) for gt in ZONES}
        for row in workers:
            predicted_zone = row[zone_key] or ("no_detection" if row["pred_index"] is None else "no_distance")
            matrix[row["gt_zone"]][predicted_zone] += 1
        assert sum(sum(row.values()) for row in matrix.values()) == len(workers)
        confusion_rows.extend({"method": method, "gt_zone": gt, **matrix[gt]} for gt in ZONES)
        comparison.append({
            "method": method, "numeric_gt": len(numeric), "numeric_predictions": len(valid),
            "numeric_coverage": scores["match_rate"], "mae_m": scores["mae_m"], "rmse_m": scores["rmse_m"],
            "bias_m": scores["bias_m"], "within_0_5m": scores["within_0_5m"], "within_1_0m": scores["within_1_0m"],
            "common_numeric_count": len(shared), "common_numeric_mae_m": common_scores["mae_m"],
            "zone_gt": len(workers), "zone_predictions": len(zone_available), "zone_correct": correct,
            "zone_accuracy_when_predicted": correct / len(zone_available),
            "zone_accuracy_all_gt": correct / len(workers),
        })
    write_csv(out / "comparison.csv", comparison)
    write_csv(out / "zone_confusion_matrices.csv", confusion_rows)
    return detector, comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the official YOLO11n person-only experiment on Work-Zone LiDAR distance GT.")
    parser.add_argument("--splits", type=Path, default=SPLITS, help="Directory containing the generated train and eval CSVs")
    parser.add_argument("--moge", type=Path, required=True, help="Local MoGe-2 ViT-S model.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    if args.epochs < 1 or not args.moge.is_file() or not torch.cuda.is_available():
        raise RuntimeError("Positive epoch count, local MoGe weights and CUDA are required")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    train_csv = args.splits / "workzone_depth.train.csv"
    eval_csv = args.splits / "workzone_depth.eval.csv"
    train_rows, eval_rows = read_csv(train_csv), read_csv(eval_csv)
    if args.smoke:
        train_rows = smoke_subset(train_rows, 48, HEAD_SEED)
        eval_rows = smoke_subset(eval_rows, 12, HEAD_SEED + 1)
        train_csv = args.out / "smoke_train.csv"
        eval_csv = args.out / "smoke_validation.csv"
        write_csv(train_csv, train_rows)
        write_csv(eval_csv, eval_rows)
    train_frames, eval_frames = load_frames(train_rows), load_frames(eval_rows)
    assert not ({f.image_id for f in train_frames} & {f.image_id for f in eval_frames})
    counts = {
        "train_images": len(train_frames), "train_workers": len(train_rows),
        "train_numeric_workers": sum(bool(r["distance_m"]) for r in train_rows),
        "validation_images": len(eval_frames), "validation_workers": len(eval_rows),
        "validation_numeric_workers": sum(bool(r["distance_m"]) for r in eval_rows),
    }
    if not args.smoke:
        assert tuple(counts.values()) == (1435, 2863, 2674, 250, 415, 339)
    config = {
        "smoke": args.smoke, "cuda_device": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "train_csv": str(train_csv), "validation_csv": str(eval_csv), "counts": counts,
        "distance_target": "distance_m", "detector": str(WEIGHT), "detector_class": 0,
        "detector_imgsz": IMGSZ, "detector_confidence": CONFIDENCE, "detector_nms_iou": NMS_IOU,
        "detector_half": False, "match_iou": MATCH_IOU, "moge_weights": str(args.moge),
        "moge_geometry_size": GEOM_SIZE, "moge_tokens": TOKENS, "moge_half": True,
        "max_head_epochs": 1 if args.smoke else args.epochs, "head_batch_size": BATCH_SIZE,
        "inner_validation_fraction": INNER_VAL_FRACTION, "head_seed": HEAD_SEED, "mlp_seed": MLP_SEED,
    }
    write_json(args.out / "configuration.json", config)
    print(f"CUDA: {config['cuda_device']}. Dataset: {counts}", flush=True)
    estimator = DistanceEstimator(
        detector=str(WEIGHT), geometry_model=str(args.moge), imgsz=IMGSZ, conf=CONFIDENCE,
        geom_size=GEOM_SIZE, num_tokens=TOKENS, device=str(device), half=True, detector_half=False,
    )
    if estimator.segmenter.model.names[0] != "person" or len(estimator.segmenter.model.names) != 80:
        raise ValueError("Expected the official COCO YOLO11n detector")
    estimator.segmenter.model.overrides["iou"] = NMS_IOU
    train_records = collect_moge(estimator, train_frames, args.out / "train_shared_detections_moge.jsonl")
    eval_records = collect_moge(estimator, eval_frames, args.out / "validation_shared_detections_moge.jsonl")
    del estimator
    gc.collect()
    torch.cuda.empty_cache()
    calibrator, calibration_workers = fit_moge(train_frames, train_records, args.out)
    model, head, strides, head_fit = fit_head(train_csv, args.out, config["max_head_epochs"], device)
    assert head_fit["refit_workers"] == counts["train_numeric_workers"]
    workers, detections = evaluate(eval_frames, eval_records, model, head, strides, calibrator, args.out, device)
    detector, comparison = summarize(workers, detections, args.out)
    summary = {
        "configuration": config, "detector_validation": detector, "head_fit": head_fit,
        "moge_mlp_matched_training_workers": calibration_workers, "comparison": comparison,
    }
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
