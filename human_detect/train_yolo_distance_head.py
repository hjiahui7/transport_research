from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .yolo_distance_head import (
    YoloGridDistanceHead,
    distance_head_loss,
    extract_yolo_detect_features,
    load_frozen_yolo_feature_model,
)


@dataclass(frozen=True)
class DistanceHeadSample:
    image_path: Path
    labels: tuple[tuple[float, float, float, float, float], ...]


class RawalkDistanceCsvDataset(Dataset):
    def __init__(self, samples: list[DistanceHeadSample], *, image_size: int) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            original_w, original_h = image.size
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()

        scale_x = self.image_size / float(original_w)
        scale_y = self.image_size / float(original_h)
        labels = []
        for x1, y1, x2, y2, distance in sample.labels:
            sx1, sx2 = x1 * scale_x, x2 * scale_x
            sy1, sy2 = y1 * scale_y, y2 * scale_y
            labels.append(((sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0, sx2 - sx1, sy2 - sy1, distance))
        return tensor, torch.asarray(labels, dtype=torch.float32), sample.image_path.as_posix()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small YOLO-grid distance regression head on Rawalk ego GT.")
    parser.add_argument("--labels", required=True, help="CSV from human_detect.rawalk_ego_depth.")
    parser.add_argument("--val-labels", default=None, help="Optional fixed eval CSV. If omitted, --labels is split by image.")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO detect weights used as frozen feature extractor.")
    parser.add_argument("--out", default=r"runs\yolo_distance_head.pt", help="Output checkpoint path.")
    parser.add_argument("--metrics-out", default=None, help="Optional metrics JSON path.")
    parser.add_argument("--distance-column", default="distance_m", choices=["distance_m", "depth_m", "torso_distance_m", "torso_depth_m"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_samples = load_distance_samples(args.labels, distance_column=args.distance_column)
    if args.val_labels:
        val_samples = load_distance_samples(args.val_labels, distance_column=args.distance_column)
        split_mode = "fixed_csv"
        if not train_samples:
            raise SystemExit(f"Need at least 1 train image in {args.labels}")
        if not val_samples:
            raise SystemExit(f"Need at least 1 eval image in {args.val_labels}")
    else:
        split_mode = "random_by_image"
        if len(train_samples) < 2:
            raise SystemExit(f"Need at least 2 images in {args.labels}, got {len(train_samples)}")
        train_samples, val_samples = split_samples(train_samples, val_fraction=args.val_fraction, seed=args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    yolo_model, channels, strides = load_frozen_yolo_feature_model(args.model, device=str(device))
    head = YoloGridDistanceHead(channels, init_distance_m=mean_distance(train_samples)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        RawalkDistanceCsvDataset(train_samples, image_size=args.imgsz),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_distance_batch,
    )
    val_loader = DataLoader(
        RawalkDistanceCsvDataset(val_samples, image_size=args.imgsz),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_distance_batch,
    )

    best_val = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            yolo_model,
            head,
            train_loader,
            device=device,
            strides=strides,
            image_size=args.imgsz,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            yolo_model,
            head,
            val_loader,
            device=device,
            strides=strides,
            image_size=args.imgsz,
            optimizer=None,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"train_mae={train_metrics['mae_m']:.3f} val_loss={val_metrics['loss']:.4f} val_mae={val_metrics['mae_m']:.3f}"
        )
        if val_metrics["mae_m"] < best_val:
            best_val = val_metrics["mae_m"]
            save_checkpoint(
                args.out,
                head=head,
                model=args.model,
                image_size=args.imgsz,
                channels=channels,
                strides=strides,
                distance_column=args.distance_column,
                metrics={"best_epoch": epoch, "best_val_mae_m": best_val},
            )

    metrics = {
        "labels": str(Path(args.labels)),
        "val_labels": None if args.val_labels is None else str(Path(args.val_labels)),
        "split_mode": split_mode,
        "model": args.model,
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "train_persons": sum(len(sample.labels) for sample in train_samples),
        "val_persons": sum(len(sample.labels) for sample in val_samples),
        "best_val_mae_m": best_val,
        "history": history,
    }
    metrics_path = Path(args.metrics_out) if args.metrics_out else Path(args.out).with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved head:    {args.out}")
    print(f"saved metrics: {metrics_path}")
    return 0


def load_distance_samples(csv_path: str | Path, *, distance_column: str) -> list[DistanceHeadSample]:
    grouped: dict[str, list[tuple[float, float, float, float, float]]] = defaultdict(list)
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            distance_raw = row.get(distance_column)
            if distance_raw in {"", None}:
                continue
            distance = float(distance_raw)
            if not np.isfinite(distance) or distance <= 0.0:
                continue
            grouped[row["image_path"]].append(
                (
                    float(row["bbox_x1"]),
                    float(row["bbox_y1"]),
                    float(row["bbox_x2"]),
                    float(row["bbox_y2"]),
                    distance,
                )
            )
    samples = [DistanceHeadSample(Path(path), tuple(labels)) for path, labels in grouped.items() if Path(path).exists()]
    samples.sort(key=lambda sample: sample.image_path.as_posix())
    return samples


def split_samples(samples: list[DistanceHeadSample], *, val_fraction: float, seed: int) -> tuple[list[DistanceHeadSample], list[DistanceHeadSample]]:
    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)
    val_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * max(0.0, min(0.9, val_fraction)))))
    return shuffled[val_count:], shuffled[:val_count]


def mean_distance(samples: list[DistanceHeadSample]) -> float:
    values = [label[-1] for sample in samples for label in sample.labels]
    return float(np.mean(values)) if values else 3.0


def collate_distance_batch(batch):
    images, labels, paths = zip(*batch)
    return torch.stack(images, dim=0), list(labels), list(paths)


def run_epoch(
    yolo_model,
    head: YoloGridDistanceHead,
    loader: DataLoader,
    *,
    device: torch.device,
    strides: list[float],
    image_size: int,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    head.train(training)
    total_loss = total_mae = total_count = 0.0
    for images, labels, _paths in loader:
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            features = extract_yolo_detect_features(yolo_model, images)
        predictions = head(features)
        loss, count = distance_head_loss(predictions, labels, strides=strides, image_size=image_size)
        if count == 0:
            continue
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        mae = batch_target_cell_mae(predictions, labels, strides=strides)
        total_loss += float(loss.detach().cpu()) * count
        total_mae += mae * count
        total_count += count
    if total_count == 0:
        return {"loss": 0.0, "mae_m": 0.0, "samples": 0}
    return {"loss": total_loss / total_count, "mae_m": total_mae / total_count, "samples": int(total_count)}


@torch.no_grad()
def batch_target_cell_mae(predictions, labels, *, strides: list[float]) -> float:
    from .yolo_distance_head import select_distance_level

    errors: list[float] = []
    for batch_index, label_tensor in enumerate(labels):
        label_tensor = label_tensor.to(predictions[0].device)
        for target in label_tensor:
            cx, cy, width, height, distance = target[:5]
            level = select_distance_level(float(width), float(height), strides)
            stride = float(strides[level])
            grid_y = int(torch.clamp(torch.floor(cy / stride), 0, predictions[level].shape[1] - 1).item())
            grid_x = int(torch.clamp(torch.floor(cx / stride), 0, predictions[level].shape[2] - 1).item())
            pred = float(torch.exp(predictions[level][batch_index, grid_y, grid_x]).detach().cpu())
            errors.append(abs(pred - float(distance.detach().cpu())))
    return float(np.mean(errors)) if errors else 0.0


def save_checkpoint(
    path: str | Path,
    *,
    head: YoloGridDistanceHead,
    model: str,
    image_size: int,
    channels: list[int],
    strides: list[float],
    distance_column: str,
    metrics: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "model": model,
            "image_size": image_size,
            "channels": channels,
            "strides": strides,
            "distance_column": distance_column,
            "metrics": metrics,
        },
        path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
