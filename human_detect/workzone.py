from __future__ import annotations

import csv
import json
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_WIDTH = 960
IMAGE_HEIGHT = 720


@dataclass(frozen=True)
class WorkzoneRow:
    worker_key: str
    image_id: str
    recording: str
    scene_type: str
    frame_id: str
    worker_index: str
    bbox_xyxy: tuple[float, float, float, float]
    depth_m: float | None
    distance_class_3: str
    depth_source: str

    @property
    def depth_usable(self) -> bool:
        return self.depth_m is not None and self.depth_m > 0.0


def read_workzone_rows(dataset_root: str | Path) -> list[WorkzoneRow]:
    root = Path(dataset_root)
    csv_path = root / "annotations" / "worker_gt_merged.csv"
    rows: list[WorkzoneRow] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            depth_raw = row.get("depth_z_m")
            try:
                depth_m = None if depth_raw in {"", None} else float(depth_raw)
            except ValueError:
                depth_m = None
            rows.append(
                WorkzoneRow(
                    worker_key=row["worker_key"],
                    image_id=row["image_id"],
                    recording=row.get("recording", ""),
                    scene_type=row.get("scene_type", ""),
                    frame_id=row.get("frame_id", ""),
                    worker_index=row.get("worker_index", ""),
                    bbox_xyxy=(
                        float(row["bbox_x1"]),
                        float(row["bbox_y1"]),
                        float(row["bbox_x2"]),
                        float(row["bbox_y2"]),
                    ),
                    depth_m=depth_m,
                    distance_class_3=row.get("distance_class_3", ""),
                    depth_source=row.get("depth_source", ""),
                )
            )
    return rows


def split_image_ids(rows: Iterable[WorkzoneRow], *, val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    by_recording: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.image_id not in by_recording[row.recording]:
            by_recording[row.recording].append(row.image_id)

    train_ids: set[str] = set()
    val_ids: set[str] = set()
    rng = random.Random(seed)
    for recording, image_ids in sorted(by_recording.items()):
        shuffled = image_ids[:]
        rng.shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_fraction))
        val_count = min(val_count, len(shuffled) - 1) if len(shuffled) > 1 else len(shuffled)
        val_ids.update(shuffled[:val_count])
        train_ids.update(shuffled[val_count:])
    return train_ids, val_ids


def prepare_workzone_outputs(
    *,
    dataset_root: str | Path,
    yolo_out: str | Path,
    labels_out: str | Path,
    val_fraction: float = 0.2,
    seed: int = 7,
    link_mode: str = "hardlink",
) -> dict:
    root = Path(dataset_root).resolve()
    rows = read_workzone_rows(root)
    train_ids, val_ids = split_image_ids(rows, val_fraction=val_fraction, seed=seed)
    by_image: dict[str, list[WorkzoneRow]] = defaultdict(list)
    for row in rows:
        by_image[row.image_id].append(row)

    yolo_root = Path(yolo_out)
    labels_root = Path(labels_out)
    for split in ["train", "val"]:
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    split_rows = {"train": [], "val": []}
    image_sets = {"train": train_ids, "val": val_ids}
    for split, image_ids in image_sets.items():
        for image_id in sorted(image_ids):
            image_path = root / "images" / f"{image_id}.png"
            if not image_path.exists():
                continue
            dest_image = yolo_root / "images" / split / image_path.name
            _place_file(image_path, dest_image, link_mode=link_mode)
            label_lines = [yolo_label_line(row.bbox_xyxy) for row in by_image[image_id]]
            (yolo_root / "labels" / split / f"{image_id}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            for row in by_image[image_id]:
                split_rows[split].append(row)

    yaml_path = yolo_root / "workzone_person.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {yolo_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: person",
                "",
            ]
        ),
        encoding="utf-8",
    )

    train_csv = labels_root / "workzone_depth.train.csv"
    eval_csv = labels_root / "workzone_depth.eval.csv"
    write_distance_csv(split_rows["train"], root=root, out_path=train_csv)
    write_distance_csv(split_rows["val"], root=root, out_path=eval_csv)

    summary = {
        "dataset_root": str(root),
        "yolo_out": str(yolo_root),
        "labels_out": str(labels_root),
        "seed": seed,
        "val_fraction": val_fraction,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "total_images": len(by_image),
        "total_workers": len(rows),
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "train_workers_all": len(split_rows["train"]),
        "val_workers_all": len(split_rows["val"]),
        "train_depth_rows": sum(row.depth_usable for row in split_rows["train"]),
        "val_depth_rows": sum(row.depth_usable for row in split_rows["val"]),
        "yaml": str(yaml_path),
        "train_csv": str(train_csv),
        "eval_csv": str(eval_csv),
    }
    (labels_root / "workzone_prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def yolo_label_line(bbox_xyxy: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    cx = ((x1 + x2) / 2.0) / IMAGE_WIDTH
    cy = ((y1 + y2) / 2.0) / IMAGE_HEIGHT
    width = (x2 - x1) / IMAGE_WIDTH
    height = (y2 - y1) / IMAGE_HEIGHT
    return f"0 {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}"


def write_distance_csv(rows: list[WorkzoneRow], *, root: Path, out_path: Path) -> None:
    fieldnames = [
        "image_path",
        "image_id",
        "sequence",
        "frame_id",
        "viewer",
        "human_name",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "depth_m",
        "distance_m",
        "distance_class_3",
        "depth_source",
        "scene_type",
        "recording",
        "worker_key",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            x1, y1, x2, y2 = row.bbox_xyxy
            depth_value = "" if row.depth_m is None else f"{row.depth_m:.6f}"
            writer.writerow(
                {
                    "image_path": str((root / "images" / f"{row.image_id}.png").resolve()),
                    "image_id": row.image_id,
                    "sequence": row.recording,
                    "frame_id": row.frame_id,
                    "viewer": "rgb",
                    "human_name": row.worker_key,
                    "bbox_x1": f"{x1:.6f}",
                    "bbox_y1": f"{y1:.6f}",
                    "bbox_x2": f"{x2:.6f}",
                    "bbox_y2": f"{y2:.6f}",
                    "depth_m": depth_value,
                    "distance_m": depth_value,
                    "distance_class_3": row.distance_class_3,
                    "depth_source": row.depth_source,
                    "scene_type": row.scene_type,
                    "recording": row.recording,
                    "worker_key": row.worker_key,
                }
            )


def _place_file(src: Path, dst: Path, *, link_mode: str) -> None:
    if dst.exists():
        return
    if link_mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
