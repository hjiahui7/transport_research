from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .workzone import yolo_label_line


@dataclass(frozen=True)
class V2Worker:
    worker_key: str
    image_id: str
    image_path: Path
    recording: str
    scene_type: str
    frame_id: str
    worker_index: str
    bbox_xyxy: tuple[float, float, float, float]
    depth_m: float | None
    distance_class_3: str
    depth_source: str
    data_source: str

    @property
    def depth_usable(self) -> bool:
        return self.depth_m is not None and self.depth_m > 0.0


CSV_COLUMNS = [
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
    "data_source",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the combined 500 + Wave 2 work-zone v2 dataset.")
    parser.add_argument("--dataset-root", default="work-zone-safety-rgbd-dataset")
    parser.add_argument("--validation-images", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--yolo-out", default=r"data\workzone_v2_yolo_person")
    parser.add_argument("--artifact-out", default=r"artifacts\workzone_v2")
    parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = prepare_workzone_v2(
        dataset_root=Path(args.dataset_root),
        validation_images=args.validation_images,
        seed=args.seed,
        yolo_out=Path(args.yolo_out),
        artifact_out=Path(args.artifact_out),
        link_mode=args.link_mode,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def prepare_workzone_v2(
    *,
    dataset_root: Path,
    validation_images: int,
    seed: int,
    yolo_out: Path,
    artifact_out: Path,
    link_mode: str,
) -> dict[str, object]:
    root = dataset_root.resolve()
    old_rows = read_wave1_rows(root)
    wave2_rows = read_wave2_rows(root / "workzone_rgbd_dataset_wave2")
    old_image_ids = {row.image_id for row in old_rows}
    wave2_image_ids = {row.image_id for row in wave2_rows}
    overlap = old_image_ids & wave2_image_ids
    if overlap:
        raise ValueError(f"Wave 1 and Wave 2 image_id values overlap: {sorted(overlap)[:3]}")
    if not 0 < validation_images < len(old_image_ids):
        raise ValueError(f"validation_images must be between 1 and {len(old_image_ids) - 1}")

    validation_ids = stratified_validation_ids(old_rows, count=validation_images, seed=seed)
    all_rows = old_rows + wave2_rows
    train_rows = [row for row in all_rows if row.image_id not in validation_ids]
    validation_rows = [row for row in old_rows if row.image_id in validation_ids]

    artifact_out.mkdir(parents=True, exist_ok=True)
    data_splits = artifact_out / "data_splits"
    models_dir = artifact_out / "models"
    results_dir = artifact_out / "results"
    for directory in [data_splits, models_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    write_distance_csv(data_splits / "workzone_depth.train.csv", train_rows)
    write_distance_csv(data_splits / "workzone_depth.eval.csv", validation_rows)
    write_distance_csv(data_splits / "workzone_depth.all1685.csv", all_rows)
    write_yolo_dataset(yolo_out, train_rows, validation_rows, link_mode=link_mode)

    v1_prompt = artifact_out.parent / "workzone_v1" / "qwen_prompt.md"
    v2_prompt = artifact_out / "qwen_prompt.md"
    if v1_prompt.exists() and not v2_prompt.exists():
        shutil.copy2(v1_prompt, v2_prompt)

    summary: dict[str, object] = {
        "dataset_root": str(root),
        "seed": seed,
        "total_images": len(old_image_ids | wave2_image_ids),
        "total_workers": len(all_rows),
        "wave1_images": len(old_image_ids),
        "wave2_images": len(wave2_image_ids),
        "train_images": len({row.image_id for row in train_rows}),
        "validation_images": len(validation_ids),
        "train_workers_all": len(train_rows),
        "validation_workers_all": len(validation_rows),
        "train_depth_rows": sum(row.depth_usable for row in train_rows),
        "validation_depth_rows": sum(row.depth_usable for row in validation_rows),
        "validation_recording_images": validation_recording_counts(old_rows, validation_ids),
        "yolo_yaml": str((yolo_out / "workzone_person.yaml").resolve()),
        "train_csv": str((data_splits / "workzone_depth.train.csv").resolve()),
        "eval_csv": str((data_splits / "workzone_depth.eval.csv").resolve()),
        "all_csv": str((data_splits / "workzone_depth.all1685.csv").resolve()),
    }
    (results_dir / "workzone_prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def read_wave1_rows(root: Path) -> list[V2Worker]:
    rows: list[V2Worker] = []
    with (root / "annotations" / "worker_gt_merged.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"]
            rows.append(
                worker_from_csv(
                    row,
                    image_path=root / "images" / f"{image_id}.png",
                    recording=row.get("recording", ""),
                    scene_type=row.get("scene_type", ""),
                    frame_id=row.get("frame_id", ""),
                    depth_source=row.get("depth_source", ""),
                    data_source="workzone_wave1",
                )
            )
    return rows


def read_wave2_rows(root: Path) -> list[V2Worker]:
    rows: list[V2Worker] = []
    with (root / "annotations.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"]
            recording, frame_id = image_id.rsplit("_", 1)
            scene_type = "garage" if "garage" in recording else "pavement"
            rows.append(
                worker_from_csv(
                    row,
                    image_path=root / row["image_path"],
                    recording=recording,
                    scene_type=scene_type,
                    frame_id=frame_id,
                    depth_source="lidar_wave2",
                    data_source="workzone_wave2",
                )
            )
    return rows


def worker_from_csv(
    row: dict[str, str],
    *,
    image_path: Path,
    recording: str,
    scene_type: str,
    frame_id: str,
    depth_source: str,
    data_source: str,
) -> V2Worker:
    depth_raw = row.get("depth_z_m", "")
    try:
        depth_m = float(depth_raw) if depth_raw else None
    except ValueError:
        depth_m = None
    return V2Worker(
        worker_key=row["worker_key"],
        image_id=row["image_id"],
        image_path=image_path.resolve(),
        recording=recording,
        scene_type=scene_type,
        frame_id=frame_id,
        worker_index=row.get("worker_index", ""),
        bbox_xyxy=(float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"])),
        depth_m=depth_m,
        distance_class_3=normalized_distance_class(depth_m, row.get("distance_class_3", "")),
        depth_source=depth_source,
        data_source=data_source,
    )


def normalized_distance_class(depth_m: float | None, label: str) -> str:
    if depth_m is not None and depth_m > 0:
        if depth_m < 3:
            return "<3"
        if depth_m <= 6:
            return "3-6"
        return ">6"
    return {"3-5": "3-6", ">5": ">6"}.get(label, label)


def stratified_validation_ids(rows: list[V2Worker], *, count: int, seed: int) -> set[str]:
    by_recording: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.image_id not in by_recording[row.recording]:
            by_recording[row.recording].append(row.image_id)

    total = sum(len(ids) for ids in by_recording.values())
    quotas: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    for recording, image_ids in sorted(by_recording.items()):
        exact = len(image_ids) * count / total
        quotas[recording] = int(exact)
        fractions.append((exact - int(exact), recording))
    for _, recording in sorted(fractions, key=lambda item: (-item[0], item[1]))[: count - sum(quotas.values())]:
        quotas[recording] += 1

    rng = random.Random(seed)
    selected: set[str] = set()
    for recording, image_ids in sorted(by_recording.items()):
        shuffled = sorted(image_ids)
        rng.shuffle(shuffled)
        selected.update(shuffled[: quotas[recording]])
    if len(selected) != count:
        raise RuntimeError(f"Expected {count} validation images, selected {len(selected)}")
    return selected


def validation_recording_counts(rows: list[V2Worker], validation_ids: set[str]) -> dict[str, int]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.image_id in validation_ids:
            result[row.recording].add(row.image_id)
    return {recording: len(image_ids) for recording, image_ids in sorted(result.items())}


def write_distance_csv(path: Path, rows: list[V2Worker]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            x1, y1, x2, y2 = row.bbox_xyxy
            depth = "" if row.depth_m is None else f"{row.depth_m:.6f}"
            writer.writerow(
                {
                    "image_path": str(row.image_path),
                    "image_id": row.image_id,
                    "sequence": row.recording,
                    "frame_id": row.frame_id,
                    "viewer": "rgb",
                    "human_name": row.worker_key,
                    "bbox_x1": f"{x1:.6f}",
                    "bbox_y1": f"{y1:.6f}",
                    "bbox_x2": f"{x2:.6f}",
                    "bbox_y2": f"{y2:.6f}",
                    "depth_m": depth,
                    "distance_m": depth,
                    "distance_class_3": row.distance_class_3,
                    "depth_source": row.depth_source,
                    "scene_type": row.scene_type,
                    "recording": row.recording,
                    "worker_key": row.worker_key,
                    "data_source": row.data_source,
                }
            )


def write_yolo_dataset(yolo_root: Path, train_rows: list[V2Worker], validation_rows: list[V2Worker], *, link_mode: str) -> None:
    for split, rows in [("train", train_rows), ("val", validation_rows)]:
        image_dir = yolo_root / "images" / split
        label_dir = yolo_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        by_image: dict[str, list[V2Worker]] = defaultdict(list)
        for row in rows:
            by_image[row.image_id].append(row)
        for image_id, workers in sorted(by_image.items()):
            source = workers[0].image_path
            if not source.exists():
                raise FileNotFoundError(source)
            place_file(source, image_dir / f"{image_id}.png", link_mode=link_mode)
            labels = "\n".join(yolo_label_line(worker.bbox_xyxy) for worker in workers) + "\n"
            (label_dir / f"{image_id}.txt").write_text(labels, encoding="utf-8")

    (yolo_root / "workzone_person.yaml").write_text(
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


def place_file(source: Path, destination: Path, *, link_mode: str) -> None:
    if destination.exists():
        return
    if link_mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


if __name__ == "__main__":
    raise SystemExit(main())
