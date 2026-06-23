from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from PIL import Image


RAWALK_TAGGING_SUFFIX = Path("disk1/rawalk/datasets/ego_exo/camera_ready/01_tagging")
SplitName = Literal["train", "val"]
StreamMode = Literal["exo", "ego", "all"]
LinkMode = Literal["hardlink", "copy"]


@dataclass(frozen=True)
class RawalkObject:
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    human_name: str | None = None
    human_id: int | None = None


@dataclass(frozen=True)
class RawalkFrame:
    sequence: str
    camera: str
    channel: str
    frame_id: str
    image_path: Path
    bbox_path: Path
    objects: tuple[RawalkObject, ...]

    @property
    def stream_key(self) -> str:
        return f"{self.camera}_{self.channel}"


@dataclass(frozen=True)
class PrepareStats:
    tagging_root: Path
    out_root: Path
    yaml_path: Path
    manifest_path: Path
    train_images: int
    val_images: int
    boxes: int
    skipped_missing_images: int
    skipped_empty_labels: int


def resolve_tagging_root(rawalk_root: str | Path) -> Path:
    """Accept either the outer Rawalk folder or the final 01_tagging directory."""
    root = Path(rawalk_root)
    candidates = [
        root,
        root / RAWALK_TAGGING_SUFFIX,
        root / "rawalk/datasets/ego_exo/camera_ready/01_tagging",
        root / "datasets/ego_exo/camera_ready/01_tagging",
        root / "ego_exo/camera_ready/01_tagging",
    ]
    for candidate in candidates:
        if candidate.exists() and any((p / "processed_data").exists() for p in candidate.iterdir() if p.is_dir()):
            return candidate
    raise FileNotFoundError(f"Could not find Rawalk 01_tagging root under: {root}")


def iter_rawalk_frames(
    rawalk_root: str | Path,
    *,
    streams: StreamMode = "exo",
    frame_step: int = 10,
    min_score: float = 0.2,
) -> Iterable[RawalkFrame]:
    tagging_root = resolve_tagging_root(rawalk_root)
    frame_step = max(1, frame_step)

    for sequence_dir in sorted(p for p in tagging_root.iterdir() if p.is_dir()):
        bbox_root = sequence_dir / "processed_data" / "bboxes"
        if not bbox_root.exists():
            continue
        for bbox_path in sorted(bbox_root.glob("*/*/*.npy")):
            camera = bbox_path.parent.parent.name
            channel = bbox_path.parent.name
            if streams == "exo" and not camera.startswith("cam"):
                continue
            if streams == "ego" and not camera.startswith("aria"):
                continue

            try:
                frame_number = int(bbox_path.stem)
            except ValueError:
                frame_number = 0
            if (frame_number - 1) % frame_step != 0:
                continue

            image_path = _image_path_for_bbox(sequence_dir, camera, channel, bbox_path.stem)
            objects = tuple(obj for obj in load_bbox_objects(bbox_path) if obj.score >= min_score)
            yield RawalkFrame(
                sequence=sequence_dir.name,
                camera=camera,
                channel=channel,
                frame_id=bbox_path.stem,
                image_path=image_path,
                bbox_path=bbox_path,
                objects=objects,
            )


def load_bbox_objects(bbox_path: str | Path) -> list[RawalkObject]:
    raw = np.load(bbox_path, allow_pickle=True)
    items = raw.tolist()
    if isinstance(items, dict):
        items = [items]

    objects: list[RawalkObject] = []
    for item in items:
        if not isinstance(item, dict) or "bbox" not in item:
            continue
        bbox = np.asarray(item["bbox"], dtype=np.float32).reshape(-1)
        if bbox.size < 4:
            continue
        score = float(bbox[4]) if bbox.size >= 5 else 1.0
        objects.append(
            RawalkObject(
                bbox_xyxy=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                score=score,
                human_name=item.get("human_name"),
                human_id=_optional_int(item.get("human_id")),
            )
        )
    return objects


def prepare_rawalk_yolo_dataset(
    rawalk_root: str | Path,
    out_root: str | Path,
    *,
    streams: StreamMode = "exo",
    frame_step: int = 10,
    val_fraction: float = 0.2,
    min_score: float = 0.2,
    min_box_px: float = 4.0,
    max_images: int | None = None,
    link_mode: LinkMode = "hardlink",
    overwrite: bool = False,
) -> PrepareStats:
    tagging_root = resolve_tagging_root(rawalk_root)
    out_root = Path(out_root)
    discovered_frames = list(iter_rawalk_frames(tagging_root, streams=streams, frame_step=frame_step, min_score=min_score))
    skipped_missing = sum(1 for frame in discovered_frames if not frame.image_path.exists())
    frames = [frame for frame in discovered_frames if frame.image_path.exists()]
    if max_images is not None:
        frames = _evenly_limit_frames(frames, max_images)

    val_sequences = _val_sequences(frames, val_fraction)
    manifest_rows: list[dict[str, str | int]] = []
    train_images = val_images = boxes = skipped_empty = 0

    for frame in frames:
        split: SplitName = "val" if frame.sequence in val_sequences else "train"
        width, height = image_size(frame.image_path)
        yolo_lines = [
            line
            for obj in frame.objects
            if (line := bbox_to_yolo_line(obj.bbox_xyxy, width, height, min_box_px=min_box_px)) is not None
        ]
        if not yolo_lines:
            skipped_empty += 1
            continue

        stem = f"{frame.sequence}_{frame.camera}_{frame.channel}_{frame.frame_id}"
        image_out = out_root / "images" / split / f"{stem}{frame.image_path.suffix.lower()}"
        label_out = out_root / "labels" / split / f"{stem}.txt"
        _place_image(frame.image_path, image_out, link_mode=link_mode, overwrite=overwrite)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not label_out.exists():
            label_out.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        if split == "train":
            train_images += 1
        else:
            val_images += 1
        boxes += len(yolo_lines)
        manifest_rows.append(
            {
                "split": split,
                "sequence": frame.sequence,
                "camera": frame.camera,
                "channel": frame.channel,
                "frame_id": frame.frame_id,
                "image": image_out.as_posix(),
                "source_image": frame.image_path.as_posix(),
                "labels": label_out.as_posix(),
                "boxes": len(yolo_lines),
            }
        )

    yaml_path = out_root / "rawalk_person.yaml"
    manifest_path = out_root / "manifest.csv"
    _write_dataset_yaml(yaml_path, out_root)
    _write_manifest(manifest_path, manifest_rows)

    return PrepareStats(
        tagging_root=tagging_root,
        out_root=out_root,
        yaml_path=yaml_path,
        manifest_path=manifest_path,
        train_images=train_images,
        val_images=val_images,
        boxes=boxes,
        skipped_missing_images=skipped_missing,
        skipped_empty_labels=skipped_empty,
    )


def bbox_to_yolo_line(
    bbox_xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    *,
    min_box_px: float = 4.0,
) -> str | None:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0.0, min(float(image_width), x1))
    x2 = max(0.0, min(float(image_width), x2))
    y1 = max(0.0, min(float(image_height), y1))
    y2 = max(0.0, min(float(image_height), y2))
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < min_box_px or box_h < min_box_px:
        return None
    cx = (x1 + x2) / 2.0 / image_width
    cy = (y1 + y2) / 2.0 / image_height
    nw = box_w / image_width
    nh = box_h / image_height
    return f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def image_size(image_path: str | Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def _image_path_for_bbox(sequence_dir: Path, camera: str, channel: str, frame_id: str) -> Path:
    filename = f"{frame_id}.jpg"
    if camera.startswith("cam"):
        return sequence_dir / "exo" / camera / "images" / filename
    return sequence_dir / "ego" / camera / "images" / channel / filename


def _val_sequences(frames: list[RawalkFrame], val_fraction: float) -> set[str]:
    sequences = sorted({frame.sequence for frame in frames})
    if not sequences:
        return set()
    val_count = max(1, round(len(sequences) * max(0.0, min(0.9, val_fraction))))
    return set(sequences[-val_count:])


def _evenly_limit_frames(frames: list[RawalkFrame], max_images: int) -> list[RawalkFrame]:
    if max_images <= 0:
        return []
    if len(frames) <= max_images:
        return frames
    if max_images == 1:
        return [frames[0]]
    step = (len(frames) - 1) / float(max_images - 1)
    indexes = sorted({round(i * step) for i in range(max_images)})
    return [frames[index] for index in indexes]


def _place_image(source: Path, destination: Path, *, link_mode: LinkMode, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not overwrite:
            return
        destination.unlink()
    if link_mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            shutil.copy2(source, destination)
            return
    shutil.copy2(source, destination)


def _write_dataset_yaml(yaml_path: Path, out_root: Path) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            f"path: {out_root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: person",
            "",
        ]
    )
    yaml_path.write_text(text, encoding="utf-8")


def _write_manifest(manifest_path: Path, rows: list[dict[str, str | int]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "sequence", "camera", "channel", "frame_id", "image", "source_image", "labels", "boxes"]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _optional_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
