from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from human_detect.rawalk import bbox_to_yolo_line, prepare_rawalk_yolo_dataset


def test_bbox_to_yolo_line_clips_and_normalizes() -> None:
    line = bbox_to_yolo_line((-10.0, 20.0, 50.0, 120.0), image_width=100, image_height=200)
    assert line == "0 0.250000 0.350000 0.500000 0.500000"


def test_prepare_rawalk_yolo_dataset_writes_labels(tmp_path: Path) -> None:
    rawalk_root = tmp_path / "rawalk" / "disk1" / "rawalk" / "datasets" / "ego_exo" / "camera_ready" / "01_tagging"
    _write_rawalk_frame(rawalk_root, "001_tagging", "00001")
    _write_rawalk_frame(rawalk_root, "002_tagging", "00001")

    out_root = tmp_path / "yolo"
    stats = prepare_rawalk_yolo_dataset(
        rawalk_root=tmp_path / "rawalk",
        out_root=out_root,
        frame_step=1,
        val_fraction=0.5,
        link_mode="copy",
    )

    assert stats.train_images == 1
    assert stats.val_images == 1
    assert stats.boxes == 2
    assert (out_root / "rawalk_person.yaml").exists()
    train_labels = sorted((out_root / "labels" / "train").glob("*.txt"))
    val_labels = sorted((out_root / "labels" / "val").glob("*.txt"))
    assert len(train_labels) == 1
    assert len(val_labels) == 1
    assert train_labels[0].read_text(encoding="utf-8").strip() == "0 0.300000 0.400000 0.400000 0.400000"


def _write_rawalk_frame(rawalk_root: Path, sequence: str, frame_id: str) -> None:
    image_dir = rawalk_root / sequence / "exo" / "cam01" / "images"
    bbox_dir = rawalk_root / sequence / "processed_data" / "bboxes" / "cam01" / "rgb"
    image_dir.mkdir(parents=True, exist_ok=True)
    bbox_dir.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (100, 200), color=(10, 20, 30)).save(image_dir / f"{frame_id}.jpg")
    item = {
        "bbox": np.array([10.0, 40.0, 50.0, 120.0, 0.99], dtype=np.float32),
        "human_name": "aria01",
        "human_id": 0,
    }
    np.save(bbox_dir / f"{frame_id}.npy", np.array([item], dtype=object), allow_pickle=True)
