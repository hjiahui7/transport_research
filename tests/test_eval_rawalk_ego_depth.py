from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from human_detect.eval_rawalk_ego_depth import greedy_rawalk_matches, load_rawalk_gt_groups, row_from_match


def test_load_rawalk_gt_groups_filters_distance_outliers(tmp_path: Path) -> None:
    image_path = tmp_path / "00001.jpg"
    Image.new("RGB", (100, 100)).save(image_path)
    labels_path = tmp_path / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence",
                "frame_id",
                "viewer",
                "human_name",
                "image_path",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "depth_m",
                "distance_m",
            ],
        )
        writer.writeheader()
        writer.writerow(_label_row(image_path, distance="3.0"))
        writer.writerow(_label_row(image_path, distance="200000.0"))

    groups = load_rawalk_gt_groups(labels_path, min_distance=0.2, max_distance=20.0)

    assert list(groups) == [image_path.as_posix()]
    assert len(groups[image_path.as_posix()]) == 1
    assert groups[image_path.as_posix()][0].distance_gt == 3.0


def test_greedy_rawalk_matches_and_calibration_row(tmp_path: Path) -> None:
    image_path = tmp_path / "00001.jpg"
    Image.new("RGB", (100, 100)).save(image_path)
    labels_path = tmp_path / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_label_row(image_path)))
        writer.writeheader()
        writer.writerow(_label_row(image_path, distance="3.0"))
    gt = load_rawalk_gt_groups(labels_path)[image_path.as_posix()]
    pred = {"bbox_xyxy": [12.0, 12.0, 48.0, 90.0], "z_depth_m": 2.8, "distance_m": 3.1, "score": 0.9, "mask_area_px": 100}

    matches = greedy_rawalk_matches([pred], gt, iou_threshold=0.3)
    row = row_from_match(
        {"image_size": {"width": 100, "height": 100}, "camera": {"fov_deg": 90.0}},
        pred,
        gt[0],
        0,
        0,
        matches[0].iou,
    )

    assert len(matches) == 1
    assert row["distance_gt"] == 3.0
    assert row["z_gt"] == 2.5
    assert row["image_id"] == "001_tagging_aria01_00001"


def _label_row(image_path: Path, *, distance: str = "3.0") -> dict[str, str]:
    return {
        "sequence": "001_tagging",
        "frame_id": "00001",
        "viewer": "aria01",
        "human_name": "aria02",
        "image_path": image_path.as_posix(),
        "bbox_x1": "10.0",
        "bbox_y1": "10.0",
        "bbox_x2": "50.0",
        "bbox_y2": "90.0",
        "depth_m": "2.5",
        "distance_m": distance,
    }
