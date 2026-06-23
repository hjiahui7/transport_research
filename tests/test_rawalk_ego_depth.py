from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from human_detect.rawalk_ego_depth import (
    build_world_to_cam,
    cam_from_world,
    image_from_cam,
    label_visible_people_for_frame,
    prepare_rawalk_ego_depth,
    raw_sensor_to_jpg_frame,
    read_aria_calibration,
)


def test_aria_projection_depth_and_jpg_rotation(tmp_path: Path) -> None:
    calib_path = tmp_path / "00001.txt"
    _write_calib(calib_path)
    calibration = read_aria_calibration(calib_path)
    world_to_cam = build_world_to_cam(
        calibration,
        {"aria01": np.eye(4), "aria02": np.eye(4)},
        viewer="aria01",
        anchor="aria01",
    )

    points_cam = cam_from_world(np.array([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]]), world_to_cam)
    points_raw = image_from_cam(points_cam, calibration.intrinsics)
    points_jpg = raw_sensor_to_jpg_frame(points_raw, image_width=100, image_height=100)

    assert np.allclose(points_cam[:, 2], [2.0, 2.0])
    assert np.allclose(points_raw[0], [50.0, 50.0])
    assert np.allclose(points_jpg[0], [50.0, 50.0])
    assert points_jpg[1, 1] > points_jpg[0, 1]


def test_label_visible_people_for_frame_excludes_viewer_and_writes_distances(tmp_path: Path) -> None:
    sequence_dir = _write_minimal_ego_sequence(tmp_path)

    labels = label_visible_people_for_frame(sequence_dir, frame_id="00001", viewer="aria01", bbox_padding=1.0)

    assert len(labels) == 1
    label = labels[0]
    assert label.human_name == "aria02"
    assert label.depth_m == 2.0
    assert label.distance_m >= label.depth_m
    assert label.num_visible_keypoints == 17
    x1, y1, x2, y2 = label.bbox_xyxy
    assert 0 <= x1 < x2 <= 100
    assert 0 <= y1 < y2 <= 100


def test_prepare_rawalk_ego_depth_writes_csv(tmp_path: Path) -> None:
    rawalk_root = tmp_path / "rawalk" / "disk1" / "rawalk" / "datasets" / "ego_exo" / "camera_ready" / "01_tagging"
    _write_minimal_ego_sequence(rawalk_root)
    out_path = tmp_path / "labels.csv"

    stats = prepare_rawalk_ego_depth(
        tmp_path / "rawalk",
        out_path,
        sequences=["001_tagging"],
        viewers=["aria01"],
        frame_step=1,
    )

    assert stats.rows == 1
    with out_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["sequence"] == "001_tagging"
    assert rows[0]["viewer"] == "aria01"
    assert rows[0]["human_name"] == "aria02"
    assert float(rows[0]["depth_m"]) == 2.0


def _write_minimal_ego_sequence(root: Path) -> Path:
    sequence_dir = root / "001_tagging"
    calib_dir = sequence_dir / "ego" / "aria01" / "calib"
    image_dir = sequence_dir / "ego" / "aria01" / "images" / "rgb"
    pose_dir = sequence_dir / "processed_data" / "fit_poses3d"
    work_dir = sequence_dir / "colmap" / "workplace"
    calib_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    _write_calib(calib_dir / "00001.txt")
    Image.new("RGB", (100, 100), color=(20, 30, 40)).save(image_dir / "00001.jpg")
    with (work_dir / "colmap_from_aria_transforms.pkl").open("wb") as handle:
        pickle.dump({"aria01": np.eye(4), "aria02": np.eye(4)}, handle)

    viewer_keypoints = _person_keypoints(z=1.5, x_offset=-0.5)
    other_keypoints = _person_keypoints(z=2.0, x_offset=0.0)
    np.save(
        pose_dir / "00001.npy",
        {"aria01": viewer_keypoints, "aria02": other_keypoints},
        allow_pickle=True,
    )
    return sequence_dir


def _person_keypoints(*, z: float, x_offset: float) -> np.ndarray:
    keypoints = np.zeros((17, 4), dtype=np.float64)
    for index in range(17):
        keypoints[index, 0] = x_offset + ((index % 5) - 2) * 0.03
        keypoints[index, 1] = ((index // 5) - 1) * 0.04
        keypoints[index, 2] = z
        keypoints[index, 3] = 1.0
    return keypoints


def _write_calib(path: Path) -> None:
    intrinsics = [100.0, 50.0, 50.0] + [0.0] * 12
    extrinsics = np.eye(4, dtype=np.float64)[:3, :].T.reshape(-1)
    text = "\n".join(
        [
            "Serial, intrinsics (radtanthinprsim), extrinsic (3x4)",
            "aria",
            " ".join(str(value) for value in intrinsics),
            " ".join(str(float(value)) for value in extrinsics),
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
