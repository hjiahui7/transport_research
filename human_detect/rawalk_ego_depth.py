from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from .rawalk import resolve_tagging_root


DEFAULT_VIEWERS = ("aria01", "aria02", "aria03", "aria04")


@dataclass(frozen=True)
class AriaCalibration:
    intrinsics: np.ndarray
    extrinsics: np.ndarray


@dataclass(frozen=True)
class RawalkEgoPersonLabel:
    sequence: str
    frame_id: str
    viewer: str
    human_name: str
    image_path: Path
    bbox_xyxy: tuple[float, float, float, float]
    depth_m: float
    distance_m: float
    torso_depth_m: float | None
    torso_distance_m: float | None
    num_visible_keypoints: int
    keypoint_depths_m: tuple[float, ...]


@dataclass(frozen=True)
class RawalkEgoDepthStats:
    tagging_root: Path
    out_path: Path
    rows: int
    skipped_missing_files: int
    skipped_not_visible: int


def read_aria_calibration(calib_path: str | Path) -> AriaCalibration:
    """Read the RGB camera intrinsics/extrinsics from an EgoHumans Aria calib txt."""
    lines = Path(calib_path).read_text(encoding="utf-8").strip().splitlines()[1:]
    lines = [line.strip() for line in lines if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"Aria calibration file is too short: {calib_path}")
    intrinsics = np.asarray([float(value) for value in lines[1].split()], dtype=np.float64)
    extrinsics = np.asarray([float(value) for value in lines[2].split()], dtype=np.float64).reshape(4, 3).T
    if intrinsics.shape != (15,):
        raise ValueError(f"Expected 15 Aria intrinsics values in {calib_path}, got {intrinsics.shape}")
    if extrinsics.shape != (3, 4):
        raise ValueError(f"Expected 3x4 Aria extrinsics in {calib_path}, got {extrinsics.shape}")
    return AriaCalibration(intrinsics=intrinsics, extrinsics=extrinsics)


def load_colmap_from_aria(sequence_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(sequence_dir) / "colmap" / "workplace" / "colmap_from_aria_transforms.pkl"
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    return {str(key): np.asarray(value, dtype=np.float64) for key, value in raw.items()}


def build_world_to_cam(
    calibration: AriaCalibration,
    colmap_from_aria: dict[str, np.ndarray],
    *,
    viewer: str,
    anchor: str = "aria01",
) -> np.ndarray:
    """Build a 4x4 transform from the shared EgoHumans world frame to the viewer RGB camera."""
    if viewer not in colmap_from_aria:
        raise KeyError(f"Viewer {viewer!r} missing from colmap_from_aria transforms")
    if anchor not in colmap_from_aria:
        raise KeyError(f"Anchor {anchor!r} missing from colmap_from_aria transforms")

    extrinsics4 = np.eye(4, dtype=np.float64)
    extrinsics4[:3, :] = calibration.extrinsics
    world_to_viewer = np.linalg.inv(colmap_from_aria[viewer]) @ colmap_from_aria[anchor]
    return extrinsics4 @ world_to_viewer


def cam_from_world(points_world: np.ndarray, world_to_cam: np.ndarray) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    homogeneous = np.concatenate([points[:, :3], np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    projected = (world_to_cam @ homogeneous.T).T
    return projected[:, :3] / projected[:, 3:4]


def image_from_cam(points_cam: np.ndarray, intrinsics: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    """Project camera-coordinate points with the Aria FishEye62 model."""
    points = np.asarray(points_cam, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 3)

    output = np.full((points.shape[0], 2), -1.0, dtype=np.float64)
    in_front = points[:, 2] > 0
    if not np.any(in_front):
        return output

    cam = points[in_front]
    start_k, num_k = 3, 6
    start_p = start_k + num_k
    start_s = start_p + 2

    ab = cam[:, :2] / cam[:, 2:3]
    radius = np.sqrt(np.sum(ab * ab, axis=1))
    theta = np.arctan(radius)
    theta_sq = theta * theta

    radial = np.ones_like(theta)
    theta_power = theta_sq.copy()
    for index in range(num_k):
        radial += theta_power * intrinsics[start_k + index]
        theta_power *= theta_sq

    theta_div_radius = np.ones_like(radius)
    nonzero_radius = radius >= eps
    theta_div_radius[nonzero_radius] = theta[nonzero_radius] / radius[nonzero_radius]
    distorted = (radial * theta_div_radius)[:, None] * ab
    distorted_sq = np.sum(distorted * distorted, axis=1)

    uv = distorted.copy()
    tangential_coeff = intrinsics[start_p : start_p + 2]
    tangential = 2.0 * distorted * tangential_coeff
    uv += tangential * distorted + distorted_sq[:, None] * tangential_coeff

    radial_powers = np.stack([distorted_sq, distorted_sq * distorted_sq], axis=1)
    uv[:, 0] += np.sum(intrinsics[start_s : start_s + 2] * radial_powers, axis=1)
    uv[:, 1] += np.sum(intrinsics[start_s + 2 : start_s + 4] * radial_powers, axis=1)

    focal, cu, cv = intrinsics[0], intrinsics[1], intrinsics[2]
    output[in_front] = focal * uv + np.array([cu, cv], dtype=np.float64)
    return output


def raw_sensor_to_jpg_frame(points_2d_raw: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Rotate Aria raw RGB sensor coordinates into the stored jpg coordinate frame."""
    points = np.asarray(points_2d_raw, dtype=np.float64)
    x_jpg = image_height - points[..., 1]
    y_jpg = points[..., 0]
    return np.stack([x_jpg, y_jpg], axis=-1)


def label_visible_people_for_frame(
    sequence_dir: str | Path,
    *,
    frame_id: str,
    viewer: str,
    anchor: str = "aria01",
    pose_source: str = "fit_poses3d",
    min_visible_keypoints: int = 5,
    bbox_padding: float = 1.4,
) -> list[RawalkEgoPersonLabel]:
    sequence_dir = Path(sequence_dir)
    image_path = sequence_dir / "ego" / viewer / "images" / "rgb" / f"{frame_id}.jpg"
    calib_path = sequence_dir / "ego" / viewer / "calib" / f"{frame_id}.txt"
    pose_path = sequence_dir / "processed_data" / pose_source / f"{frame_id}.npy"
    if not image_path.exists() or not calib_path.exists() or not pose_path.exists():
        missing = [str(path) for path in [image_path, calib_path, pose_path] if not path.exists()]
        raise FileNotFoundError("; ".join(missing))

    image_width, image_height = _image_size(image_path)
    calibration = read_aria_calibration(calib_path)
    colmap_from_aria = load_colmap_from_aria(sequence_dir)
    world_to_cam = build_world_to_cam(calibration, colmap_from_aria, viewer=viewer, anchor=anchor)
    poses3d = np.load(pose_path, allow_pickle=True).item()

    labels: list[RawalkEgoPersonLabel] = []
    for human_name, keypoints in poses3d.items():
        human_name = str(human_name)
        if human_name == viewer:
            continue
        label = _label_one_person(
            sequence=sequence_dir.name,
            frame_id=frame_id,
            viewer=viewer,
            human_name=human_name,
            image_path=image_path,
            keypoints_3d=np.asarray(keypoints, dtype=np.float64),
            world_to_cam=world_to_cam,
            intrinsics=calibration.intrinsics,
            image_width=image_width,
            image_height=image_height,
            min_visible_keypoints=min_visible_keypoints,
            bbox_padding=bbox_padding,
        )
        if label is not None:
            labels.append(label)
    return labels


def iter_ego_depth_labels(
    rawalk_root: str | Path,
    *,
    sequences: Sequence[str] | None = None,
    viewers: Sequence[str] = DEFAULT_VIEWERS,
    anchor: str = "aria01",
    pose_source: str = "fit_poses3d",
    frame_step: int = 1,
    max_frames_per_viewer: int | None = None,
    min_visible_keypoints: int = 5,
    bbox_padding: float = 1.4,
) -> Iterable[RawalkEgoPersonLabel]:
    tagging_root = resolve_tagging_root(rawalk_root)
    wanted_sequences = set(sequences or [])
    frame_step = max(1, frame_step)

    for sequence_dir in sorted(path for path in tagging_root.iterdir() if path.is_dir()):
        if wanted_sequences and sequence_dir.name not in wanted_sequences:
            continue
        pose_dir = sequence_dir / "processed_data" / pose_source
        if not pose_dir.exists():
            continue
        all_frame_ids = [path.stem for path in sorted(pose_dir.glob("*.npy"))]
        frame_ids = [frame_id for frame_id in all_frame_ids if _use_frame(frame_id, frame_step)]
        if max_frames_per_viewer is not None:
            frame_ids = frame_ids[: max(0, max_frames_per_viewer)]

        available_viewers = [viewer for viewer in viewers if (sequence_dir / "ego" / viewer).exists()]
        for viewer in available_viewers:
            for frame_id in frame_ids:
                try:
                    yield from label_visible_people_for_frame(
                        sequence_dir,
                        frame_id=frame_id,
                        viewer=viewer,
                        anchor=anchor,
                        pose_source=pose_source,
                        min_visible_keypoints=min_visible_keypoints,
                        bbox_padding=bbox_padding,
                    )
                except FileNotFoundError:
                    continue


def write_ego_depth_csv(rows: Sequence[RawalkEgoPersonLabel], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_csv(row))


def prepare_rawalk_ego_depth(
    rawalk_root: str | Path,
    out_path: str | Path,
    *,
    sequences: Sequence[str] | None = None,
    viewers: Sequence[str] = DEFAULT_VIEWERS,
    anchor: str = "aria01",
    pose_source: str = "fit_poses3d",
    frame_step: int = 1,
    max_frames_per_viewer: int | None = None,
    min_visible_keypoints: int = 5,
    bbox_padding: float = 1.4,
) -> RawalkEgoDepthStats:
    tagging_root = resolve_tagging_root(rawalk_root)
    rows = list(
        iter_ego_depth_labels(
            tagging_root,
            sequences=sequences,
            viewers=viewers,
            anchor=anchor,
            pose_source=pose_source,
            frame_step=frame_step,
            max_frames_per_viewer=max_frames_per_viewer,
            min_visible_keypoints=min_visible_keypoints,
            bbox_padding=bbox_padding,
        )
    )
    write_ego_depth_csv(rows, out_path)
    return RawalkEgoDepthStats(
        tagging_root=tagging_root,
        out_path=Path(out_path),
        rows=len(rows),
        skipped_missing_files=0,
        skipped_not_visible=0,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ego-view Rawalk person bbox and distance labels from 3D pose GT.")
    parser.add_argument(
        "--rawalk-root",
        default=r"data\media\rawalk",
        help="Rawalk root. Accepts either data/media/rawalk or the final 01_tagging folder.",
    )
    parser.add_argument("--out", default=r"runs\rawalk_ego_depth.csv", help="Output CSV path.")
    parser.add_argument("--sequences", nargs="*", default=None, help="Optional sequence names such as 001_tagging.")
    parser.add_argument("--viewers", nargs="*", default=list(DEFAULT_VIEWERS), help="Ego viewers to process.")
    parser.add_argument("--anchor", default="aria01", help="Anchor ego camera used by the shared 3D world frame.")
    parser.add_argument(
        "--pose-source",
        choices=["fit_poses3d", "refine_poses3d", "poses3d"],
        default="fit_poses3d",
        help="3D pose folder to use for labels.",
    )
    parser.add_argument("--frame-step", type=int, default=10, help="Use every Nth frame per viewer.")
    parser.add_argument("--max-frames-per-viewer", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--min-visible-keypoints", type=int, default=5, help="Drop people with fewer visible keypoints.")
    parser.add_argument("--bbox-padding", type=float, default=1.4, help="Expand projected keypoint bbox by this factor.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    stats = prepare_rawalk_ego_depth(
        rawalk_root=Path(args.rawalk_root),
        out_path=Path(args.out),
        sequences=args.sequences,
        viewers=args.viewers,
        anchor=args.anchor,
        pose_source=args.pose_source,
        frame_step=args.frame_step,
        max_frames_per_viewer=args.max_frames_per_viewer,
        min_visible_keypoints=args.min_visible_keypoints,
        bbox_padding=args.bbox_padding,
    )
    print(f"Rawalk root: {stats.tagging_root}")
    print(f"Output CSV:  {stats.out_path}")
    print(f"Rows:        {stats.rows}")
    return 0


def _label_one_person(
    *,
    sequence: str,
    frame_id: str,
    viewer: str,
    human_name: str,
    image_path: Path,
    keypoints_3d: np.ndarray,
    world_to_cam: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    min_visible_keypoints: int,
    bbox_padding: float,
) -> RawalkEgoPersonLabel | None:
    points_cam = cam_from_world(keypoints_3d[:, :3], world_to_cam)
    depths = points_cam[:, 2]
    distances = np.linalg.norm(points_cam, axis=1)
    points_raw = image_from_cam(points_cam, intrinsics)
    points_jpg = raw_sensor_to_jpg_frame(points_raw, image_width, image_height)

    keypoint_valid = keypoints_3d[:, 3] > 0 if keypoints_3d.shape[1] >= 4 else np.ones(len(keypoints_3d), dtype=bool)
    in_front = depths > 0
    in_bounds = (
        (points_jpg[:, 0] >= 0)
        & (points_jpg[:, 0] < image_width)
        & (points_jpg[:, 1] >= 0)
        & (points_jpg[:, 1] < image_height)
    )
    finite = np.isfinite(points_jpg).all(axis=1) & np.isfinite(depths) & np.isfinite(distances)
    visible = keypoint_valid & in_front & in_bounds & finite
    if int(visible.sum()) < min_visible_keypoints:
        return None

    bbox = _padded_bbox(points_jpg[visible], image_width=image_width, image_height=image_height, padding=bbox_padding)
    torso_visible = visible[[11, 12]] if len(visible) > 12 else np.asarray([], dtype=bool)
    torso_depth_m = None
    torso_distance_m = None
    if torso_visible.size == 2 and np.any(torso_visible):
        torso_indexes = np.asarray([11, 12], dtype=np.int64)[torso_visible]
        torso_depth_m = float(np.mean(depths[torso_indexes]))
        torso_distance_m = float(np.mean(distances[torso_indexes]))

    return RawalkEgoPersonLabel(
        sequence=sequence,
        frame_id=frame_id,
        viewer=viewer,
        human_name=human_name,
        image_path=image_path,
        bbox_xyxy=bbox,
        depth_m=float(np.mean(depths[visible])),
        distance_m=float(np.mean(distances[visible])),
        torso_depth_m=torso_depth_m,
        torso_distance_m=torso_distance_m,
        num_visible_keypoints=int(visible.sum()),
        keypoint_depths_m=tuple(float(value) for value in depths),
    )


def _padded_bbox(
    points_xy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[float, float, float, float]:
    x1, y1 = points_xy.min(axis=0)
    x2, y2 = points_xy.max(axis=0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    box_w, box_h = (x2 - x1) * padding, (y2 - y1) * padding
    padded = (
        max(0.0, cx - box_w / 2.0),
        max(0.0, cy - box_h / 2.0),
        min(float(image_width), cx + box_w / 2.0),
        min(float(image_height), cy + box_h / 2.0),
    )
    return tuple(float(value) for value in padded)


def _image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def _use_frame(frame_id: str, frame_step: int) -> bool:
    try:
        frame_number = int(frame_id)
    except ValueError:
        return True
    return (frame_number - 1) % frame_step == 0


def _csv_fieldnames() -> list[str]:
    return [
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
        "torso_depth_m",
        "torso_distance_m",
        "num_visible_keypoints",
        "keypoint_depths_m",
    ]


def _row_to_csv(row: RawalkEgoPersonLabel) -> dict[str, str | int | float]:
    x1, y1, x2, y2 = row.bbox_xyxy
    return {
        "sequence": row.sequence,
        "frame_id": row.frame_id,
        "viewer": row.viewer,
        "human_name": row.human_name,
        "image_path": row.image_path.as_posix(),
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "depth_m": row.depth_m,
        "distance_m": row.distance_m,
        "torso_depth_m": "" if row.torso_depth_m is None else row.torso_depth_m,
        "torso_distance_m": "" if row.torso_distance_m is None else row.torso_distance_m,
        "num_visible_keypoints": row.num_visible_keypoints,
        "keypoint_depths_m": " ".join(f"{value:.6f}" for value in row.keypoint_depths_m),
    }


if __name__ == "__main__":
    raise SystemExit(main())
