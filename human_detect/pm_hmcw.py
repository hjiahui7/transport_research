from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Iterator
from zipfile import ZipFile

from .geometry import CameraIntrinsics, intrinsics_from_projection

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class KittiObject:
    category: str
    truncation: float
    occlusion: int
    alpha: float
    bbox_xyxy: tuple[float, float, float, float]
    dimensions_hwl: tuple[float, float, float]
    location_xyz: tuple[float, float, float]
    rotation_y: float
    score: float | None = None

    @property
    def z_gt(self) -> float:
        # KITTI/PM-HMCW label location is camera-coordinate (x, y, z); z is forward depth.
        return self.location_xyz[2]

    @property
    def distance_gt(self) -> float:
        # The Euclidean distance is the person 3D-box center distance from the camera.
        x, y, z = self.location_xyz
        return sqrt(x * x + y * y + z * z)


@dataclass(frozen=True)
class PmHmcwSample:
    image_path: Path
    label_path: Path
    calib_path: Path
    split_dir: Path

    @property
    def image_id(self) -> str:
        return self.image_path.stem

    def load_objects(self) -> list[KittiObject]:
        return parse_label_text(self.label_path.read_text(encoding="utf-8"))

    def load_intrinsics(self) -> CameraIntrinsics:
        return parse_calib_text(self.calib_path.read_text(encoding="utf-8"))


def parse_label_text(text: str) -> list[KittiObject]:
    objects: list[KittiObject] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 15:
            raise ValueError(f"KITTI label line {line_no} has {len(parts)} fields; expected at least 15")
        # PM-HMCW real/virtual exports use KITTI-style labels:
        # type trunc occl alpha bbox(4) dims(h,w,l) location(x,y,z) rotation_y [score].
        category = parts[0]
        truncation = float(parts[1])
        occlusion = int(float(parts[2]))
        alpha = float(parts[3])
        bbox = tuple(float(v) for v in parts[4:8])
        dimensions = tuple(float(v) for v in parts[8:11])
        location = tuple(float(v) for v in parts[11:14])
        rotation_y = float(parts[14])
        score = float(parts[15]) if len(parts) > 15 else None
        objects.append(
            KittiObject(
                category=category,
                truncation=truncation,
                occlusion=occlusion,
                alpha=alpha,
                bbox_xyxy=bbox,  # type: ignore[arg-type]
                dimensions_hwl=dimensions,  # type: ignore[arg-type]
                location_xyz=location,  # type: ignore[arg-type]
                rotation_y=rotation_y,
                score=score,
            )
        )
    return objects


def parse_calib_text(text: str) -> CameraIntrinsics:
    entries: dict[str, list[float]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        entries[key.strip()] = [float(v) for v in value.split()]
    if "P2" not in entries:
        raise ValueError("Calibration text does not contain P2")
    return intrinsics_from_projection(entries["P2"])


def read_zip_text(zip_path: str | Path, member: str) -> str:
    with ZipFile(zip_path) as archive:
        with archive.open(member) as handle:
            return handle.read().decode("utf-8")


def parse_label_from_zip(zip_path: str | Path, member: str) -> list[KittiObject]:
    return parse_label_text(read_zip_text(zip_path, member))


def parse_calib_from_zip(zip_path: str | Path, member: str) -> CameraIntrinsics:
    return parse_calib_text(read_zip_text(zip_path, member))


def find_split_dirs(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(root_path)

    split_dirs: list[Path] = []
    candidates = [root_path]
    candidates.extend(path.parent for path in root_path.rglob("image_2") if path.is_dir())
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (candidate / "image_2").is_dir() and (candidate / "label_2").is_dir() and (candidate / "calib").is_dir():
            split_dirs.append(candidate)
    return sorted(split_dirs)


def iter_samples(root: str | Path, *, category: str | None = None) -> Iterator[PmHmcwSample]:
    for split_dir in find_split_dirs(root):
        image_dir = split_dir / "image_2"
        label_dir = split_dir / "label_2"
        calib_dir = split_dir / "calib"
        for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
            label_path = label_dir / f"{image_path.stem}.txt"
            calib_path = calib_dir / f"{image_path.stem}.txt"
            if not label_path.exists() or not calib_path.exists():
                continue
            sample = PmHmcwSample(image_path=image_path, label_path=label_path, calib_path=calib_path, split_dir=split_dir)
            if category is None:
                yield sample
            else:
                if any(obj.category == category for obj in sample.load_objects()):
                    yield sample
