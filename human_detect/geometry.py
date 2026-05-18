from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, radians, sqrt, tan
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int | None = None
    height: int | None = None
    source: str = "unknown"

    def scaled(self, sx: float, sy: float, *, width: int | None = None, height: int | None = None) -> "CameraIntrinsics":
        return CameraIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=width if width is not None else self.width,
            height=height if height is not None else self.height,
            source=self.source,
        )

    def to_json(self, fov_deg: float | None = None) -> dict[str, float | int | str | None]:
        return {
            "source": self.source,
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "fov_deg": None if fov_deg is None else float(fov_deg),
        }


def intrinsics_from_projection(values: Iterable[float], *, width: int | None = None, height: int | None = None) -> CameraIntrinsics:
    matrix = list(values)
    if len(matrix) != 12:
        raise ValueError(f"Expected 12 projection values, got {len(matrix)}")
    return CameraIntrinsics(
        fx=matrix[0],
        fy=matrix[5],
        cx=matrix[2],
        cy=matrix[6],
        width=width,
        height=height,
        source="intrinsics",
    )


def intrinsics_from_fov(width: int, height: int, fov_deg: float, *, source: str = "estimated_fov") -> CameraIntrinsics:
    if not np.isfinite(fov_deg) or fov_deg <= 0.0 or fov_deg >= 179.0:
        raise ValueError(f"Invalid horizontal FOV: {fov_deg}")
    fx = width / (2.0 * tan(radians(fov_deg) / 2.0))
    return CameraIntrinsics(
        fx=fx,
        fy=fx,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        width=width,
        height=height,
        source=source,
    )


def intrinsics_from_normalized_matrix(matrix: np.ndarray, width: int, height: int, *, source: str = "moge_intrinsics") -> CameraIntrinsics:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics matrix, got {arr.shape}")

    fx, fy, cx, cy = float(arr[0, 0]), float(arr[1, 1]), float(arr[0, 2]), float(arr[1, 2])
    if max(abs(fx), abs(fy), abs(cx), abs(cy)) <= 10.0:
        fx *= width
        fy *= height
        cx *= width
        cy *= height

    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, source=source)


def fov_x_from_intrinsics(intrinsics: CameraIntrinsics, width: int | None = None) -> float | None:
    image_width = width or intrinsics.width
    if image_width is None or intrinsics.fx <= 0.0:
        return None
    return degrees(2.0 * atan2(image_width, 2.0 * intrinsics.fx))


def backproject_pixel(u: float, v: float, z: float, intrinsics: CameraIntrinsics) -> tuple[float, float, float]:
    if not np.isfinite(z) or z <= 0.0:
        raise ValueError(f"Depth must be a positive finite value, got {z}")
    if intrinsics.fx == 0.0 or intrinsics.fy == 0.0:
        raise ValueError("Camera focal length cannot be zero")
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return float(x), float(y), float(z)


def distance_and_angles(x: float, y: float, z: float) -> tuple[float, float, float]:
    distance = sqrt(x * x + y * y + z * z)
    bearing_yaw = degrees(atan2(x, z))
    elevation_pitch = degrees(atan2(y, z))
    return float(distance), float(bearing_yaw), float(elevation_pitch)


def finite_positive_median(values: np.ndarray) -> float | None:
    valid = values[np.isfinite(values) & (values > 0.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))

