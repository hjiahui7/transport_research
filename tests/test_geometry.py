from __future__ import annotations

import math

import numpy as np

from human_detect.geometry import CameraIntrinsics, backproject_pixel, distance_and_angles
from human_detect.pooling import pool_person_depth


def test_backprojection_center_pixel_has_zero_xy() -> None:
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0)
    x, y, z = backproject_pixel(50.0, 40.0, 10.0, intrinsics)
    assert (x, y, z) == (0.0, 0.0, 10.0)
    distance, yaw, pitch = distance_and_angles(x, y, z)
    assert distance == 10.0
    assert yaw == 0.0
    assert pitch == 0.0


def test_backprojection_angles() -> None:
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0)
    x, y, z = backproject_pixel(100.0, 100.0, 100.0, intrinsics)
    distance, yaw, pitch = distance_and_angles(x, y, z)
    assert math.isclose(distance, math.sqrt(30000.0))
    assert math.isclose(yaw, 45.0)
    assert math.isclose(pitch, 45.0)


def test_pool_person_depth_uses_lower_region_when_available() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:9, 4:7] = True
    depth = np.ones((10, 10), dtype=np.float32) * 2.0
    depth[5:9, 4:7] = 4.0
    pooled = pool_person_depth(mask, depth, min_pixels=3)
    assert pooled.used_region == "lower_60"
    assert pooled.z_depth_m == 4.0
    assert pooled.mask_area_px == 21
    assert pooled.depth_stats["depth_p50_m"] == 4.0
    assert pooled.depth_stats["lower_depth_p50_m"] == 4.0


def test_pool_person_depth_empty_mask_is_stable() -> None:
    pooled = pool_person_depth(np.zeros((3, 3), dtype=bool), np.ones((3, 3), dtype=np.float32))
    assert pooled.z_depth_m is None
    assert pooled.centroid_uv is None
    assert pooled.mask_area_px == 0
    assert pooled.depth_stats["depth_p50_m"] is None
