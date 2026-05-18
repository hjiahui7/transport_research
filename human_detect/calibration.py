from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_COLUMNS = [
    "z_depth_m",
    "distance_m",
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_area_norm",
    "mask_area_norm",
    "center_x_norm",
    "center_y_norm",
    "score",
    "bearing_yaw_deg",
    "elevation_pitch_deg",
    "fov_deg",
    "depth_p10_m",
    "depth_p25_m",
    "depth_p50_m",
    "depth_p75_m",
    "depth_p90_m",
    "lower_depth_p10_m",
    "lower_depth_p25_m",
    "lower_depth_p50_m",
    "lower_depth_p75_m",
    "lower_depth_p90_m",
]


@dataclass
class CalibratorBundle:
    z_model: Any
    distance_model: Any
    model_type: str
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    metrics: dict[str, Any] = field(default_factory=dict)

    def predict_person(self, person: dict[str, Any], camera: dict[str, Any], image_width: int, image_height: int) -> dict[str, float] | None:
        if person.get("z_depth_m") is None or person.get("distance_m") is None:
            return None
        # This predicts corrected person-level distances only; it does not alter the depth map.
        features = person_feature_vector(person, camera, image_width, image_height)
        z_pred = float(self.z_model.predict([features])[0])
        distance_pred = float(self.distance_model.predict([features])[0])
        return {
            "z_depth_calibrated_m": max(0.0, z_pred),
            "distance_calibrated_m": max(0.0, distance_pred),
        }


def person_feature_vector(person: dict[str, Any], camera: dict[str, Any], image_width: int, image_height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in person["bbox_xyxy"]]
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    bbox_w = max(0.0, x2 - x1)
    bbox_h = max(0.0, y2 - y1)
    bbox_area = bbox_w * bbox_h
    image_area = width * height
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    # The head gets only cheap, per-person geometry signals so it stays fast and hard to overfit.
    depth_stats = person.get("depth_stats") or {}
    values = {
        "z_depth_m": person.get("z_depth_m"),
        "distance_m": person.get("distance_m"),
        "bbox_width_norm": bbox_w / width,
        "bbox_height_norm": bbox_h / height,
        "bbox_area_norm": bbox_area / image_area,
        "mask_area_norm": float(person.get("mask_area_px") or 0.0) / image_area,
        "center_x_norm": center_x / width,
        "center_y_norm": center_y / height,
        "score": person.get("score"),
        "bearing_yaw_deg": person.get("bearing_yaw_deg"),
        "elevation_pitch_deg": person.get("elevation_pitch_deg"),
        "fov_deg": camera.get("fov_deg"),
        "depth_p10_m": depth_stats.get("depth_p10_m", person.get("depth_p10_m")),
        "depth_p25_m": depth_stats.get("depth_p25_m", person.get("depth_p25_m")),
        "depth_p50_m": depth_stats.get("depth_p50_m", person.get("depth_p50_m")),
        "depth_p75_m": depth_stats.get("depth_p75_m", person.get("depth_p75_m")),
        "depth_p90_m": depth_stats.get("depth_p90_m", person.get("depth_p90_m")),
        "lower_depth_p10_m": depth_stats.get("lower_depth_p10_m", person.get("lower_depth_p10_m")),
        "lower_depth_p25_m": depth_stats.get("lower_depth_p25_m", person.get("lower_depth_p25_m")),
        "lower_depth_p50_m": depth_stats.get("lower_depth_p50_m", person.get("lower_depth_p50_m")),
        "lower_depth_p75_m": depth_stats.get("lower_depth_p75_m", person.get("lower_depth_p75_m")),
        "lower_depth_p90_m": depth_stats.get("lower_depth_p90_m", person.get("lower_depth_p90_m")),
    }
    return [_clean_float(values[name]) for name in FEATURE_COLUMNS]


def row_feature_vector(row: dict[str, str]) -> list[float]:
    return [_clean_float(row.get(name)) for name in FEATURE_COLUMNS]


def save_calibrator(bundle: CalibratorBundle, path: str | Path) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_calibrator(path: str | Path) -> CalibratorBundle:
    import joblib

    return joblib.load(path)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(sqrt(float(np.mean(err * err))))
    denom = np.maximum(np.abs(y_true), 1e-6)
    absrel = float(np.mean(np.abs(err) / denom))
    return {"mae": mae, "rmse": rmse, "absrel": absrel}


def _clean_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not isfinite(out):
        return 0.0
    return out
