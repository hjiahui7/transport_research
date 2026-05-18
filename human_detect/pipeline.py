from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from .calibration import CalibratorBundle
from .geometry import CameraIntrinsics, backproject_pixel, distance_and_angles, fov_x_from_intrinsics
from .geometry_moge import MogeGeometry
from .pm_hmcw import parse_calib_text
from .pooling import pool_person_depth
from .segmenter import YoloPersonSegmenter


class DistanceEstimator:
    def __init__(
        self,
        *,
        detector: str = "yolo11n-seg.pt",
        geometry_model: str = "Ruicheng/moge-2-vits-normal",
        imgsz: int = 640,
        geom_size: int = 768,
        num_tokens: int | None = 1200,
        device: str = "cuda:0",
        half: bool = True,
        calibrator: CalibratorBundle | None = None,
    ) -> None:
        self.segmenter = YoloPersonSegmenter(detector, device=device, imgsz=imgsz, half=half)
        self.geometry = MogeGeometry(geometry_model, device=device, half=half, geom_size=geom_size, num_tokens=num_tokens)
        self.calibrator = calibrator

    def infer(self, image_path: str | Path, *, calib_path: str | Path | None = None) -> tuple[dict[str, Any], list]:
        image_path = Path(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image_height, image_width = image.shape[:2]

        # Keep the heavy models frozen: YOLO gives person masks, MoGe gives a metric depth map.
        # Everything after this point is geometric post-processing plus an optional tiny calibrator.
        persons = self.segmenter.predict(image_path)
        geometry_result = self.geometry.infer(image_path)
        intrinsics = _load_intrinsics_override(calib_path) if calib_path else geometry_result.intrinsics
        if calib_path:
            intrinsics = CameraIntrinsics(
                fx=intrinsics.fx,
                fy=intrinsics.fy,
                cx=intrinsics.cx,
                cy=intrinsics.cy,
                width=geometry_result.depth.shape[1],
                height=geometry_result.depth.shape[0],
                source="intrinsics",
            )
        fov_deg = fov_x_from_intrinsics(intrinsics, geometry_result.depth.shape[1])
        camera_json = intrinsics.to_json(fov_deg)

        output_persons: list[dict[str, Any]] = []
        output_masks = []
        for idx, person in enumerate(persons):
            # Estimate one robust depth per person from the instance mask, then backproject the
            # representative pixel into camera coordinates.
            pooled = pool_person_depth(person.mask, geometry_result.depth)
            person_json: dict[str, Any] = {
                "id": idx,
                "score": person.score,
                "bbox_xyxy": [float(v) for v in person.bbox_xyxy],
                "mask_area_px": pooled.mask_area_px,
                "z_depth_m": None,
                "distance_m": None,
                "bearing_yaw_deg": None,
                "elevation_pitch_deg": None,
                "facing_yaw_deg": None,
                "depth_stats": pooled.depth_stats,
            }
            if pooled.z_depth_m is not None and pooled.centroid_uv is not None:
                u, v = pooled.centroid_uv
                x, y, z = backproject_pixel(u, v, pooled.z_depth_m, intrinsics)
                distance, yaw, pitch = distance_and_angles(x, y, z)
                person_json.update(
                    {
                        "z_depth_m": z,
                        "distance_m": distance,
                        "bearing_yaw_deg": yaw,
                        "elevation_pitch_deg": pitch,
                }
            )

            if self.calibrator is not None:
                # The calibrator is a small post-hoc regressor trained on PM-HMCW person-level GT.
                calibrated = self.calibrator.predict_person(person_json, camera_json, image_width, image_height)
                if calibrated is not None:
                    person_json.update(calibrated)

            output_persons.append(person_json)
            output_masks.append(person.mask)

        result = {
            "image_path": str(image_path),
            "image_size": {"width": image_width, "height": image_height},
            "camera": camera_json,
            "persons": output_persons,
        }
        return result, output_masks


def _load_intrinsics_override(calib_path: str | Path) -> CameraIntrinsics:
    return parse_calib_text(Path(calib_path).read_text(encoding="utf-8"))
