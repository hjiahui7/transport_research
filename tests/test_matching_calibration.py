from __future__ import annotations

from human_detect.calibration import FEATURE_COLUMNS, person_feature_vector
from human_detect.matching import bbox_iou, greedy_bbox_matches
from human_detect.pm_hmcw import KittiObject


def test_bbox_iou_and_greedy_matching() -> None:
    predictions = [
        {"bbox_xyxy": [0.0, 0.0, 10.0, 10.0]},
        {"bbox_xyxy": [100.0, 100.0, 120.0, 120.0]},
    ]
    gt = [
        KittiObject("Pedestrian", 0.0, 0, 0.0, (1.0, 1.0, 11.0, 11.0), (1.0, 1.0, 1.0), (0.0, 0.0, 2.0), 0.0),
        KittiObject("Pedestrian", 0.0, 0, 0.0, (99.0, 99.0, 121.0, 121.0), (1.0, 1.0, 1.0), (0.0, 0.0, 5.0), 0.0),
    ]
    assert bbox_iou(predictions[0]["bbox_xyxy"], gt[0].bbox_xyxy) > 0.6
    matches = greedy_bbox_matches(predictions, gt, iou_threshold=0.3)
    assert [(m.pred_index, m.gt_index) for m in matches] == [(0, 0), (1, 1)]


def test_person_feature_vector_is_finite() -> None:
    person = {
        "bbox_xyxy": [10.0, 20.0, 110.0, 220.0],
        "mask_area_px": 1000,
        "score": 0.9,
        "z_depth_m": 3.0,
        "distance_m": 3.2,
        "bearing_yaw_deg": -5.0,
        "elevation_pitch_deg": 2.0,
    }
    camera = {"fov_deg": 60.0}
    features = person_feature_vector(person, camera, 200, 400)
    assert len(features) == len(FEATURE_COLUMNS)
    assert all(isinstance(value, float) for value in features)
    assert features[2] == 0.5
    assert features[3] == 0.5
