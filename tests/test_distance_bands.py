from __future__ import annotations

from human_detect.distance_eval import distance_band
from human_detect.workzone import WorkzoneRow, distance_class_3_from_row
from human_detect.workzone_report import distance_band_from_meters


def test_workzone_distance_band_uses_three_and_six_meter_thresholds() -> None:
    assert distance_band_from_meters(2.99) == "Close"
    assert distance_band_from_meters(3.0) == "Careful"
    assert distance_band_from_meters(6.0) == "Careful"
    assert distance_band_from_meters(6.01) == "Safe"


def test_distance_eval_band_uses_matching_short_labels() -> None:
    assert distance_band(2.99) == "<3"
    assert distance_band(3.0) == "3-6"
    assert distance_band(6.0) == "3-6"
    assert distance_band(6.01) == ">6"


def test_workzone_split_distance_class_normalizes_legacy_labels() -> None:
    row = WorkzoneRow(
        worker_key="w1",
        image_id="img1",
        recording="rec",
        scene_type="scene",
        frame_id="1",
        worker_index="1",
        bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
        depth_m=None,
        distance_class_3="3-5",
        depth_source="none",
    )
    assert distance_class_3_from_row(row) == "3-6"
