from pathlib import Path

from human_detect.prepare_workzone_v2 import V2Worker, normalized_distance_class, stratified_validation_ids


def worker(image_id: str, recording: str) -> V2Worker:
    return V2Worker(
        worker_key=f"{image_id}#1",
        image_id=image_id,
        image_path=Path(f"{image_id}.png"),
        recording=recording,
        scene_type="garage",
        frame_id="1",
        worker_index="1",
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        depth_m=2.0,
        distance_class_3="<3",
        depth_source="test",
        data_source="test",
    )


def test_validation_split_has_exact_count_and_is_reproducible() -> None:
    rows = [worker(f"a_{index}", "a") for index in range(6)]
    rows += [worker(f"b_{index}", "b") for index in range(4)]
    first = stratified_validation_ids(rows, count=5, seed=7)
    second = stratified_validation_ids(rows, count=5, seed=7)
    assert first == second
    assert len(first) == 5
    assert len([image_id for image_id in first if image_id.startswith("a_")]) == 3
    assert len([image_id for image_id in first if image_id.startswith("b_")]) == 2


def test_numeric_depth_controls_distance_class() -> None:
    assert normalized_distance_class(2.9, ">6") == "<3"
    assert normalized_distance_class(6.0, "<3") == "3-6"
    assert normalized_distance_class(6.1, "<3") == ">6"
    assert normalized_distance_class(None, "3-5") == "3-6"
