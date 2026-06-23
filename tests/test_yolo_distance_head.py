from __future__ import annotations

import torch

from human_detect.yolo_distance_head import YoloGridDistanceHead, distance_head_loss, select_distance_level


def test_yolo_grid_distance_head_shapes() -> None:
    head = YoloGridDistanceHead([8, 16, 32], init_distance_m=3.0)
    features = [
        torch.zeros(2, 8, 10, 10),
        torch.zeros(2, 16, 5, 5),
        torch.zeros(2, 32, 3, 3),
    ]

    outputs = head(features)

    assert [tuple(output.shape) for output in outputs] == [(2, 10, 10), (2, 5, 5), (2, 3, 3)]


def test_distance_head_loss_uses_target_grid_cell() -> None:
    predictions = [torch.zeros(1, 10, 10), torch.zeros(1, 5, 5), torch.zeros(1, 3, 3)]
    labels = [torch.tensor([[40.0, 40.0, 32.0, 48.0, 2.0]], dtype=torch.float32)]

    loss, count = distance_head_loss(predictions, labels, strides=[8.0, 16.0, 32.0], image_size=80)

    assert count == 1
    assert loss.item() > 0.0


def test_select_distance_level_prefers_matching_object_scale() -> None:
    assert select_distance_level(24.0, 32.0, [8.0, 16.0, 32.0]) == 0
    assert select_distance_level(64.0, 64.0, [8.0, 16.0, 32.0]) == 1
    assert select_distance_level(200.0, 200.0, [8.0, 16.0, 32.0]) == 2
