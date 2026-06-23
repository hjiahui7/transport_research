from __future__ import annotations

import numpy as np

from human_detect.infer_distance_head import _bbox_mask, _scale_boxes_to_square


def test_scale_boxes_to_square_matches_training_resize() -> None:
    boxes = [(10.0, 20.0, 50.0, 100.0)]

    scaled = _scale_boxes_to_square(boxes, image_width=100, image_height=200, image_size=640)

    assert scaled == [(64.0, 64.0, 320.0, 320.0)]


def test_bbox_mask_clips_to_image() -> None:
    mask = _bbox_mask((-5.0, 2.0, 5.0, 20.0), width=10, height=10)

    assert mask.shape == (10, 10)
    assert mask.dtype == np.bool_
    assert int(mask.sum()) == 40
