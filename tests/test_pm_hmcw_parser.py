from __future__ import annotations

from pathlib import Path

import pytest

from human_detect.pm_hmcw import parse_calib_from_zip, parse_calib_text, parse_label_from_zip, parse_label_text


LABEL_TEXT = """Pedestrian 0.00 0 0.00 195.21 396.14 711.35 1080.00 1.2 0.63 0.66 -0.95 0.93 3.34 5.63 1.0000
Pedestrian 0.00 0 0.00 614.93 261.38 1261.15 1080.00 0.75 0.54 0.31 -0.06 0.50 1.79 4.72 1.0000
"""

CALIB_TEXT = """P0: 0 0 0 0 0 0 0 0 0 0 0 0
P1: 0 0 0 0 0 0 0 0 0 0 0 0
P2: 1.964088378906e+03 0.000000000000e+00 1.007396423340e+03 0.000000000000e+00 0.000000000000e+00 1.964088378906e+03 5.551860961914e+02 0.000000000000e+00 0 0 1 0
"""


def test_parse_kitti_labels_and_gt_distance() -> None:
    objects = parse_label_text(LABEL_TEXT)
    assert len(objects) == 2
    assert objects[0].category == "Pedestrian"
    assert objects[0].z_gt == 3.34
    assert objects[0].distance_gt > objects[0].z_gt


def test_parse_calib_p2_intrinsics() -> None:
    intrinsics = parse_calib_text(CALIB_TEXT)
    assert intrinsics.fx == pytest.approx(1964.088378906)
    assert intrinsics.fy == pytest.approx(1964.088378906)
    assert intrinsics.cx == pytest.approx(1007.39642334)
    assert intrinsics.cy == pytest.approx(555.1860961914)


def test_parse_known_local_pm_hmcw_zips_when_available() -> None:
    data_root = Path(r"C:\Users\jiahu\OneDrive\Desktop\PM-HMCW\Dataset")
    virtual_zip = data_root / "virtual" / "test.zip"
    real_zip = data_root / "real-world.zip"
    if not virtual_zip.exists() or not real_zip.exists():
        pytest.skip("Local PM-HMCW zips are not present")

    virtual_objects = parse_label_from_zip(virtual_zip, "label_2/000000.txt")
    real_objects = parse_label_from_zip(real_zip, "real-world/test/label_2/000248.txt")
    real_intrinsics = parse_calib_from_zip(real_zip, "real-world/test/calib/000248.txt")

    assert len(virtual_objects) == 6
    assert len(real_objects) == 2
    assert real_intrinsics.fx > 0

