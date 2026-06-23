from __future__ import annotations

import csv
from pathlib import Path

import pytest

from human_detect.split_rawalk_ego_depth import filter_rows_by_distance, split_rows, write_csv_rows
from human_detect.train_yolo_distance_head import split_samples


def test_split_rows_keeps_people_from_same_image_together() -> None:
    rows = [
        {"image_path": "a.jpg", "human_name": "p1"},
        {"image_path": "a.jpg", "human_name": "p2"},
        {"image_path": "b.jpg", "human_name": "p3"},
        {"image_path": "c.jpg", "human_name": "p4"},
    ]

    train_rows, eval_rows, summary = split_rows(rows, eval_fraction=0.34, seed=3)

    train_images = {row["image_path"] for row in train_rows}
    eval_images = {row["image_path"] for row in eval_rows}
    assert train_images.isdisjoint(eval_images)
    assert train_images | eval_images == {"a.jpg", "b.jpg", "c.jpg"}
    assert summary["train_groups"] == 2
    assert summary["eval_groups"] == 1


def test_split_rows_requires_at_least_two_groups() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        split_rows([{"image_path": "a.jpg"}])


def test_filter_rows_by_distance_drops_unusable_outliers() -> None:
    rows = [
        {"image_path": "a.jpg", "distance_m": "0.1"},
        {"image_path": "b.jpg", "distance_m": "3.0"},
        {"image_path": "c.jpg", "distance_m": "200000.0"},
        {"image_path": "d.jpg", "distance_m": ""},
    ]

    kept = filter_rows_by_distance(rows, distance_column="distance_m", min_distance=0.2, max_distance=20.0)

    assert kept == [{"image_path": "b.jpg", "distance_m": "3.0"}]


def test_write_csv_rows_preserves_header(tmp_path: Path) -> None:
    out_path = tmp_path / "labels.csv"

    write_csv_rows(out_path, ["image_path", "distance_m"], [{"image_path": "a.jpg", "distance_m": "2.0"}])

    with out_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["image_path", "distance_m"]
        assert list(reader) == [{"image_path": "a.jpg", "distance_m": "2.0"}]


def test_random_split_never_empties_train_set() -> None:
    samples = [object(), object()]

    train_samples, val_samples = split_samples(samples, val_fraction=0.9, seed=7)  # type: ignore[arg-type]

    assert len(train_samples) == 1
    assert len(val_samples) == 1
