from __future__ import annotations

from math import sqrt
from typing import Iterable

import numpy as np


def distance_band(distance_m: float) -> str:
    if distance_m < 3.0:
        return "<3"
    if distance_m <= 5.0:
        return "3-5"
    return ">5"


def summarize_distance_errors(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    *,
    total_gt: int | None = None,
) -> dict[str, float | int]:
    true = np.asarray(list(y_true), dtype=np.float64)
    pred = np.asarray(list(y_pred), dtype=np.float64)
    if true.size == 0:
        return {
            "matched": 0,
            "total_gt": int(total_gt or 0),
            "match_rate": 0.0,
            "mae_m": 0.0,
            "rmse_m": 0.0,
            "bias_m": 0.0,
            "absrel": 0.0,
            "within_0_5m": 0.0,
            "within_1_0m": 0.0,
            "within_0_5m_all_gt": 0.0,
            "within_1_0m_all_gt": 0.0,
            "band_accuracy": 0.0,
        }

    errors = pred - true
    abs_errors = np.abs(errors)
    matched = int(true.size)
    gt_count = int(total_gt if total_gt is not None else matched)
    gt_count = max(gt_count, matched)
    within_05_count = int(np.sum(abs_errors <= 0.5))
    within_10_count = int(np.sum(abs_errors <= 1.0))
    band_correct = sum(distance_band(float(t)) == distance_band(float(p)) for t, p in zip(true, pred))
    return {
        "matched": matched,
        "total_gt": gt_count,
        "match_rate": matched / gt_count if gt_count else 0.0,
        "mae_m": float(np.mean(abs_errors)),
        "rmse_m": float(sqrt(float(np.mean(errors * errors)))),
        "bias_m": float(np.mean(errors)),
        "absrel": float(np.mean(abs_errors / np.maximum(np.abs(true), 1e-6))),
        "within_0_5m": within_05_count / matched,
        "within_1_0m": within_10_count / matched,
        "within_0_5m_all_gt": within_05_count / gt_count if gt_count else 0.0,
        "within_1_0m_all_gt": within_10_count / gt_count if gt_count else 0.0,
        "band_accuracy": band_correct / matched,
    }
