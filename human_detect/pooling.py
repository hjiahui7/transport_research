from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import finite_positive_median


@dataclass(frozen=True)
class PooledDepth:
    z_depth_m: float | None
    centroid_uv: tuple[float, float] | None
    mask_area_px: int
    used_region: str
    depth_stats: dict[str, float | None]


def pool_person_depth(mask: np.ndarray, depth: np.ndarray, *, min_pixels: int = 50) -> PooledDepth:
    if mask.shape != depth.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match depth shape {depth.shape}")

    mask_bool = mask.astype(bool)
    ys, xs = np.nonzero(mask_bool)
    mask_area = int(xs.size)
    if mask_area == 0:
        return PooledDepth(z_depth_m=None, centroid_uv=None, mask_area_px=0, used_region="empty", depth_stats=_empty_stats())

    centroid = (float(xs.mean()), float(ys.mean()))
    y_min = int(ys.min())
    y_max = int(ys.max())
    # Lower-mask depth is usually closer to the person's supporting body/feet region and is
    # less affected by helmet/upper-body shape or background pixels leaking through the mask.
    lower_cutoff = y_min + int(round((y_max - y_min + 1) * 0.4))
    lower_mask = mask_bool.copy()
    lower_mask[:lower_cutoff, :] = False

    lower_values = depth[lower_mask]
    full_stats = _depth_stats(depth[mask_bool], prefix="depth")
    lower_stats = _depth_stats(lower_values, prefix="lower_depth")
    stats = {**full_stats, **lower_stats}
    if np.count_nonzero(np.isfinite(lower_values) & (lower_values > 0.0)) >= min_pixels:
        pooled = finite_positive_median(lower_values)
        return PooledDepth(z_depth_m=pooled, centroid_uv=centroid, mask_area_px=mask_area, used_region="lower_60", depth_stats=stats)

    pooled = finite_positive_median(depth[mask_bool])
    return PooledDepth(z_depth_m=pooled, centroid_uv=centroid, mask_area_px=mask_area, used_region="full_mask", depth_stats=stats)


def _depth_stats(values: np.ndarray, *, prefix: str) -> dict[str, float | None]:
    # Quantiles give the correction head a compact signal about depth spread and outliers.
    valid = values[np.isfinite(values) & (values > 0.0)]
    keys = ["p10", "p25", "p50", "p75", "p90"]
    if valid.size == 0:
        return {f"{prefix}_{key}_m": None for key in keys}
    quantiles = np.percentile(valid, [10, 25, 50, 75, 90])
    return {f"{prefix}_{key}_m": float(value) for key, value in zip(keys, quantiles)}


def _empty_stats() -> dict[str, float | None]:
    return {
        **{f"depth_{key}_m": None for key in ["p10", "p25", "p50", "p75", "p90"]},
        **{f"lower_depth_{key}_m": None for key in ["p10", "p25", "p50", "p75", "p90"]},
    }
