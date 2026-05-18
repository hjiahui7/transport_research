from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .calibration import CalibratorBundle, regression_metrics, row_feature_vector, save_calibrator


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a lightweight PM-HMCW distance/depth calibrator.")
    parser.add_argument("--preds", required=True, help="CSV from human_detect.eval_pm_hmcw.")
    parser.add_argument("--out", required=True, help="Output joblib path.")
    parser.add_argument("--metrics-out", default=None, help="Optional metrics JSON path.")
    parser.add_argument("--model", default="best", choices=["best", "scale_bias", "linear", "ridge", "gbr", "mlp"], help="Model to save.")
    parser.add_argument("--val-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--group-column", default="image_id", help="Split train/val by this CSV column; empty for row-random split.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = _read_rows(args.preds)
    if len(rows) < 4:
        raise SystemExit(f"Need at least 4 matched rows to fit calibration, got {len(rows)}")

    x = np.asarray([row_feature_vector(row) for row in rows], dtype=np.float64)
    y_z = np.asarray([float(row["z_gt"]) for row in rows], dtype=np.float64)
    y_distance = np.asarray([float(row["distance_gt"]) for row in rows], dtype=np.float64)
    raw_z = np.asarray([float(row["z_depth_m"]) for row in rows], dtype=np.float64)
    raw_distance = np.asarray([float(row["distance_m"]) for row in rows], dtype=np.float64)

    train_idx, val_idx = _split_indices(rows, args.val_frac, args.seed, args.group_column)
    candidates = {
        "scale_bias": _fit_scale_bias_pair,
        "linear": _fit_linear_pair,
        "ridge": _fit_ridge_pair,
        "gbr": _fit_gbr_pair,
        "mlp": _fit_mlp_pair,
    }
    metrics: dict[str, dict] = {
        "raw": {
            "z": regression_metrics(y_z[val_idx], raw_z[val_idx]),
            "distance": regression_metrics(y_distance[val_idx], raw_distance[val_idx]),
            "n_train": int(train_idx.size),
            "n_val": int(val_idx.size),
        }
    }
    bundles: dict[str, CalibratorBundle] = {}

    for name, fit_fn in candidates.items():
        z_model, distance_model = fit_fn(x[train_idx], y_z[train_idx], y_distance[train_idx], args.seed)
        z_pred = z_model.predict(x[val_idx])
        distance_pred = distance_model.predict(x[val_idx])
        model_metrics = {
            "z": regression_metrics(y_z[val_idx], z_pred),
            "distance": regression_metrics(y_distance[val_idx], distance_pred),
            "n_train": int(train_idx.size),
            "n_val": int(val_idx.size),
        }
        metrics[name] = model_metrics
        bundles[name] = CalibratorBundle(z_model=z_model, distance_model=distance_model, model_type=name, metrics=model_metrics)

    if args.model == "best":
        selected = min(bundles, key=lambda name: metrics[name]["distance"]["mae"])
    else:
        selected = args.model
    # After choosing the model family on the held-out split, refit that family on all matched
    # rows so the saved calibrator uses every available PM-HMCW person sample.
    z_final, distance_final = candidates[selected](x, y_z, y_distance, args.seed)
    bundle = CalibratorBundle(
        z_model=z_final,
        distance_model=distance_final,
        model_type=selected,
        metrics=metrics[selected],
    )
    bundle.metrics = {"selected": selected, "all": metrics}
    save_calibrator(bundle, args.out)

    metrics_path = Path(args.metrics_out) if args.metrics_out else Path(args.out).with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(bundle.metrics, indent=2), encoding="utf-8")

    print(json.dumps(bundle.metrics, indent=2))
    print(f"saved {selected} calibrator to {args.out}")
    return 0


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row.get("z_depth_m") not in {"", None}
        and row.get("distance_m") not in {"", None}
        and row.get("z_gt") not in {"", None}
        and row.get("distance_gt") not in {"", None}
    ]


def _split_indices(rows: list[dict[str, str]], val_frac: float, seed: int, group_column: str) -> tuple[np.ndarray, np.ndarray]:
    count = len(rows)
    rng = np.random.default_rng(seed)
    if not group_column:
        indices = np.arange(count)
        rng.shuffle(indices)
        val_count = max(1, int(round(count * val_frac)))
        val_count = min(val_count, count - 1)
        return indices[val_count:], indices[:val_count]

    # Split by image_id by default so people from the same image do not leak across train/val.
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row.get(group_column, str(index)), []).append(index)

    group_keys = np.asarray(list(groups))
    rng.shuffle(group_keys)
    val_group_count = max(1, int(round(len(group_keys) * val_frac)))
    val_group_count = min(val_group_count, len(group_keys) - 1)
    val_groups = set(group_keys[:val_group_count])
    val_idx = np.asarray([idx for group in val_groups for idx in groups[group]], dtype=np.int64)
    train_idx = np.asarray([idx for group in group_keys[val_group_count:] for idx in groups[group]], dtype=np.int64)
    return train_idx, val_idx


def _fit_scale_bias_pair(x: np.ndarray, y_z: np.ndarray, y_distance: np.ndarray, seed: int):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    z_model = make_pipeline(ColumnTransformer([("raw_z", "passthrough", [0])]), LinearRegression())
    distance_model = make_pipeline(ColumnTransformer([("raw_distance", "passthrough", [1])]), LinearRegression())
    return z_model.fit(x, y_z), distance_model.fit(x, y_distance)


def _fit_linear_pair(x: np.ndarray, y_z: np.ndarray, y_distance: np.ndarray, seed: int):
    from sklearn.linear_model import LinearRegression

    return LinearRegression().fit(x, y_z), LinearRegression().fit(x, y_distance)


def _fit_ridge_pair(x: np.ndarray, y_z: np.ndarray, y_distance: np.ndarray, seed: int):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    z_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    distance_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return z_model.fit(x, y_z), distance_model.fit(x, y_distance)


def _fit_gbr_pair(x: np.ndarray, y_z: np.ndarray, y_distance: np.ndarray, seed: int):
    from sklearn.ensemble import GradientBoostingRegressor

    z_model = GradientBoostingRegressor(
        n_estimators=80,
        learning_rate=0.05,
        max_depth=2,
        random_state=seed,
        min_samples_leaf=2,
    ).fit(x, y_z)
    distance_model = GradientBoostingRegressor(
        n_estimators=80,
        learning_rate=0.05,
        max_depth=2,
        random_state=seed,
        min_samples_leaf=2,
    ).fit(x, y_distance)
    return z_model, distance_model


def _fit_mlp_pair(x: np.ndarray, y_z: np.ndarray, y_distance: np.ndarray, seed: int):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def make_model() -> object:
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                batch_size=min(64, max(8, x.shape[0])),
                learning_rate_init=1e-3,
                max_iter=2500,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=80,
                random_state=seed,
            ),
        )

    z_model = make_model().fit(x, y_z)
    distance_model = make_model().fit(x, y_distance)
    return z_model, distance_model


if __name__ == "__main__":
    raise SystemExit(main())
