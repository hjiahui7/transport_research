"""Generate publication figures from the Workzone v2 evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


ZONE_LABELS = ["Danger\n(<3 m)", "Caution\n(3-6 m)", "Lower-risk\n(>6 m)"]
ZONE_NAMES = ["Close", "Careful", "Safe"]
ZONE_COLORS = {"Close": "#C62828", "Careful": "#E6A700", "Safe": "#2E7D32"}
ZONE_DISPLAY = {"Close": "DANGER", "Careful": "CAUTION", "Safe": "LOWER-RISK"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def worker_key(row: dict[str, str]) -> tuple[str, int]:
    return row["image_id"], int(row["gt_worker_index"])


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, *, dpi: int = 300) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def confusion_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=int)
    index = {name: idx for idx, name in enumerate(ZONE_NAMES)}
    for row in rows:
        gt = row.get("gt_distance_band", "")
        pred = row.get("pred_distance_band", "")
        if gt in index and pred in index:
            matrix[index[gt], index[pred]] += 1
    return matrix


def draw_confusion_axis(axis: plt.Axes, matrix: np.ndarray, title: str):
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row_idx in range(3):
        for col_idx in range(3):
            value = normalized[row_idx, col_idx]
            text_color = "white" if value > 0.55 else "#17202A"
            axis.text(
                col_idx,
                row_idx,
                f"{matrix[row_idx, col_idx]}\n({value:.1%})",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold" if row_idx == col_idx else "normal",
            )
    for col_idx in (1, 2):
        axis.add_patch(Rectangle((col_idx - 0.49, -0.49), 0.98, 0.98, fill=False, edgecolor="#B71C1C", lw=2))
    accuracy = np.trace(matrix) / matrix.sum()
    danger_recall = matrix[0, 0] / matrix[0].sum()
    axis.set_title(f"{title}\nAccuracy {accuracy:.1%} | Danger recall {danger_recall:.1%}")
    axis.set_xticks(range(3), ZONE_LABELS)
    axis.set_yticks(range(3), ZONE_LABELS)
    axis.set_xlabel("Predicted zone")
    axis.set_ylabel("Ground-truth zone")
    axis.tick_params(length=0)
    return image


def plot_zone_confusions(
    calibrated_rows: list[dict[str, str]], direct_rows: list[dict[str, str]], output_dir: Path
) -> dict[str, object]:
    calibrated = {worker_key(row): row for row in calibrated_rows if row.get("gt_distance_m")}
    direct = {worker_key(row): row for row in direct_rows if row.get("gt_distance_m")}
    common = sorted(set(calibrated) & set(direct))
    matrices = [
        confusion_matrix([calibrated[key] for key in common]),
        confusion_matrix([direct[key] for key in common]),
    ]
    titles = ["Depth pipeline (MoGe + calibration)", "VLM direct distance estimation"]

    individual_dir = output_dir / "individual"
    stems = ["fig1a_depth_zone_confusion", "fig1b_vlm_direct_zone_confusion"]
    for matrix, title, stem in zip(matrices, titles, stems):
        single_fig, single_axis = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
        single_image = draw_confusion_axis(single_axis, matrix, title)
        single_colorbar = single_fig.colorbar(single_image, ax=single_axis, fraction=0.045, pad=0.04)
        single_colorbar.set_label("Row-normalized proportion")
        save_figure(single_fig, individual_dir, stem)

    return {
        "common_workers": len(common),
        "depth_matrix": matrices[0].tolist(),
        "vlm_direct_matrix": matrices[1].tolist(),
    }


def join_distance_rows(
    raw_rows: list[dict[str, str]],
    calibrated_rows: list[dict[str, str]],
    head_rows: list[dict[str, str]],
    direct_rows: list[dict[str, str]],
) -> list[dict[str, float | str | int]]:
    raw = {
        (Path(row["image_path"]).name, int(row["gt_id"]) + 1): row
        for row in raw_rows
        if row.get("distance_gt")
    }
    calibrated = {worker_key(row): row for row in calibrated_rows if row.get("gt_distance_m")}
    head = {
        (f"{row['image_id']}.png", int(row["gt_id"]) + 1): row
        for row in head_rows
        if row.get("gt_distance_m")
    }
    direct = {worker_key(row): row for row in direct_rows if row.get("gt_distance_m")}
    common = sorted(set(raw) & set(calibrated) & set(head) & set(direct))
    joined: list[dict[str, float | str | int]] = []
    for image_id, gt_index in common:
        joined.append(
            {
                "image_id": image_id,
                "gt_index": gt_index,
                "gt": float(raw[(image_id, gt_index)]["distance_gt"]),
                "raw": float(raw[(image_id, gt_index)]["distance_m"]),
                "calibrated": float(calibrated[(image_id, gt_index)]["pred_distance_m"]),
                "head": float(head[(image_id, gt_index)]["pred_distance_m"]),
                "vlm_direct": float(direct[(image_id, gt_index)]["pred_distance_m"]),
            }
        )
    return joined


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


def draw_distance_scatter(
    axis: plt.Axes,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    title: str,
    color: str,
    axis_max: float,
) -> dict[str, float | int]:
    clipped = prediction > axis_max
    axis.scatter(ground_truth[~clipped], prediction[~clipped], s=15, alpha=0.55, color=color, edgecolors="none")
    if clipped.any():
        axis.scatter(ground_truth[clipped], np.full(clipped.sum(), axis_max), marker="^", s=28, color=color)
        axis.text(0.03, 0.95, f"{clipped.sum()} points clipped at {axis_max:.0f} m", transform=axis.transAxes, va="top")
    axis.plot([0, axis_max], [0, axis_max], linestyle="--", color="#333333", lw=1, label="Ideal (y=x)")
    for threshold in (3, 6):
        axis.axvline(threshold, color="#8A8A8A", linestyle=":", lw=0.9)
        axis.axhline(threshold, color="#8A8A8A", linestyle=":", lw=0.9)
    errors = prediction - ground_truth
    axis.set_title(f"{title}\nMAE {np.mean(np.abs(errors)):.3f} m | Bias {np.mean(errors):+.3f} m")
    axis.set_xlabel("Ground-truth distance (m)")
    axis.set_ylabel("Predicted distance (m)")
    axis.set_xlim(0, axis_max)
    axis.set_ylim(0, axis_max)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18)
    return {
        "n": len(errors),
        "mae_m": float(np.mean(np.abs(errors))),
        "bias_m": float(np.mean(errors)),
    }


def draw_error_cdf(
    axis: plt.Axes,
    joined: list[dict[str, float | str | int]],
    ground_truth: np.ndarray,
    methods: list[tuple[str, str, str, str]],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for label, key, color, linestyle in methods:
        prediction = np.array([float(row[key]) for row in joined])
        error = np.abs(prediction - ground_truth)
        x, y = empirical_cdf(error)
        axis.plot(x, y, label=label, color=color, linestyle=linestyle, lw=2)
        metrics[key] = {
            "mae_m": float(error.mean()),
            "within_0_5m": float((error <= 0.5).mean()),
            "within_1_0m": float((error <= 1.0).mean()),
        }
    axis.axvline(0.5, color="#8A8A8A", linestyle=":", lw=0.9)
    axis.axvline(1.0, color="#8A8A8A", linestyle=":", lw=0.9)
    axis.set_xlim(0, 3.0)
    axis.set_ylim(0, 1.01)
    axis.set_xlabel("Absolute distance error (m)")
    axis.set_ylabel("Cumulative proportion")
    axis.set_title(f"Absolute-error distribution (n={len(joined)})")
    axis.grid(alpha=0.18)
    axis.legend(loc="lower right", fontsize=8)
    return metrics


def plot_calibration_and_cdf(
    raw_rows: list[dict[str, str]],
    calibrated_rows: list[dict[str, str]],
    joined: list[dict[str, float | str | int]],
    output_dir: Path,
) -> dict[str, object]:
    raw_gt = np.array([float(row["distance_gt"]) for row in raw_rows if row.get("distance_gt")])
    raw_prediction = np.array([float(row["distance_m"]) for row in raw_rows if row.get("distance_gt")])
    calibrated_valid = [row for row in calibrated_rows if row.get("gt_distance_m")]
    calibrated_gt = np.array([float(row["gt_distance_m"]) for row in calibrated_valid])
    calibrated_prediction = np.array([float(row["pred_distance_m"]) for row in calibrated_valid])
    cdf_gt = np.array([float(row["gt"]) for row in joined])
    axis_max = max(20.0, float(np.ceil(max(raw_gt.max(), calibrated_gt.max()))))

    scatter_specs = [
        (raw_gt, raw_prediction, "Raw MoGe", "#D55E00"),
        (calibrated_gt, calibrated_prediction, "MoGe + MLP calibration", "#0072B2"),
    ]
    methods = [
        ("Distance head", "head", "#009E73", "-."),
        ("Raw MoGe", "raw", "#D55E00", ":"),
        ("MoGe + calibration", "calibrated", "#0072B2", "-"),
        ("VLM direct", "vlm_direct", "#CC79A7", "--"),
    ]
    individual_dir = output_dir / "individual"
    scatter_stems = ["fig2a_raw_moge_scatter", "fig2b_calibrated_scatter"]
    scatter_metrics: dict[str, dict[str, float | int]] = {}
    for (scatter_gt, prediction, title, color), stem in zip(scatter_specs, scatter_stems):
        single_fig, single_axis = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
        scatter_metrics[title] = draw_distance_scatter(single_axis, scatter_gt, prediction, title, color, axis_max)
        save_figure(single_fig, individual_dir, stem)
    cdf_fig, cdf_single_axis = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    cdf_metrics = draw_error_cdf(cdf_single_axis, joined, cdf_gt, methods)
    save_figure(cdf_fig, individual_dir, "fig2c_absolute_error_cdf")
    return {"scatter": scatter_metrics, "cdf_common_workers": len(joined), "methods": cdf_metrics}


def truthy_label(value: str) -> str:
    lower = value.strip().lower()
    if lower == "true":
        return "yes"
    if lower == "false":
        return "no"
    return value


def plot_qualitative(
    raw_rows: list[dict[str, str]], calibrated_rows: list[dict[str, str]], output_dir: Path
) -> dict[str, object]:
    raw_by_prediction = defaultdict(list)
    for row in raw_rows:
        raw_by_prediction[Path(row["image_path"]).name].append(row)
    calibrated_by_prediction = defaultdict(list)
    for row in calibrated_rows:
        calibrated_by_prediction[row["image_id"]].append(row)

    cases = [
        ("Garage4_000144.png", "Near-field workers"),
        ("Pave1_001686.png", "Partial occlusion"),
        ("Garage1_002064.png", "Calibration correction"),
        ("Garage4_001530.png", "Multiple workers and zones"),
    ]
    fig = plt.figure(figsize=(11.5, 9.0), constrained_layout=True)
    outer = fig.add_gridspec(2, 2)
    panel_letters = ["(a)", "(b)", "(c)", "(d)"]
    case_details: list[dict[str, object]] = []

    for panel_spec, (image_id, title), panel_letter in zip(outer, cases, panel_letters):
        panel = panel_spec.subgridspec(2, 1, height_ratios=(1.0, 0.18), hspace=0.02)
        axis = fig.add_subplot(panel[0])
        footer = fig.add_subplot(panel[1])
        raw_image_rows = raw_by_prediction[image_id]
        if not raw_image_rows:
            raise ValueError(f"No matched raw rows for qualitative image: {image_id}")
        image_path = Path(raw_image_rows[0]["image_path"])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        axis.imshow(image)
        calibrated_map = {
            int(row["pred_worker_index"]): row for row in calibrated_by_prediction[image_id]
        }
        workers = []
        for raw_row in sorted(raw_image_rows, key=lambda item: int(item["pred_id"])):
            pred_index = int(raw_row["pred_id"]) + 1
            result_index = pred_index
            if image_id == "Pave1_001686.png":
                result_index = {1: 2, 2: 1}.get(pred_index, pred_index)
            elif image_id == "Garage4_001530.png":
                result_index = {2: 3, 3: 2}.get(pred_index, pred_index)
            calibrated_row = calibrated_map.get(result_index)
            if calibrated_row is None:
                continue
            center_x = float(raw_row["center_x_norm"]) * width
            center_y = float(raw_row["center_y_norm"]) * height
            bbox_width = float(raw_row["bbox_width_norm"]) * width
            bbox_height = float(raw_row["bbox_height_norm"]) * height
            x1, y1 = center_x - bbox_width / 2, center_y - bbox_height / 2
            zone = calibrated_row["pred_distance_band"]
            color = ZONE_COLORS[zone]
            axis.add_patch(Rectangle((x1, y1), bbox_width, bbox_height, fill=False, edgecolor=color, lw=2.5))
            distance = float(calibrated_row["pred_distance_m"])
            label_x = min(max(2, x1), width - 115)
            label_y = min(max(5, y1 + 5), height - 25)
            if image_id == "Pave1_001686.png":
                if pred_index == 1:
                    label_x = min(width - 115, x1 + bbox_width + 8)
                elif pred_index == 2:
                    label_x = max(2, x1 - 105)
            axis.text(
                label_x,
                label_y,
                f"W{pred_index}  {distance:.1f} m",
                color="white",
                fontsize=7.5,
                va="top",
                bbox={"facecolor": color, "alpha": 0.94, "edgecolor": "none", "pad": 2.0},
            )
            workers.append(
                {
                    "worker_index": pred_index,
                    "result_worker_index": result_index,
                    "distance_m": distance,
                    "zone": zone,
                    "gt_distance_m": float(calibrated_row["gt_distance_m"]),
                    "raw_moge_m": float(raw_row["distance_m"]),
                }
            )
        axis.set_title(f"{panel_letter} {title}", loc="left", fontweight="bold")
        axis.axis("off")
        footer.axis("off")
        footer_lines = []
        for worker in workers:
            pred_index = int(worker["worker_index"])
            calibrated_row = calibrated_map[int(worker["result_worker_index"])]
            prefix = f"W{pred_index}: {float(worker['distance_m']):.1f} m {ZONE_DISPLAY[str(worker['zone'])]}"
            if image_id == "Garage1_002064.png":
                prefix = (
                    f"W{pred_index}: raw {float(worker['raw_moge_m']):.1f} -> "
                    f"calibrated {float(worker['distance_m']):.1f} m (GT {float(worker['gt_distance_m']):.1f} m)"
                )
            attrs = (
                f"vest {truthy_label(calibrated_row['pred_high_visibility_vest'])}, "
                f"helmet {calibrated_row['pred_helmet_status']}, "
                f"{calibrated_row['pred_orientation']}, {calibrated_row['pred_occlusion_level']}"
            )
            footer_lines.append(f"{prefix} | {attrs}")
        footer.text(0.0, 0.98, "\n".join(footer_lines), ha="left", va="top", fontsize=7.3, linespacing=1.25)
        case_details.append({"image_id": image_id, "workers": workers})
    save_figure(fig, output_dir, "fig3_qualitative_pipeline", dpi=220)

    composite_path = output_dir / "fig3_qualitative_pipeline.png"
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)
    stems = [
        "fig3a_near_field_workers",
        "fig3b_partial_occlusion",
        "fig3c_calibration_correction",
        "fig3d_multiple_workers_and_zones",
    ]
    with Image.open(composite_path) as composite:
        width, height = composite.size
        boxes = [
            (0, 0, width // 2, height // 2),
            (width // 2, 0, width, height // 2),
            (0, height // 2, width // 2, height),
            (width // 2, height // 2, width, height),
        ]
        for box, stem in zip(boxes, stems):
            panel_image = composite.crop(box).convert("RGB")
            panel_image.save(individual_dir / f"{stem}.png", dpi=(220, 220), optimize=True)
            panel_fig, panel_axis = plt.subplots(
                figsize=(panel_image.width / 220, panel_image.height / 220),
                constrained_layout=True,
            )
            panel_axis.imshow(panel_image)
            panel_axis.axis("off")
            panel_fig.savefig(individual_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0, facecolor="white")
            plt.close(panel_fig)
    composite_path.unlink()
    (output_dir / "fig3_qualitative_pipeline.pdf").unlink()
    return {"cases": case_details}


def plot_vlm_attributes(model_paths: list[tuple[str, Path]], output_dir: Path) -> dict[str, object]:
    model_rows = [(name, {worker_key(row): row for row in read_csv(path)}) for name, path in model_paths]
    common = set.intersection(*(set(rows) for _, rows in model_rows))
    attributes = [
        ("Vest", "high_visibility_vest"),
        ("Helmet", "helmet_status"),
        ("Orientation", "orientation"),
        ("Occlusion", "occlusion_level"),
    ]
    accuracies = np.zeros((len(model_rows), len(attributes)), dtype=float)
    denominators = np.zeros_like(accuracies, dtype=int)
    for model_idx, (_, rows) in enumerate(model_rows):
        for attr_idx, (_, attr) in enumerate(attributes):
            valid = [
                rows[key]
                for key in common
                if rows[key].get(f"gt_{attr}") not in {None, "", "uncertain"}
            ]
            denominators[model_idx, attr_idx] = len(valid)
            accuracies[model_idx, attr_idx] = np.mean(
                [row.get(f"pred_{attr}") == row.get(f"gt_{attr}") for row in valid]
            )

    fig, axis = plt.subplots(figsize=(9.2, 3.9), constrained_layout=True)
    x = np.arange(len(attributes))
    width = 0.19
    colors = ["#0072B2", "#56B4E9", "#D55E00", "#009E73"]
    hatches = [None, "//", "..", "xx"]
    for model_idx, ((name, _), color, hatch) in enumerate(zip(model_paths, colors, hatches)):
        offset = (model_idx - (len(model_paths) - 1) / 2) * width
        bars = axis.bar(x + offset, accuracies[model_idx] * 100, width, label=name, color=color, hatch=hatch)
        for bar, value in zip(bars, accuracies[model_idx]):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{value:.1%}", ha="center", va="bottom", fontsize=7, rotation=90)
    axis.set_xticks(x, [label for label, _ in attributes])
    axis.set_ylim(0, 111)
    axis.set_ylabel("Accuracy (%)")
    axis.set_title(f"VLM attribute accuracy on common matched workers (n={len(common)})", pad=32)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.10), fontsize=8)
    individual_dir = output_dir / "individual"
    save_figure(fig, individual_dir, "fig4_vlm_attribute_accuracy")
    return {
        "common_workers": len(common),
        "models": {
            name: {
                attr_label: {
                    "accuracy": float(accuracies[model_idx, attr_idx]),
                    "n": int(denominators[model_idx, attr_idx]),
                }
                for attr_idx, (attr_label, _) in enumerate(attributes)
            }
            for model_idx, (name, _) in enumerate(model_paths)
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/workzone_v2/figures"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve()
    results = root / "artifacts/workzone_v2/results"

    calibrated_rows = read_csv(results / "qwen3_6_flash_eval250_moge_calibrated_per_worker.csv")
    direct_rows = read_csv(results / "qwen3_6_flash_vlm_direct_distance_eval250_cached_per_worker.csv")
    raw_rows = read_csv(results / "scheme2_moge_raw_eval_predictions.csv")
    head_rows = read_csv(results / "scheme1_finetuned_yolo_distance_eval_matches.csv")
    joined = join_distance_rows(raw_rows, calibrated_rows, head_rows, direct_rows)

    set_paper_style()
    metadata = {
        "zone_confusion": plot_zone_confusions(calibrated_rows, direct_rows, output_dir),
        "distance_analysis": plot_calibration_and_cdf(raw_rows, calibrated_rows, joined, output_dir),
        "qualitative": plot_qualitative(raw_rows, calibrated_rows, output_dir),
        "vlm_attributes": plot_vlm_attributes(
            [
                ("Qwen3.6-Flash (cached)", results / "qwen3_6_flash_eval250_cached_per_worker.csv"),
                ("Qwen3-VL-Flash", results / "qwen3_vl_flash_eval250_cached_per_worker.csv"),
                ("Qwen2.5-VL-3B", results / "qwen2_5_vl_3b_local_eval250_cached_per_worker.csv"),
                ("Qwen3.6-Flash (v2)", results / "qwen3_6_flash_eval250_moge_calibrated_per_worker.csv"),
            ],
            output_dir,
        ),
    }
    (output_dir / "figure_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote paper figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
