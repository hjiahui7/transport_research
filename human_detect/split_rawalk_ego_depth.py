from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split Rawalk ego distance-label CSV into fixed train/eval CSVs.")
    parser.add_argument("--labels", required=True, help="CSV from human_detect.rawalk_ego_depth.")
    parser.add_argument("--train-out", required=True, help="Output train CSV path.")
    parser.add_argument("--eval-out", required=True, help="Output eval CSV path.")
    parser.add_argument("--summary-out", default=None, help="Optional split summary JSON path.")
    parser.add_argument("--eval-fraction", type=float, default=0.2, help="Fraction of groups reserved for eval.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic shuffle seed.")
    parser.add_argument("--distance-column", default="distance_m", help="Distance column used by optional min/max filters.")
    parser.add_argument("--min-distance", type=float, default=None, help="Drop rows below this distance in meters.")
    parser.add_argument("--max-distance", type=float, default=None, help="Drop rows above this distance in meters.")
    parser.add_argument(
        "--group-column",
        default="image_path",
        help="CSV column used for grouping. Use image_path to keep all people from one image in the same split.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    fieldnames, rows = read_csv_rows(args.labels)
    original_rows = len(rows)
    rows = filter_rows_by_distance(
        rows,
        distance_column=args.distance_column,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
    )
    train_rows, eval_rows, summary = split_rows(
        rows,
        group_column=args.group_column,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    write_csv_rows(args.train_out, fieldnames, train_rows)
    write_csv_rows(args.eval_out, fieldnames, eval_rows)

    summary.update(
        {
            "labels": str(Path(args.labels)),
            "train_out": str(Path(args.train_out)),
            "eval_out": str(Path(args.eval_out)),
            "original_rows": original_rows,
            "filtered_rows": original_rows - len(rows),
            "distance_column": args.distance_column,
            "min_distance": args.min_distance,
            "max_distance": args.max_distance,
        }
    )
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


def read_csv_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv_rows(path: str | Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_rows_by_distance(
    rows: list[dict[str, str]],
    *,
    distance_column: str,
    min_distance: float | None = None,
    max_distance: float | None = None,
) -> list[dict[str, str]]:
    if min_distance is None and max_distance is None:
        return rows
    kept: list[dict[str, str]] = []
    for row in rows:
        try:
            distance = float(row[distance_column])
        except (KeyError, TypeError, ValueError):
            continue
        if min_distance is not None and distance < min_distance:
            continue
        if max_distance is not None and distance > max_distance:
            continue
        kept.append(row)
    return kept


def split_rows(
    rows: list[dict[str, str]],
    *,
    group_column: str = "image_path",
    eval_fraction: float = 0.2,
    seed: int = 7,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot split an empty CSV")
    if group_column not in rows[0]:
        raise ValueError(f"Missing group column {group_column!r}")

    group_keys = sorted({row[group_column] for row in rows})
    if len(group_keys) < 2:
        raise ValueError(f"Need at least 2 {group_column} groups, got {len(group_keys)}")

    shuffled = group_keys[:]
    random.Random(seed).shuffle(shuffled)
    eval_count = _eval_group_count(len(shuffled), eval_fraction)
    eval_keys = set(shuffled[:eval_count])
    train_keys = set(shuffled[eval_count:])

    train_rows = [row for row in rows if row[group_column] in train_keys]
    eval_rows = [row for row in rows if row[group_column] in eval_keys]
    summary = {
        "group_column": group_column,
        "seed": seed,
        "eval_fraction": eval_fraction,
        "train_groups": len(train_keys),
        "eval_groups": len(eval_keys),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
    }
    return train_rows, eval_rows, summary


def _eval_group_count(num_groups: int, eval_fraction: float) -> int:
    clamped_fraction = max(0.0, min(0.9, eval_fraction))
    return min(num_groups - 1, max(1, round(num_groups * clamped_fraction)))


if __name__ == "__main__":
    raise SystemExit(main())
