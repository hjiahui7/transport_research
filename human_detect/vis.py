from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def save_visualization(image_path: str | Path, persons: list[dict[str, Any]], masks: list[np.ndarray], out_path: str | Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)

    overlay = image.copy()
    colors = _colors(len(persons))
    for person, mask, color in zip(persons, masks, colors):
        overlay[mask.astype(bool)] = (0.55 * overlay[mask.astype(bool)] + 0.45 * np.array(color)).astype(np.uint8)
        x1, y1, x2, y2 = [int(round(v)) for v in person["bbox_xyxy"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        distance = person.get("distance_m")
        yaw = person.get("bearing_yaw_deg")
        if distance is None:
            text = f"#{person['id']} depth n/a"
        elif yaw is None:
            text = f"#{person['id']} {distance:.2f}m"
        else:
            text = f"#{person['id']} {distance:.2f}m yaw {yaw:.1f}"
        y_text = max(18, y1 - 8)
        cv2.putText(overlay, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def _colors(count: int) -> list[tuple[int, int, int]]:
    palette = [
        (46, 204, 113),
        (52, 152, 219),
        (241, 196, 15),
        (231, 76, 60),
        (155, 89, 182),
        (26, 188, 156),
    ]
    return [palette[i % len(palette)] for i in range(count)]
