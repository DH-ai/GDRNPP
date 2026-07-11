"""Drawing utilities for the high-level YOLOX prediction API."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


def _color(class_id: int) -> Tuple[int, int, int]:
    # Stable, well-separated BGR colors without global random state.
    hue = int((class_id * 47) % 180)
    pixel = np.uint8([[[hue, 210, 255]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_detections(
    image: np.ndarray,
    detections: Sequence[Dict[str, Any]],
    *,
    class_names: Optional[Sequence[str]] = None,
    score_thres: float = 0.0,
    box_thickness: int = 2,
    font_scale: float = 0.5,
    show_labels: bool = True,
    show_scores: bool = True,
) -> np.ndarray:
    """Return a copy of ``image`` with original-coordinate boxes drawn."""
    output = image.copy()
    for detection in detections:
        score = float(detection["score"])
        if score < score_thres:
            continue
        class_id = int(detection["class_id"])
        x1, y1, x2, y2 = (int(round(value)) for value in detection["bbox"])
        color = _color(class_id)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)
        if not (show_labels or show_scores):
            continue
        name = detection.get("class_name")
        if name is None and class_names is not None and 0 <= class_id < len(class_names):
            name = class_names[class_id]
        label_parts = [str(name if name is not None else class_id)] if show_labels else []
        if show_scores:
            label_parts.append(f"{score:.2f}")
        label = " ".join(label_parts)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_y = max(text_h + baseline, y1)
        cv2.rectangle(output, (x1, text_y - text_h - baseline), (x1 + text_w + 2, text_y + baseline), color, -1)
        cv2.putText(
            output,
            label,
            (x1 + 1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return output


__all__ = ["draw_detections"]
