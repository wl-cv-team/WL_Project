"""Draw YOLO boxes on images and tile them into contact sheets."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

Box = tuple[int, float, float, float, float]


def draw_yolo_boxes(image: np.ndarray, boxes: Sequence[Box], color=(0, 255, 0)) -> np.ndarray:
    """Return a copy of a BGR image with normalized YOLO boxes drawn as rectangles."""
    out = image.copy()
    h, w = out.shape[:2]
    for _, cx, cy, bw, bh in boxes:
        x1, y1 = round((cx - bw / 2) * w), round((cy - bh / 2) * h)
        x2, y2 = round((cx + bw / 2) * w), round((cy + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return out


def tile_images(
    images: Sequence[np.ndarray], columns: int, cell: tuple[int, int] = (320, 240)
) -> np.ndarray:
    """Resize each image to `cell` (width, height) and place them row by row on a black canvas."""
    if columns < 1:
        raise ValueError("columns must be >= 1")
    cw, ch = cell
    rows = max(1, -(-len(images) // columns))
    sheet = np.zeros((rows * ch, columns * cw, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, columns)
        sheet[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw] = cv2.resize(img, (cw, ch))
    return sheet


def label_image(image: np.ndarray, text: str) -> np.ndarray:
    """Return a copy with `text` written at the top-left corner."""
    out = image.copy()
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return out
