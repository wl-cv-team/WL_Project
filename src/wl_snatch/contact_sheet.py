"""Render images with YOLO boxes into fixed-size cells and tile them into contact sheets.

Each image is letterboxed into the cell first (aspect ratio kept, padded with
black), then boxes and the label are drawn at cell scale, so only cell-sized
arrays need to be kept in memory.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

Box = tuple[int, float, float, float, float]
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def letterbox(image: np.ndarray, cell: tuple[int, int]) -> tuple[np.ndarray, float, int, int]:
    """Fit an image into `cell` (width, height) keeping aspect ratio.

    Returns (cell_image, scale, offset_x, offset_y); the image content occupies
    the centered region starting at (offset_x, offset_y), the rest is black.
    """
    cw, ch = cell
    h, w = image.shape[:2]
    scale = min(cw / w, ch / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((ch, cw, 3), dtype=np.uint8)
    ox, oy = (cw - nw) // 2, (ch - nh) // 2
    out[oy : oy + nh, ox : ox + nw] = resized
    return out, scale, ox, oy


def _draw_label(img: np.ndarray, text: str) -> None:
    """Write text at the top-left, shrinking the font and wrapping into two lines to fit the width."""
    if not text:
        return
    width = img.shape[1] - 8
    base = max(1, round(min(img.shape[:2]) / 270))
    for lines in ([text], [text[: (len(text) + 1) // 2], text[(len(text) + 1) // 2 :]]):
        scale = 0.6
        while scale > 0.25 and max(cv2.getTextSize(s, _FONT, scale, base)[0][0] for s in lines) > width:
            scale -= 0.05
        if max(cv2.getTextSize(s, _FONT, scale, base)[0][0] for s in lines) <= width:
            break
    line_h = cv2.getTextSize("Ag", _FONT, scale, base)[0][1] + 6
    for i, s in enumerate(lines):
        cv2.putText(img, s, (4, line_h * (i + 1) - 2), _FONT, scale, TEXT_COLOR, base, cv2.LINE_AA)


def render_cell(
    image: np.ndarray, boxes: Sequence[Box], cell: tuple[int, int], text: str
) -> np.ndarray:
    """Letterbox `image` into `cell`, draw normalized YOLO boxes (outline only) and a text label."""
    out, scale, ox, oy = letterbox(image, cell)
    h, w = image.shape[:2]
    thickness = max(1, round(min(cell) / 135))
    for _, cx, cy, bw, bh in boxes:
        x1 = ox + round((cx - bw / 2) * w * scale)
        y1 = oy + round((cy - bh / 2) * h * scale)
        x2 = ox + round((cx + bw / 2) * w * scale)
        y2 = oy + round((cy + bh / 2) * h * scale)
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, thickness)
    _draw_label(out, text)
    return out


def tile_images(
    images: Sequence[np.ndarray], columns: int, cell: tuple[int, int] = (270, 480)
) -> np.ndarray:
    """Place cell-sized images (width, height = `cell`) row by row on a black canvas.

    Raises ValueError for columns < 1 or an image whose size is not the cell size
    (images are never resized here, so nothing is stretched).
    """
    if columns < 1:
        raise ValueError("columns must be >= 1")
    cw, ch = cell
    rows = max(1, -(-len(images) // columns))
    sheet = np.zeros((rows * ch, columns * cw, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        if img.shape[:2] != (ch, cw):
            raise ValueError(f"image {i} has size {img.shape[1]}x{img.shape[0]}, expected {cw}x{ch}")
        r, c = divmod(i, columns)
        sheet[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw] = img
    return sheet
