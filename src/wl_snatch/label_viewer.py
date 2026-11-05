"""Draw label boxes on a frame held in memory.

Functions here return arrays. Showing an array on screen is done by the
notebook, and no function in this module writes an image file (issue #11:
the videos show real people's faces).

Only boxes are handled. Joint points and phase marks are added to the same
structure once their formats are fixed (#8, #9).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from wl_snatch.frames import read_frame

NormBox = tuple[float, float, float, float]  # cx, cy, w, h, normalized to 0..1
BOX_COLOR_BGR = (0, 255, 0)
BOX_THICKNESS = 2


@dataclass(frozen=True)
class Overlay:
    """Boxes to draw, plus reasons for entries left out because their coordinates are N/A."""

    boxes: tuple[NormBox, ...]
    skipped: tuple[str, ...]


def flag_box_issues(
    boxes: tuple[NormBox, ...] | list[NormBox],
    min_side: float = 0.02,
    max_side: float = 0.9,
) -> list[str]:
    """Descriptions of boxes with a side below `min_side` or above `max_side` (normalized units)."""
    issues: list[str] = []
    for number, (cx, cy, w, h) in enumerate(boxes):
        reasons = []
        if min(w, h) < min_side:
            reasons.append(f"a side is below {min_side:g}")
        if max(w, h) > max_side:
            reasons.append(f"a side is above {max_side:g}")
        if reasons:
            issues.append(f"box {number}: w={w:.4g}, h={h:.4g}: " + "; ".join(reasons))
    return issues


def draw_overlay(frame: np.ndarray, overlay: Overlay) -> np.ndarray:
    """Copy of `frame` with the boxes drawn. The input array is not changed."""
    result = frame.copy()
    height, width = result.shape[:2]
    for cx, cy, w, h in overlay.boxes:
        x1 = round((cx - w / 2) * width)
        y1 = round((cy - h / 2) * height)
        x2 = round((cx + w / 2) * width)
        y2 = round((cy + h / 2) * height)
        cv2.rectangle(result, (x1, y1), (x2, y2), BOX_COLOR_BGR, BOX_THICKNESS)
    return result


def render_label_frame(
    video_path: Path, frame_index: int, overlay: Overlay
) -> tuple[np.ndarray, list[str]]:
    """Frame `frame_index` of the video with boxes drawn, and notes for the viewer.

    Notes are the box size descriptions from `flag_box_issues` followed by the
    skipped reasons. Nothing is displayed or saved.
    """
    frame = read_frame(video_path, frame_index)
    notes = flag_box_issues(overlay.boxes) + list(overlay.skipped)
    return draw_overlay(frame, overlay), notes
