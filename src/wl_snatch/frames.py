"""Read frames from a local video into memory.

Frame numbers are decode order, starting at 0 (issue #10). Frames are decoded
from the start of the file in sequence. Seeking is not used and the container's
frame-count metadata is not used, because both can differ from the decoded
frames. Nothing in this module writes a file.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def _open(video_path: Path) -> cv2.VideoCapture:
    path = Path(video_path)
    if not path.is_file():
        raise ValueError(f"video file not found: {path.name}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"video file cannot be opened: {path.name}")
    return capture


def _decode(capture: cv2.VideoCapture) -> Iterator[tuple[np.ndarray, float]]:
    """Yield (frame, timestamp in ms as reported by the decoder) in decode order."""
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        yield frame, capture.get(cv2.CAP_PROP_POS_MSEC)


def read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    """BGR array of the `frame_index`-th decoded frame (0-based).

    Raises ValueError, naming the video file and the index, when the video
    cannot be read or has no frame at that index.
    """
    name = Path(video_path).name
    if frame_index < 0:
        raise ValueError(f"{name}: frame index {frame_index} is negative")
    try:
        capture = _open(video_path)
    except ValueError as error:
        raise ValueError(f"{error} (frame index {frame_index})") from error
    try:
        for position, (frame, _) in enumerate(_decode(capture)):
            if position == frame_index:
                return frame
    finally:
        capture.release()
    raise ValueError(f"{name}: frame index {frame_index} is out of range (video has fewer frames)")


def count_frames(video_path: Path) -> int:
    """Number of frames obtained by decoding the whole video in sequence."""
    capture = _open(video_path)
    try:
        return sum(1 for _ in _decode(capture))
    finally:
        capture.release()


def frame_times_ms(video_path: Path) -> list[float]:
    """Decoder timestamp (ms) of each decoded frame, shifted so the first frame is 0."""
    capture = _open(video_path)
    try:
        times = [t for _, t in _decode(capture)]
    finally:
        capture.release()
    if not times:
        return []
    start = times[0]
    return [t - start for t in times]
