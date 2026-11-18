"""Read frames from a local video into memory.

Frame numbers are decode order, starting at 0 (issue #10). Frames are decoded
from the start of the file in sequence. Seeking is not used and the container's
frame-count metadata is not used, because both can differ from the decoded
frames. Nothing in this module writes a file.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np


class VideoUnreadableError(ValueError):
    """The video file is missing or cannot be opened."""


def _open(video_path: Path) -> cv2.VideoCapture:
    path = Path(video_path)
    if not path.is_file():
        raise VideoUnreadableError(f"video file not found: {path.name}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoUnreadableError(f"video file cannot be opened: {path.name}")
    return capture


def check_readable(video_path: Path) -> None:
    """Open the video and release it; no frame is decoded.

    Raises VideoUnreadableError when the file is missing or cannot be opened.
    """
    _open(video_path).release()


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


def read_frames(video_path: Path, frame_indices: Sequence[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Decode the video once, in sequence, and yield the requested frames.

    Yields (frame index, BGR array) in ascending index order, each index once,
    whatever the order or repetition of `frame_indices`. Decoding stops after the
    largest requested index. Frames that exist are yielded before the error is
    raised: when an index is at or beyond the decoded length, ValueError naming
    the video and the first such index is raised after the last decoded frame.
    A negative index raises ValueError before any decoding. Nothing is written.
    """
    name = Path(video_path).name
    wanted = sorted(set(frame_indices))
    for index in wanted:
        if index < 0:
            raise ValueError(f"{name}: frame index {index} is negative")
    if not wanted:
        return
    capture = _open(video_path)
    try:
        pending = iter(wanted)
        target = next(pending)
        last = wanted[-1]
        decoded = 0
        for position, (frame, _) in enumerate(_decode(capture)):
            decoded = position + 1
            if position == target:
                yield position, frame
                if position == last:
                    return
                target = next(pending)
    finally:
        capture.release()
    raise ValueError(
        f"{name}: frame index {target} is out of range (video has {decoded} frames)"
    )


def count_frames(video_path: Path) -> int:
    """Number of frames obtained by decoding the whole video in sequence."""
    capture = _open(video_path)
    try:
        return sum(1 for _ in _decode(capture))
    finally:
        capture.release()


def raw_frame_times_ms(video_path: Path) -> list[float]:
    """Decoder timestamp (ms) of each decoded frame, as reported, without any shift.

    Raises ValueError when there are at least 2 frames and all report the same time.
    """
    capture = _open(video_path)
    try:
        times = [t for _, t in _decode(capture)]
    finally:
        capture.release()
    if len(times) > 1 and len(set(times)) == 1:
        raise ValueError(
            f"{Path(video_path).name}: timestamps unavailable "
            f"(all {len(times)} frames report {times[0]} ms)"
        )
    return times


def frame_times_ms(video_path: Path) -> list[float]:
    """Decoder timestamp (ms) of each decoded frame, shifted so the first frame is 0."""
    times = raw_frame_times_ms(video_path)
    if not times:
        return []
    start = times[0]
    return [t - start for t in times]


def frame_size(video_path: Path) -> tuple[int, int]:
    """(width, height) of the first decoded frame."""
    capture = _open(video_path)
    try:
        for frame, _ in _decode(capture):
            height, width = frame.shape[:2]
            return width, height
    finally:
        capture.release()
    raise ValueError(f"{Path(video_path).name}: video has no decodable frame")
