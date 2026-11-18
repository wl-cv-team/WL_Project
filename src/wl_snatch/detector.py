"""Plate detector adapter: runs a model function over frames and returns FrameDetections.

The model is injected as a `ModelFn`, so tests use a fake. Ultralytics is
imported only inside `load_ultralytics_model`. Frames are read with a single
sequential decode and no image is written.

Frames that cannot be processed become N/A entries instead of errors:
`video_unreadable` when the video cannot be opened, `frame_out_of_range` when a
requested index is negative or at/beyond the decoded length. A processed frame
without a remaining box is a zero-detection frame (`na_reason=None`, no boxes).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from wl_snatch.config import DetectorConfig, resolve_device, weight_name
from wl_snatch.contracts import BoxDetection, FrameDetections
from wl_snatch.frames import VideoUnreadableError, count_frames, read_frames

LABEL = "plate"
VIDEO_UNREADABLE = "video_unreadable"
FRAME_OUT_OF_RANGE = "frame_out_of_range"

ModelFn = Callable[[np.ndarray], Sequence[tuple[float, float, float, float, float]]]

_log = logging.getLogger(__name__)


def _na(video_id: str, index: int, reason: str) -> FrameDetections:
    return FrameDetections(video_id, index, (), reason)


def detect_frames(
    video_path: Path,
    video_id: str,
    frame_indices: Sequence[int],
    model: ModelFn,
    config: DetectorConfig,
) -> list[FrameDetections]:
    """One entry per requested frame, ascending, duplicates removed.

    Boxes with confidence below `config.conf` are dropped. The video is decoded
    once; frames reached before the end of the video are processed even when
    other requested indices are out of range.
    """
    wanted = sorted(set(frame_indices))
    results: dict[int, FrameDetections] = {}
    frames = read_frames(video_path, [i for i in wanted if i >= 0])
    while True:
        try:
            index, frame = next(frames)
        except StopIteration:
            break
        except VideoUnreadableError:
            return [_na(video_id, i, VIDEO_UNREADABLE) for i in wanted]
        except ValueError:
            # read_frames raises this only after yielding every reachable frame.
            break
        boxes = tuple(
            BoxDetection(index, x1, y1, x2, y2, score, LABEL)
            for x1, y1, x2, y2, score in model(frame)
            if score >= config.conf
        )
        results[index] = FrameDetections(video_id, index, boxes, None)
    return [results.get(i) or _na(video_id, i, FRAME_OUT_OF_RANGE) for i in wanted]


def detect_video(
    video_path: Path, video_id: str, model: ModelFn, config: DetectorConfig
) -> list[FrameDetections]:
    """`detect_frames` over every decoded frame; an unreadable video gives an empty list."""
    try:
        total = count_frames(video_path)
    except ValueError:
        return []
    return detect_frames(video_path, video_id, range(total), model, config)


def load_ultralytics_model(config: DetectorConfig) -> ModelFn:
    """Load Ultralytics weights named by `config.model_name` and wrap them as a ModelFn.

    Ultralytics and torch are imported here only. Not covered by tests: it needs
    the weight file and network access.
    """
    import torch
    from ultralytics import YOLO

    device, reason = resolve_device(
        config.device,
        mps_available=torch.backends.mps.is_available(),
        cuda_available=torch.cuda.is_available(),
    )
    weights = weight_name(config.model_name)
    _log.info("loading %s on device %s", weights, device)
    if reason is not None:
        _log.warning("device fallback: %s", reason)
    yolo = YOLO(weights)

    def model(frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        result = yolo(
            frame, imgsz=config.imgsz, conf=config.conf, device=device, verbose=False
        )[0]
        xyxy = result.boxes.xyxy.tolist()
        scores = result.boxes.conf.tolist()
        return [(x1, y1, x2, y2, s) for (x1, y1, x2, y2), s in zip(xyxy, scores)]

    return model
