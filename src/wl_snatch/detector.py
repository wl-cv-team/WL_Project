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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from wl_snatch.config import DetectorConfig, resolve_device
from wl_snatch.contracts import BoxDetection, FrameDetections
from wl_snatch.frames import VideoUnreadableError, check_readable, count_frames, read_frames

LABEL = "plate"
VIDEO_UNREADABLE = "video_unreadable"
FRAME_OUT_OF_RANGE = "frame_out_of_range"

ModelFn = Callable[[np.ndarray], Sequence[tuple[float, float, float, float, float]]]

# Like ModelFn, but each box also carries the class name given by the model.
LabeledModelFn = Callable[[np.ndarray], Sequence[tuple[float, float, float, float, float, str]]]

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
    """One entry per requested frame, ascending, duplicates removed; every box is "plate".

    Boxes with confidence below `config.conf` are dropped. The video is decoded
    once; frames reached before the end of the video are processed even when
    other requested indices are out of range. A thin layer over
    `detect_frames_labeled` that fixes the label.
    """

    def labeled(frame: np.ndarray):
        return [(*box, LABEL) for box in model(frame)]

    return detect_frames_labeled(video_path, video_id, frame_indices, labeled, config)


def detect_frames_labeled(
    video_path: Path,
    video_id: str,
    frame_indices: Sequence[int],
    model: LabeledModelFn,
    config: DetectorConfig,
) -> list[FrameDetections]:
    """Same rules as `detect_frames`, but `BoxDetection.label` is the name the model gives."""
    wanted = sorted(set(frame_indices))
    results: dict[int, FrameDetections] = {}
    if not wanted:
        return []
    # Checked first so an unopenable video is unreadable for every index,
    # negative ones included (read_frames is not called when none is >= 0).
    try:
        check_readable(video_path)
    except VideoUnreadableError:
        return [_na(video_id, i, VIDEO_UNREADABLE) for i in wanted]
    frames = read_frames(video_path, [i for i in wanted if i >= 0])
    try:
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
                BoxDetection(index, x1, y1, x2, y2, score, label)
                for x1, y1, x2, y2, score, label in model(frame)
                if score >= config.conf
            )
            results[index] = FrameDetections(video_id, index, boxes, None)
    finally:
        frames.close()
    return [results.get(i) or _na(video_id, i, FRAME_OUT_OF_RANGE) for i in wanted]


def detect_video(
    video_path: Path, video_id: str, model: ModelFn, config: DetectorConfig
) -> list[FrameDetections]:
    """`detect_frames` over every decoded frame.

    The number of frames is found by decoding once (`count_frames`). When the
    video cannot be opened the number of frames is unknown, so the result is one
    N/A entry for frame 0 with reason `video_unreadable`.
    """
    try:
        total = count_frames(video_path)
    except VideoUnreadableError:
        return [_na(video_id, 0, VIDEO_UNREADABLE)]
    return detect_frames(video_path, video_id, range(total), model, config)


_NOT_PLATE_MODEL = (
    "the model must be a single-class plate detector trained for this project; "
    "COCO-pretrained weights (yolo26n.pt and the like) do not detect plates"
)


def check_plate_names(names: Mapping[int, str]) -> None:
    """Raise ValueError unless the model's class names are exactly {0: "plate"}."""
    if dict(names) != {0: LABEL}:
        shown = dict(list(dict(names).items())[:5])
        suffix = " ..." if len(names) > 5 else ""
        raise ValueError(f"{_NOT_PLATE_MODEL}; class names found: {shown}{suffix}")


def load_ultralytics_model(config: DetectorConfig) -> ModelFn:
    """Load the plate-detector weights at `config.weights_path` and wrap them as a ModelFn.

    `weights_path` is required: the pretrained names in `WEIGHT_NAMES` are
    COCO-pretrained starting points for training and do not detect plates. The
    loaded model's class names must be exactly {0: "plate"} (`check_plate_names`),
    so no other class is ever written with the label "plate".

    Ultralytics and torch are imported here only, after the path check. Not
    covered by tests: it needs a weight file.
    """
    if config.weights_path is None:
        raise ValueError(f"weights_path is required: {_NOT_PLATE_MODEL}")
    import torch
    from ultralytics import YOLO

    device, reason = resolve_device(
        config.device,
        mps_available=torch.backends.mps.is_available(),
        cuda_available=torch.cuda.is_available(),
    )
    _log.info("loading %s (%s) on device %s", config.weights_path, config.model_name, device)
    if reason is not None:
        _log.warning("device fallback: %s", reason)
    yolo = YOLO(str(config.weights_path))
    check_plate_names(yolo.names)

    def model(frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        result = yolo(
            frame, imgsz=config.imgsz, conf=config.conf, device=device, verbose=False
        )[0]
        xyxy = result.boxes.xyxy.tolist()
        scores = result.boxes.conf.tolist()
        return [(x1, y1, x2, y2, s) for (x1, y1, x2, y2), s in zip(xyxy, scores)]

    return model
