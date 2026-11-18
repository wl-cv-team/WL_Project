"""Run COCO-pretrained detectors as they are and summarize what they output.

This is an exploration of whether a model loads and runs, which COCO classes it
reports on the labeled frames, and how long inference takes. It is NOT a plate
detection measurement: no plate label is compared, no metric against labels is
computed, and no box is ever renamed to "plate". Trained plate weights go through
`detector.load_ultralytics_model` instead.

Ultralytics is imported only inside `load_coco_model`. Nothing here writes an
image; the report is plain text.
"""

from __future__ import annotations

import importlib.metadata
import logging
import platform
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wl_snatch.config import DetectorConfig, resolve_device, weight_name
from wl_snatch.contracts import FrameDetections, GroundTruthFrame
from wl_snatch.detector import LabeledModelFn, detect_frames_labeled

COCO_CLASS_COUNT = 80
NOT_INSTALLED = "not installed"
NOT_PLATE_PERFORMANCE = (
    "These numbers are not plate-detection performance: the models are COCO-pretrained "
    "and no label is compared."
)

_log = logging.getLogger(__name__)


def check_coco_names(names: Mapping[int, str]) -> None:
    """Raise ValueError unless `names` has 80 entries with keys 0..79 and no "plate" class."""
    found = list(dict(names).values())[:5]
    if len(names) != COCO_CLASS_COUNT or set(names) != set(range(COCO_CLASS_COUNT)):
        raise ValueError(
            f"expected the {COCO_CLASS_COUNT} COCO classes (keys 0..{COCO_CLASS_COUNT - 1}), "
            f"got {len(names)} entries; first names: {found}"
        )
    if any(str(name).lower() == "plate" for name in names.values()):
        raise ValueError(
            "the model has a class named plate, so it is not a COCO-pretrained model; "
            "trained plate weights go through the plate path (load_ultralytics_model); "
            f"first names: {found}"
        )


def load_coco_model(config: DetectorConfig) -> LabeledModelFn:
    """Load the COCO-pretrained weights named by `config.model_name`, keeping class names.

    Raises ValueError when `config.weights_path` is set (trained weights belong to the
    plate path), before importing Ultralytics. The loaded model's `names` must pass
    `check_coco_names`. Not covered by tests: it needs a weight file and network access.
    """
    if config.weights_path is not None:
        raise ValueError(
            "weights_path is not allowed in COCO mode; trained weights go through "
            "load_ultralytics_model (plate path)"
        )
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
    check_coco_names(yolo.names)
    names = yolo.names

    def model(frame: np.ndarray) -> list[tuple[float, float, float, float, float, str]]:
        result = yolo(
            frame, imgsz=config.imgsz, conf=config.conf, device=device, verbose=False
        )[0]
        xyxy = result.boxes.xyxy.tolist()
        scores = result.boxes.conf.tolist()
        classes = result.boxes.cls.tolist()
        return [
            (x1, y1, x2, y2, s, names[int(c)])
            for (x1, y1, x2, y2), s, c in zip(xyxy, scores, classes)
        ]

    return model


def _installed_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return NOT_INSTALLED


def environment_record(
    config: DetectorConfig,
    device: str,
    reason: str | None,
    versions: Mapping[str, str] | None = None,
    nms_iou: str | None = None,
) -> dict[str, str]:
    """Run conditions as plain strings.

    `versions` may give "python", "ultralytics", "torch", "opencv" (tests inject them);
    missing ones are read with `importlib.metadata` without importing the libraries,
    and a package that is not installed gives "not installed". `nms_iou` is recorded
    only when the caller passes it, as text that says where the value comes from.
    """
    given = dict(versions or {})
    record = {
        "python": given.get("python", platform.python_version()),
        "ultralytics": given.get("ultralytics", _installed_version("ultralytics")),
        "torch": given.get("torch", _installed_version("torch")),
        "opencv": given.get("opencv", _installed_version("opencv-python")),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "requested_device": config.device,
        "device": device,
        "device_reason": reason if reason is not None else "none",
        "weights": weight_name(config.model_name),
        "imgsz": str(config.imgsz),
        "conf": str(config.conf),
        "seed": str(config.seed),
    }
    if nms_iou is not None:
        record["nms_iou"] = nms_iou
    return record


def timed(model: LabeledModelFn) -> tuple[LabeledModelFn, list[float]]:
    """Wrap `model`; each call appends its duration in milliseconds to the returned list."""
    times: list[float] = []

    def wrapped(frame: np.ndarray):
        start = time.perf_counter()
        result = model(frame)
        times.append((time.perf_counter() - start) * 1000.0)
        return result

    return wrapped, times


@dataclass(frozen=True)
class CocoSummary:
    class_counts: dict[str, int]
    zero_box_frames: int
    processed_frames: int
    na_frames: int
    na_reasons: dict[str, int]
    first_call_ms: float | None
    # Over the calls after the first one; None when there are none.
    median_rest_ms: float | None
    mean_rest_ms: float | None
    max_rest_ms: float | None


def summarize_coco_run(
    frames: Sequence[FrameDetections], times_ms: Sequence[float]
) -> CocoSummary:
    """Counts per class name, processed and N/A frames, and timing of the model calls."""
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    processed = zero = 0
    for frame in frames:
        if frame.na_reason is not None:
            reasons[frame.na_reason] += 1
            continue
        processed += 1
        if not frame.boxes:
            zero += 1
        counts.update(box.label for box in frame.boxes)
    rest = list(times_ms[1:])
    return CocoSummary(
        class_counts=dict(counts),
        zero_box_frames=zero,
        processed_frames=processed,
        na_frames=sum(reasons.values()),
        na_reasons=dict(reasons),
        first_call_ms=times_ms[0] if times_ms else None,
        median_rest_ms=statistics.median(rest) if rest else None,
        mean_rest_ms=statistics.fmean(rest) if rest else None,
        max_rest_ms=max(rest) if rest else None,
    )


def group_frames_by_video(truths: Sequence[GroundTruthFrame]) -> dict[str, list[int]]:
    """Labeled frame numbers per video id (ascending, no duplicates). Boxes are not used."""
    grouped: dict[str, set[int]] = defaultdict(set)
    for truth in truths:
        grouped[truth.video_id].add(truth.frame_index)
    return {video_id: sorted(grouped[video_id]) for video_id in sorted(grouped)}


def run_coco_baseline(
    truths_by_role: Mapping[str, Sequence[GroundTruthFrame]],
    video_paths: Mapping[str, Path],
    model: LabeledModelFn,
    config: DetectorConfig,
) -> dict[str, CocoSummary]:
    """Run `model` on the labeled frames of each role and summarize per role.

    Only the video ids and frame numbers of `truths_by_role` are used. Roles run in the
    given order, videos in ascending id; the timing of a role covers its own calls
    (the process's first call, which includes any warm-up, falls into the first role).
    Raises ValueError naming a video id without an entry in `video_paths`, before any
    detection runs.
    """
    grouped = {role: group_frames_by_video(t) for role, t in truths_by_role.items()}
    for per_video in grouped.values():
        for video_id in per_video:
            if video_id not in video_paths:
                raise ValueError(f"{video_id}: no video path given")
    wrapped, times = timed(model)
    summaries: dict[str, CocoSummary] = {}
    for role, per_video in grouped.items():
        before = len(times)
        frames: list[FrameDetections] = []
        for video_id, indices in per_video.items():
            frames += detect_frames_labeled(video_paths[video_id], video_id, indices, wrapped, config)
        summaries[role] = summarize_coco_run(frames, times[before:])
    return summaries


def format_report(record: Mapping[str, str], summaries: Mapping[str, CocoSummary]) -> str:
    """Plain-text report: run conditions, then per role counts and timing."""
    lines = ["COCO pretrained run (exploration)", ""]
    lines += [f"{key}: {value}" for key, value in record.items()]
    for role, s in summaries.items():
        lines += ["", f"[{role}]"]
        lines.append(f"frames processed: {s.processed_frames}")
        lines.append(f"frames with no box: {s.zero_box_frames}")
        lines.append(f"N/A frames: {s.na_frames}")
        for reason, count in sorted(s.na_reasons.items()):
            lines.append(f"  N/A {reason}: {count}")
        lines.append("class counts (boxes):")
        if not s.class_counts:
            lines.append("  none")
        for name, count in sorted(s.class_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  {name}: {count}")
        lines.append(f"first call ms: {_ms(s.first_call_ms)}")
        lines.append(f"later calls median ms: {_ms(s.median_rest_ms)}")
        lines.append(f"later calls mean ms: {_ms(s.mean_rest_ms)}")
        lines.append(f"later calls max ms: {_ms(s.max_rest_ms)}")
    lines += ["", NOT_PLATE_PERFORMANCE, ""]
    return "\n".join(lines)


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"
