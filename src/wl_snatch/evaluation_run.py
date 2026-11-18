"""Run a detector over ground truth frames and evaluate it: the loop between the layers.

Groups the labeled frames by video, calls `detect_frames` once per video (one
decode per video) and passes the detections and the truths to the evaluator.
No image is read from disk beyond video decoding and none is written.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from wl_snatch.config import DetectorConfig
from wl_snatch.contracts import GroundTruthFrame
from wl_snatch.detector import ModelFn, detect_frames
from wl_snatch.eval_detection import DetectionMetrics, evaluate_detections


def evaluate_ground_truth(
    truths: Sequence[GroundTruthFrame],
    video_paths: Mapping[str, Path],
    model: ModelFn,
    config: DetectorConfig,
    iou_threshold: float,
) -> DetectionMetrics:
    """Detect on every labeled frame (videos in ascending id) and evaluate.

    Raises ValueError naming a `video_id` of `truths` that has no entry in
    `video_paths`, before any detection runs.
    """
    frames_by_video: dict[str, list[int]] = defaultdict(list)
    for truth in truths:
        frames_by_video[truth.video_id].append(truth.frame_index)
    for video_id in frames_by_video:
        if video_id not in video_paths:
            raise ValueError(f"{video_id}: no video path given")
    predictions = []
    for video_id in sorted(frames_by_video):
        predictions += detect_frames(
            video_paths[video_id], video_id, sorted(frames_by_video[video_id]), model, config
        )
    return evaluate_detections(predictions, list(truths), iou_threshold)
