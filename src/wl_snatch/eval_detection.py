"""Detection evaluator: IoU matching, precision, recall and false positives per frame.

Reports numbers only. Evaluation runs on labeled frames (the frames in `truths`).
Frames the detector could not process (N/A) are left out of the metrics and counted
per reason. No image is read or written.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from wl_snatch.contracts import FrameDetections, GroundTruthFrame  # noqa: F401 (re-exported)

Box = tuple[float, float, float, float]

PREDICTION_MISSING = "prediction_missing"


@dataclass(frozen=True)
class DetectionMetrics:
    precision: float | None
    recall: float | None
    false_positives_per_frame: float | None
    frames_evaluated: int
    frames_na: int
    na_reasons: dict[str, int]
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes.

    A box with zero area or inverted corners has no area; the result is 0.0.
    """
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0.0 or height <= 0.0:
        return 0.0
    inter = width * height
    return inter / (area_a + area_b - inter)


def _index_unique(items: Sequence, what: str) -> dict[tuple[str, int], object]:
    indexed: dict[tuple[str, int], object] = {}
    for item in items:
        key = (item.video_id, item.frame_index)
        if key in indexed:
            raise ValueError(f"duplicate {what} for frame {key}")
        indexed[key] = item
    return indexed


def _match_frame(
    frame: FrameDetections, truth_boxes: tuple[Box, ...], iou_threshold: float
) -> tuple[int, int, int]:
    """Greedy matching by descending confidence. Returns (tp, fp, fn)."""
    ordered = sorted(frame.boxes, key=lambda b: b.confidence, reverse=True)
    unmatched = list(truth_boxes)
    tp = 0
    fp = 0
    for box in ordered:
        pred: Box = (box.x1, box.y1, box.x2, box.y2)
        best_index = -1
        best_iou = 0.0
        for i, truth in enumerate(unmatched):
            value = iou(pred, truth)
            if value > best_iou:
                best_index, best_iou = i, value
        if best_index >= 0 and best_iou >= iou_threshold:
            del unmatched[best_index]
            tp += 1
        else:
            fp += 1
    return tp, fp, len(unmatched)


def evaluate_detections(
    predictions: list[FrameDetections],
    truths: list[GroundTruthFrame],
    iou_threshold: float,
) -> DetectionMetrics:
    """Evaluate predictions against ground truth on the labeled frames.

    Predictions on frames not in `truths` are ignored. A truth frame with no
    prediction entry is N/A with reason "prediction_missing". A frame whose
    prediction has `na_reason` is N/A with that reason. Duplicate
    (video_id, frame_index) keys raise ValueError. All predicted boxes count,
    whatever their label.

    Predictions of equal confidence are matched in input order. A frame whose
    prediction has `na_reason` is N/A, so its truth boxes are not counted as
    misses (false negatives).
    """
    if isinstance(iou_threshold, bool) or not 0 < iou_threshold <= 1:
        raise ValueError(f"iou_threshold must satisfy 0 < iou_threshold <= 1, got {iou_threshold}")
    pred_by_key = _index_unique(predictions, "prediction")
    truth_by_key = _index_unique(truths, "ground truth")

    tp = fp = fn = 0
    evaluated = 0
    na_reasons: Counter[str] = Counter()
    for key, truth in truth_by_key.items():
        frame = pred_by_key.get(key)
        if frame is None:
            na_reasons[PREDICTION_MISSING] += 1
            continue
        if frame.na_reason is not None:
            na_reasons[frame.na_reason] += 1
            continue
        frame_tp, frame_fp, frame_fn = _match_frame(frame, truth.boxes, iou_threshold)
        tp += frame_tp
        fp += frame_fp
        fn += frame_fn
        evaluated += 1

    return DetectionMetrics(
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        false_positives_per_frame=fp / evaluated if evaluated else None,
        frames_evaluated=evaluated,
        frames_na=sum(na_reasons.values()),
        na_reasons=dict(na_reasons),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )
