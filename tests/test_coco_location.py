from pathlib import Path

from wl_snatch.config import DetectorConfig
from wl_snatch.contracts import BoxDetection, FrameDetections, GroundTruthFrame
from wl_snatch import coco_location


def test_location_keeps_coco_labels_and_counts_unmatched_and_na(monkeypatch):
    predictions = [
        FrameDetections("v", 0, (
            BoxDetection(0, 0, 0, 10, 10, .8, "frisbee"),
            BoxDetection(0, 20, 20, 30, 30, .9, "person"),
        ), None),
        FrameDetections("v", 1, (), None),
        FrameDetections("v", 2, (), "video_unreadable"),
    ]
    monkeypatch.setattr(coco_location, "detect_frames_labeled", lambda *args: predictions)
    truths = [GroundTruthFrame("v", i, ((0, 0, 10, 10),)) for i in range(3)]
    metrics, output = coco_location.measure_location(
        truths, Path("."), object(), DetectorConfig(model_name="yolo26"), .5
    )
    assert output == predictions
    assert output[0].boxes[0].label == "frisbee"
    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (1, 1, 1)
    assert metrics.frames_evaluated == 2
    assert metrics.frames_na == 1
    assert metrics.precision == .5
    assert metrics.recall == .5
