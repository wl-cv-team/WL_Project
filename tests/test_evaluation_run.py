from pathlib import Path

import pytest

from wl_snatch.config import DetectorConfig
from wl_snatch.contracts import GroundTruthFrame
from wl_snatch.evaluation_run import evaluate_ground_truth

CONFIG = DetectorConfig(model_name="yolo26")


class CountingModel:
    """Counts model calls; every call returns no box."""

    def __init__(self):
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        return []


def test_missing_video_path_raises_value_error_naming_the_video(tmp_path: Path):
    truths = [GroundTruthFrame("a01", 0, ()), GroundTruthFrame("b01", 0, ())]
    with pytest.raises(ValueError, match="b01"):
        evaluate_ground_truth(truths, {"a01": tmp_path / "a01.mp4"}, CountingModel(), CONFIG, 0.5)


def test_frames_of_one_video_are_requested_in_one_detect_frames_call(
    tmp_path: Path, make_video, monkeypatch: pytest.MonkeyPatch
):
    import wl_snatch.evaluation_run as module

    requests = []
    original = module.detect_frames

    def spy(path, video_id, frame_indices, model, config):
        requests.append((video_id, list(frame_indices)))
        return original(path, video_id, frame_indices, model, config)

    monkeypatch.setattr(module, "detect_frames", spy)
    paths = {v: make_video(tmp_path / f"{v}.mp4") for v in ("a01", "b01")}
    truths = [
        GroundTruthFrame("b01", 1, ()),
        GroundTruthFrame("a01", 3, ()),
        GroundTruthFrame("a01", 0, ()),
    ]
    model = CountingModel()
    metrics = evaluate_ground_truth(truths, paths, model, CONFIG, 0.5)
    assert requests == [("a01", [0, 3]), ("b01", [1])]
    assert model.calls == 3
    assert metrics.frames_evaluated == 3


def test_empty_truths_give_none_ratios_and_no_evaluated_frames(tmp_path: Path):
    metrics = evaluate_ground_truth([], {}, CountingModel(), CONFIG, 0.5)
    assert metrics.frames_evaluated == 0
    assert metrics.precision is None
    assert metrics.recall is None
    assert metrics.false_positives_per_frame is None
