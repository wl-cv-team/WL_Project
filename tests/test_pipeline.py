"""Integration: ground_truth.load_ground_truth -> detector.detect_frames -> evaluate_detections.

Synthetic data only: small videos from `make_video`, label files written by hand, a
fake model function. Frame size is 64x48, so a normalized box maps to easy pixels.
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from wl_snatch.config import DetectorConfig
from wl_snatch.detector import detect_frames
from wl_snatch.eval_detection import DetectionMetrics, evaluate_detections
from wl_snatch.ground_truth import load_ground_truth

WIDTH, HEIGHT = 64, 48
CONFIG = DetectorConfig(model_name="yolo26", conf=0.25)

# Pixel boxes (x1, y1, x2, y2) and their normalized label lines "0 cx cy w h".
BOX_A = (16.0, 12.0, 48.0, 36.0)
LINE_A = "0 0.500000 0.500000 0.500000 0.500000\n"
BOX_B = (0.0, 0.0, 32.0, 24.0)
LINE_B = "0 0.250000 0.250000 0.500000 0.500000\n"
# Same size as BOX_A, shifted 24 px right: IoU = 8*24 / (2*768 - 8*24) < 0.5.
BOX_SHIFTED = (40.0, 12.0, 72.0, 36.0)


class World:
    """Temp folders with synthetic videos and labels, plus the two tables."""

    def __init__(self, tmp_path: Path, make_video: Callable[..., Path]):
        self.make_video = make_video
        self.videos_dir = tmp_path / "videos"
        self.boxes_dir = tmp_path / "boxes"
        self.videos_dir.mkdir()
        self.boxes_dir.mkdir()
        self.rows: list[dict] = []
        self.roles: list[dict] = []

    def add_video(self, video_id: str, role: str, frame_count: int | None = None) -> None:
        path = self.make_video(self.videos_dir / f"{video_id}.mp4", size=(WIDTH, HEIGHT))
        self.rows.append(
            {
                "video_id": video_id,
                "width": WIDTH,
                "height": HEIGHT,
                "frame_count": 5 if frame_count is None else frame_count,
            }
        )
        self.roles.append({"video_id": video_id, "role": role})
        assert path.is_file()

    def label(self, video_id: str, frame: int, line: str) -> None:
        (self.boxes_dir / f"{video_id}_f{frame}.txt").write_text(line)

    def video_path(self, video_id: str) -> Path:
        return self.videos_dir / f"{video_id}.mp4"

    def truths(self, role: str):
        frames, _ = load_ground_truth(
            self.boxes_dir, pd.DataFrame(self.rows), pd.DataFrame(self.roles), role
        )
        return frames

    def run(
        self,
        role: str,
        model_for: Callable[[str, list[tuple[float, ...]]], Callable],
        video_dir_override: Path | None = None,
    ) -> DetectionMetrics:
        truths = self.truths(role)
        predictions = []
        for video_id in sorted({t.video_id for t in truths}):
            mine = [t for t in truths if t.video_id == video_id]
            path = (video_dir_override or self.videos_dir) / f"{video_id}.mp4"
            model = model_for(video_id, [t.boxes for t in mine])
            predictions += detect_frames(path, video_id, [t.frame_index for t in mine], model, CONFIG)
        return evaluate_detections(predictions, truths, 0.5)


def echo_truth(video_id, boxes_per_frame):
    """Fake model that returns the truth boxes of the frame it is called for.

    detect_frames calls the model once per requested frame, in ascending frame
    order, and `truths` is sorted by frame; so the n-th call is the n-th truth frame.
    A call counter is therefore a reliable key (the synthetic frames themselves
    do not identify their index uniquely).
    """
    calls = iter(boxes_per_frame)

    def model(frame):
        return [(*box, 0.9) for box in next(calls)]

    return model


def scripted(per_call):
    """Fake model factory returning `per_call[n]` (a list of boxes) on the n-th call."""

    def factory(video_id, boxes_per_frame):
        calls = iter(per_call[video_id])

        def model(frame):
            return [(*box, 0.9) for box in next(calls)]

        return model

    return factory


def make_world(tmp_path: Path, make_video) -> World:
    world = World(tmp_path, make_video)
    world.add_video("a01", "train")
    world.add_video("b01", "validation")
    world.label("a01", 0, LINE_A)
    world.label("a01", 2, LINE_B)
    world.label("a01", 4, "")
    world.label("b01", 1, LINE_A)
    world.label("b01", 3, LINE_B)
    return world


def test_perfect_model_gives_precision_and_recall_one_across_the_layers(
    tmp_path: Path, make_video
):
    world = make_world(tmp_path, make_video)
    metrics = world.run("train", echo_truth)
    labeled = len(world.truths("train"))
    assert labeled == 3
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.false_positives == 0
    assert metrics.frames_evaluated == labeled
    assert metrics.na_reasons == {}


def test_box_on_a_labeled_empty_frame_is_a_false_positive_across_the_layers(
    tmp_path: Path, make_video
):
    world = make_world(tmp_path, make_video)
    # train frames in order: 0 (box A), 2 (box B), 4 (empty label)
    model_for = scripted({"a01": [[BOX_A], [BOX_B], [BOX_A]]})
    metrics = world.run("train", model_for)
    assert metrics.false_positives == 1
    assert metrics.true_positives == 2
    assert metrics.frames_evaluated == 3
    assert metrics.frames_na == 0
    assert metrics.recall == 1.0
    assert metrics.precision == 2 / 3


def test_shifted_box_below_threshold_is_a_miss_and_a_false_positive_across_the_layers(
    tmp_path: Path, make_video
):
    world = make_world(tmp_path, make_video)
    model_for = scripted({"a01": [[BOX_SHIFTED], [BOX_B], []]})
    metrics = world.run("train", model_for)
    assert metrics.true_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.frames_evaluated == 3
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5


def test_unreadable_video_is_counted_as_na_by_reason_across_the_layers(
    tmp_path: Path, make_video
):
    world = make_world(tmp_path, make_video)
    missing = tmp_path / "no_videos_here"
    metrics = world.run("train", echo_truth, video_dir_override=missing)
    labeled = len(world.truths("train"))
    assert metrics.frames_na == labeled
    assert metrics.na_reasons == {"video_unreadable": labeled}
    assert metrics.frames_evaluated == 0
    assert metrics.precision is None
    assert metrics.recall is None


def test_frame_beyond_the_video_is_na_frame_out_of_range_across_the_layers(
    tmp_path: Path, make_video
):
    world = World(tmp_path, make_video)
    # The table says 8 frames; the synthetic video decodes to 5.
    world.add_video("c01", "train", frame_count=8)
    world.label("c01", 1, LINE_A)
    world.label("c01", 6, LINE_A)
    world.label("c01", 7, LINE_B)
    # Only frame 1 is reachable, so the model is called once.
    metrics = world.run("train", scripted({"c01": [[BOX_A]]}))
    assert metrics.na_reasons == {"frame_out_of_range": 2}
    assert metrics.frames_na == 2
    assert metrics.frames_evaluated == 1
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0


def test_only_the_requested_role_flows_through(tmp_path: Path, make_video):
    world = make_world(tmp_path, make_video)
    train = world.truths("train")
    validation = world.truths("validation")
    assert {t.video_id for t in train} == {"a01"}
    assert {t.video_id for t in validation} == {"b01"}
    train_metrics = world.run("train", echo_truth)
    validation_metrics = world.run("validation", echo_truth)
    assert train_metrics.frames_evaluated == 3
    assert validation_metrics.frames_evaluated == 2
    assert train_metrics.true_positives == 2
    assert validation_metrics.true_positives == 2
    assert train_metrics.frames_na == validation_metrics.frames_na == 0


def test_pipeline_writes_no_image_file(tmp_path: Path, make_video):
    world = make_world(tmp_path, make_video)
    world.run("train", echo_truth)
    world.run("validation", echo_truth)
    files = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    expected = {
        "videos/a01.mp4",
        "videos/b01.mp4",
        "boxes/a01_f0.txt",
        "boxes/a01_f2.txt",
        "boxes/a01_f4.txt",
        "boxes/b01_f1.txt",
        "boxes/b01_f3.txt",
    }
    assert files == expected
