import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from wl_snatch.config import DetectorConfig
from wl_snatch.contracts import BoxDetection, FrameDetections, GroundTruthFrame
from wl_snatch.detection_viewer import (
    DetectionViewSession,
    build_view_session,
    class_color,
    draw_detections,
    draw_truth,
)
from wl_snatch.frames import read_frame

CONFIG = DetectorConfig(model_name="yolo26", conf=0.25)
GREEN = (0, 200, 0)
YELLOW = (0, 255, 255)  # BGR; the fixed color of the reference truth overlay


def box(frame_index, x1, y1, x2, y2, confidence=0.9, label="person"):
    return BoxDetection(frame_index, x1, y1, x2, y2, confidence, label)


def black() -> np.ndarray:
    return np.zeros((48, 64, 3), dtype=np.uint8)


def px(image: np.ndarray, x: int, y: int) -> tuple[int, int, int]:
    return tuple(int(v) for v in image[y, x])


@pytest.fixture
def video(tmp_path: Path, make_video) -> Path:
    # 64x48, 5 frames; frame colors (BGR) blue, green, red, blue, green
    return make_video(tmp_path / "v1.mp4")


def frames_of(*specs):
    """FrameDetections from (frame_index, [boxes]) pairs."""
    return [FrameDetections("v1", i, tuple(boxes), None) for i, boxes in specs]


def test_draw_detections_returns_a_copy_and_keeps_the_input():
    frame = black()
    result = draw_detections(frame, [box(0, 10, 20, 50, 40)], color_for=lambda label: GREEN)
    assert result is not frame
    assert result.shape == frame.shape and result.dtype == frame.dtype
    assert not frame.any()
    assert result.any()
    # right edge of the 1 px rectangle (hand-placed at x=50, y=30), interior untouched
    assert px(result, 50, 30) == GREEN
    assert px(result, 30, 30) == (0, 0, 0)


def test_class_color_is_stable_and_differs_between_labels():
    labels = ["person", "frisbee", "clock", "sports ball"]
    colors = [class_color(label) for label in labels]
    assert colors == [class_color(label) for label in labels]
    assert len(set(colors)) == len(labels)
    for color in colors:
        assert len(color) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in color)
        assert color != YELLOW
    code = "from wl_snatch.detection_viewer import class_color; print(class_color('person'))"
    other = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert other.stdout.strip() == str(class_color("person"))


def test_draw_detections_clips_boxes_outside_the_frame():
    result = draw_detections(
        black(), [box(0, -10, -10, 20, 20), box(0, 50, 30, 200, 200)], color_for=lambda label: GREEN
    )
    # first box clipped to x 0..20, y 0..20: left edge, right edge, bottom edge
    assert px(result, 0, 18) == GREEN
    assert px(result, 20, 18) == GREEN
    assert px(result, 10, 20) == GREEN
    # second box clipped to x 50..63, y 30..47
    assert px(result, 50, 40) == GREEN
    assert px(result, 63, 40) == GREEN
    assert px(result, 55, 47) == GREEN
    # a box entirely outside draws nothing
    outside = draw_detections(black(), [box(0, 100, 100, 120, 120)], color_for=lambda label: GREEN)
    assert not outside.any()


def test_draw_detections_filters_by_label_and_confidence():
    boxes = [box(0, 5, 5, 20, 20, 0.9, "person"), box(0, 30, 5, 45, 20, 0.3, "frisbee")]
    only_person = draw_detections(black(), boxes, label_filter={"person"}, color_for=lambda label: GREEN)
    assert px(only_person, 20, 12) == GREEN and px(only_person, 45, 12) == (0, 0, 0)
    confident = draw_detections(black(), boxes, min_confidence=0.5, color_for=lambda label: GREEN)
    assert px(confident, 20, 12) == GREEN and px(confident, 45, 12) == (0, 0, 0)


def test_draw_truth_uses_a_color_other_than_class_colors():
    result = draw_truth(black(), [box(0, 5, 5, 30, 25, 1.0, "x")])
    assert px(result, 30, 15) == YELLOW
    assert class_color("person") != YELLOW


def test_session_shows_only_frames_present_in_detections(
    video: Path, monkeypatch: pytest.MonkeyPatch
):
    opened = []
    original = cv2.VideoCapture.__init__

    def counting(self, *args, **kwargs):
        opened.append(args)
        original(self, *args, **kwargs)

    monkeypatch.setattr(cv2.VideoCapture, "__init__", counting)
    session = DetectionViewSession(video, "v1", frames_of((1, []), (3, [])))
    assert session.frame_indices == [1, 3]
    image, _ = session.view()
    # frame 1 is green, frame 3 is blue (conftest colors; lossy codec, so thresholds)
    b, g, r = px(image, 5, 5)
    assert g > 200 and b < 60 and r < 60
    session.goto(1)
    b, g, r = px(session.view()[0], 5, 5)
    assert b > 200 and g < 60 and r < 60
    assert len(opened) == 1  # one read for the whole session


def test_goto_and_step_clamp_at_both_ends(video: Path):
    session = DetectionViewSession(video, "v1", frames_of((0, []), (2, []), (4, [])))
    assert session.position == 0
    session.step(-1)
    assert session.position == 0
    session.goto(99)
    assert session.position == 2 and session.current_frame_index == 4
    session.step(1)
    assert session.position == 2
    session.goto(-5)
    assert session.position == 0
    session.goto(1)
    assert session.current_frame_index == 2
    session.step(5)
    assert session.position == 2


def test_view_filters_by_label_and_min_confidence(video: Path):
    boxes = [
        box(2, 5, 5, 20, 20, 0.9, "person"),
        box(2, 5, 5, 20, 20, 0.3, "person"),
        box(2, 30, 5, 45, 20, 0.6, "frisbee"),
    ]
    session = DetectionViewSession(video, "v1", frames_of((2, boxes)))
    raw = read_frame(video, 2)
    everything, lines = session.view()
    assert px(everything, 45, 12) != px(raw, 45, 12)
    assert px(everything, 20, 12) != px(raw, 20, 12)
    assert "person 2, frisbee 1" in lines
    only_person, lines = session.view(label_filter={"person"}, min_confidence=0.5)
    assert px(only_person, 45, 12) == px(raw, 45, 12)
    assert px(only_person, 20, 12) != px(raw, 20, 12)
    assert "person 1" in lines and not any("frisbee" in line for line in lines)
    assert session.classes() == ["frisbee", "person"]


def test_truth_overlay_is_off_by_default_and_uses_another_color(video: Path):
    truths = [GroundTruthFrame("v1", 2, ((5.0, 5.0, 30.0, 25.0),))]
    session = DetectionViewSession(
        video, "v1", frames_of((2, [box(2, 40, 30, 60, 45)])), truths=truths
    )
    raw = read_frame(video, 2)
    default, default_lines = session.view()
    assert px(default, 30, 15) == px(raw, 30, 15)  # no overlay by default
    assert not any("참고용 원판 라벨" in line for line in default_lines)
    shown, shown_lines = session.view(show_truth=True)
    assert px(shown, 30, 15) == YELLOW
    assert any("참고용 원판 라벨" in line for line in shown_lines)
    assert px(shown, 60, 38) == class_color("person")  # the detection box keeps its own color
    assert class_color("person") != YELLOW
    joined = " ".join(shown_lines).lower()
    for word in ("iou", "precision", "recall"):
        assert word not in joined


def test_info_lines_contain_frame_role_and_class_counts(video: Path):
    boxes = [
        box(2, 5, 5, 20, 20, 0.9, "person"),
        box(2, 6, 6, 21, 21, 0.8, "person"),
        box(2, 30, 5, 45, 20, 0.6, "frisbee"),
    ]
    session = DetectionViewSession(
        video, "v1", frames_of((0, []), (2, boxes)), roles={0: "train", 2: "validation"}
    )
    session.goto(1)
    _, lines = session.view()
    assert lines == ["프레임 2", "역할: validation", "person 2, frisbee 1"]
    session.goto(0)
    _, lines = session.view()
    assert lines[:2] == ["프레임 0", "역할: train"]
    assert lines[2] == "박스 없음"


def test_na_frame_is_shown_with_its_reason(video: Path):
    detections = [
        FrameDetections("v1", 1, (), "frame_out_of_range"),
        FrameDetections("v1", 9, (), "frame_out_of_range"),
    ]
    session = DetectionViewSession(video, "v1", detections)
    image, lines = session.view()
    assert "N/A: frame_out_of_range" in lines
    assert image.ndim == 3 and image.shape[2] == 3
    session.goto(1)  # frame 9 does not exist in the 5-frame video: placeholder, reason shown
    image, lines = session.view()
    assert "N/A: frame_out_of_range" in lines
    assert not image.any()


def test_view_png_is_encoded_in_memory_and_viewer_writes_no_image_file(
    video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def refuse(*args, **kwargs):
        raise AssertionError("an image file must not be written")

    monkeypatch.setattr(cv2, "imwrite", refuse)
    monkeypatch.setattr(cv2, "VideoWriter", refuse)
    cwd = Path.cwd()
    tree_before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    cwd_before = {p.name for p in cwd.iterdir()}
    session = DetectionViewSession(
        video,
        "v1",
        frames_of((0, [box(0, 5, 5, 20, 20)]), (2, [])),
        truths=[GroundTruthFrame("v1", 0, ())],
    )
    session.step(1)
    session.goto(0)
    session.view(show_truth=True, label_filter={"person"}, min_confidence=0.5)
    png = session.view_png(show_truth=True)
    assert png.startswith(b"\x89PNG")
    assert session.show() is not None
    assert {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")} == tree_before
    assert {p.name for p in cwd.iterdir()} == cwd_before


def test_module_import_does_not_import_ultralytics_or_ipywidgets():
    code = (
        "import sys\n"
        "import wl_snatch.detection_viewer\n"
        "bad = [m for m in ('ultralytics', 'ipywidgets', 'torch') if m in sys.modules]\n"
        "sys.exit(1 if bad else 0)\n"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


class CountingModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        return [(1.0, 1.0, 9.0, 9.0, 0.9, "person")]


def test_build_view_session_runs_the_model_only_on_labeled_frames_of_that_video(
    tmp_path: Path, make_video
):
    videos_dir = tmp_path / "videos"
    boxes_dir = tmp_path / "boxes"
    videos_dir.mkdir()
    boxes_dir.mkdir()
    for video_id in ("a01", "b01"):
        make_video(videos_dir / f"{video_id}.mp4", size=(64, 48))
    for name in ("a01_f0", "a01_f2", "b01_f1"):
        (boxes_dir / f"{name}.txt").write_text("0 0.500000 0.500000 0.250000 0.500000\n")
    (boxes_dir / "a01_f4.txt").write_text("")
    videos = pd.DataFrame(
        [{"video_id": v, "width": 64, "height": 48, "frame_count": 5} for v in ("a01", "b01")]
    )
    split = pd.DataFrame(
        [{"video_id": "a01", "role": "train"}, {"video_id": "b01", "role": "validation"}]
    )
    model = CountingModel()
    session = build_view_session(CONFIG, model, "a01", videos_dir, boxes_dir, videos, split)
    assert model.calls == 3  # frames 0, 2, 4 of a01 only; b01 frame 1 is never run
    assert session.frame_indices == [0, 2, 4]
    _, lines = session.view()
    assert "역할: train" in lines and "person 1" in lines
    session.goto(2)
    assert session.view()[1][2] == "person 1"
    other = build_view_session(CONFIG, CountingModel(), "b01", videos_dir, boxes_dir, videos, split)
    assert other.frame_indices == [1]
    assert "역할: validation" in other.view()[1]
    with pytest.raises(ValueError, match="zzz"):
        build_view_session(CONFIG, model, "zzz", videos_dir, boxes_dir, videos, split)


def test_label_text_and_confidence_are_drawn_above_the_box():
    # box top edge at y=30; text is drawn on rows above it (about y 17..26), the
    # rectangle itself only touches rows 30..45, so any pixel in rows 15..28 is text.
    def region(confidence):
        image = draw_detections(
            black(), [box(0, 10, 30, 50, 45, confidence, "person")], color_for=lambda label: GREEN
        )
        return image[15:29, 10:60]

    high, low = region(0.9), region(0.1)
    assert high.any() and low.any()
    assert not np.array_equal(high, low)  # the confidence text changes the pixels
    other = draw_detections(
        black(), [box(0, 10, 30, 50, 45, 0.9, "frisbee")], color_for=lambda label: GREEN
    )[15:29, 10:60]
    assert not np.array_equal(high, other)  # the label text changes the pixels


def test_unreadable_frame_with_boxes_says_it_could_not_be_read(video: Path):
    session = DetectionViewSession(video, "v1", frames_of((9, [box(9, 5, 5, 20, 20)])))
    image, lines = session.view()
    assert "프레임을 읽지 못함: 검은 화면 위에 표시" in lines
    assert px(image, 20, 12) == class_color("person")  # boxes still drawn on the placeholder
    readable = DetectionViewSession(video, "v1", frames_of((2, [box(2, 5, 5, 20, 20)])))
    assert not any("읽지 못함" in line for line in readable.view()[1])


def test_negative_na_frame_does_not_hide_valid_frames(video):
    detections = [
        FrameDetections("v1", -1, (), "frame_out_of_range"),
        FrameDetections("v1", 0, (), None),
    ]
    session = DetectionViewSession(video, "v1", detections)
    assert session.frame_indices == [-1, 0]
    _, info = session.view()
    assert any("N/A: frame_out_of_range" in line for line in info)
    session.goto(1)
    image, info = session.view()
    np.testing.assert_array_equal(image, read_frame(video, 0))
    assert not any("프레임을 읽지 못함" in line for line in info)
