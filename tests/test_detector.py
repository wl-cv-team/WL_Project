import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wl_snatch.config import DetectorConfig
from wl_snatch.contracts import FrameDetections
from wl_snatch.detector import check_plate_names, detect_frames, detect_video

CONFIG = DetectorConfig(model_name="yolo26", conf=0.25)


@pytest.fixture
def video(tmp_path: Path, make_video) -> Path:
    return make_video(tmp_path / "synthetic.mp4")


class FakeModel:
    def __init__(self, boxes_by_channel=None):
        self.calls: list[int] = []
        self.boxes_by_channel = boxes_by_channel or {}

    def __call__(self, frame: np.ndarray):
        channel = int(np.argmax(frame.reshape(-1, 3).mean(axis=0)))
        self.calls.append(channel)
        return self.boxes_by_channel.get(channel, [])


def test_detect_frames_calls_injected_model_once_per_requested_frame(video: Path):
    model = FakeModel()
    result = detect_frames(video, "v1", [3, 1, 1], model, CONFIG)
    assert len(model.calls) == 2
    assert [f.frame_index for f in result] == [1, 3]
    assert all(f.video_id == "v1" for f in result)


def test_detect_frames_output_is_ascending_without_duplicates(video: Path):
    result = detect_frames(video, "v1", [4, 0, 2, 0], FakeModel(), CONFIG)
    assert [f.frame_index for f in result] == [0, 2, 4]


def test_detect_frames_drops_boxes_below_conf_and_keeps_zero_detection_frames_as_not_na(
    video: Path,
):
    # channel per frame: 0 -> 0, 1 -> 1, 2 -> 2, 3 -> 0
    model = FakeModel(
        {
            0: [(1.0, 2.0, 30.0, 40.0, 0.9), (5.0, 5.0, 9.0, 9.0, 0.1)],
            1: [(5.0, 5.0, 9.0, 9.0, 0.1)],
            2: [(1.0, 1.0, 2.0, 2.0, 0.25)],
        }
    )
    result = detect_frames(video, "v1", [0, 1, 2], model, CONFIG)
    first, second, third = result
    assert len(first.boxes) == 1
    box = first.boxes[0]
    assert (box.x1, box.y1, box.x2, box.y2, box.confidence) == (1.0, 2.0, 30.0, 40.0, 0.9)
    assert box.label == "plate"
    assert box.frame_index == 0
    assert first.na_reason is None
    assert second.boxes == () and second.na_reason is None
    assert len(third.boxes) == 1 and third.boxes[0].frame_index == 2
    assert third.na_reason is None


def test_detect_frames_marks_out_of_range_frame_as_na_and_continues(video: Path):
    model = FakeModel({0: [(1.0, 1.0, 5.0, 5.0, 0.8)]})
    result = detect_frames(video, "v1", [0, 2, 9, 12, -1], model, CONFIG)
    by_index = {f.frame_index: f for f in result}
    assert [f.frame_index for f in result] == [-1, 0, 2, 9, 12]
    assert len(by_index[0].boxes) == 1 and by_index[0].na_reason is None
    assert by_index[2].na_reason is None
    for index in (-1, 9, 12):
        assert by_index[index].na_reason == "frame_out_of_range"
        assert by_index[index].boxes == ()
    assert len(model.calls) == 2


def test_detect_frames_mixes_na_and_non_na_in_one_call(video: Path):
    result = detect_frames(video, "v1", [4, 5, 1], FakeModel(), CONFIG)
    assert [(f.frame_index, f.na_reason) for f in result] == [
        (1, None),
        (4, None),
        (5, "frame_out_of_range"),
    ]


def test_unreadable_video_gives_na_for_every_requested_frame(tmp_path: Path):
    model = FakeModel()
    result = detect_frames(tmp_path / "absent.mp4", "v1", [2, 0, 2], model, CONFIG)
    assert [(f.frame_index, f.na_reason) for f in result] == [
        (0, "video_unreadable"),
        (2, "video_unreadable"),
    ]
    assert model.calls == []
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    result = detect_frames(broken, "v1", [0, 1], model, CONFIG)
    assert [f.na_reason for f in result] == ["video_unreadable"] * 2


def test_detect_frames_empty_request_returns_empty_list(video: Path):
    assert detect_frames(video, "v1", [], FakeModel(), CONFIG) == []


def test_detect_video_covers_every_frame(video: Path):
    model = FakeModel()
    result = detect_video(video, "v1", model, CONFIG)
    assert [f.frame_index for f in result] == [0, 1, 2, 3, 4]
    assert len(model.calls) == 5


def test_detect_video_unreadable_returns_one_na_entry(tmp_path: Path):
    result = detect_video(tmp_path / "absent.mp4", "v1", FakeModel(), CONFIG)
    assert result == [FrameDetections("v1", 0, (), "video_unreadable")]


def test_check_plate_names_accepts_single_plate_class():
    assert check_plate_names({0: "plate"}) is None


@pytest.mark.parametrize(
    "names",
    [
        {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus"},
        {},
        {0: "plate", 1: "person"},
        {0: "plates"},
        {1: "plate"},
    ],
)
def test_check_plate_names_rejects_other_names_and_shows_them(names):
    with pytest.raises(ValueError) as excinfo:
        check_plate_names(names)
    message = str(excinfo.value)
    assert "plate" in message and "COCO" in message
    for name in list(names.values())[:5]:
        assert name in message
    if len(names) > 5:
        assert "bus" not in message  # only the first 5 names are shown


def test_load_ultralytics_model_without_weights_path_raises_before_importing_ultralytics():
    code = (
        "import sys\n"
        "from wl_snatch.config import DetectorConfig\n"
        "from wl_snatch.detector import load_ultralytics_model\n"
        "try:\n"
        "    load_ultralytics_model(DetectorConfig(model_name='yolo26'))\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    sys.exit(2)\n"
        "bad = [m for m in sys.modules if m == 'ultralytics' or m.startswith('ultralytics.')]\n"
        "sys.exit(1 if bad else 0)\n"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_module_import_does_not_import_ultralytics():
    code = (
        "import sys\n"
        "import wl_snatch.config, wl_snatch.detector\n"
        "bad = [m for m in sys.modules if m == 'ultralytics' or m.startswith('ultralytics.')]\n"
        "sys.exit(1 if bad else 0)\n"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_detector_writes_no_image_file(video: Path, tmp_path: Path):
    before = set(tmp_path.rglob("*"))
    detect_frames(video, "v1", [0, 1, 2], FakeModel(), CONFIG)
    after = set(tmp_path.rglob("*"))
    assert after == before
    images = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    assert not [p for p in after if p.suffix.lower() in images]


@pytest.mark.parametrize("request_", [[-1], [-1, -2], [-1, 3], [3]])
def test_unreadable_video_is_unreadable_for_every_index_including_negative(
    tmp_path: Path, request_: list[int]
):
    model = FakeModel()
    result = detect_frames(tmp_path / "absent.mp4", "v1", request_, model, CONFIG)
    assert [f.frame_index for f in result] == sorted(request_)
    assert all(f.na_reason == "video_unreadable" for f in result)
    assert model.calls == []


def test_readable_video_with_only_negative_indices_is_out_of_range_without_model_call(
    video: Path,
):
    model = FakeModel()
    result = detect_frames(video, "v1", [-1, -3], model, CONFIG)
    assert [(f.frame_index, f.na_reason) for f in result] == [
        (-3, "frame_out_of_range"),
        (-1, "frame_out_of_range"),
    ]
    assert model.calls == []


def test_frame_generator_is_closed_when_model_raises(
    video: Path, monkeypatch: pytest.MonkeyPatch
):
    import wl_snatch.detector as detector

    closed = []
    original = detector.read_frames

    def tracking(path, indices):
        try:
            yield from original(path, indices)
        finally:
            closed.append(True)

    monkeypatch.setattr(detector, "read_frames", tracking)

    def boom(frame):
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError) as excinfo:
        detect_frames(video, "v1", [0, 1], boom, CONFIG)
    assert closed == [True]  # closed while the exception is still held
    assert excinfo.value is not None
