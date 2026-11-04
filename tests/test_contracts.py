from pathlib import Path

import pytest

from wl_snatch.contracts import (
    BoxDetection,
    FrameDetections,
    read_detections_csv,
    write_detections_csv,
)


def _box(frame_index: int, x1: float = 10.0) -> BoxDetection:
    return BoxDetection(frame_index, x1, 20.0, 110.5, 220.25, 0.87, "plate")


def _sample_frames() -> list[FrameDetections]:
    return [
        FrameDetections("vid_a", 0, (_box(0), _box(0, 300.0)), None),
        FrameDetections("vid_a", 1, (), None),
        FrameDetections("vid_a", 2, (), "frame unreadable"),
        FrameDetections("vid_a", 3, (_box(3),), None),
    ]


def test_na_frame_must_have_no_boxes():
    with pytest.raises(ValueError):
        FrameDetections("vid_a", 0, (_box(0),), "frame unreadable")


def test_csv_roundtrip_preserves_frames(tmp_path: Path):
    frames = _sample_frames()
    path = tmp_path / "det.csv"
    write_detections_csv(frames, path)
    assert read_detections_csv(path) == frames


def test_zero_detections_differs_from_na(tmp_path: Path):
    path = tmp_path / "det.csv"
    write_detections_csv(_sample_frames(), path)
    frames = read_detections_csv(path)
    zero, na = frames[1], frames[2]
    assert zero.boxes == () and zero.na_reason is None
    assert na.boxes == () and na.na_reason == "frame unreadable"
    assert zero != na
