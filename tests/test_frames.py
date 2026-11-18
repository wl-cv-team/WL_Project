from pathlib import Path

import cv2
import numpy as np
import pytest

from wl_snatch.frames import (
    count_frames,
    frame_size,
    frame_times_ms,
    raw_frame_times_ms,
    read_frame,
    read_frames,
    VideoUnreadableError,
    check_readable,
)

@pytest.fixture
def video(tmp_path: Path, make_video) -> Path:
    return make_video(tmp_path / "synthetic.mp4")


def test_read_frame_returns_requested_index(video: Path, dominant_channels):
    for index, channel in enumerate(dominant_channels):
        frame = read_frame(video, index)
        assert frame.shape == (48, 64, 3)
        means = frame.reshape(-1, 3).mean(axis=0)
        assert int(np.argmax(means)) == channel
        assert means[channel] > 150


def test_read_frame_out_of_range_raises_value_error_with_name_and_index(video: Path):
    with pytest.raises(ValueError) as excinfo:
        read_frame(video, 5)
    assert "synthetic.mp4" in str(excinfo.value)
    assert "5" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        read_frame(video, -1)
    assert "synthetic.mp4" in str(excinfo.value)
    assert "-1" in str(excinfo.value)


def test_read_frame_missing_video_raises_value_error(tmp_path: Path):
    with pytest.raises(ValueError) as excinfo:
        read_frame(tmp_path / "absent.mp4", 3)
    assert "absent.mp4" in str(excinfo.value)
    assert "3" in str(excinfo.value)


def test_count_frames_matches_decoded_frames(video: Path):
    assert count_frames(video) == 5


def test_count_frames_missing_video_raises_value_error(tmp_path: Path):
    with pytest.raises(ValueError, match="absent.mp4"):
        count_frames(tmp_path / "absent.mp4")


@pytest.mark.parametrize("fps", [10.0, 25.0])
def test_frame_times_start_at_zero_and_have_one_entry_per_frame(
    tmp_path: Path, make_video, fps: float
):
    times = frame_times_ms(make_video(tmp_path / "v.mp4", fps=fps))
    expected = [i * 1000.0 / fps for i in range(5)]
    assert len(times) == 5
    assert times[0] == 0.0
    assert times == pytest.approx(expected, abs=3.0)


def test_frame_times_raises_when_backend_reports_identical_timestamps(
    video: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(cv2.VideoCapture, "get", lambda self, prop: 0.0)
    with pytest.raises(ValueError, match="synthetic.mp4.*timestamps unavailable"):
        frame_times_ms(video)


def test_frame_size_returns_width_height_of_decoded_frame(tmp_path: Path, make_video):
    video = make_video(tmp_path / "wide.mp4", size=(64, 48))
    assert frame_size(video) == (64, 48)


def test_frame_size_empty_video_raises_value_error(
    video: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(cv2.VideoCapture, "read", lambda self: (False, None))
    with pytest.raises(ValueError, match="synthetic.mp4"):
        frame_size(video)


def test_raw_frame_times_not_shifted_and_frame_times_are_shifted_copy(
    video: Path, monkeypatch: pytest.MonkeyPatch
):
    reported = iter([40.0, 80.0, 120.0, 160.0, 200.0])
    monkeypatch.setattr(cv2.VideoCapture, "get", lambda self, prop: next(reported))
    assert raw_frame_times_ms(video) == [40.0, 80.0, 120.0, 160.0, 200.0]
    reported = iter([40.0, 80.0, 120.0, 160.0, 200.0])
    monkeypatch.setattr(cv2.VideoCapture, "get", lambda self, prop: next(reported))
    assert frame_times_ms(video) == [0.0, 40.0, 80.0, 120.0, 160.0]


def _dominant(frame: np.ndarray) -> int:
    return int(np.argmax(frame.reshape(-1, 3).mean(axis=0)))


def test_read_frames_returns_requested_frames_in_ascending_order_once_each(
    video: Path, dominant_channels, monkeypatch: pytest.MonkeyPatch
):
    opened = []
    original = cv2.VideoCapture.__init__

    def counting_init(self, *args, **kwargs):
        opened.append(args)
        original(self, *args, **kwargs)

    monkeypatch.setattr(cv2.VideoCapture, "__init__", counting_init)
    result = list(read_frames(video, [3, 1, 3, 0]))
    assert [index for index, _ in result] == [0, 1, 3]
    for index, frame in result:
        assert frame.shape == (48, 64, 3)
        assert _dominant(frame) == dominant_channels[index]
    assert len(opened) == 1


def test_read_frames_out_of_range_raises_value_error_with_name_and_index(video: Path):
    with pytest.raises(ValueError) as excinfo:
        list(read_frames(video, [1, 7]))
    assert "synthetic.mp4" in str(excinfo.value)
    assert "7" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        list(read_frames(video, [-2]))
    assert "synthetic.mp4" in str(excinfo.value)
    assert "-2" in str(excinfo.value)


def test_read_frames_yields_reachable_frames_before_raising(video: Path):
    seen = []
    with pytest.raises(ValueError, match="7"):
        for index, _ in read_frames(video, [0, 2, 7]):
            seen.append(index)
    assert seen == [0, 2]


def test_read_frames_empty_request_yields_nothing(video: Path):
    assert list(read_frames(video, [])) == []


def test_read_frames_missing_video_raises_value_error(tmp_path: Path):
    with pytest.raises(ValueError, match="absent.mp4"):
        list(read_frames(tmp_path / "absent.mp4", [0]))


def test_check_readable_accepts_readable_video_without_decoding(
    video: Path, monkeypatch: pytest.MonkeyPatch
):
    def fail(self):
        raise AssertionError("must not decode")

    monkeypatch.setattr(cv2.VideoCapture, "read", fail)
    assert check_readable(video) is None


def test_check_readable_raises_unreadable_error_for_missing_and_corrupt(tmp_path: Path):
    with pytest.raises(VideoUnreadableError, match="absent.mp4"):
        check_readable(tmp_path / "absent.mp4")
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(VideoUnreadableError, match="broken.mp4"):
        check_readable(broken)
