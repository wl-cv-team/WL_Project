from pathlib import Path

import numpy as np
import pytest

from wl_snatch.frames import count_frames, frame_times_ms, read_frame

DOMINANT = [0, 1, 2, 0, 1]


@pytest.fixture
def video(tmp_path: Path, make_video) -> Path:
    return make_video(tmp_path / "synthetic.mp4")


def test_read_frame_returns_requested_index(video: Path):
    for index, channel in enumerate(DOMINANT):
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


def test_frame_times_start_at_zero_and_have_one_entry_per_frame(video: Path):
    times = frame_times_ms(video)
    assert len(times) == 5
    assert times[0] == 0.0
    assert all(b > a for a, b in zip(times, times[1:]))
