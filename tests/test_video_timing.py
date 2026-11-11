from pathlib import Path

import pytest

from wl_snatch.video_timing import (
    TimingRow,
    analyze_timing,
    unreadable_row,
    write_timing_csv,
)


def _times(gaps: list[float], start: float = 0.0) -> list[float]:
    times = [start]
    for gap in gaps:
        times.append(times[-1] + gap)
    return times


def test_constant_30fps_has_no_note_and_matches_statistics():
    row = analyze_timing("v", 720, 1280, _times([1000 / 30] * 29))
    assert row.video_id == "v"
    assert (row.width, row.height, row.frames) == (720, 1280, 30)
    assert row.gap_min_ms == pytest.approx(1000 / 30)
    assert row.gap_median_ms == pytest.approx(1000 / 30)
    assert row.gap_max_ms == pytest.approx(1000 / 30)
    assert row.fps_from_median_gap == pytest.approx(30.0)
    assert row.first_frame_ms == 0.0
    assert row.timestamp_note == ""
    assert row.irregular_gap_frames == ()


def test_first_frame_offset_is_noted():
    row = analyze_timing("v", 10, 10, _times([1000 / 30] * 9, start=33.3))
    assert row.first_frame_ms == pytest.approx(33.3)
    assert row.timestamp_note == "first_frame_offset"
    assert row.irregular_gap_frames == ()


def test_small_first_frame_time_within_tolerance_is_not_noted():
    row = analyze_timing("v", 10, 10, _times([1000 / 30] * 9, start=0.4))
    assert row.timestamp_note == ""


def test_irregular_gap_is_noted_with_frame_number():
    gaps = [41.7] * 10
    gaps[4] = 8.7
    row = analyze_timing("v", 10, 10, _times(gaps))
    assert row.irregular_gap_frames == (4,)
    assert row.timestamp_note == "irregular_gap"
    assert row.gap_min_ms == pytest.approx(8.7)
    assert row.gap_median_ms == pytest.approx(41.7)


def test_timebase_rounding_at_60fps_is_not_flagged():
    gaps = [16.7, 16.7, 18.3] * 6
    row = analyze_timing("v", 10, 10, _times(gaps))
    assert row.irregular_gap_frames == ()
    assert row.timestamp_note == ""


def test_large_deviation_at_60fps_is_flagged():
    gaps = [16.7] * 10
    gaps[7] = 5.0
    row = analyze_timing("v", 10, 10, _times(gaps))
    assert row.irregular_gap_frames == (7,)
    assert row.timestamp_note == "irregular_gap"


def test_notes_join_both_with_semicolon_in_fixed_order():
    gaps = [41.7] * 10
    gaps[2] = 8.7
    row = analyze_timing("v", 10, 10, _times(gaps, start=20.0))
    assert row.timestamp_note == "first_frame_offset;irregular_gap"


@pytest.mark.parametrize("times", [[], [5.0]])
def test_fewer_than_two_frames_gives_na_note(times: list[float]):
    row = analyze_timing("v", 10, 10, times)
    assert row.timestamp_note == "N/A: fewer than 2 frames"
    assert row.frames == len(times)
    assert row.gap_min_ms is None
    assert row.gap_median_ms is None
    assert row.gap_max_ms is None
    assert row.fps_from_median_gap is None
    assert row.irregular_gap_frames == ()


def test_unreadable_row_has_na_note_and_empty_numbers():
    row = unreadable_row("v", "cannot be opened")
    assert row.video_id == "v"
    assert row.timestamp_note == "N/A: cannot be opened"
    for name in (
        "width",
        "height",
        "frames",
        "fps_from_median_gap",
        "gap_min_ms",
        "gap_median_ms",
        "gap_max_ms",
        "first_frame_ms",
    ):
        assert getattr(row, name) is None
    assert row.irregular_gap_frames == ()


def test_write_timing_csv_header_and_values(tmp_path: Path):
    rows = [
        TimingRow(
            video_id="a",
            width=720,
            height=1280,
            frames=10,
            fps_from_median_gap=24.0,
            gap_min_ms=8.7,
            gap_median_ms=41.666,
            gap_max_ms=41.7,
            first_frame_ms=0.0,
            timestamp_note="irregular_gap",
            irregular_gap_frames=(3, 7),
        ),
        unreadable_row("b", "cannot be opened"),
    ]
    out = tmp_path / "sub" / "timing.csv"
    write_timing_csv(rows, out)
    lines = out.read_text().splitlines()
    assert lines[0] == (
        "video_id,width,height,frames,fps_from_median_gap,gap_min_ms,"
        "gap_median_ms,gap_max_ms,first_frame_ms,timestamp_note,irregular_gap_frames"
    )
    assert lines[1] == "a,720,1280,10,24.00,8.70,41.67,41.70,0.00,irregular_gap,3;7"
    assert lines[2] == "b,,,,,,,,,N/A: cannot be opened,"
