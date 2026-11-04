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
        FrameDetections("vid_a", 0, (_box(0), _box(0, 100.0)), None),
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


# ---- validation and read-time rules ----

_HEADER = "video_id,frame_index,x1,y1,x2,y2,confidence,label,na_reason\n"


def _write_raw(tmp_path: Path, *rows: str) -> Path:
    path = tmp_path / "raw.csv"
    path.write_text(_HEADER + "".join(r + "\n" for r in rows), encoding="utf-8")
    return path


def test_empty_na_reason_rejected():
    with pytest.raises(ValueError):
        FrameDetections("v", 0, (), "")


def test_box_rejects_nan():
    with pytest.raises(ValueError):
        BoxDetection(0, float("nan"), 0.0, 1.0, 1.0, 0.5, "plate")
    with pytest.raises(ValueError):
        BoxDetection(0, 0.0, 0.0, 1.0, 1.0, float("nan"), "plate")


def test_box_rejects_inverted_coordinates():
    with pytest.raises(ValueError):
        BoxDetection(0, 5.0, 0.0, 1.0, 1.0, 0.5, "plate")
    with pytest.raises(ValueError):
        BoxDetection(0, 0.0, 5.0, 1.0, 1.0, 0.5, "plate")


def test_box_allows_negative_coordinates():
    BoxDetection(0, -3.0, -2.0, 1.0, 1.0, 0.5, "plate")


def test_boxless_frame_as_last_row_roundtrips(tmp_path: Path):
    frames = [
        FrameDetections("v", 0, (_box(0),), None),
        FrameDetections("v", 1, (), None),
    ]
    path = tmp_path / "det.csv"
    write_detections_csv(frames, path)
    assert read_detections_csv(path) == frames


def test_duplicate_non_contiguous_key_rejected(tmp_path: Path):
    path = _write_raw(
        tmp_path,
        "v,0,1,2,3,4,0.5,plate,",
        "v,1,1,2,3,4,0.5,plate,",
        "v,0,5,6,7,8,0.5,plate,",
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_detections_csv(path)


def test_conflicting_na_reason_for_same_key_rejected(tmp_path: Path):
    path = _write_raw(
        tmp_path,
        "v,0,1,2,3,4,0.5,plate,",
        "v,0,,,,,,,unreadable",
    )
    with pytest.raises(ValueError, match="na_reason"):
        read_detections_csv(path)


def test_boxless_row_mixed_with_box_rows_rejected(tmp_path: Path):
    path = _write_raw(
        tmp_path,
        "v,0,1,2,3,4,0.5,plate,",
        "v,0,,,,,,,",
    )
    with pytest.raises(ValueError):
        read_detections_csv(path)


@pytest.mark.parametrize(
    "row",
    [
        "v,0,1,2,3,4,0.5,plate,unreadable",  # na_reason with coordinates
        "v,0,1,2,,4,0.5,plate,",  # partial coordinates
        "v,0,1,abc,3,4,0.5,plate,",  # non-numeric
        "v,0,1,2,3,4,0.5,,",  # box row without label
    ],
)
def test_malformed_row_names_file_and_row_number(tmp_path: Path, row: str):
    path = _write_raw(tmp_path, "v,0,1,2,3,4,0.5,plate,", row)
    with pytest.raises(ValueError) as exc:
        read_detections_csv(path)
    assert str(path) in str(exc.value)
    assert "row 3" in str(exc.value)


def test_missing_required_column_rejected(tmp_path: Path):
    path = tmp_path / "raw.csv"
    path.write_text("video_id,frame_index,x1\nv,0,1\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        read_detections_csv(path)
    assert str(path) in str(exc.value)
    assert "y1" in str(exc.value)
