import os
from pathlib import Path

import pytest

from wl_snatch import plate_boxes
from wl_snatch.frames import read_frame
from wl_snatch.label_viewer import BOX_COLOR_BGR
from wl_snatch.plate_boxes import (
    BoxLabelingSession,
    FrameRange,
    PlateBoxStore,
    corners_to_norm_box,
    format_yolo_line,
    parse_yolo_line,
    read_frame_ranges,
)

YELLOW_BGR = (0, 255, 255)
VIDEO_ID = "v01"


def close(a, b, tolerance=1e-9):
    return all(abs(x - y) <= tolerance for x, y in zip(a, b, strict=True))


def write_ranges(path: Path, *rows: str) -> Path:
    path.write_text("video_id,start_frame,end_frame,reason\n" + "".join(r + "\n" for r in rows))
    return path


def make_store(tmp_path: Path, *range_rows: str) -> PlateBoxStore:
    ranges = write_ranges(tmp_path / "ranges.csv", *range_rows) if range_rows else None
    return PlateBoxStore(tmp_path / "plate_boxes", ranges)


# corners_to_norm_box


def test_corners_to_norm_box_matches_hand_computed_values():
    box = corners_to_norm_box(40, 80, 120, 160, 200, 400)
    assert close(box, (0.4, 0.3, 0.4, 0.2))


def test_corner_order_does_not_matter():
    expected = corners_to_norm_box(40, 80, 120, 160, 200, 400)
    assert close(corners_to_norm_box(120, 160, 40, 80, 200, 400), expected)
    assert close(corners_to_norm_box(120, 80, 40, 160, 200, 400), expected)


def test_corners_to_norm_box_same_for_proportional_display_and_original_size():
    shown = corners_to_norm_box(90, 160, 270, 480, 360, 640)
    original = corners_to_norm_box(270, 480, 810, 1440, 1080, 1920)
    assert close(shown, original, 1e-12)


def test_corners_are_clipped_to_frame_and_box_never_leaves_zero_to_one():
    cx, cy, w, h = corners_to_norm_box(-50, -20, 300, 500, 200, 400)
    assert close((cx, cy, w, h), (0.5, 0.5, 1.0, 1.0))
    cx, cy, w, h = corners_to_norm_box(150, 300, 260, 460, 200, 400)
    assert close((cx, cy, w, h), (0.875, 0.875, 0.25, 0.25))
    for low, high in ((cx - w / 2, cx + w / 2), (cy - h / 2, cy + h / 2)):
        assert 0 <= low and high <= 1 + 1e-12


def test_box_with_no_area_after_clipping_raises_value_error():
    with pytest.raises(ValueError):
        corners_to_norm_box(250, 50, 300, 100, 200, 400)  # entirely right of the frame
    with pytest.raises(ValueError):
        corners_to_norm_box(50, 50, 50, 100, 200, 400)  # width 0
    with pytest.raises(ValueError):
        corners_to_norm_box(50, 100, 80, 100, 200, 400)  # height 0


# YOLO line


def test_format_yolo_line_has_class_zero_and_six_decimals():
    line = format_yolo_line((0.4, 0.3, 0.4, 0.2))
    assert line == "0 0.400000 0.300000 0.400000 0.200000"
    assert "\n" not in format_yolo_line((1 / 3, 0.5, 0.25, 0.125))
    assert format_yolo_line((1 / 3, 0.5, 0.25, 0.125)) == "0 0.333333 0.500000 0.250000 0.125000"


def test_parse_yolo_line_round_trips_and_rejects_bad_lines():
    box = (0.123457, 0.5, 0.25, 0.125)
    assert close(parse_yolo_line(format_yolo_line(box), "x.txt"), box, 1e-9)
    assert close(parse_yolo_line("0 0.5 0.5 0.5 0.5\n", "x.txt"), (0.5, 0.5, 0.5, 0.5))
    bad = [
        "",
        "0 0.5 0.5 0.5",
        "0 0.5 0.5 0.5 0.5 0.5",
        "1 0.5 0.5 0.5 0.5",
        "0 1.5 0.5 0.5 0.5",
        "0 -0.1 0.5 0.5 0.5",
        "0 0.5 0.5 0 0.5",
        "0 0.5 0.5 0.5 0",
        "0 abc 0.5 0.5 0.5",
        "0 nan 0.5 0.5 0.5",
    ]
    for text in bad:
        with pytest.raises(ValueError, match="source_marker.txt"):
            parse_yolo_line(text, "source_marker.txt")


# store


def test_saved_box_is_read_back_equal_within_six_decimals(tmp_path):
    store = make_store(tmp_path)
    box = (1 / 3, 0.4567891, 0.2, 0.1234567)
    store.save_box(VIDEO_ID, 2, box, frame_count=5)
    assert close(store.read(VIDEO_ID, 2), box, 5e-7)
    assert store.path_for(VIDEO_ID, 2) == tmp_path / "plate_boxes" / "v01_f2.txt"


def test_status_unlabeled_empty_and_box(tmp_path):
    store = make_store(tmp_path)
    assert store.status(VIDEO_ID, 0) == "unlabeled"
    store.save_empty(VIDEO_ID, 0, frame_count=5)
    assert store.status(VIDEO_ID, 0) == "empty"
    store.save_box(VIDEO_ID, 1, (0.5, 0.5, 0.2, 0.2), frame_count=5)
    assert store.status(VIDEO_ID, 1) == "box"
    store.path_for(VIDEO_ID, 3).write_text("  \n")
    assert store.status(VIDEO_ID, 3) == "empty"


def test_empty_file_and_missing_file_are_different_states(tmp_path):
    store = make_store(tmp_path)
    store.save_empty(VIDEO_ID, 0, frame_count=5)
    assert store.path_for(VIDEO_ID, 0).exists()
    assert store.path_for(VIDEO_ID, 0).stat().st_size == 0
    assert not store.path_for(VIDEO_ID, 1).exists()
    assert store.read(VIDEO_ID, 0) is None
    assert store.read(VIDEO_ID, 1) is None
    assert store.status(VIDEO_ID, 0) != store.status(VIDEO_ID, 1)


def test_read_rejects_malformed_files_with_file_name(tmp_path):
    store = make_store(tmp_path)
    store.path_for(VIDEO_ID, 0).parent.mkdir(parents=True)
    store.path_for(VIDEO_ID, 0).write_text("0 0.5 0.5 0.5 0.5\n0 0.5 0.5 0.1 0.1\n")
    with pytest.raises(ValueError, match="v01_f0.txt"):
        store.read(VIDEO_ID, 0)
    store.path_for(VIDEO_ID, 1).write_text("0 0.5 0.5\n")
    with pytest.raises(ValueError, match="v01_f1.txt"):
        store.read(VIDEO_ID, 1)


def test_saving_again_replaces_box_with_empty_and_empty_with_box(tmp_path):
    store = make_store(tmp_path)
    store.save_box(VIDEO_ID, 1, (0.5, 0.5, 0.2, 0.2), frame_count=5)
    store.save_empty(VIDEO_ID, 1, frame_count=5)
    assert store.status(VIDEO_ID, 1) == "empty"
    assert store.read(VIDEO_ID, 1) is None
    store.save_box(VIDEO_ID, 1, (0.3, 0.3, 0.1, 0.1), frame_count=5)
    assert close(store.read(VIDEO_ID, 1), (0.3, 0.3, 0.1, 0.1))
    store.save_box(VIDEO_ID, 1, (0.6, 0.6, 0.2, 0.2), frame_count=5)
    assert close(store.read(VIDEO_ID, 1), (0.6, 0.6, 0.2, 0.2))
    assert len(list(store.boxes_dir.iterdir())) == 1


def test_frame_outside_video_raises_value_error(tmp_path):
    store = make_store(tmp_path)
    for frame in (-1, 5, 99):
        with pytest.raises(ValueError, match=f"{VIDEO_ID}.*{frame}"):
            store.save_empty(VIDEO_ID, frame, frame_count=5)
        with pytest.raises(ValueError, match=VIDEO_ID):
            store.save_box(VIDEO_ID, frame, (0.5, 0.5, 0.2, 0.2), frame_count=5)
    assert not store.boxes_dir.exists() or not list(store.boxes_dir.iterdir())


def test_bool_or_float_frame_raises_value_error(tmp_path):
    store = make_store(tmp_path)
    for frame in (True, False, 1.0, 2.5, "1"):
        with pytest.raises(ValueError, match=VIDEO_ID):
            store.save_empty(VIDEO_ID, frame, frame_count=5)
        with pytest.raises(ValueError, match=VIDEO_ID):
            store.save_box(VIDEO_ID, frame, (0.5, 0.5, 0.2, 0.2), frame_count=5)


def test_blank_video_id_raises_value_error(tmp_path):
    store = make_store(tmp_path)
    for video_id in ("", "   ", None):
        with pytest.raises(ValueError):
            store.save_empty(video_id, 0, frame_count=5)
        with pytest.raises(ValueError):
            store.save_box(video_id, 0, (0.5, 0.5, 0.2, 0.2), frame_count=5)


def test_box_outside_zero_to_one_or_without_area_is_not_saved(tmp_path):
    store = make_store(tmp_path)
    for box in ((0.5, 0.5, 0.0, 0.2), (0.5, 0.5, 0.2, -0.1), (1.2, 0.5, 0.2, 0.2)):
        with pytest.raises(ValueError):
            store.save_box(VIDEO_ID, 0, box, frame_count=5)
    assert store.status(VIDEO_ID, 0) == "unlabeled"


def test_failed_save_leaves_existing_file_unchanged(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.save_box(VIDEO_ID, 1, (0.5, 0.5, 0.2, 0.2), frame_count=5)
    path = store.path_for(VIDEO_ID, 1)
    before = path.read_bytes()

    def failing_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(plate_boxes.os, "replace", failing_replace)
    with pytest.raises(OSError):
        store.save_box(VIDEO_ID, 1, (0.1, 0.1, 0.1, 0.1), frame_count=5)
    with pytest.raises(OSError):
        store.save_empty(VIDEO_ID, 1, frame_count=5)
    assert path.read_bytes() == before
    assert [p.name for p in store.boxes_dir.iterdir()] == [path.name]


# frame range exceptions


def test_read_frame_ranges_valid_and_missing_file(tmp_path):
    assert read_frame_ranges(tmp_path / "missing.csv") == {}
    path = write_ranges(tmp_path / "r.csv", "b02,0,53,plate_leaves_frame", "c01,10,20,other")
    assert read_frame_ranges(path) == {
        "b02": FrameRange(0, 53, "plate_leaves_frame"),
        "c01": FrameRange(10, 20, "other"),
    }


def test_read_frame_ranges_rejects_bad_rows_with_line_number(tmp_path):
    cases = [
        ("line 3", ("a,0,5,x", "b,zero,5,x")),
        ("line 2", ("a,9,5,x",)),
        ("line 3", ("a,0,5,x", "a,1,2,y")),
        ("line 2", (",0,5,x",)),
    ]
    for expected, rows in cases:
        path = write_ranges(tmp_path / "r.csv", *rows)
        with pytest.raises(ValueError, match=expected):
            read_frame_ranges(path)
    wrong_header = tmp_path / "h.csv"
    wrong_header.write_text("video,start,end,reason\n")
    with pytest.raises(ValueError, match="line 1"):
        read_frame_ranges(wrong_header)


def test_frame_outside_range_exception_is_rejected_for_box_and_empty(tmp_path):
    store = make_store(tmp_path, "v01,1,3,plate_leaves_frame")
    for frame in (0, 4):
        with pytest.raises(ValueError, match=f"{VIDEO_ID}.*{frame}"):
            store.save_empty(VIDEO_ID, frame, frame_count=10)
        with pytest.raises(ValueError, match=VIDEO_ID):
            store.save_box(VIDEO_ID, frame, (0.5, 0.5, 0.2, 0.2), frame_count=10)
        assert store.status(VIDEO_ID, frame) == "unlabeled"


def test_frame_at_range_edges_is_accepted(tmp_path):
    store = make_store(tmp_path, "v01,1,3,plate_leaves_frame")
    store.save_empty(VIDEO_ID, 1, frame_count=10)
    store.save_box(VIDEO_ID, 3, (0.5, 0.5, 0.2, 0.2), frame_count=10)
    assert store.status(VIDEO_ID, 1) == "empty"
    assert store.status(VIDEO_ID, 3) == "box"


def test_video_without_range_exception_accepts_any_frame_in_video(tmp_path):
    store = make_store(tmp_path, "other_video,1,3,x")
    store.save_empty(VIDEO_ID, 0, frame_count=10)
    store.save_empty(VIDEO_ID, 9, frame_count=10)
    assert store.status(VIDEO_ID, 9) == "empty"


# session


def test_session_step_clamps_and_goto_out_of_range_raises(make_video, tmp_path):
    session = BoxLabelingSession(make_video(tmp_path / "v.mp4"), VIDEO_ID, make_store(tmp_path))
    assert (session.frame, session.frame_count) == (0, 5)
    session.step(-1)
    assert session.frame == 0
    session.step(2)
    assert session.frame == 2
    session.step(100)
    assert session.frame == 4
    session.goto(1)
    assert session.frame == 1
    for frame in (-1, 5):
        with pytest.raises(ValueError):
            session.goto(frame)
    assert session.frame == 1


def test_session_move_clears_draft(make_video, tmp_path):
    session = BoxLabelingSession(make_video(tmp_path / "v.mp4"), VIDEO_ID, make_store(tmp_path))
    session.set_draft_from_corners(10, 10, 50, 40, 64, 48)
    assert session.draft is not None
    session.step(1)
    assert session.draft is None
    session.set_draft_from_corners(10, 10, 50, 40, 64, 48)
    session.goto(3)
    assert session.draft is None
    session.set_draft_from_corners(10, 10, 50, 40, 64, 48)
    session.clear_draft()
    assert session.draft is None


def test_session_save_without_draft_raises(make_video, tmp_path):
    store = make_store(tmp_path)
    session = BoxLabelingSession(make_video(tmp_path / "v.mp4"), VIDEO_ID, store)
    with pytest.raises(ValueError):
        session.save_box()
    assert store.status(VIDEO_ID, 0) == "unlabeled"


def test_session_save_box_and_empty_use_current_frame(make_video, tmp_path):
    store = make_store(tmp_path)
    session = BoxLabelingSession(make_video(tmp_path / "v.mp4"), VIDEO_ID, store)
    session.goto(2)
    session.set_draft_from_corners(16, 12, 48, 36, 64, 48)
    session.save_box()
    assert close(store.read(VIDEO_ID, 2), (0.5, 0.5, 0.5, 0.5))
    session.goto(3)
    session.save_empty()
    assert store.status(VIDEO_ID, 3) == "empty"


def drag(session, start, end, width, height):
    base = {"boundingRectWidth": width, "boundingRectHeight": height}
    session.handle_pointer_event({"type": "mousedown", "relativeX": start[0], "relativeY": start[1], **base})
    session.handle_pointer_event({"type": "mousemove", "relativeX": 100, "relativeY": 100, **base})
    session.handle_pointer_event({"type": "mouseup", "relativeX": end[0], "relativeY": end[1], **base})


def portrait_session(make_video, tmp_path, frames=2):
    video = make_video(tmp_path / "p.mp4", colors=[(255, 0, 0)] * frames, size=(72, 128))
    return BoxLabelingSession(video, VIDEO_ID, make_store(tmp_path))


def test_pointer_drag_events_set_draft_equal_to_corner_computation(make_video, tmp_path):
    session = portrait_session(make_video, tmp_path)
    drag(session, (90, 160), (270, 480), 360, 640)
    expected = corners_to_norm_box(90, 160, 270, 480, 360, 640)
    assert close(session.draft, expected)
    shown = session.draft
    session.clear_draft()
    drag(session, (18, 32), (54, 96), 72, 128)  # same drag at the original size
    assert close(session.draft, shown, 1e-12)
    session.clear_draft()
    drag(session, (270, 480), (90, 160), 360, 640)  # reverse direction
    assert close(session.draft, shown, 1e-12)


def test_pointer_mouseup_without_mousedown_is_ignored(make_video, tmp_path):
    session = portrait_session(make_video, tmp_path)
    session.handle_pointer_event(
        {"type": "mouseup", "relativeX": 10, "relativeY": 10, "boundingRectWidth": 360, "boundingRectHeight": 640}
    )
    assert session.draft is None


def test_pointer_mouseup_clears_start_so_a_second_mouseup_is_ignored(make_video, tmp_path):
    session = portrait_session(make_video, tmp_path)
    drag(session, (90, 160), (270, 480), 360, 640)
    session.clear_draft()
    session.handle_pointer_event(
        {"type": "mouseup", "relativeX": 300, "relativeY": 500, "boundingRectWidth": 360, "boundingRectHeight": 640}
    )
    assert session.draft is None


def test_pointer_mousemove_updates_text_but_not_draft(make_video, tmp_path):
    session = portrait_session(make_video, tmp_path)
    assert session.pointer_text == ""
    base = {"boundingRectWidth": 360, "boundingRectHeight": 640}
    session.handle_pointer_event({"type": "mousemove", "relativeX": 36, "relativeY": 64, **base})
    assert session.pointer_text != ""
    assert session.draft is None
    session.handle_pointer_event({"type": "mousedown", "relativeX": 90, "relativeY": 160, **base})
    session.handle_pointer_event({"type": "mousemove", "relativeX": 200, "relativeY": 300, **base})
    assert session.draft is None
    session.handle_pointer_event({"type": "mouseup", "relativeX": 270, "relativeY": 480, **base})
    assert session.draft is not None  # the start point survived the mousemove


def test_pointer_event_with_zero_bounding_rect_is_ignored(make_video, tmp_path):
    session = portrait_session(make_video, tmp_path)
    for width, height in ((0, 640), (360, 0), (-1, 640), (360, -5)):
        session.handle_pointer_event(
            {"type": "mousedown", "relativeX": 10, "relativeY": 10, "boundingRectWidth": width, "boundingRectHeight": height}
        )
        session.handle_pointer_event(
            {"type": "mousemove", "relativeX": 10, "relativeY": 10, "boundingRectWidth": width, "boundingRectHeight": height}
        )
    assert session.pointer_text == ""
    # a mousedown that was ignored does not provide a start point
    session.handle_pointer_event(
        {"type": "mouseup", "relativeX": 90, "relativeY": 90, "boundingRectWidth": 360, "boundingRectHeight": 640}
    )
    assert session.draft is None


def test_pointer_unknown_event_type_is_ignored(make_video, tmp_path):
    session = portrait_session(make_video, tmp_path)
    base = {"relativeX": 10, "relativeY": 10, "boundingRectWidth": 360, "boundingRectHeight": 640}
    for kind in ("click", "mouseleave", "wheel", ""):
        session.handle_pointer_event({"type": kind, **base})
    session.handle_pointer_event({"relativeX": 10})  # no type, no size
    assert session.draft is None
    assert session.pointer_text == ""


def test_pointer_click_without_drag_raises_and_leaves_no_start_point(make_video, tmp_path):
    session = portrait_session(make_video, tmp_path)
    base = {"relativeX": 90, "relativeY": 160, "boundingRectWidth": 360, "boundingRectHeight": 640}
    session.handle_pointer_event({"type": "mousedown", **base})
    with pytest.raises(ValueError):
        session.handle_pointer_event({"type": "mouseup", **base})
    assert session.draft is None
    session.handle_pointer_event({"type": "mouseup", **base})  # start point is gone
    assert session.draft is None


def test_pointer_text_shows_fractions_and_original_pixels(make_video, tmp_path):
    session = portrait_session(make_video, tmp_path)  # original size 72 x 128
    session.handle_pointer_event(
        {"type": "mousemove", "relativeX": 90, "relativeY": 160, "boundingRectWidth": 360, "boundingRectHeight": 640}
    )
    assert session.pointer_text == "x 0.250  y 0.250  pixel (18, 32)"
    session.handle_pointer_event(
        {"type": "mousemove", "relativeX": 360, "relativeY": 640, "boundingRectWidth": 360, "boundingRectHeight": 640}
    )
    assert session.pointer_text == "x 1.000  y 1.000  pixel (71, 127)"


def test_session_view_draws_saved_box_green_and_draft_yellow(make_video, tmp_path):
    store = make_store(tmp_path)
    video = make_video(tmp_path / "v.mp4", colors=[(0, 0, 0)] * 3, size=(64, 48))
    session = BoxLabelingSession(video, VIDEO_ID, store)
    store.save_box(VIDEO_ID, 0, (0.25, 0.5, 0.25, 0.5), frame_count=3)  # x 8..24, y 12..36
    session.set_draft_from_corners(40, 12, 56, 36, 64, 48)
    image, _ = session.view()
    assert tuple(image[24, 8]) == BOX_COLOR_BGR
    assert tuple(image[12, 16]) == BOX_COLOR_BGR
    assert tuple(image[24, 40]) == YELLOW_BGR
    assert tuple(image[36, 48]) == YELLOW_BGR
    assert tuple(image[24, 8]) != YELLOW_BGR
    assert tuple(image[24, 40]) != BOX_COLOR_BGR
    # the frame itself is not modified on disk or in memory by drawing
    assert image.shape == read_frame(video, 0).shape
    session.goto(1)
    other, _ = session.view()
    assert tuple(other[24, 8]) != BOX_COLOR_BGR  # frame 1 has no saved box, and the draft is cleared
    assert tuple(other[24, 40]) != YELLOW_BGR


def test_session_view_text_reports_status_and_range_exception(make_video, tmp_path):
    store = make_store(tmp_path, "v01,1,3,plate_leaves_frame")
    session = BoxLabelingSession(make_video(tmp_path / "v.mp4"), VIDEO_ID, store)
    session.goto(1)
    _, lines = session.view()
    text = "\n".join(lines)
    assert "frame 1 / 5" in text
    assert "unlabeled" in text
    assert "1" in text and "3" in text and "plate_leaves_frame" in text
    session.save_empty()
    _, lines = session.view()
    assert any("empty" in line for line in lines)
    plain = BoxLabelingSession(make_video(tmp_path / "w.mp4"), "w01", make_store(tmp_path))
    _, plain_lines = plain.view()
    assert "plate_leaves_frame" not in "\n".join(plain_lines)


def test_session_view_text_includes_box_size_warning(make_video, tmp_path):
    store = make_store(tmp_path)
    session = BoxLabelingSession(make_video(tmp_path / "v.mp4"), VIDEO_ID, store)
    session.set_draft_from_corners(10, 10, 11, 40, 64, 48)  # width 1/64 is below min_side
    _, lines = session.view()
    assert any("width" in line and "below" in line for line in lines)


def test_session_and_store_write_no_image_file(make_video, tmp_path):
    video = make_video(tmp_path / "v.mp4")
    store = make_store(tmp_path)
    session = BoxLabelingSession(video, VIDEO_ID, store)
    session.set_draft_from_corners(10, 10, 50, 40, 64, 48)
    session.view()
    session.save_box()
    session.step(1)
    session.save_empty()
    session.view()
    written = sorted(p.name for p in store.boxes_dir.iterdir())
    assert written == ["v01_f0.txt", "v01_f1.txt"]
    everything = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert everything == ["plate_boxes/v01_f0.txt", "plate_boxes/v01_f1.txt", "v.mp4"]
    assert not any(name.startswith(".") for name in os.listdir(store.boxes_dir))
