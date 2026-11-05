import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from wl_snatch.label_viewer import (
    BOX_COLOR_BGR,
    Overlay,
    draw_overlay,
    flag_box_issues,
    render_label_frame,
)


def blank(h: int = 100, w: int = 200) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_draw_overlay_does_not_modify_input():
    frame = blank()
    before = frame.copy()
    result = draw_overlay(frame, Overlay(boxes=((0.5, 0.5, 0.4, 0.4),), skipped=()))
    assert np.array_equal(frame, before)
    assert result is not frame


def test_draw_overlay_changes_pixels_at_box_edge():
    frame = blank()
    # box spans x 60..140, y 30..70 in pixels
    result = draw_overlay(frame, Overlay(boxes=((0.5, 0.5, 0.4, 0.4),), skipped=()))
    color = np.array(BOX_COLOR_BGR)
    for y, x in [(50, 60), (50, 140), (30, 100), (70, 100)]:
        assert np.array_equal(result[y, x], color)
    for y, x in [(50, 59), (50, 141), (29, 100), (71, 100)]:  # one step outside
        assert np.array_equal(result[y, x], frame[y, x])
    for y, x in [(50, 61), (50, 139), (31, 100), (69, 100)]:  # one step inside
        assert np.array_equal(result[y, x], frame[y, x])


def test_draw_overlay_rounds_non_integer_corners():
    # x edges at 59.7 and 140.3 pixels -> 60 and 140
    result = draw_overlay(blank(), Overlay(boxes=((0.5, 0.5, 0.403, 0.4),), skipped=()))
    color = np.array(BOX_COLOR_BGR)
    assert np.array_equal(result[50, 60], color)
    assert np.array_equal(result[50, 140], color)
    assert not result[50, 59].any()
    assert not result[50, 141].any()


def test_draw_overlay_box_clipped_by_frame_edge_draws_in_frame_edges():
    # x spans 150..230 (frame width 200), y spans 30..70
    result = draw_overlay(blank(), Overlay(boxes=((0.95, 0.5, 0.4, 0.4),), skipped=()))
    color = np.array(BOX_COLOR_BGR)
    assert np.array_equal(result[50, 150], color)  # left edge, inside the frame
    assert np.array_equal(result[30, 199], color)  # top edge reaches the frame border
    assert np.array_equal(result[70, 199], color)
    assert not result[50, 151].any()


def test_draw_overlay_empty_boxes_returns_equal_copy():
    frame = np.random.default_rng(0).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    result = draw_overlay(frame, Overlay(boxes=(), skipped=()))
    assert np.array_equal(result, frame)
    assert result is not frame


def test_draw_overlay_box_partly_outside_frame_does_not_raise():
    overlay = Overlay(boxes=((0.95, 0.95, 0.4, 0.4), (-0.1, 0.5, 0.5, 0.5)), skipped=())
    result = draw_overlay(blank(), overlay)
    assert result.shape == (100, 200, 3)


def test_flag_box_issues_flags_width_0_024_height_0_0047():
    issues = flag_box_issues(((0.5, 0.5, 0.024, 0.0047),))
    assert len(issues) == 1
    assert "height" in issues[0]
    assert "0.0047" in issues[0]
    assert "width" not in issues[0]


def test_flag_box_issues_boundary_values():
    assert flag_box_issues(((0.5, 0.5, 0.02, 0.9),)) == []
    below = flag_box_issues(((0.5, 0.5, 0.0199, 0.5),))
    assert len(below) == 1 and "width" in below[0]
    above = flag_box_issues(((0.5, 0.5, 0.5, 0.9001),))
    assert len(above) == 1 and "height" in above[0]


def test_flag_box_issues_passes_normal_plate_box():
    assert flag_box_issues(((0.5, 0.5, 0.2, 0.2),)) == []


def test_flag_box_issues_threshold_is_configurable():
    boxes = ((0.5, 0.5, 0.2, 0.2),)
    assert flag_box_issues(boxes, min_side=0.3) != []
    assert flag_box_issues(boxes, max_side=0.1) != []
    assert flag_box_issues(((0.5, 0.5, 0.95, 0.5),)) != []
    assert flag_box_issues(((0.5, 0.5, 0.95, 0.5),), max_side=0.99) == []


def test_render_label_frame_reports_skipped_reasons(tmp_path: Path, make_video):
    video = make_video(tmp_path / "v.mp4")
    overlay = Overlay(
        boxes=((0.5, 0.5, 0.3, 0.3), (0.5, 0.5, 0.024, 0.0047)),
        skipped=("frame 2: N/A (label not provided)",),
    )
    array, notes = render_label_frame(video, 2, overlay)
    assert array.shape == (48, 64, 3)
    assert "frame 2: N/A (label not provided)" in notes
    assert any("0.0047" in n for n in notes)


def test_no_image_file_is_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_video):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    video = make_video(tmp_path / "v.mp4")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    def listing() -> set[str]:
        return {
            os.path.join(root, name)
            for base in (tmp_path, Path(tempfile.gettempdir()))
            for root, _, names in os.walk(base)
            for name in names
        }

    before = listing()
    render_label_frame(video, 1, Overlay(boxes=((0.5, 0.5, 0.3, 0.3),), skipped=()))
    assert listing() == before
