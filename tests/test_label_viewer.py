import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from wl_snatch.label_viewer import Overlay, draw_overlay, flag_box_issues, render_label_frame


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
    assert result[50, 60:62].any()
    assert result[30:32, 100].any()
    assert not result[50, 100].any()  # interior is unchanged


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
    assert "0.0047" in issues[0]


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
