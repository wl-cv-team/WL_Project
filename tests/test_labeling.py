import os
from pathlib import Path

import pytest

from wl_snatch import labeling
from wl_snatch.labeling import (
    EVENTS,
    LabelingSession,
    PointLabel,
    PointLabelStore,
    ignored_events,
    required_events,
    shuffled_order,
)


@pytest.fixture
def store(tmp_path: Path) -> PointLabelStore:
    return PointLabelStore(tmp_path / "labels" / "point_labels.csv")


def add(store, video_id="v1", event="ready", frame=1, pass_no=1, completion="f", frame_count=10):
    store.add(video_id, event, frame, pass_no, completion=completion, frame_count=frame_count)


def test_recorded_point_is_saved_and_read_back_equal(store):
    add(store, "v1", "drive", 4, 1)
    assert store.read() == [PointLabel("v1", "drive", 4, 1)]
    assert PointLabelStore(store.path).read() == [PointLabel("v1", "drive", 4, 1)]


def test_recording_same_video_event_pass_overwrites(store):
    add(store, "v1", "drive", 4, 1)
    add(store, "v1", "drive", 6, 1)
    assert store.read() == [PointLabel("v1", "drive", 6, 1)]


def test_pass_one_and_pass_two_are_kept_separately_even_for_same_frame(store):
    add(store, "v1", "drive", 4, 1)
    add(store, "v1", "drive", 4, 2)
    assert store.read() == [PointLabel("v1", "drive", 4, 1), PointLabel("v1", "drive", 4, 2)]
    assert store.read(pass_no=2) == [PointLabel("v1", "drive", 4, 2)]


def test_pass_one_cannot_be_added_or_changed_after_pass_two_exists_for_that_video(store):
    add(store, "v1", "ready", 2, 1)
    add(store, "v1", "ready", 3, 2)
    with pytest.raises(ValueError) as excinfo:
        add(store, "v1", "ready", 5, 1)
    assert "v1" in str(excinfo.value)
    with pytest.raises(ValueError):
        add(store, "v1", "drive", 5, 1)
    assert store.read(pass_no=1) == [PointLabel("v1", "ready", 2, 1)]


def test_pass_one_of_another_video_is_unaffected_by_pass_two(store):
    add(store, "v1", "ready", 3, 2)
    add(store, "v2", "ready", 5, 1)
    add(store, "v2", "ready", 6, 1)
    assert store.read("v2") == [PointLabel("v2", "ready", 6, 1)]


def test_unknown_event_raises_value_error(store):
    with pytest.raises(ValueError) as excinfo:
        add(store, "v1", "lift", 2, 1)
    assert "v1" in str(excinfo.value)
    assert "lift" in str(excinfo.value)
    assert not store.path.exists()


@pytest.mark.parametrize("pass_no", [0, 3, -1])
def test_pass_other_than_one_or_two_raises_value_error(store, pass_no):
    with pytest.raises(ValueError) as excinfo:
        add(store, "v1", "ready", 2, pass_no)
    assert "v1" in str(excinfo.value)
    assert str(pass_no) in str(excinfo.value)


@pytest.mark.parametrize("frame", [-1, 10, 99])
def test_frame_outside_video_raises_value_error(store, frame):
    with pytest.raises(ValueError) as excinfo:
        add(store, "v1", "ready", frame, 1, frame_count=10)
    assert "v1" in str(excinfo.value)
    assert str(frame) in str(excinfo.value)


def test_half_video_rejects_catch_and_finish(store):
    for event in ("catch", "finish"):
        with pytest.raises(ValueError) as excinfo:
            add(store, "h1", event, 3, 1, completion="h")
        assert "h1" in str(excinfo.value)
        assert event in str(excinfo.value)
    add(store, "h1", "drive", 3, 1, completion="h")
    assert not any(p.event in ("catch", "finish") for p in store.read())


def test_required_events_full_and_half():
    assert required_events("f") == EVENTS
    assert required_events("h") == ("ready", "transition", "drive")


def test_ignored_events_full_and_half():
    assert ignored_events("f") == ()
    assert ignored_events("h") == ("catch", "finish")


def test_unknown_completion_raises_value_error():
    for function in (required_events, ignored_events):
        with pytest.raises(ValueError) as excinfo:
            function("x")
        assert "x" in str(excinfo.value)


def test_missing_events_lists_only_required_events_without_a_record(store):
    add(store, "h1", "ready", 1, 1, completion="h")
    add(store, "h1", "drive", 3, 2, completion="h")
    assert store.missing_events("h1", "h", 1) == ("transition", "drive")
    assert store.missing_events("h1", "h", 2) == ("ready", "transition")
    assert store.missing_events("h1", "f", 1) == ("transition", "drive", "catch", "finish")
    assert store.missing_events("other", "h", 1) == ("ready", "transition", "drive")


def test_malformed_row_raises_value_error_with_line_number(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("video_id,event,frame,pass\nv1,ready,2,1\nv1,drive,abc,1\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        PointLabelStore(path).read()
    assert "line 3" in str(excinfo.value)
    path.write_text("video_id,event,frame,pass\nv1,lift,2,1\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        PointLabelStore(path).read()
    assert "line 2" in str(excinfo.value)
    path.write_text("video_id,event,frame,pass\nv1,ready,2\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        PointLabelStore(path).read()
    assert "line 2" in str(excinfo.value)


def test_failed_add_leaves_existing_file_unchanged(store, monkeypatch):
    add(store, "v1", "ready", 2, 1)
    before = store.path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(labeling.os, "replace", fail)
    with pytest.raises(OSError):
        add(store, "v1", "drive", 4, 1)
    assert store.path.read_bytes() == before
    assert [p.name for p in store.path.parent.iterdir()] == [store.path.name]


def test_rows_are_sorted_by_video_pass_and_event_order(store):
    add(store, "b", "finish", 9, 1)
    add(store, "a", "catch", 7, 1)
    add(store, "a", "ready", 2, 1)
    add(store, "a", "drive", 5, 2)
    add(store, "a", "ready", 1, 2)
    keys = [(p.video_id, p.pass_no, p.event) for p in store.read()]
    assert keys == [
        ("a", 1, "ready"),
        ("a", 1, "catch"),
        ("a", 2, "ready"),
        ("a", 2, "drive"),
        ("b", 1, "finish"),
    ]


def test_shuffled_order_is_deterministic_for_seed_and_a_permutation():
    ids = [f"v{i}" for i in range(20)]
    first = shuffled_order(ids, 7)
    assert first == shuffled_order(ids, 7)
    assert sorted(first) == sorted(ids)
    assert first != shuffled_order(ids, 8)
    assert ids == [f"v{i}" for i in range(20)]


@pytest.fixture
def session_parts(tmp_path: Path, make_video):
    video = make_video(tmp_path / "synthetic.mp4")
    store = PointLabelStore(tmp_path / "ann" / "point_labels.csv")
    return video, store


def make_session(session_parts, pass_no=1, completion="f", video_id="v1"):
    video, store = session_parts
    return LabelingSession(video, video_id, store, pass_no, completion)


def test_step_clamps_at_first_and_last_frame(session_parts):
    session = make_session(session_parts)
    assert session.frame_count == 5
    assert session.frame == 0
    session.step(-1)
    assert session.frame == 0
    session.step(2)
    assert session.frame == 2
    session.step(100)
    assert session.frame == 4
    session.step(1)
    assert session.frame == 4


def test_goto_out_of_range_raises_value_error(session_parts):
    session = make_session(session_parts)
    for frame in (-1, 5):
        with pytest.raises(ValueError) as excinfo:
            session.goto(frame)
        assert str(frame) in str(excinfo.value)
    session.goto(3)
    assert session.frame == 3


def test_record_uses_current_frame(session_parts):
    session = make_session(session_parts)
    session.goto(3)
    session.record("drive")
    assert session_parts[1].read() == [PointLabel("v1", "drive", 3, 1)]
    assert session.visible_labels() == [PointLabel("v1", "drive", 3, 1)]


def test_pass_two_session_hides_pass_one_labels(session_parts):
    first = make_session(session_parts, pass_no=1)
    first.goto(2)
    first.record("ready")
    second = make_session(session_parts, pass_no=2)
    assert second.visible_labels() == []
    second.goto(3)
    second.record("ready")
    assert second.visible_labels() == [PointLabel("v1", "ready", 3, 2)]
    text = " ".join(second.view()[1])
    assert "ready=3" in text
    assert "ready=2" not in text


def test_view_text_reports_missing_and_ignored_events(session_parts, dominant_channels):
    session = make_session(session_parts, completion="h")
    session.goto(2)
    session.record("ready")
    frame, lines = session.view()
    assert frame.shape == (48, 64, 3)
    assert int(frame.reshape(-1, 3).mean(axis=0).argmax()) == dominant_channels[2]
    text = "\n".join(lines)
    assert "frame 2 / 5" in text
    assert "ready=2" in text
    assert "missing: transition, drive" in text
    assert "ignored: catch, finish" in text


def test_session_and_store_write_no_image_file(session_parts, tmp_path):
    session = make_session(session_parts)
    session.goto(1)
    session.record("ready")
    session.view()
    files = {p for p in tmp_path.rglob("*") if p.is_file()}
    assert files == {session_parts[0], session_parts[1].path}
    assert not any(p.suffix.lower() in {".png", ".jpg", ".jpeg"} for p in files)
    assert os.path.exists(session_parts[1].path)


@pytest.mark.parametrize(
    ("frame", "pass_no"),
    [(True, 1), (3.0, 1), (2, True), (2, 1.0), (2, 2.0), ("2", 1)],
)
def test_non_int_frame_or_pass_is_rejected_and_file_unchanged(store, frame, pass_no):
    add(store, "v1", "ready", 2, 1)
    before = store.path.read_bytes()
    with pytest.raises(ValueError) as excinfo:
        add(store, "v1", "drive", frame, pass_no)
    assert "v1" in str(excinfo.value)
    assert repr(frame if frame in (True, 3.0, "2") else pass_no) in str(excinfo.value)
    assert store.path.read_bytes() == before
    assert store.read() == [PointLabel("v1", "ready", 2, 1)]


@pytest.mark.parametrize("video_id", ["", "  ", "\t\n"])
def test_empty_or_blank_video_id_is_rejected_and_file_stays_readable(store, video_id):
    add(store, "v1", "ready", 2, 1)
    with pytest.raises(ValueError):
        add(store, video_id, "ready", 2, 1)
    assert store.read() == [PointLabel("v1", "ready", 2, 1)]


def test_blank_video_id_is_rejected_when_no_file_exists(store):
    with pytest.raises(ValueError):
        add(store, "", "ready", 2, 1)
    assert not store.path.exists()


def test_zero_byte_file_reads_as_empty_and_add_writes_header_and_row(tmp_path):
    path = tmp_path / "p.csv"
    path.write_bytes(b"")
    store = PointLabelStore(path)
    assert store.read() == []
    add(store, "v1", "ready", 2, 1)
    assert path.read_text(encoding="utf-8").splitlines() == [
        "video_id,event,frame,pass",
        "v1,ready,2,1",
    ]


def test_duplicate_key_rows_in_file_raise_value_error_with_line_numbers(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text(
        "video_id,event,frame,pass\nv1,ready,2,1\nv1,drive,4,1\nv1,ready,3,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        PointLabelStore(path).read()
    message = str(excinfo.value)
    assert "line 4" in message
    assert "line 2" in message
    assert "v1" in message
    assert "ready" in message


def test_video_id_with_comma_quote_and_newline_round_trips(store):
    ids = ['a,b', 'a"b', "a\nb", 'x\n,"y']
    for index, video_id in enumerate(ids):
        add(store, video_id, "ready", index, 1)
    labels = store.read()
    assert sorted(label.video_id for label in labels) == sorted(ids)
    assert len(labels) == len(ids)
    assert {label.video_id: label.frame for label in labels} == {v: i for i, v in enumerate(ids)}


def test_guidance_html_escapes_markup_in_video_id_and_messages():
    html_text = labeling.guidance_html(["<b>x</b>  pass 1", "a & b"])
    assert "<b>" not in html_text
    assert "&lt;b&gt;x&lt;/b&gt;" in html_text
    assert "a &amp; b" in html_text
    assert "<br>" in html_text
