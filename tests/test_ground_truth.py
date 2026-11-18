import pandas as pd
import pytest

from wl_snatch.eval_detection import GroundTruthFrame
from wl_snatch.ground_truth import (
    ROLES,
    GroundTruthReport,
    load_ground_truth,
    parse_label_name,
)


def make_videos(*rows):
    return pd.DataFrame(
        [
            {"video_id": v, "width": w, "height": h, "frame_count": n}
            for v, w, h, n in rows
        ]
    )


def make_split(*rows):
    return pd.DataFrame([{"video_id": v, "role": r} for v, r in rows])


@pytest.fixture
def videos():
    return make_videos(
        ("a01_sn_f_01", 720, 1280, 100),
        ("a02_sn_f_01", 1280, 720, 100),
        ("a03_sn_f_01", 720, 1280, 100),
    )


@pytest.fixture
def split():
    return make_split(
        ("a01_sn_f_01", "train"), ("a02_sn_f_01", "validation"), ("a03_sn_f_01", "train")
    )


def write(directory, video_id, frame, text):
    (directory / f"{video_id}_f{frame}.txt").write_text(text, encoding="utf-8")


def test_parse_label_name_reads_video_id_containing_f_and_frame_number():
    assert parse_label_name("a01_sn_f_01_f3.txt") == ("a01_sn_f_01", 3)
    assert parse_label_name("a01_sn_f_01_f120.txt") == ("a01_sn_f_01", 120)


@pytest.mark.parametrize(
    "name", ["a01_sn_f_01.txt", "a01_sn_f_01_f3.png", "a01_f.txt", "a01_f3x.txt", "_f3.txt", "x_f-1.txt"]
)
def test_parse_label_name_rejects_other_names_with_the_name(name):
    with pytest.raises(ValueError, match=name):
        parse_label_name(name)


def test_roles_are_train_and_validation():
    assert ROLES == ("train", "validation")


def test_box_file_becomes_pixel_box_from_normalized_values(tmp_path, videos, split):
    write(tmp_path, "a01_sn_f_01", 5, "0 0.5 0.25 0.5 0.125\n")
    frames, _ = load_ground_truth(tmp_path, videos, split, "train")
    # 720 x 1280: x1 = (0.5 - 0.25) * 720, y1 = (0.25 - 0.0625) * 1280
    assert frames == [GroundTruthFrame("a01_sn_f_01", 5, ((180.0, 240.0, 540.0, 400.0),))]


def test_pixel_conversion_uses_the_size_of_each_video(tmp_path, videos, split):
    write(tmp_path, "a02_sn_f_01", 1, "0 0.5 0.5 0.5 0.5\n")
    frames, _ = load_ground_truth(tmp_path, videos, split, "validation")
    # 1280 x 720
    assert frames[0].boxes == ((320.0, 180.0, 960.0, 540.0),)


def test_empty_file_becomes_frame_without_boxes(tmp_path, videos, split):
    write(tmp_path, "a01_sn_f_01", 2, "")
    frames, _ = load_ground_truth(tmp_path, videos, split, "train")
    assert frames == [GroundTruthFrame("a01_sn_f_01", 2, ())]


def test_only_requested_role_is_returned(tmp_path, videos, split):
    write(tmp_path, "a01_sn_f_01", 1, "")
    write(tmp_path, "a02_sn_f_01", 1, "")
    train, _ = load_ground_truth(tmp_path, videos, split, "train")
    validation, _ = load_ground_truth(tmp_path, videos, split, "validation")
    assert [f.video_id for f in train] == ["a01_sn_f_01"]
    assert [f.video_id for f in validation] == ["a02_sn_f_01"]


def test_unknown_role_raises_value_error(tmp_path, videos, split):
    with pytest.raises(ValueError, match="test"):
        load_ground_truth(tmp_path, videos, split, "test")


def test_unknown_role_value_in_split_raises_value_error_naming_the_video(tmp_path, videos):
    bad = make_split(("a01_sn_f_01", "train"), ("a02_sn_f_01", "holdout"))
    write(tmp_path, "a01_sn_f_01", 1, "")
    with pytest.raises(ValueError, match="a02_sn_f_01"):
        load_ground_truth(tmp_path, videos, bad, "train")


def test_same_video_never_appears_in_both_roles(tmp_path, videos, split):
    for video_id in ("a01_sn_f_01", "a02_sn_f_01", "a03_sn_f_01"):
        write(tmp_path, video_id, 1, "")
    train, _ = load_ground_truth(tmp_path, videos, split, "train")
    validation, _ = load_ground_truth(tmp_path, videos, split, "validation")
    assert {f.video_id for f in train}.isdisjoint({f.video_id for f in validation})
    assert len(train) + len(validation) == 3


def test_video_not_in_split_is_skipped_and_counted(tmp_path, videos):
    only_one = make_split(("a01_sn_f_01", "train"))
    write(tmp_path, "a01_sn_f_01", 1, "")
    write(tmp_path, "a02_sn_f_01", 1, "")
    write(tmp_path, "a03_sn_f_01", 1, "")
    frames, report = load_ground_truth(tmp_path, videos, only_one, "train")
    assert [f.video_id for f in frames] == ["a01_sn_f_01"]
    assert report.skipped_not_in_split == 2


def test_video_in_split_but_not_in_videos_raises_value_error(tmp_path, split):
    short = make_videos(("a01_sn_f_01", 720, 1280, 100))
    write(tmp_path, "a02_sn_f_01", 1, "")
    with pytest.raises(ValueError, match="a02_sn_f_01"):
        load_ground_truth(tmp_path, short, split, "train")


@pytest.mark.parametrize("frame", [100, 250])
def test_frame_outside_video_raises_value_error(tmp_path, videos, split, frame):
    write(tmp_path, "a01_sn_f_01", frame, "")
    with pytest.raises(ValueError, match=rf"a01_sn_f_01.*{frame}"):
        load_ground_truth(tmp_path, videos, split, "train")


def test_last_frame_inside_video_is_accepted(tmp_path, videos, split):
    write(tmp_path, "a01_sn_f_01", 99, "")
    frames, _ = load_ground_truth(tmp_path, videos, split, "train")
    assert len(frames) == 1


def test_results_are_sorted_by_video_then_numeric_frame(tmp_path, videos, split):
    write(tmp_path, "a03_sn_f_01", 1, "")
    write(tmp_path, "a01_sn_f_01", 10, "")
    write(tmp_path, "a01_sn_f_01", 3, "")
    write(tmp_path, "a01_sn_f_01", 2, "")
    frames, _ = load_ground_truth(tmp_path, videos, split, "train")
    assert [(f.video_id, f.frame_index) for f in frames] == [
        ("a01_sn_f_01", 2),
        ("a01_sn_f_01", 3),
        ("a01_sn_f_01", 10),
        ("a03_sn_f_01", 1),
    ]


def test_report_counts_boxes_and_empty_files(tmp_path, videos, split):
    write(tmp_path, "a01_sn_f_01", 1, "0 0.5 0.5 0.2 0.2\n")
    write(tmp_path, "a01_sn_f_01", 2, "")
    write(tmp_path, "a03_sn_f_01", 1, "0 0.4 0.4 0.2 0.2\n")
    write(tmp_path, "a02_sn_f_01", 1, "")
    frames, report = load_ground_truth(tmp_path, videos, split, "train")
    assert report == GroundTruthReport(
        frames=3,
        with_box=2,
        empty=1,
        skipped_not_in_split=0,
        video_ids=("a01_sn_f_01", "a03_sn_f_01"),
    )
    assert len(frames) == report.frames


def test_hidden_temp_files_are_ignored(tmp_path, videos, split):
    write(tmp_path, "a01_sn_f_01", 1, "")
    (tmp_path / ".a01_sn_f_01_f2.txt.abc.tmp").write_text("junk")
    (tmp_path / ".hidden_f3.txt").write_text("junk")
    frames, report = load_ground_truth(tmp_path, videos, split, "train")
    assert len(frames) == 1
    assert report.skipped_not_in_split == 0


def test_loader_reads_and_writes_no_image_file(tmp_path, videos, split):
    write(tmp_path, "a01_sn_f_01", 1, "0 0.5 0.5 0.2 0.2\n")
    write(tmp_path, "a01_sn_f_01", 2, "")
    before = sorted(p.name for p in tmp_path.iterdir())
    load_ground_truth(tmp_path, videos, split, "train")
    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after
    assert all(name.endswith(".txt") for name in after)


def write_ranges(path, *rows):
    lines = ["video_id,start_frame,end_frame,reason"]
    lines += [f"{v},{s},{e},{reason}" for v, s, e, reason in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_label_inside_range_exception_is_accepted(tmp_path, videos, split):
    boxes = tmp_path / "boxes"
    boxes.mkdir()
    ranges = tmp_path / "ranges.csv"
    write_ranges(ranges, ("a01_sn_f_01", 10, 20, "pilot"))
    write(boxes, "a01_sn_f_01", 10, "")
    write(boxes, "a01_sn_f_01", 20, "")
    write(boxes, "a03_sn_f_01", 50, "")
    frames, _ = load_ground_truth(boxes, videos, split, "train", ranges)
    assert len(frames) == 3


def test_label_outside_range_exception_raises_value_error(tmp_path, videos, split):
    boxes = tmp_path / "boxes"
    boxes.mkdir()
    ranges = tmp_path / "ranges.csv"
    write_ranges(ranges, ("a01_sn_f_01", 10, 20, "pilot"))
    write(boxes, "a01_sn_f_01", 21, "")
    with pytest.raises(ValueError, match=r"a01_sn_f_01_f21\.txt.*10.*20"):
        load_ground_truth(boxes, videos, split, "train", ranges)


def test_range_is_not_checked_without_ranges_path(tmp_path, videos, split):
    write(tmp_path, "a01_sn_f_01", 21, "")
    frames, _ = load_ground_truth(tmp_path, videos, split, "train")
    assert len(frames) == 1


@pytest.mark.parametrize("name", ["a_f03.txt", "a_f00.txt", "a_f٣.txt", "a_f007.txt"])
def test_parse_label_name_rejects_frames_not_written_like_the_labeling_tool(name):
    with pytest.raises(ValueError, match=name):
        parse_label_name(name)


def test_parse_label_name_accepts_frame_zero():
    assert parse_label_name("a_f0.txt") == ("a", 0)


def test_leading_zero_label_file_raises_instead_of_becoming_empty_frame(tmp_path, videos, split):
    write(tmp_path, "a01_sn_f_01", "03", "0 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match="a01_sn_f_01_f03.txt"):
        load_ground_truth(tmp_path, videos, split, "train")


def test_leading_zero_and_plain_file_for_same_frame_raise(tmp_path, videos, split):
    write(tmp_path, "a01_sn_f_01", "03", "0 0.5 0.5 0.2 0.2\n")
    write(tmp_path, "a01_sn_f_01", 3, "0 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match="a01_sn_f_01_f03.txt"):
        load_ground_truth(tmp_path, videos, split, "train")


def test_uppercase_txt_suffix_raises_naming_the_file(tmp_path, videos, split):
    (tmp_path / "a01_sn_f_01_f4.TXT").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match=r"a01_sn_f_01_f4\.TXT"):
        load_ground_truth(tmp_path, videos, split, "train")


def test_other_suffixes_are_ignored(tmp_path, videos, split):
    (tmp_path / "a01_sn_f_01_f4.png").write_bytes(b"")
    write(tmp_path, "a01_sn_f_01", 1, "")
    frames, _ = load_ground_truth(tmp_path, videos, split, "train")
    assert len(frames) == 1


@pytest.mark.parametrize(
    "column, value",
    [("frame_count", "100.0"), ("width", "abc"), ("height", "x"), ("frame_count", None)],
)
def test_bad_video_table_value_raises_naming_video_and_column(tmp_path, split, column, value):
    row = {"video_id": "a01_sn_f_01", "width": 720, "height": 1280, "frame_count": 100}
    row[column] = value
    write(tmp_path, "a01_sn_f_01", 1, "")
    with pytest.raises(ValueError, match=rf"a01_sn_f_01.*{column}"):
        load_ground_truth(tmp_path, pd.DataFrame([row]), split, "train")


@pytest.mark.parametrize("frame_count, width", [("100", "720"), (100.0, 720.0), (100, 720)])
def test_int_like_video_table_values_work(tmp_path, split, frame_count, width):
    row = {"video_id": "a01_sn_f_01", "width": width, "height": 1280, "frame_count": frame_count}
    write(tmp_path, "a01_sn_f_01", 1, "0 0.5 0.5 0.5 0.5\n")
    frames, _ = load_ground_truth(tmp_path, pd.DataFrame([row]), split, "train")
    assert frames[0].boxes == ((180.0, 320.0, 540.0, 960.0),)


def test_directory_named_like_label_file_is_skipped(tmp_path, videos, split):
    (tmp_path / "a01_sn_f_01_f4.txt").mkdir()
    write(tmp_path, "a01_sn_f_01", 1, "")
    frames, _ = load_ground_truth(tmp_path, videos, split, "train")
    assert [f.frame_index for f in frames] == [1]


def test_duplicate_video_in_split_raises(tmp_path, videos):
    dup = make_split(("a01_sn_f_01", "train"), ("a01_sn_f_01", "validation"))
    with pytest.raises(ValueError, match="a01_sn_f_01"):
        load_ground_truth(tmp_path, videos, dup, "train")


def test_duplicate_video_in_videos_raises(tmp_path, split):
    dup = make_videos(("a01_sn_f_01", 720, 1280, 100), ("a01_sn_f_01", 720, 1280, 100))
    with pytest.raises(ValueError, match="a01_sn_f_01"):
        load_ground_truth(tmp_path, dup, split, "train")


def test_missing_boxes_dir_raises_file_not_found(tmp_path, videos, split):
    with pytest.raises(FileNotFoundError):
        load_ground_truth(tmp_path / "nope", videos, split, "train")


def test_empty_boxes_dir_gives_empty_result(tmp_path, videos, split):
    frames, report = load_ground_truth(tmp_path, videos, split, "train")
    assert frames == []
    assert report == GroundTruthReport(0, 0, 0, 0, ())


def test_range_exception_of_other_role_video_still_raises(tmp_path, videos, split):
    boxes = tmp_path / "boxes"
    boxes.mkdir()
    ranges = tmp_path / "ranges.csv"
    write_ranges(ranges, ("a02_sn_f_01", 10, 20, "pilot"))
    write(boxes, "a02_sn_f_01", 30, "")
    with pytest.raises(ValueError, match=r"a02_sn_f_01_f30\.txt"):
        load_ground_truth(boxes, videos, split, "train", ranges)
