from pathlib import Path

from wl_snatch.label_audit import (
    AuditReport,
    audit_yolo_labels,
    describe,
    filename_overlap,
    parse_frame_name,
    person_id_of,
    read_boxes,
)

OK = "0 0.5 0.5 0.2 0.2\n"


def make(tmp_path: Path, images: dict[str, str | None], orphans: dict[str, str] | None = None):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    for name, label in images.items():
        (img_dir / name).write_bytes(b"x")
        if label is not None:
            (lbl_dir / (Path(name).stem + ".txt")).write_text(label)
    for name, label in (orphans or {}).items():
        (lbl_dir / name).write_text(label)
    return img_dir, lbl_dir


def kinds(report: AuditReport) -> list[str]:
    return [i.kind for i in report.issues]


def test_clean_dataset_has_no_issues(tmp_path):
    img, lbl = make(tmp_path, {"a.jpg": OK, "b.JPG": OK + "0 0.3 0.3 0.1 0.1\n"})
    report = audit_yolo_labels(img, lbl, frozenset({0}))
    assert report.images == 2
    assert report.labeled_boxes == 3
    assert report.issues == ()


def test_image_without_label_file_is_reported(tmp_path):
    img, lbl = make(tmp_path, {"a.jpg": None})
    report = audit_yolo_labels(img, lbl, frozenset({0}))
    assert kinds(report) == ["missing_label"]
    assert report.issues[0].image == "a.jpg"


def test_empty_label_file_is_reported(tmp_path):
    img, lbl = make(tmp_path, {"a.jpg": "  \n"})
    report = audit_yolo_labels(img, lbl, frozenset({0}))
    assert kinds(report) == ["empty_label"]
    assert report.labeled_boxes == 0


def test_coordinate_outside_zero_one_is_reported(tmp_path):
    img, lbl = make(
        tmp_path,
        {
            "a.jpg": "0 1.2 0.5 0.2 0.2\n",
            "b.jpg": "0 0.5 0.5 0 0.2\n",
            "c.jpg": "0 0.5 0.5 abc 0.2\n",
            "d.jpg": "0 0.5 0.5 0.2\n",
            "e.jpg": "0 -0.1 0.5 0.2 0.2\n",
            "f.jpg": "0 0.05 0.5 0.4 0.2\n",
        },
    )
    report = audit_yolo_labels(img, lbl, frozenset({0}))
    assert sorted(i.image for i in report.issues) == [
        "a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg"
    ]
    assert set(kinds(report)) == {"out_of_range"}
    assert report.labeled_boxes == 0


def test_unknown_class_id_is_reported(tmp_path):
    img, lbl = make(tmp_path, {"a.jpg": "1 0.5 0.5 0.2 0.2\n", "b.jpg": "x 0.5 0.5 0.2 0.2\n"})
    report = audit_yolo_labels(img, lbl, frozenset({0}))
    assert kinds(report) == ["bad_class", "bad_class"]


def test_label_without_image_is_reported(tmp_path):
    img, lbl = make(tmp_path, {"a.jpg": OK}, orphans={"zz.txt": OK})
    report = audit_yolo_labels(img, lbl, frozenset({0}))
    assert kinds(report) == ["orphan_label"]
    assert report.issues[0].image == "zz.txt"


def test_two_identical_boxes_are_reported_as_duplicate(tmp_path):
    img, lbl = make(tmp_path, {"a.jpg": OK + OK})
    report = audit_yolo_labels(img, lbl, frozenset({0}))
    assert kinds(report) == ["duplicate_box"]
    assert report.labeled_boxes == 2


def test_undecodable_label_and_missing_dirs_do_not_raise(tmp_path):
    img, lbl = make(tmp_path, {"a.jpg": None})
    (lbl / "a.txt").write_bytes(b"\xff\xfe\x00bad")
    assert audit_yolo_labels(img, lbl, frozenset({0})).issues
    report = audit_yolo_labels(tmp_path / "nope", tmp_path / "nope2", frozenset({0}))
    assert report.images == 0


def test_read_boxes_skips_malformed_lines(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.2 0.4\nbad line\n0 0.5 0.5 nan 0.1\n")
    assert read_boxes(p) == [(0, 0.5, 0.5, 0.2, 0.4)]
    assert read_boxes(tmp_path / "missing.txt") == []


def test_parse_frame_name_and_person_id():
    assert parse_frame_name("c02_sn_f_01_frame_0109.jpg") == ("c02_sn_f_01", 109)
    assert parse_frame_name("weird.jpg") is None
    assert person_id_of("c02_sn_f_01") == "c02"
    assert person_id_of("zzz") is None


def test_describe_and_overlap():
    assert describe([]) == {"min": None, "median": None, "max": None}
    assert describe([1.0, 3.0, 2.0]) == {"min": 1.0, "median": 2.0, "max": 3.0}
    assert filename_overlap(["a", "b"], ["b", "c"]) == ["b"]


def test_summarize_directory_counts(tmp_path):
    from wl_snatch.label_audit import summarize_directory

    img, lbl = make(
        tmp_path,
        {
            "c02_sn_f_01_frame_0001.jpg": "0 0.5 0.5 0.2 0.4\n",
            "c02_sn_f_02_frame_0001.JPG": OK + "0 0.3 0.3 0.1 0.1\n",
            "odd.jpg": None,
        },
    )
    s = summarize_directory(img, lbl, frozenset({0}))
    assert s["images"] == 3 and s["labeled_boxes"] == 3
    assert s["issues_by_kind"]["missing_label"] == 1
    assert s["frames_per_video"] == {"c02_sn_f_01": 1, "c02_sn_f_02": 1}
    assert s["frames_per_person"] == {"c02": 2}
    assert s["unparsed_names"] == ["odd.jpg"]
    assert s["box_width_normalized"]["max"] == 0.2
    assert s["boxes_per_image_histogram"] == {"0": 1, "1": 1, "2": 1}
