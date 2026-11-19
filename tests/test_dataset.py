import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from wl_snatch.contracts import GroundTruthFrame
from wl_snatch.dataset import (
    DatasetReport,
    assert_ignored_output,
    export_yolo_dataset,
    verify_exported_dataset,
)

REPO = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 64, 48  # non-square on purpose: a width/height swap changes the labels

# Hand-computed: pixel box (8, 6, 24, 42) in a 64x48 frame
#   cx = (8+24)/2/64 = 0.25, cy = (6+42)/2/48 = 0.5, w = 16/64 = 0.25, h = 36/48 = 0.75
BOX = (8.0, 6.0, 24.0, 42.0)
BOX_LINE = "0 0.250000 0.500000 0.250000 0.750000\n"
# Hand-computed: pixel box (32, 12, 48, 24): cx = 40/64 = 0.625, cy = 18/48 = 0.375,
#   w = 16/64 = 0.25, h = 12/48 = 0.25
BOX2 = (32.0, 12.0, 48.0, 24.0)
BOX2_LINE = "0 0.625000 0.375000 0.250000 0.250000\n"


def make_videos(tmp_path: Path, make_video, ids=("a01_sn_f_01", "b01_sn_f_01")) -> Path:
    root = tmp_path / "videos"
    root.mkdir()
    for video_id in ids:
        make_video(root / f"{video_id}.mp4", size=(WIDTH, HEIGHT))  # 5 frames each
    return root


def videos_table(persons=None) -> pd.DataFrame:
    persons = persons or {"a01_sn_f_01": "a01", "b01_sn_f_01": "b01"}
    return pd.DataFrame(
        [
            {"video_id": v, "person_id": p, "width": WIDTH, "height": HEIGHT, "frame_count": 5}
            for v, p in persons.items()
        ]
    )


def standard_frames():
    return {
        "train": [
            GroundTruthFrame("a01_sn_f_01", 0, (BOX,)),
            GroundTruthFrame("a01_sn_f_01", 2, ()),
        ],
        "validation": [GroundTruthFrame("b01_sn_f_01", 1, (BOX2,))],
    }


@pytest.fixture
def exported(tmp_path: Path, make_video):
    root = make_videos(tmp_path, make_video)
    out = tmp_path / "out"
    yaml_path = export_yolo_dataset(standard_frames(), videos_table(), root, out, REPO)
    return tmp_path, root, out, yaml_path


def test_export_places_frames_by_role_and_writes_yaml(exported):
    _, _, out, yaml_path = exported
    assert yaml_path == out / "dataset.yaml"
    names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert names == [
        "dataset.yaml",
        "images/train/a01_sn_f_01_f0.jpg",
        "images/train/a01_sn_f_01_f2.jpg",
        "images/validation/b01_sn_f_01_f1.jpg",
        "labels/train/a01_sn_f_01_f0.txt",
        "labels/train/a01_sn_f_01_f2.txt",
        "labels/validation/b01_sn_f_01_f1.txt",
        "manifest.json",
    ]
    image = cv2.imdecode(
        np.fromfile(out / "images/validation/b01_sn_f_01_f1.jpg", dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert image.shape == (HEIGHT, WIDTH, 3)
    b, g, r = (float(v) for v in image.reshape(-1, 3).mean(axis=0))
    assert g > 200 and b < 60 and r < 60  # frame 1 of the synthetic video is green


def test_yaml_uses_relative_paths_and_single_class_plate(exported):
    _, _, out, yaml_path = exported
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    assert not any(line.startswith("path:") for line in lines)  # Ultralytics then uses the yaml folder
    assert "train: images/train" in lines
    assert "val: images/validation" in lines
    assert lines[lines.index("names:") + 1] == "  0: plate"
    text = yaml_path.read_text(encoding="utf-8")
    assert str(out) not in text and "/" not in text.replace("images/", "")


def test_label_text_is_normalized_from_pixel_box_and_empty_for_no_plate(exported):
    _, _, out, _ = exported
    assert (out / "labels/train/a01_sn_f_01_f0.txt").read_text() == BOX_LINE
    assert (out / "labels/validation/b01_sn_f_01_f1.txt").read_text() == BOX2_LINE
    assert (out / "labels/train/a01_sn_f_01_f2.txt").read_text() == ""


def test_empty_label_frames_are_exported_with_empty_label_files(exported):
    _, _, out, _ = exported
    assert (out / "images/train/a01_sn_f_01_f2.jpg").is_file()
    assert (out / "labels/train/a01_sn_f_01_f2.txt").stat().st_size == 0


def test_export_rejects_non_empty_output_dir(tmp_path: Path, make_video):
    root = make_videos(tmp_path, make_video)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        export_yolo_dataset(standard_frames(), videos_table(), root, out, REPO)
    assert (out / "keep.txt").read_text() == "x"
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_export_rejects_unknown_role(tmp_path: Path, make_video):
    root = make_videos(tmp_path, make_video)
    frames = {"val": [GroundTruthFrame("a01_sn_f_01", 0, ())]}
    with pytest.raises(ValueError, match="val"):
        export_yolo_dataset(frames, videos_table(), root, tmp_path / "out", REPO)


def test_assert_ignored_output_rejects_tracked_path_inside_repo():
    with pytest.raises(ValueError, match="not ignored"):
        assert_ignored_output(REPO / "src" / "dataset_out", REPO)


def test_assert_ignored_output_allows_ignored_and_outside_paths(tmp_path: Path):
    assert_ignored_output(REPO / "data" / "derived" / "plate_dataset", REPO)
    assert_ignored_output(tmp_path / "anywhere", REPO)


def test_same_frame_in_two_roles_raises_value_error(tmp_path: Path, make_video):
    root = make_videos(tmp_path, make_video)
    frames = {
        "train": [GroundTruthFrame("a01_sn_f_01", 1, ())],
        "validation": [GroundTruthFrame("a01_sn_f_01", 1, ())],
    }
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="a01_sn_f_01.*1|frame"):
        export_yolo_dataset(frames, videos_table(), root, out, REPO)
    assert not out.exists() or not any(out.iterdir())


def test_same_person_in_both_roles_raises_value_error(tmp_path: Path, make_video):
    root = make_videos(tmp_path, make_video)
    table = videos_table({"a01_sn_f_01": "p7", "b01_sn_f_01": "p7"})
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="p7"):
        export_yolo_dataset(standard_frames(), table, root, out, REPO)
    assert not out.exists() or not any(out.iterdir())


def test_unreadable_frame_raises_and_cleans_up_what_it_created(tmp_path: Path, make_video):
    root = make_videos(tmp_path, make_video)
    frames = {
        "train": [GroundTruthFrame("a01_sn_f_01", 0, (BOX,)), GroundTruthFrame("a01_sn_f_01", 9, ())],
        "validation": [GroundTruthFrame("b01_sn_f_01", 1, ())],
    }
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="a01_sn_f_01.*9"):
        export_yolo_dataset(frames, videos_table(), root, out, REPO)
    assert not out.exists() or not any(out.iterdir())
    # an existing empty output folder is kept, but emptied
    out.mkdir(exist_ok=True)
    with pytest.raises(ValueError):
        export_yolo_dataset(frames, videos_table(), root, out, REPO)
    assert out.is_dir() and not any(out.iterdir())


def test_table_size_different_from_decoded_frame_raises_and_cleans_up(
    tmp_path: Path, make_video
):
    root = make_videos(tmp_path, make_video)
    table = videos_table()
    table["width"], table["height"] = HEIGHT, WIDTH  # swapped relative to the real frames
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="a01_sn_f_01"):
        export_yolo_dataset(standard_frames(), table, root, out, REPO)
    assert not out.exists() or not any(out.iterdir())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_records_counts_and_sha256_of_every_file(exported):
    _, _, out, _ = exported
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["jpeg_quality"] == 95
    assert manifest["roles"]["train"] == {
        "images": 2,
        "with_plate": 1,
        "without_plate": 1,
        "video_ids": ["a01_sn_f_01"],
    }
    assert manifest["roles"]["validation"] == {
        "images": 1,
        "with_plate": 1,
        "without_plate": 0,
        "video_ids": ["b01_sn_f_01"],
    }
    expected_files = sorted(
        p.relative_to(out).as_posix()
        for p in out.rglob("*")
        if p.is_file() and p.suffix in (".jpg", ".txt")
    )
    assert sorted(manifest["files"]) == expected_files
    for name in expected_files:
        assert manifest["files"][name] == sha(out / name)
    joined = "".join(f"{name}:{sha(out / name)}\n" for name in expected_files)
    assert manifest["overall_sha256"] == hashlib.sha256(joined.encode("ascii")).hexdigest()


def test_same_inputs_give_same_overall_hash(tmp_path: Path, make_video):
    root = make_videos(tmp_path, make_video)
    first = export_yolo_dataset(standard_frames(), videos_table(), root, tmp_path / "o1", REPO)
    second = export_yolo_dataset(standard_frames(), videos_table(), root, tmp_path / "o2", REPO)
    assert verify_exported_dataset(first).overall_sha256 == verify_exported_dataset(second).overall_sha256


def test_verify_accepts_a_fresh_export_and_rejects_a_modified_label_or_image(exported):
    _, _, out, yaml_path = exported
    report = verify_exported_dataset(yaml_path)
    assert dataclasses.is_dataclass(DatasetReport)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.overall_sha256 = "x"
    assert report.images_by_role == {"train": 2, "validation": 1}
    assert report.with_plate_by_role == {"train": 1, "validation": 1}
    assert report.without_plate_by_role == {"train": 1, "validation": 0}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert report.overall_sha256 == manifest["overall_sha256"]

    label = out / "labels/train/a01_sn_f_01_f0.txt"
    original = label.read_text()
    label.write_text("0 0.300000 0.500000 0.250000 0.750000\n")  # valid line, different value
    with pytest.raises(ValueError, match="a01_sn_f_01_f0.txt"):
        verify_exported_dataset(yaml_path)
    label.write_text(original)
    verify_exported_dataset(yaml_path)

    image = out / "images/validation/b01_sn_f_01_f1.jpg"
    data = image.read_bytes()
    image.write_bytes(data[:-3] + b"\x00\x00\x00")
    with pytest.raises(ValueError, match="b01_sn_f_01_f1.jpg"):
        verify_exported_dataset(yaml_path)


def rewrite_manifest_hash(out: Path, name: str) -> None:
    """Keep the manifest consistent with a hand-edited file so other checks are reached."""
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][name] = sha(out / name)
    joined = "".join(f"{n}:{manifest['files'][n]}\n" for n in sorted(manifest["files"]))
    manifest["overall_sha256"] = hashlib.sha256(joined.encode("ascii")).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_verify_rejects_label_outside_the_image_and_orphan_files(exported):
    _, _, out, yaml_path = exported
    name = "labels/train/a01_sn_f_01_f0.txt"
    (out / name).write_text("0 0.900000 0.500000 0.500000 0.500000\n")  # x2 = 1.15
    rewrite_manifest_hash(out, name)
    with pytest.raises(ValueError, match="a01_sn_f_01_f0.txt"):
        verify_exported_dataset(yaml_path)
    (out / name).write_text("1 0.500000 0.500000 0.500000 0.500000\n")  # wrong class id
    rewrite_manifest_hash(out, name)
    with pytest.raises(ValueError, match="a01_sn_f_01_f0.txt"):
        verify_exported_dataset(yaml_path)
    (out / name).write_text("0 1.200000 0.500000 0.100000 0.100000\n")  # value above 1
    rewrite_manifest_hash(out, name)
    with pytest.raises(ValueError, match="a01_sn_f_01_f0.txt"):
        verify_exported_dataset(yaml_path)


def test_verify_rejects_orphan_label_and_orphan_image(exported):
    _, _, out, yaml_path = exported
    (out / "labels/train/stray_f3.txt").write_text("")
    with pytest.raises(ValueError, match="stray_f3"):
        verify_exported_dataset(yaml_path)
    (out / "labels/train/stray_f3.txt").unlink()
    (out / "images/validation/lonely_f4.jpg").write_bytes(b"x")
    with pytest.raises(ValueError, match="lonely_f4"):
        verify_exported_dataset(yaml_path)


def test_verify_rejects_same_file_name_in_train_and_val(exported):
    _, _, out, yaml_path = exported
    for kind, suffix in (("images", ".jpg"), ("labels", ".txt")):
        source = out / kind / "train" / f"a01_sn_f_01_f2{suffix}"
        (out / kind / "validation" / f"a01_sn_f_01_f2{suffix}").write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="a01_sn_f_01_f2"):
        verify_exported_dataset(yaml_path)


def test_verify_is_read_only(exported):
    _, _, out, yaml_path = exported
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file()}
    verify_exported_dataset(yaml_path)
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file()}
    assert before == after


def test_export_writes_files_only_inside_out_dir(
    tmp_path: Path, make_video, monkeypatch: pytest.MonkeyPatch
):
    root = make_videos(tmp_path, make_video)
    outside_before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    cwd_before = {p.name for p in Path.cwd().iterdir()}
    out = tmp_path / "nested" / "out"
    export_yolo_dataset(standard_frames(), videos_table(), root, out, REPO)
    outside_after = {
        p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file() and out not in p.parents
    }
    assert outside_after == outside_before
    assert {p.name for p in Path.cwd().iterdir()} == cwd_before


def test_dataset_module_import_does_not_import_ultralytics():
    code = (
        "import sys\n"
        "import wl_snatch.dataset\n"
        "bad = [m for m in ('ultralytics', 'torch', 'ipywidgets') if m in sys.modules]\n"
        "sys.exit(1 if bad else 0)\n"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "box",
    [
        (float("nan"), 6.0, 24.0, 42.0),
        (8.0, 6.0, float("inf"), 42.0),
        (10.0, 10.0, 10.0, 30.0),  # zero width
        (10.0, 20.0, 30.0, 20.0),  # zero height
        (24.0, 6.0, 8.0, 42.0),  # x1 > x2: negative width
        (8.0, 6.0, 70.0, 42.0),  # x2 = 70 is beyond the 64 px width
        (8.0, -3.0, 24.0, 42.0),  # y1 = -3 is above the frame
        (8.0, 6.0, 24.0, 50.0),  # y2 = 50 is below the 48 px height
    ],
)
def test_export_rejects_bad_pixel_box_naming_video_and_frame(tmp_path: Path, make_video, box):
    root = make_videos(tmp_path, make_video)
    frames = {"train": [GroundTruthFrame("a01_sn_f_01", 3, (box,))]}
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="a01_sn_f_01_f3"):
        export_yolo_dataset(frames, videos_table(), root, out, REPO)
    assert not out.exists() or not any(out.iterdir())


def test_verify_rejects_manifest_counts_that_differ_from_the_files(exported):
    _, _, out, yaml_path = exported
    manifest_path = out / "manifest.json"
    original = manifest_path.read_text(encoding="utf-8")
    for key, wrong in (("images", 3), ("with_plate", 0), ("without_plate", 2)):
        manifest = json.loads(original)
        manifest["roles"]["train"][key] = wrong
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="train"):
            verify_exported_dataset(yaml_path)
    manifest_path.write_text(original, encoding="utf-8")
    verify_exported_dataset(yaml_path)


def test_verify_rejects_stray_folders_and_files_under_images_and_labels(exported):
    _, _, out, yaml_path = exported
    (out / "images" / "test").mkdir()
    (out / "images" / "test" / "a.jpg").write_bytes(b"x")
    with pytest.raises(ValueError, match="test"):
        verify_exported_dataset(yaml_path)
    (out / "images" / "test" / "a.jpg").unlink()
    (out / "images" / "test").rmdir()
    (out / "labels" / "extra").mkdir()
    with pytest.raises(ValueError, match="extra"):
        verify_exported_dataset(yaml_path)
    (out / "labels" / "extra").rmdir()
    (out / "images" / "loose.jpg").write_bytes(b"x")
    with pytest.raises(ValueError, match="loose.jpg"):
        verify_exported_dataset(yaml_path)


def test_assert_ignored_output_names_the_first_open_probe_path():
    # In data/metadata/ the .gitignore rules ignore images (any .jpg under data/) but track
    # text files, so the image probe passes and the label probe is the first open one.
    with pytest.raises(ValueError, match="not ignored") as excinfo:
        assert_ignored_output(REPO / "data" / "metadata" / "x", REPO)
    assert "data/metadata/x/labels/x.txt" in str(excinfo.value)


def test_assert_ignored_output_checks_all_four_export_paths(monkeypatch: pytest.MonkeyPatch):
    import wl_snatch.dataset as module

    probed = []

    def fake_run(command, **kwargs):
        probed.append(command[-1])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert_ignored_output(REPO / "data" / "derived" / "d", REPO)
    assert probed == [
        "data/derived/d/images/x.jpg",
        "data/derived/d/labels/x.txt",
        "data/derived/d/dataset.yaml",
        "data/derived/d/manifest.json",
    ]


def test_assert_ignored_output_allows_runs_folder_for_training_output():
    assert_ignored_output(REPO / "runs" / "plate", REPO)


@pytest.mark.parametrize("dimensions", ["0 0.2", "0.2 0"])
def test_verify_rejects_zero_area_label_before_hash_check(exported, dimensions):
    _, _, out, yaml_path = exported
    label = out / "labels/train/a01_sn_f_01_f0.txt"
    label.write_text(f"0 0.5 0.5 {dimensions}\n")
    with pytest.raises(ValueError, match="positive area"):
        verify_exported_dataset(yaml_path)
