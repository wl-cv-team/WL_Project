import zipfile
from pathlib import Path

import pytest

from wl_snatch.data_import import import_archive


def make_zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return path


def test_raw_video_folder_is_mapped_to_data_raw_videos(tmp_path):
    z = make_zip(
        tmp_path / "a.zip",
        {
            "raw video/c05_sn_f_05.mp4": b"video",
            "metadata/videos.csv": b"id\n",
        },
    )
    data = tmp_path / "data"
    report = import_archive(z, data)
    assert (data / "raw" / "videos" / "c05_sn_f_05.mp4").read_bytes() == b"video"
    assert (data / "metadata" / "videos.csv").exists()
    assert not (data / "raw video").exists()
    assert report.videos_copied == 1
    assert report.files_copied == 2


def test_dataset_yaml_path_is_rewritten_to_relative(tmp_path):
    yaml_text = b"path: /Users/someone/proj/plate_dataset\ntrain: images/train\nnames:\n  0: plate\n"
    z = make_zip(tmp_path / "a.zip", {"plate_dataset/dataset.yaml": yaml_text})
    data = tmp_path / "data"
    report = import_archive(z, data)
    text = (data / "plate_dataset" / "dataset.yaml").read_text()
    assert "path: .\n" in text
    assert "/Users/" not in text
    assert "train: images/train" in text
    assert report.rewritten_paths == ("plate_dataset/dataset.yaml",)


def test_zip_slip_entry_is_rejected(tmp_path):
    z = make_zip(
        tmp_path / "a.zip",
        {"metadata/videos.csv": b"id\n", "../evil.txt": b"x"},
    )
    data = tmp_path / "data"
    with pytest.raises(ValueError):
        import_archive(z, data)
    assert not (tmp_path / "evil.txt").exists()
    assert not data.exists() or not any(data.rglob("*"))


@pytest.mark.parametrize("name", ["/etc/evil.txt", "C:/evil.txt"])
def test_absolute_path_entry_is_rejected(tmp_path, name):
    z = make_zip(tmp_path / "a.zip", {"metadata/videos.csv": b"id\n", name: b"x"})
    data = tmp_path / "data"
    with pytest.raises(ValueError):
        import_archive(z, data)
    assert not data.exists() or not any(data.rglob("*"))


def test_existing_file_is_not_overwritten(tmp_path):
    data = tmp_path / "data"
    (data / "metadata").mkdir(parents=True)
    (data / "metadata" / "videos.csv").write_text("mine")
    z = make_zip(
        tmp_path / "a.zip",
        {"metadata/videos.csv": b"theirs", "splits/s.csv": b"s"},
    )
    report = import_archive(z, data)
    assert (data / "metadata" / "videos.csv").read_text() == "mine"
    assert report.files_copied == 1
    assert report.skipped_existing == ("metadata/videos.csv",)


def test_unlisted_cache_and_macos_entries_are_reported_not_imported(tmp_path):
    z = make_zip(
        tmp_path / "a.zip",
        {
            "metadata/videos.csv": b"id\n",
            "plate_dataset/labels.cache": b"c",
            "plate_dataset_v2/labels/train.cache": b"c",
            "other/file.txt": b"o",
            "__MACOSX/metadata/._videos.csv": b"j",
            "metadata/.DS_Store": b"j",
        },
    )
    data = tmp_path / "data"
    report = import_archive(z, data)
    assert report.files_copied == 1
    assert set(report.not_imported) == {
        "plate_dataset/labels.cache",
        "plate_dataset_v2/labels/train.cache",
        "other/file.txt",
        "__MACOSX/metadata/._videos.csv",
        "metadata/.DS_Store",
    }
    assert not (data / "other").exists()
    assert not (data / "plate_dataset" / "labels.cache").exists()


@pytest.mark.parametrize("name", ["metadata/..\\..\\evil.txt", "__MACOSX/../x", "other/../x"])
def test_backslash_and_parent_segments_are_rejected_under_any_prefix(tmp_path, name):
    z = make_zip(tmp_path / "a.zip", {"metadata/videos.csv": b"id\n", name: b"x"})
    data = tmp_path / "data"
    with pytest.raises(ValueError):
        import_archive(z, data)
    assert not data.exists() or not any(data.rglob("*"))


def test_non_utf8_dataset_yaml_is_rejected_before_any_write(tmp_path):
    z = make_zip(
        tmp_path / "a.zip",
        {"metadata/videos.csv": b"id\n", "plate_dataset/dataset.yaml": b"path: \xff\xfe\n"},
    )
    data = tmp_path / "data"
    with pytest.raises(ValueError, match="plate_dataset/dataset.yaml"):
        import_archive(z, data)
    assert not data.exists() or not any(data.rglob("*"))


def test_second_run_copies_nothing_and_changes_nothing(tmp_path):
    z = make_zip(
        tmp_path / "a.zip",
        {
            "metadata/videos.csv": b"id\n",
            "raw video/v.mp4": b"v",
            "plate_dataset/dataset.yaml": b"path: /x/y\n",
        },
    )
    data = tmp_path / "data"
    import_archive(z, data)
    before = {p: p.read_bytes() for p in data.rglob("*") if p.is_file()}
    report = import_archive(z, data)
    after = {p: p.read_bytes() for p in data.rglob("*") if p.is_file()}
    assert report.files_copied == 0
    assert report.videos_copied == 0
    assert report.rewritten_paths == ()
    assert len(report.skipped_existing) == 3
    assert before == after


def test_existing_dataset_yaml_is_left_unrewritten(tmp_path):
    data = tmp_path / "data"
    (data / "plate_dataset").mkdir(parents=True)
    (data / "plate_dataset" / "dataset.yaml").write_text("path: /mine\n")
    z = make_zip(tmp_path / "a.zip", {"plate_dataset/dataset.yaml": b"path: /theirs\n"})
    report = import_archive(z, data)
    assert (data / "plate_dataset" / "dataset.yaml").read_text() == "path: /mine\n"
    assert report.rewritten_paths == ()
    assert report.skipped_existing == ("plate_dataset/dataset.yaml",)


def test_dataset_yaml_rewrite_handles_bom_and_crlf(tmp_path):
    raw = b"\xef\xbb\xbfpath: /Users/a/b\r\ntrain: images/train\r\n"
    z = make_zip(tmp_path / "a.zip", {"plate_dataset/dataset.yaml": raw})
    data = tmp_path / "data"
    import_archive(z, data)
    out = (data / "plate_dataset" / "dataset.yaml").read_bytes()
    assert out == b"\xef\xbb\xbfpath: .\r\ntrain: images/train\r\n"


def test_file_versus_directory_conflict_within_archive_is_rejected(tmp_path):
    z = make_zip(
        tmp_path / "a.zip",
        {"splits/ok.csv": b"s", "metadata/a": b"f", "metadata/a/b": b"g"},
    )
    data = tmp_path / "data"
    with pytest.raises(ValueError):
        import_archive(z, data)
    assert not data.exists() or not any(data.rglob("*"))


def test_existing_file_where_directory_is_needed_is_rejected(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "metadata").write_text("i am a file")
    z = make_zip(tmp_path / "a.zip", {"splits/ok.csv": b"s", "metadata/videos.csv": b"id\n"})
    with pytest.raises(ValueError):
        import_archive(z, data)
    assert not (data / "splits").exists()


def test_corrupt_entry_leaves_no_partial_file(tmp_path):
    zpath = tmp_path / "a.zip"
    payload = b"A" * 64
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("metadata/videos.csv", payload)
    blob = bytearray(zpath.read_bytes())
    idx = blob.index(payload)
    blob[idx] = ord("B")
    zpath.write_bytes(bytes(blob))
    data = tmp_path / "data"
    with pytest.raises(zipfile.BadZipFile):
        import_archive(zpath, data)
    leftovers = [p for p in data.rglob("*") if p.is_file()]
    assert leftovers == []
