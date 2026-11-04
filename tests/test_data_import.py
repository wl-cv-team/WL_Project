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


@pytest.mark.parametrize("name", ["/etc/evil.txt", "C:/evil.txt", "metadata/..\\..\\evil.txt"])
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
