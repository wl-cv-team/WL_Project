import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _check_ignore(path: str) -> int:
    result = subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=REPO, capture_output=True, text=True
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"git check-ignore failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.returncode


def test_package_imports():
    import wl_snatch

    # 네임스페이스 패키지는 __file__이 None이다. 실제 패키지여야 한다.
    assert wl_snatch.__file__ is not None
    assert Path(wl_snatch.__file__).resolve().is_relative_to(REPO / "src")


def test_data_dir_is_git_ignored():
    assert _check_ignore("data/raw/anything.mp4") == 0


def test_weights_are_git_ignored():
    assert _check_ignore("model.pt") == 0


import pytest  # noqa: E402


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/videos/a01_sn_f_01.mp4",
        "data/raw/anything.csv",
        "data/plate_dataset/images/x.jpg",
        "data/annotations/frames/x.jpg",
        "data/metadata/preview.png",
        "data/other_dir/file.csv",
    ],
)
def test_videos_images_and_unlisted_data_are_ignored(path: str):
    # 영상과 영상에서 뽑은 이미지에는 얼굴이 나온다. 목록에 없는 data/ 파일도 추적하지 않는다.
    assert _check_ignore(path) == 0


@pytest.mark.parametrize(
    "path",
    [
        "data/README.md",
        "data/metadata/videos.csv",
        "data/annotations/phase_events.csv",
        "data/annotations/labels/c08_sn_f_01_frame_0092.txt",
        "data/splits/person_split.csv",
    ],
)
def test_metadata_annotations_and_splits_are_tracked(path: str):
    assert _check_ignore(path) == 1
