import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _check_ignore(path: str) -> int:
    return subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=REPO, capture_output=True
    ).returncode


def test_package_imports():
    import wl_snatch  # noqa: F401


def test_data_dir_is_git_ignored():
    assert _check_ignore("data/raw/anything.mp4") == 0


def test_weights_are_git_ignored():
    assert _check_ignore("model.pt") == 0
