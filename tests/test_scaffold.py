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
