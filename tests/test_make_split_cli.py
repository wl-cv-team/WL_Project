import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_split.py"


def _data_dir(tmp_path: Path) -> Path:
    rows = []
    for group, n in (("A", 4), ("B", 3), ("C", 1)):
        for i in range(n):
            for v in range(2):
                rows.append({"video_id": f"{group}{i}_{v}", "person_id": f"{group}{i}",
                             "group": group, "path": "p", "has_plate": "yes",
                             "read_status": "ok"})
    rows[0]["read_status"] = "error"
    meta = tmp_path / "data" / "metadata"
    meta.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(meta / "videos.csv", index=False)
    return tmp_path / "data"


def _run(data_dir: Path, *extra: str):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--data-dir", str(data_dir), *extra],
        capture_output=True, text=True,
    )


def test_cli_prints_counts_and_writes_split(tmp_path: Path):
    data = _data_dir(tmp_path)
    result = _run(data)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "persons=8 videos=16" in out
    assert "group=A role=train" in out and "group=A role=validation" in out
    assert "group=C role=train persons=1 videos=2" in out
    assert "single_person_groups=['C']" in out
    assert "read_status ok=15" in out and "read_status error=1" in out
    written = pd.read_csv(data / "splits" / "person_split.csv")
    assert len(written) == 16


def test_cli_refuses_overwrite_without_force(tmp_path: Path):
    data = _data_dir(tmp_path)
    assert _run(data).returncode == 0
    target = data / "splits" / "person_split.csv"
    target.write_text("sentinel", encoding="utf-8")
    refused = _run(data)
    assert refused.returncode != 0
    assert "--force" in refused.stderr
    assert target.read_text(encoding="utf-8") == "sentinel"
    forced = _run(data, "--force")
    assert forced.returncode == 0, forced.stderr
    assert target.read_text(encoding="utf-8") != "sentinel"
