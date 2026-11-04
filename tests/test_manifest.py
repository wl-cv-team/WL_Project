from pathlib import Path

import pytest

from wl_snatch.manifest import REQUIRED_COLUMNS, load_manifest

HEADER = "video_id,person_id,group,path,has_plate,read_status\n"


def _write(tmp_path: Path, body: str, header: str = HEADER) -> Path:
    path = tmp_path / "videos.csv"
    path.write_text(header + body, encoding="utf-8")
    return path


def test_missing_required_column_names_the_column(tmp_path: Path):
    path = _write(tmp_path, "v1,A,p,yes\n", header="video_id,group,path,has_plate\n")
    with pytest.raises(ValueError) as err:
        load_manifest(path)
    assert "person_id" in str(err.value)
    assert "read_status" in str(err.value)


def test_missing_columns_message_names_every_missing_column(tmp_path: Path):
    path = _write(tmp_path, "v1\n", header="video_id\n")
    with pytest.raises(ValueError) as err:
        load_manifest(path)
    for column in REQUIRED_COLUMNS[1:]:
        assert column in str(err.value)


def test_duplicate_video_id_raises(tmp_path: Path):
    path = _write(tmp_path, "v1,a01,A,p,yes,ok\nv1,a02,A,p,yes,ok\n")
    with pytest.raises(ValueError, match="v1"):
        load_manifest(path)


@pytest.mark.parametrize("column", ["video_id", "person_id", "group"])
@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_identifier_raises_naming_column(tmp_path: Path, column: str, blank: str):
    row = {"video_id": "v1", "person_id": "a01", "group": "A"}
    row[column] = blank
    body = f"{row['video_id']},{row['person_id']},{row['group']},p,yes,ok\nv2,a02,A,p,yes,ok\n"
    with pytest.raises(ValueError) as err:
        load_manifest(_write(tmp_path, body))
    assert column in str(err.value)
    assert "1 row" in str(err.value)


def test_valid_manifest_loads_with_string_ids(tmp_path: Path):
    path = _write(tmp_path, "v1,001,A,p,yes,ok\nv2,002,B,p,no,ok\n")
    df = load_manifest(path)
    assert list(df["person_id"]) == ["001", "002"]
    assert len(df) == 2
