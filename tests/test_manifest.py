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


# --- near_side (issue #7) ---------------------------------------------------

from wl_snatch.manifest import NEAR_SIDE_VALUES, require_near_side  # noqa: E402

NEAR_HEADER = HEADER.strip() + ",near_side\n"


def test_near_side_values_are_left_and_right():
    assert NEAR_SIDE_VALUES == ("left", "right")


def test_load_accepts_blank_near_side_before_it_is_filled(tmp_path: Path):
    path = _write(tmp_path, "v1,a01,A,p,yes,ok,\n", header=NEAR_HEADER)
    assert list(load_manifest(path)["near_side"]) == [""]


def test_load_rejects_value_other_than_left_right_or_blank(tmp_path: Path):
    path = _write(tmp_path, "v1,a01,A,p,yes,ok,unknown\n", header=NEAR_HEADER)
    with pytest.raises(ValueError, match="unknown"):
        load_manifest(path)


def test_load_rejects_wrong_case_near_side(tmp_path: Path):
    path = _write(tmp_path, "v1,a01,A,p,yes,ok,Left\n", header=NEAR_HEADER)
    with pytest.raises(ValueError, match="Left"):
        load_manifest(path)


def test_require_near_side_rejects_missing_column(tmp_path: Path):
    df = load_manifest(_write(tmp_path, "v1,a01,A,p,yes,ok\n"))
    with pytest.raises(ValueError, match="near_side"):
        require_near_side(df)


def test_require_near_side_names_videos_with_blank_value(tmp_path: Path):
    body = "v1,a01,A,p,yes,ok,left\nv2,a02,A,p,yes,ok,\nv3,a03,A,p,yes,ok,right\n"
    df = load_manifest(_write(tmp_path, body, header=NEAR_HEADER))
    with pytest.raises(ValueError) as err:
        require_near_side(df)
    assert "v2" in str(err.value)
    assert "v1" not in str(err.value)


def test_require_near_side_passes_when_all_filled(tmp_path: Path):
    body = "v1,a01,A,p,yes,ok,left\nv2,a02,A,p,yes,ok,right\n"
    df = load_manifest(_write(tmp_path, body, header=NEAR_HEADER))
    require_near_side(df)
