import pandas as pd
import pytest

from wl_snatch.split import assert_no_person_leak, make_person_split, single_person_groups


def _manifest(persons_by_group: dict[str, int], videos_per_person: int = 3) -> pd.DataFrame:
    rows = []
    for group, count in persons_by_group.items():
        for i in range(count):
            person = f"{group.lower()}{i:02d}"
            for v in range(videos_per_person):
                rows.append(
                    {"video_id": f"{person}_{v}", "person_id": person, "group": group,
                     "path": "p", "has_plate": "yes", "read_status": "ok"}
                )
    return pd.DataFrame(rows)


def test_split_has_no_person_in_both_roles():
    split = make_person_split(_manifest({"A": 10, "B": 8, "C": 6}), seed=0, validation_fraction=0.3)
    assert list(split.columns) == ["video_id", "person_id", "group", "role"]
    assert set(split["role"]) == {"train", "validation"}
    assert_no_person_leak(split)
    assert len(split) == 24 * 3


def test_split_is_deterministic_for_same_seed():
    m = _manifest({"A": 10, "B": 8, "C": 6})
    a = make_person_split(m, 3, 0.3)
    b = make_person_split(m.sample(frac=1, random_state=1), 3, 0.3)
    pd.testing.assert_frame_equal(a, b)
    c = make_person_split(m, 4, 0.3)
    assert not a["role"].equals(c["role"])


def test_person_with_many_videos_stays_in_one_role():
    split = make_person_split(_manifest({"A": 6, "B": 6}, videos_per_person=7), 1, 0.3)
    assert (split.groupby("person_id")["role"].nunique() == 1).all()


def test_group_with_single_person_goes_to_train_and_is_reported():
    split = make_person_split(_manifest({"A": 5, "C": 1}), 0, 0.3)
    c_roles = set(split.loc[split["group"] == "C", "role"])
    assert c_roles == {"train"}
    assert split.attrs["single_person_groups"] == ["C"]
    assert_no_person_leak(split)


def test_no_single_person_group_reports_empty():
    split = make_person_split(_manifest({"A": 5, "B": 4}), 0, 0.3)
    assert split.attrs["single_person_groups"] == []


def test_small_fraction_still_gives_each_multi_person_group_both_roles():
    split = make_person_split(_manifest({"A": 2, "B": 3}), 0, 0.01)
    for group in ("A", "B"):
        assert set(split.loc[split["group"] == group, "role"]) == {"train", "validation"}


def test_large_fraction_still_keeps_a_train_person():
    split = make_person_split(_manifest({"A": 2, "B": 3}), 0, 0.99)
    for group in ("A", "B"):
        assert set(split.loc[split["group"] == group, "role"]) == {"train", "validation"}


def test_one_person_total_goes_to_train():
    split = make_person_split(_manifest({"A": 1}), 0, 0.3)
    assert set(split["role"]) == {"train"}
    assert split.attrs["single_person_groups"] == ["A"]


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.5])
def test_invalid_validation_fraction_raises(fraction):
    with pytest.raises(ValueError):
        make_person_split(_manifest({"A": 4}), 0, fraction)


def test_person_in_two_groups_raises():
    m = _manifest({"A": 3})
    m.loc[0, "group"] = "B"
    with pytest.raises(ValueError, match="a00"):
        make_person_split(m, 0, 0.3)


def test_leak_checker_raises_on_leaky_split():
    leaky = pd.DataFrame(
        {"video_id": ["v1", "v2"], "person_id": ["a01", "a01"], "group": ["A", "A"],
         "role": ["train", "validation"]}
    )
    with pytest.raises(AssertionError, match="a01"):
        assert_no_person_leak(leaky)


def test_non_ok_read_status_row_still_gets_a_role():
    m = _manifest({"A": 4, "B": 4})
    m.loc[0, "read_status"] = "error"
    split = make_person_split(m, 0, 0.3)
    assert len(split) == len(m)
    assert split["role"].isin(["train", "validation"]).all()
    assert split.loc[split["video_id"] == m.loc[0, "video_id"], "role"].notna().all()


def test_half_way_rounding_rounds_up():
    # 5 persons * 0.3 = 1.5 -> 2 validation persons (round half up)
    split = make_person_split(_manifest({"A": 5}), 0, 0.3)
    assert split.loc[split["role"] == "validation", "person_id"].nunique() == 2


def test_person_with_single_video():
    split = make_person_split(_manifest({"A": 4}, videos_per_person=1), 0, 0.3)
    assert len(split) == 4
    assert_no_person_leak(split)


def test_single_person_groups_recomputed_from_frame():
    split = make_person_split(_manifest({"A": 5, "C": 1, "B": 1}), 0, 0.3)
    derived = split.sort_values("video_id", ascending=False).copy()
    derived.attrs.clear()
    assert single_person_groups(derived) == ("B", "C")
    assert single_person_groups(make_person_split(_manifest({"A": 3}), 0, 0.3)) == ()


@pytest.mark.parametrize("bad", [None, "", "test"])
def test_leak_checker_rejects_invalid_role(bad):
    split = pd.DataFrame(
        {"video_id": ["v1", "v2"], "person_id": ["a01", "a02"], "group": ["A", "A"],
         "role": ["train", bad]}
    )
    with pytest.raises(AssertionError, match="role"):
        assert_no_person_leak(split)
