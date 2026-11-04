import pandas as pd
import pytest

from wl_snatch.split import assert_no_person_leak, make_person_split


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
