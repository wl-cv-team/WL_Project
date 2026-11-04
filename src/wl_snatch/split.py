"""Person-level train/validation split."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRAIN = "train"
VALIDATION = "validation"


def _validation_count(n_persons: int, fraction: float) -> int:
    """Validation persons for a group: round half up, at least 1, at most n-1.

    Only called for groups with 2 or more persons.
    """
    return min(n_persons - 1, max(1, math.floor(n_persons * fraction + 0.5)))


def make_person_split(
    manifest: pd.DataFrame, seed: int, validation_fraction: float
) -> pd.DataFrame:
    """Split videos into train/validation with person_id as the unit.

    The sorted unique person_ids are shuffled once with numpy's default_rng(seed),
    then each group (A/B/C) takes its persons in that order: the first
    round(n * validation_fraction) go to validation, rounded half up, but at
    least 1 and at most n - 1, so every group with 2 or more persons has both
    roles. The result does not depend on the row order of the manifest.

    A group with a single person puts that person in train. Those group names
    are returned in ``result.attrs["single_person_groups"]`` (sorted list, empty
    when none). ``attrs`` is not carried through every pandas operation, so
    read it from the returned frame directly.

    Rows are not filtered by read_status: a video that could not be read still
    receives its person's role, because the split is about people. Later stages
    record unreadable videos as N/A with a reason.

    Returns columns video_id, person_id, group, role, sorted by video_id.
    Raises ValueError if validation_fraction is not in (0, 1) or if one
    person_id appears in more than one group.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")

    persons = manifest[["person_id", "group"]].drop_duplicates()
    multi_group = sorted(persons.loc[persons["person_id"].duplicated(), "person_id"].unique())
    if multi_group:
        raise ValueError(f"person_id appears in more than one group: {multi_group}")

    person_group = dict(zip(persons["person_id"], persons["group"]))
    ordered = sorted(person_group)
    shuffled = [ordered[i] for i in np.random.default_rng(seed).permutation(len(ordered))]

    role_of: dict[str, str] = {}
    single_person_groups: list[str] = []
    for group in sorted(set(person_group.values())):
        members = [p for p in shuffled if person_group[p] == group]
        if len(members) == 1:
            single_person_groups.append(group)
            n_validation = 0
        else:
            n_validation = _validation_count(len(members), validation_fraction)
        for i, person in enumerate(members):
            role_of[person] = VALIDATION if i < n_validation else TRAIN

    result = manifest[["video_id", "person_id", "group"]].copy()
    result["role"] = result["person_id"].map(role_of)
    result = result.sort_values("video_id").reset_index(drop=True)
    result.attrs["single_person_groups"] = single_person_groups
    return result


def single_person_groups(split: pd.DataFrame) -> tuple[str, ...]:
    """Sorted groups that contain exactly one person_id, computed from the frame."""
    counts = split.groupby("group")["person_id"].nunique()
    return tuple(sorted(counts[counts == 1].index))


def assert_no_person_leak(split: pd.DataFrame) -> None:
    """Raise AssertionError on a person in more than one role or an invalid role.

    Valid roles are exactly "train" and "validation"; NaN, blank or any other
    value in the role column also raises.
    """
    invalid = ~split["role"].isin([TRAIN, VALIDATION])
    if invalid.any():
        values = sorted({repr(v) for v in split.loc[invalid, "role"]})
        raise AssertionError(f"role has {int(invalid.sum())} invalid value(s): {values}")
    roles_per_person = split.groupby("person_id")["role"].nunique()
    leaked = sorted(roles_per_person[roles_per_person > 1].index)
    if leaked:
        raise AssertionError(f"person_id present in more than one role: {leaked}")
