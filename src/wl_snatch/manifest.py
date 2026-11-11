"""Load and validate the video manifest (data/metadata/videos.csv)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("video_id", "person_id", "group", "path", "has_plate", "read_status")

# Which side of the lifter faces the camera. Entered by hand per video (issue #7).
NEAR_SIDE_VALUES = ("left", "right")


def load_manifest(path: Path) -> pd.DataFrame:
    """Read the manifest CSV.

    Raises ValueError if any required column is missing (all missing names are
    listed) or if a video_id appears more than once. Identifier columns are read
    as strings so ids such as "001" keep their form.
    """
    df = pd.read_csv(
        path, dtype={"video_id": str, "person_id": str, "group": str}, keep_default_na=False
    )
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest {path} is missing required columns: {missing}")
    for column in ("video_id", "person_id", "group"):
        blank = df[column].isna() | (df[column].astype(str).str.strip() == "")
        if blank.any():
            raise ValueError(
                f"manifest {path} has {int(blank.sum())} row(s) with blank {column} "
                f"(csv data rows {[int(i) + 2 for i in df.index[blank]][:10]})"
            )
    duplicated = sorted(df.loc[df["video_id"].duplicated(), "video_id"].unique())
    if duplicated:
        raise ValueError(f"manifest {path} has duplicate video_id values: {duplicated}")
    if "near_side" in df.columns:
        # Blank means "not entered yet"; any other value must be exactly left or right.
        bad = ~df["near_side"].isin(("",) + NEAR_SIDE_VALUES)
        if bad.any():
            found = sorted(df.loc[bad, "near_side"].unique())
            raise ValueError(
                f"manifest {path} has invalid near_side values {found}; "
                f"allowed: {list(NEAR_SIDE_VALUES)} (or blank until entered)"
            )
    return df


def require_near_side(df: pd.DataFrame) -> None:
    """Raise ValueError unless every video has near_side set to left or right.

    Stages that need the side (for example choosing the near-side knee) call this;
    stages that do not (the person split) do not.
    """
    if "near_side" not in df.columns:
        raise ValueError("manifest has no near_side column")
    blank = df["near_side"] == ""
    if blank.any():
        ids = list(df.loc[blank, "video_id"])
        raise ValueError(f"near_side is blank for {len(ids)} video(s): {ids}")
