"""Load and validate the video manifest (data/metadata/videos.csv)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("video_id", "person_id", "group", "path", "has_plate", "read_status")


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
    return df
