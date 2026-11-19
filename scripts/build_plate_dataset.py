"""Build the YOLO plate training dataset from the labeled frames and verify it.

Reads the two roles with `load_ground_truth`, exports images and labels under `--out`
(a git-ignored folder; the images show faces and must stay local), verifies the export
and prints the counts per role and the overall SHA256. Example:

    uv run python scripts/build_plate_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wl_snatch.dataset import export_yolo_dataset, verify_exported_dataset
from wl_snatch.ground_truth import ROLES, load_ground_truth

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos-csv", type=Path, default=Path("data/metadata/videos.csv"))
    parser.add_argument("--split", type=Path, default=Path("data/splits/person_split.csv"))
    parser.add_argument("--boxes-dir", type=Path, default=Path("data/annotations/plate_boxes"))
    parser.add_argument(
        "--ranges", type=Path, default=Path("data/annotations/plate_frame_ranges.csv")
    )
    parser.add_argument("--videos-dir", type=Path, default=Path("data/raw/videos"))
    parser.add_argument("--out", type=Path, default=Path("data/derived/plate_dataset"))
    args = parser.parse_args()

    videos = pd.read_csv(args.videos_csv)
    split = pd.read_csv(args.split)
    frames_by_role = {
        role: load_ground_truth(args.boxes_dir, videos, split, role, args.ranges)[0]
        for role in ROLES
    }
    yaml_path = export_yolo_dataset(frames_by_role, videos, args.videos_dir, args.out, REPO_ROOT)
    report = verify_exported_dataset(yaml_path)
    print(f"dataset: {yaml_path}")
    for role in ROLES:
        print(
            f"{role}: {report.images_by_role[role]} images "
            f"(plate {report.with_plate_by_role[role]}, "
            f"no plate {report.without_plate_by_role[role]})"
        )
    print(f"overall sha256: {report.overall_sha256}")


if __name__ == "__main__":
    main()
