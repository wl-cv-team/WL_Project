"""Command line entry point: person-level split of data/metadata/videos.csv."""

import argparse
from pathlib import Path

from wl_snatch.manifest import load_manifest
from wl_snatch.split import assert_no_person_leak, make_person_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a person-level train/validation split.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.3)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    manifest = load_manifest(args.data_dir / "metadata" / "videos.csv")
    split = make_person_split(manifest, args.seed, args.validation_fraction)
    assert_no_person_leak(split)

    out_path = args.data_dir / "splits" / "person_split.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(out_path, index=False)

    print(f"seed={args.seed} validation_fraction={args.validation_fraction}")
    print(f"persons={split['person_id'].nunique()} videos={len(split)}")
    persons = split.drop_duplicates("person_id").groupby(["group", "role"]).size()
    videos = split.groupby(["group", "role"]).size()
    for (group, role), n_videos in videos.items():
        print(f"group={group} role={role} persons={persons[(group, role)]} videos={n_videos}")
    single = split.attrs["single_person_groups"]
    print(f"single_person_groups={single} (assigned to train)" if single else "single_person_groups=[]")
    print("assert_no_person_leak: passed")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
