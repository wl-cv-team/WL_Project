"""Write per-video timing facts for every video in data/metadata/videos.csv."""

import argparse
import csv
from collections import Counter
from pathlib import Path

from wl_snatch.frames import frame_size, raw_frame_times_ms
from wl_snatch.video_timing import TimingRow, analyze_timing, unreadable_row, write_timing_csv


def build_row(video_id: str, video_path: Path) -> TimingRow:
    try:
        width, height = frame_size(video_path)
        times = raw_frame_times_ms(video_path)
    except ValueError as error:
        return unreadable_row(video_id, str(error))
    return analyze_timing(video_id, width, height, times)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the video timing table.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("data/metadata/video_timing.csv"))
    args = parser.parse_args()

    # `path` in videos.csv is relative to the repository root (the parent of the data dir).
    root = args.data_dir.resolve().parent
    with (args.data_dir / "metadata" / "videos.csv").open(newline="") as handle:
        entries = [(r["video_id"], root / r["path"]) for r in csv.DictReader(handle)]

    rows = [build_row(video_id, path) for video_id, path in entries]
    write_timing_csv(rows, args.out)

    notes = Counter(row.timestamp_note or "(empty)" for row in rows)
    unreadable = sum(1 for row in rows if row.frames is None)
    print(f"videos={len(rows)}")
    for note, count in sorted(notes.items()):
        print(f"timestamp_note {note}={count}")
    print(f"unreadable={unreadable}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
