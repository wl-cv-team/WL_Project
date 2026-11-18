"""Ground truth for detection: plate-box label files turned into `GroundTruthFrame` lists.

Inputs are the label files `<boxes_dir>/<video_id>_f<frame>.txt` (see `plate_boxes`),
the video table (`video_id`, `width`, `height`, `frame_count`) and the person-level
split table (`video_id`, `role`). The role of a video is taken from the split table as
it is, so one video belongs to one role only.

An empty label file gives a frame with no boxes. Only `.txt` files are read; no image
is read or written and no file is created.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from wl_snatch.eval_detection import GroundTruthFrame
from wl_snatch.plate_boxes import PlateBoxStore, read_frame_ranges

ROLES = ("train", "validation")

_LABEL_NAME = re.compile(r"(?P<video_id>.+)_f(?P<frame>\d+)\.txt")


def parse_label_name(name: str) -> tuple[str, int]:
    """`("a01_sn_f_01", 3)` from `a01_sn_f_01_f3.txt`. Only the last `_f<digits>` is the frame."""
    match = _LABEL_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"{name}: label file name is not <video_id>_f<frame>.txt")
    return match["video_id"], int(match["frame"])


@dataclass(frozen=True)
class GroundTruthReport:
    frames: int
    with_box: int
    empty: int
    skipped_not_in_split: int
    video_ids: tuple[str, ...]


def _unique_rows(table: pd.DataFrame, what: str) -> dict[str, pd.Series]:
    rows: dict[str, pd.Series] = {}
    for _, row in table.iterrows():
        video_id = str(row["video_id"])
        if video_id in rows:
            raise ValueError(f"{what}: video {video_id} appears more than once")
        rows[video_id] = row
    return rows


def load_ground_truth(
    boxes_dir: Path,
    videos: pd.DataFrame,
    split: pd.DataFrame,
    role: str,
    ranges_path: Path | None = None,
) -> tuple[list[GroundTruthFrame], GroundTruthReport]:
    """Ground truth frames of the videos whose split role is `role`, sorted by video and frame.

    Raises ValueError for a `role` outside `ROLES`, a role value in `split` outside
    `ROLES`, a label file name of the wrong form, a video that is in `split` but not in
    `videos`, a frame outside `[0, frame_count)`, and (when `ranges_path` is given) a
    label file outside the frame range exception of its video. Label files of videos
    that are not in `split` are skipped and counted.
    """
    if role not in ROLES:
        raise ValueError(f"role {role!r} is not one of {ROLES}")
    split_rows = _unique_rows(split, "split")
    for video_id, row in split_rows.items():
        if row["role"] not in ROLES:
            raise ValueError(f"{video_id}: split role {row['role']!r} is not one of {ROLES}")
    video_rows = _unique_rows(videos, "videos")

    boxes_dir = Path(boxes_dir)
    store = PlateBoxStore(boxes_dir, ranges_path)
    exceptions = {} if ranges_path is None else read_frame_ranges(Path(ranges_path))

    entries: list[tuple[str, int]] = []
    skipped = 0
    for path in boxes_dir.iterdir():
        if path.name.startswith(".") or path.suffix != ".txt":
            continue
        video_id, frame = parse_label_name(path.name)
        if video_id not in split_rows:
            skipped += 1
            continue
        if video_id not in video_rows:
            raise ValueError(f"{video_id}: video is in the split but not in the video table")
        frame_count = int(video_rows[video_id]["frame_count"])
        if not 0 <= frame < frame_count:
            raise ValueError(
                f"{video_id}: frame {frame} is outside 0..{frame_count - 1} "
                f"(video has {frame_count} frames)"
            )
        allowed = exceptions.get(video_id)
        if allowed is not None and not allowed.start_frame <= frame <= allowed.end_frame:
            raise ValueError(
                f"{path.name}: frame {frame} is outside the allowed range "
                f"{allowed.start_frame}..{allowed.end_frame} ({allowed.reason})"
            )
        if split_rows[video_id]["role"] == role:
            entries.append((video_id, frame))

    entries.sort()
    frames: list[GroundTruthFrame] = []
    for video_id, frame in entries:
        width = float(video_rows[video_id]["width"])
        height = float(video_rows[video_id]["height"])
        norm = store.read(video_id, frame)
        if norm is None:
            boxes: tuple[tuple[float, float, float, float], ...] = ()
        else:
            cx, cy, w, h = norm
            boxes = (
                (
                    (cx - w / 2) * width,
                    (cy - h / 2) * height,
                    (cx + w / 2) * width,
                    (cy + h / 2) * height,
                ),
            )
        frames.append(GroundTruthFrame(video_id, frame, boxes))

    with_box = sum(1 for f in frames if f.boxes)
    report = GroundTruthReport(
        frames=len(frames),
        with_box=with_box,
        empty=len(frames) - with_box,
        skipped_not_in_split=skipped,
        video_ids=tuple(sorted({f.video_id for f in frames})),
    )
    return frames, report
