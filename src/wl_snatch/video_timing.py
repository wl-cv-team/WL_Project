"""Per-video timing facts: frame gaps, first frame time and irregular gap positions.

Reports numbers and frame numbers only. Writes a CSV; no image is created.
Frame number g in `irregular_gap_frames` means the gap between frame g and g+1
(decode order, starting at 0).
"""

from __future__ import annotations

import csv
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

COLUMNS = [
    "video_id",
    "width",
    "height",
    "frames",
    "fps_from_median_gap",
    "gap_min_ms",
    "gap_median_ms",
    "gap_max_ms",
    "first_frame_ms",
    "timestamp_note",
    "irregular_gap_frames",
]


@dataclass(frozen=True)
class TimingRow:
    video_id: str
    width: int | None
    height: int | None
    frames: int | None
    fps_from_median_gap: float | None
    gap_min_ms: float | None
    gap_median_ms: float | None
    gap_max_ms: float | None
    first_frame_ms: float | None
    timestamp_note: str
    irregular_gap_frames: tuple[int, ...]


def unreadable_row(video_id: str, reason: str) -> TimingRow:
    return TimingRow(
        video_id=video_id,
        width=None,
        height=None,
        frames=None,
        fps_from_median_gap=None,
        gap_min_ms=None,
        gap_median_ms=None,
        gap_max_ms=None,
        first_frame_ms=None,
        timestamp_note=f"N/A: {reason}",
        irregular_gap_frames=(),
    )


def analyze_timing(
    video_id: str,
    width: int,
    height: int,
    raw_times_ms: Sequence[float],
    offset_tolerance_ms: float = 0.5,
    gap_tolerance_ms: float = 2.5,
    gap_tolerance_fraction: float = 0.10,
) -> TimingRow:
    frames = len(raw_times_ms)
    first = float(raw_times_ms[0]) if frames else None
    if frames < 2:
        return TimingRow(
            video_id=video_id,
            width=width,
            height=height,
            frames=frames,
            fps_from_median_gap=None,
            gap_min_ms=None,
            gap_median_ms=None,
            gap_max_ms=None,
            first_frame_ms=first,
            timestamp_note="N/A: fewer than 2 frames",
            irregular_gap_frames=(),
        )
    gaps = [raw_times_ms[g + 1] - raw_times_ms[g] for g in range(frames - 1)]
    median = statistics.median(gaps)
    tolerance = max(gap_tolerance_ms, gap_tolerance_fraction * median)
    irregular = tuple(g for g, gap in enumerate(gaps) if abs(gap - median) > tolerance)
    notes = []
    if first > offset_tolerance_ms:
        notes.append("first_frame_offset")
    if irregular:
        notes.append("irregular_gap")
    return TimingRow(
        video_id=video_id,
        width=width,
        height=height,
        frames=frames,
        fps_from_median_gap=1000.0 / median if median > 0 else None,
        gap_min_ms=min(gaps),
        gap_median_ms=median,
        gap_max_ms=max(gaps),
        first_frame_ms=first,
        timestamp_note=";".join(notes),
        irregular_gap_frames=irregular,
    )


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_timing_csv(rows: Sequence[TimingRow], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row.video_id,
                    _cell(row.width),
                    _cell(row.height),
                    _cell(row.frames),
                    _cell(row.fps_from_median_gap),
                    _cell(row.gap_min_ms),
                    _cell(row.gap_median_ms),
                    _cell(row.gap_max_ms),
                    _cell(row.first_frame_ms),
                    row.timestamp_note,
                    ";".join(str(g) for g in row.irregular_gap_frames),
                ]
            )
