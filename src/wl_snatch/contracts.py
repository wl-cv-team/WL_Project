"""단계 사이에서 주고받는 데이터 형식. 형식 설명은 docs/contracts/detection.md."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

_COLUMNS = [
    "video_id",
    "frame_index",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
    "label",
    "na_reason",
]


@dataclass(frozen=True)
class BoxDetection:
    """한 프레임의 박스 하나. 좌표는 픽셀, 원점은 왼쪽 위."""

    frame_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2, self.confidence)
        if any(math.isnan(v) for v in values):
            raise ValueError("box values must not be NaN")
        if self.x1 > self.x2 or self.y1 > self.y2:
            raise ValueError("box must satisfy x1 <= x2 and y1 <= y2")


@dataclass(frozen=True)
class FrameDetections:
    """한 프레임의 탐지 결과. na_reason이 있으면 N/A 프레임이고 boxes는 비어 있어야 한다."""

    video_id: str
    frame_index: int
    boxes: tuple[BoxDetection, ...]
    na_reason: str | None

    def __post_init__(self) -> None:
        if self.na_reason == "":
            raise ValueError("na_reason must be None or a non-empty string")
        if self.na_reason is not None and self.boxes:
            raise ValueError("N/A frame must not have boxes")


def write_detections_csv(frames: list[FrameDetections], path: Path) -> None:
    """박스 하나를 한 행으로 쓴다. 박스가 없는 프레임은 좌표가 빈 한 행으로 쓴다."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_COLUMNS)
        for frame in frames:
            if frame.boxes:
                for b in frame.boxes:
                    writer.writerow(
                        [frame.video_id, frame.frame_index, repr(b.x1), repr(b.y1),
                         repr(b.x2), repr(b.y2), repr(b.confidence), b.label, ""]
                    )
            else:
                writer.writerow(
                    [frame.video_id, frame.frame_index, "", "", "", "", "", "",
                     frame.na_reason or ""]
                )


def read_detections_csv(path: Path) -> list[FrameDetections]:
    """write_detections_csv가 쓴 파일을 읽는다. 프레임 순서와 video_id를 유지한다.

    형식이 어긋난 행은 파일 경로와 행 번호(헤더가 1행)를 담은 ValueError로 거부한다.
    """
    frames: list[dict] = []
    seen: set[tuple[str, int]] = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in _COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        for row in reader:
            n = reader.line_num

            def bad(msg: str) -> ValueError:
                return ValueError(f"{path}: row {n}: {msg}")

            if any(row[c] is None for c in _COLUMNS):
                raise bad("too few fields")
            try:
                key = (row["video_id"], int(row["frame_index"]))
            except ValueError:
                raise bad("frame_index is not an integer") from None
            reason = row["na_reason"] or None
            coords = [row[c] for c in ("x1", "y1", "x2", "y2", "confidence")]
            filled = [c != "" for c in coords]
            box = None
            if any(filled):
                if not all(filled) or row["label"] == "":
                    raise bad("coordinates, confidence and label must all be filled")
                if reason is not None:
                    raise bad("na_reason must be empty when a box is given")
                try:
                    x1, y1, x2, y2, conf = (float(c) for c in coords)
                    box = BoxDetection(key[1], x1, y1, x2, y2, conf, row["label"])
                except ValueError as e:
                    raise bad(str(e)) from None
            elif row["label"] != "":
                raise bad("label given without coordinates")

            cur = frames[-1] if frames else None
            if cur is not None and cur["key"] == key:
                if reason != cur["reason"]:
                    raise bad("na_reason differs from earlier row of the same frame")
                if box is None or cur["empty"]:
                    raise bad("box-less row must be the only row of its frame")
            else:
                if key in seen:
                    raise bad(f"duplicate frame {key} is not contiguous")
                seen.add(key)
                cur = {"key": key, "reason": reason, "boxes": [], "empty": box is None}
                frames.append(cur)
            if box is not None:
                cur["boxes"].append(box)
    return [
        FrameDetections(c["key"][0], c["key"][1], tuple(c["boxes"]), c["reason"])
        for c in frames
    ]
