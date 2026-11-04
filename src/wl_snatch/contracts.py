"""단계 사이에서 주고받는 데이터 형식. 형식 설명은 docs/contracts/detection.md."""

import csv
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


@dataclass(frozen=True)
class FrameDetections:
    """한 프레임의 탐지 결과. na_reason이 있으면 N/A 프레임이고 boxes는 비어 있어야 한다."""

    video_id: str
    frame_index: int
    boxes: tuple[BoxDetection, ...]
    na_reason: str | None

    def __post_init__(self) -> None:
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
    """write_detections_csv가 쓴 파일을 읽는다. 프레임 순서와 video_id를 유지한다."""
    order: list[tuple[str, int]] = []
    boxes: dict[tuple[str, int], list[BoxDetection]] = {}
    reasons: dict[tuple[str, int], str | None] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["video_id"], int(row["frame_index"]))
            if key not in boxes:
                order.append(key)
                boxes[key] = []
                reasons[key] = row["na_reason"] or None
            if row["x1"] != "":
                boxes[key].append(
                    BoxDetection(
                        key[1], float(row["x1"]), float(row["y1"]), float(row["x2"]),
                        float(row["y2"]), float(row["confidence"]), row["label"],
                    )
                )
    return [FrameDetections(k[0], k[1], tuple(boxes[k]), reasons[k]) for k in order]
