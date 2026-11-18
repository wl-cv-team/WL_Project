"""Local viewer for detection results: boxes drawn over frames, in memory only.

The frames show people, so no image file is written anywhere: boxes are drawn on
copies of decoded frames and shown through `cv2.imencode` bytes. The viewer shows
the output of one model for the frames present in `detections`. The reference
overlay of the plate labels ("참고용 원판 라벨") is off by default and only draws
the label boxes; no IoU or other number is computed or shown.

`ipywidgets` is imported only inside `DetectionViewSession.show`; Ultralytics is
never imported here.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from wl_snatch.config import DetectorConfig
from wl_snatch.contracts import BoxDetection, FrameDetections, GroundTruthFrame
from wl_snatch.detector import LabeledModelFn, detect_frames_labeled
from wl_snatch.frames import read_frames
from wl_snatch.ground_truth import ROLES, load_ground_truth

TRUTH_COLOR = (0, 255, 255)  # BGR yellow, fixed for the reference overlay
TRUTH_NAME = "참고용 원판 라벨"
PLACEHOLDER_SIZE = (240, 320)  # (height, width) of the blank image for unreadable frames

Color = tuple[int, int, int]


def class_color(label: str) -> Color:
    """BGR color derived from the label with md5; the same in every run."""
    digest = hashlib.md5(label.encode("utf-8")).digest()
    color = (64 + digest[0] % 192, 64 + digest[1] % 192, 64 + digest[2] % 192)
    return (255, 0, 255) if color == TRUTH_COLOR else color


def _clipped(box: BoxDetection, width: int, height: int) -> tuple[int, int, int, int] | None:
    if box.x2 < 0 or box.y2 < 0 or box.x1 > width - 1 or box.y1 > height - 1:
        return None

    def clip(value: float, high: int) -> int:
        return max(0, min(high, round(value)))

    return clip(box.x1, width - 1), clip(box.y1, height - 1), clip(box.x2, width - 1), clip(
        box.y2, height - 1
    )


def draw_detections(
    frame: np.ndarray,
    boxes: Sequence[BoxDetection],
    *,
    min_confidence: float = 0.0,
    label_filter: Collection[str] | None = None,
    color_for: Callable[[str], Color] = class_color,
) -> np.ndarray:
    """Copy of `frame` with box, label and confidence drawn; the input is not changed.

    Coordinates outside the frame are clipped; a box entirely outside is skipped.
    """
    image = frame.copy()
    height, width = image.shape[:2]
    for box in boxes:
        if box.confidence < min_confidence:
            continue
        if label_filter is not None and box.label not in label_filter:
            continue
        clipped = _clipped(box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        color = color_for(box.label)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        text = f"{box.label} {box.confidence:.2f}"
        origin = (x1, y1 - 4) if y1 >= 14 else (x1 + 2, y1 + 12)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return image


def draw_truth(frame: np.ndarray, boxes: Sequence[BoxDetection]) -> np.ndarray:
    """Copy of `frame` with the reference label boxes in one fixed color (no text)."""
    image = frame.copy()
    height, width = image.shape[:2]
    for box in boxes:
        clipped = _clipped(box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        cv2.rectangle(image, (x1, y1), (x2, y2), TRUTH_COLOR, 1)
    return image


class DetectionViewSession:
    """Frames of one video with the detections of one model, read once.

    Shows only the frames present in `detections` (ascending). `goto` and `step`
    move over positions in `frame_indices` and stop at both ends.
    """

    def __init__(
        self,
        video_path: Path,
        video_id: str,
        detections: Sequence[FrameDetections],
        truths: Sequence[GroundTruthFrame] = (),
        roles: Mapping[int, str] | None = None,
    ):
        by_frame: dict[int, FrameDetections] = {}
        for item in detections:
            if item.frame_index in by_frame:
                raise ValueError(f"{video_id}: duplicate detections for frame {item.frame_index}")
            by_frame[item.frame_index] = item
        if not by_frame:
            raise ValueError(f"{video_id}: no frames to show")
        self.video_id = video_id
        self._detections = by_frame
        self.frame_indices = sorted(by_frame)
        self._truth = {t.frame_index: t.boxes for t in truths if t.video_id == video_id}
        self._roles = dict(roles or {})
        self._frames = self._read(Path(video_path))
        self.position = 0

    def _read(self, video_path: Path) -> dict[int, np.ndarray]:
        frames: dict[int, np.ndarray] = {}
        try:
            for index, frame in read_frames(video_path, self.frame_indices):
                frames[index] = frame
        except ValueError:
            pass  # unreadable video or frames beyond its end: those frames show a placeholder
        return frames

    @property
    def current_frame_index(self) -> int:
        return self.frame_indices[self.position]

    def goto(self, position: int) -> None:
        self.position = max(0, min(len(self.frame_indices) - 1, int(position)))

    def step(self, delta: int) -> None:
        self.goto(self.position + delta)

    def classes(self) -> list[str]:
        return sorted({b.label for d in self._detections.values() for b in d.boxes})

    def view(
        self,
        show_truth: bool = False,
        label_filter: Collection[str] | None = None,
        min_confidence: float = 0.0,
    ) -> tuple[np.ndarray, list[str]]:
        """Drawn frame and info lines (frame number, role, class counts) of the current frame."""
        index = self.current_frame_index
        frame = self._frames.get(index)
        if frame is None:
            frame = np.zeros((*PLACEHOLDER_SIZE, 3), dtype=np.uint8)
        detection = self._detections[index]
        lines = [f"프레임 {index}"]
        if index in self._roles:
            lines.append(f"역할: {self._roles[index]}")
        image = frame
        if detection.na_reason is not None:
            lines.append(f"N/A: {detection.na_reason}")
        else:
            shown = [
                b
                for b in detection.boxes
                if b.confidence >= min_confidence
                and (label_filter is None or b.label in label_filter)
            ]
            image = draw_detections(
                frame, shown, min_confidence=min_confidence, label_filter=label_filter
            )
            counts = Counter(b.label for b in shown)
            ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            lines.append(", ".join(f"{name} {n}" for name, n in ordered) or "박스 없음")
        if show_truth:
            truth_boxes = self._truth.get(index, ())
            image = draw_truth(
                image, [BoxDetection(index, *b, 1.0, TRUTH_NAME) for b in truth_boxes]
            )
            lines.append(f"{TRUTH_NAME}: {len(truth_boxes)}개")
        return image, lines

    def view_png(self, **kwargs) -> bytes:
        """PNG bytes of `view(**kwargs)`, encoded in memory (nothing is written to disk)."""
        image, _ = self.view(**kwargs)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("frame could not be encoded")
        return encoded.tobytes()

    def show(self):
        """Build and return the widget screen (slider, class choice, confidence, truth box)."""
        import ipywidgets as widgets

        slider = widgets.IntSlider(
            value=0, min=0, max=len(self.frame_indices) - 1, description="frame"
        )
        classes = widgets.SelectMultiple(
            options=self.classes(), value=tuple(self.classes()), description="class"
        )
        confidence = widgets.FloatSlider(
            value=0.0, min=0.0, max=1.0, step=0.05, description="min conf"
        )
        truth = widgets.Checkbox(value=False, description=TRUTH_NAME)
        picture = widgets.Image(format="png")
        info = widgets.HTML()

        def refresh(_=None) -> None:
            self.goto(slider.value)
            kwargs = {
                "show_truth": truth.value,
                "label_filter": set(classes.value),
                "min_confidence": confidence.value,
            }
            picture.value = self.view_png(**kwargs)
            info.value = "<br>".join(self.view(**kwargs)[1])

        for control in (slider, classes, confidence, truth):
            control.observe(refresh, names="value")
        refresh()
        return widgets.VBox([slider, classes, confidence, truth, picture, info])


def build_view_session(
    config: DetectorConfig,
    model: LabeledModelFn,
    video_id: str,
    videos_dir: Path,
    boxes_dir: Path,
    videos: pd.DataFrame,
    split: pd.DataFrame,
    ranges_path: Path | None = None,
) -> DetectionViewSession:
    """Run `model` on the labeled frames of one video and open a viewer on the result.

    The labeled frames come from `load_ground_truth`; the model runs only on them.
    The video is read from `<videos_dir>/<video_id>.mp4`.
    """
    truths: list[GroundTruthFrame] = []
    roles: dict[int, str] = {}
    for role in ROLES:
        frames, _ = load_ground_truth(boxes_dir, videos, split, role, ranges_path)
        for frame in frames:
            if frame.video_id == video_id:
                truths.append(frame)
                roles[frame.frame_index] = role
    if not truths:
        raise ValueError(f"{video_id}: no labeled frames found")
    video_path = Path(videos_dir) / f"{video_id}.mp4"
    detections = detect_frames_labeled(
        video_path, video_id, [t.frame_index for t in truths], model, config
    )
    return DetectionViewSession(video_path, video_id, detections, truths=truths, roles=roles)
