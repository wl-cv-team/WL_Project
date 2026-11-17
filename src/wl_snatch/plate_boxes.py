"""Bounding boxes around the camera-side weight plate, drawn by hand (issue #2).

One label is one file, `<boxes_dir>/<video_id>_f<frame>.txt`, with one line
`class cx cy w h` (YOLO, normalized to 0..1, class always 0).

- An empty file means the frame was checked and has no camera-side plate.
- A missing file means the frame is not labeled yet.
- Saving the same video and frame again replaces the earlier file.
- A video listed in the frame range table accepts labels only for frames in
  `start_frame..end_frame` (both included). Other videos accept every frame.

Frame numbers are decode order from 0 (issue #10). Nothing in this module
writes an image file. The frame shown by `BoxLabelingSession.show` is encoded
in memory only.
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wl_snatch.frames import count_frames, frame_size, read_frame
from wl_snatch.label_viewer import NormBox, Overlay, draw_overlay, flag_box_issues
from wl_snatch.labeling import guidance_html

DRAFT_COLOR_BGR = (0, 255, 255)
DISPLAY_WIDTH = 360
_RANGE_COLUMNS = ["video_id", "start_frame", "end_frame", "reason"]


def corners_to_norm_box(
    x1: float, y1: float, x2: float, y2: float, width: float, height: float
) -> NormBox:
    """Normalized (cx, cy, w, h) of the rectangle with these corners, clipped to the frame.

    The corners may be given in any order. Coordinates are clipped to
    [0, width] x [0, height] before normalizing. The same function serves
    screen coordinates (width and height of the displayed image) and original
    pixel coordinates (original size). Raises ValueError when the clipped
    rectangle has no width or no height.
    """
    inputs = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": width, "height": height}
    non_finite = [name for name, value in inputs.items() if not math.isfinite(value)]
    if non_finite:
        raise ValueError(f"box input {', '.join(non_finite)} is not finite: {inputs}")
    left = min(max(min(x1, x2), 0.0), width)
    right = min(max(max(x1, x2), 0.0), width)
    top = min(max(min(y1, y2), 0.0), height)
    bottom = min(max(max(y1, y2), 0.0), height)
    if right - left <= 0 or bottom - top <= 0:
        raise ValueError(
            f"box has no area after clipping to the frame: corners ({x1}, {y1}), ({x2}, {y2}), "
            f"frame size {width} x {height}"
        )
    return (
        (left + right) / 2 / width,
        (top + bottom) / 2 / height,
        (right - left) / width,
        (bottom - top) / height,
    )


def format_yolo_line(box: NormBox) -> str:
    """`0 cx cy w h` with six decimals and no line break."""
    cx, cy, w, h = box
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def parse_yolo_line(text: str, source: str) -> NormBox:
    """Box from one `0 cx cy w h` line. `source` names the file in error messages."""
    fields = text.split()
    if len(fields) != 5:
        raise ValueError(f"{source}: expected 5 fields, found {len(fields)}")
    if fields[0] != "0":
        raise ValueError(f"{source}: class {fields[0]!r} is not 0")
    try:
        values = tuple(float(field) for field in fields[1:])
    except ValueError:
        raise ValueError(f"{source}: {text.strip()!r} has a field that is not a number") from None
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"{source}: values {values} are not all within 0..1")
    if values[2] <= 0 or values[3] <= 0:
        raise ValueError(f"{source}: width {values[2]} or height {values[3]} is not above 0")
    return values  # type: ignore[return-value]


@dataclass(frozen=True)
class FrameRange:
    start_frame: int
    end_frame: int
    reason: str


def read_frame_ranges(path: Path) -> dict[str, FrameRange]:
    """Frame range exceptions by video_id. A missing or 0-byte file gives an empty dict."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    ranges: dict[str, FrameRange] = {}
    first_line: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != _RANGE_COLUMNS:
            raise ValueError(
                f"{path.name} line 1: header is {header!r}, expected {_RANGE_COLUMNS!r}"
            )
        for row in reader:
            line = reader.line_num
            where = f"{path.name} line {line}"
            if len(row) != len(_RANGE_COLUMNS):
                raise ValueError(f"{where}: expected {len(_RANGE_COLUMNS)} columns, found {len(row)}")
            video_id, start_text, end_text, reason = row
            if not video_id.strip():
                raise ValueError(f"{where}: video_id is empty")
            try:
                start, end = int(start_text), int(end_text)
            except ValueError:
                raise ValueError(
                    f"{where}: {video_id}: start_frame {start_text!r} or end_frame "
                    f"{end_text!r} is not an integer"
                ) from None
            if start < 0 or start > end:
                raise ValueError(
                    f"{where}: {video_id}: start_frame {start} and end_frame {end} "
                    f"do not satisfy 0 <= start_frame <= end_frame"
                )
            if video_id in ranges:
                raise ValueError(
                    f"{where}: {video_id} already has a range on line {first_line[video_id]}"
                )
            first_line[video_id] = line
            ranges[video_id] = FrameRange(start, end, reason)
    return ranges


class PlateBoxStore:
    """Directory of one-line label files, plus the optional frame range table."""

    def __init__(self, boxes_dir: Path, ranges_path: Path | None = None):
        self.boxes_dir = Path(boxes_dir)
        self.ranges_path = None if ranges_path is None else Path(ranges_path)

    def path_for(self, video_id: str, frame: int) -> Path:
        return self.boxes_dir / f"{video_id}_f{frame}.txt"

    def frame_range(self, video_id: str) -> FrameRange | None:
        """Range exception of this video, or None. The table is read on every call."""
        if self.ranges_path is None:
            return None
        return read_frame_ranges(self.ranges_path).get(video_id)

    def save_box(self, video_id: str, frame: int, box: NormBox, *, frame_count: int) -> None:
        """Write the box for this frame, replacing any earlier file."""
        line = format_yolo_line(box)
        self._check(video_id, frame, frame_count)
        parse_yolo_line(line, f"{video_id} frame {frame}")  # same rules as reading back
        self._write(video_id, frame, line + "\n")

    def save_empty(self, video_id: str, frame: int, *, frame_count: int) -> None:
        """Write an empty file: checked, no camera-side plate."""
        self._check(video_id, frame, frame_count)
        self._write(video_id, frame, "")

    def status(self, video_id: str, frame: int) -> str:
        """"unlabeled" (no file), "empty" (no content besides whitespace) or "box"."""
        path = self.path_for(video_id, frame)
        if not path.exists():
            return "unlabeled"
        return "box" if path.read_text(encoding="utf-8").strip() else "empty"

    def read(self, video_id: str, frame: int) -> NormBox | None:
        """The saved box, or None for an empty file or no file."""
        path = self.path_for(video_id, frame)
        if not path.exists():
            return None
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return None
        if len(lines) > 1:
            raise ValueError(f"{path.name}: expected one line, found {len(lines)}")
        return parse_yolo_line(lines[0], path.name)

    def _check(self, video_id: str, frame: int, frame_count: int) -> None:
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError(f"video_id {video_id!r} is empty or not text")
        if "/" in video_id or "\\" in video_id:
            raise ValueError(f"video_id {video_id!r} contains a path separator")
        if type(frame) is not int or not 0 <= frame < frame_count:
            raise ValueError(
                f"{video_id}: frame {frame!r} is not an integer within 0..{frame_count - 1} "
                f"(video has {frame_count} frames)"
            )
        allowed = self.frame_range(video_id)
        if allowed is not None and not allowed.start_frame <= frame <= allowed.end_frame:
            raise ValueError(
                f"{video_id}: frame {frame} is outside the allowed range "
                f"{allowed.start_frame}..{allowed.end_frame} ({allowed.reason})"
            )

    def _write(self, video_id: str, frame: int, content: str) -> None:
        target = self.path_for(video_id, frame)
        self.boxes_dir.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            dir=self.boxes_dir,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                handle.write(content)
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


class BoxLabelingSession:
    """Frame navigation, a draft box and saving for one video."""

    def __init__(self, video_path: Path, video_id: str, store: PlateBoxStore):
        self.video_path = Path(video_path)
        self.video_id = video_id
        self.store = store
        self.frame_count = count_frames(self.video_path)
        self.frame = 0
        self.draft: NormBox | None = None
        self.pointer_text = ""
        self._size: tuple[int, int] | None = None
        self._drag_start: tuple[float, float] | None = None

    def goto(self, frame: int) -> None:
        """Move to a frame and clear the draft and any drag start."""
        if type(frame) is not int or not 0 <= frame < self.frame_count:
            raise ValueError(
                f"{self.video_id}: frame {frame!r} is not an integer within "
                f"0..{self.frame_count - 1}"
            )
        self.frame = frame
        self.clear_draft()

    def step(self, delta: int) -> None:
        """Move by `delta` frames, stopping at the first and last frame. Clears the draft."""
        self.frame = min(max(self.frame + delta, 0), self.frame_count - 1)
        self.clear_draft()

    def set_draft_from_corners(
        self, x1: float, y1: float, x2: float, y2: float, width: float, height: float
    ) -> None:
        self.draft = corners_to_norm_box(x1, y1, x2, y2, width, height)

    def clear_draft(self) -> None:
        """Clear the draft and the drag start."""
        self.draft = None
        self._drag_start = None

    def save_box(self) -> None:
        if self.draft is None:
            raise ValueError(f"{self.video_id}: frame {self.frame} has no draft box to save")
        self.store.save_box(self.video_id, self.frame, self.draft, frame_count=self.frame_count)

    def save_empty(self) -> None:
        self.store.save_empty(self.video_id, self.frame, frame_count=self.frame_count)

    def view(self) -> tuple[np.ndarray, list[str]]:
        """Current frame with the saved box (green) and the draft (yellow), and guidance lines."""
        image = read_frame(self.video_path, self.frame)
        saved = self.store.read(self.video_id, self.frame)
        lines = [
            f"{self.video_id}",
            f"frame {self.frame} / {self.frame_count}",
            f"status: {self.store.status(self.video_id, self.frame)}",
        ]
        allowed = self.store.frame_range(self.video_id)
        if allowed is not None:
            lines.append(
                f"allowed frames {allowed.start_frame}..{allowed.end_frame}: {allowed.reason}"
            )
        shown = (("saved", saved), ("draft", self.draft))
        for name, box in shown:
            if box is not None:
                lines.extend(f"{name} {issue}" for issue in flag_box_issues([box]))
        if saved is not None:
            image = draw_overlay(image, Overlay(boxes=(saved,), skipped=()))
        if self.draft is not None:
            image = draw_overlay(image, Overlay(boxes=(self.draft,), skipped=()), DRAFT_COLOR_BGR)
        return image, lines

    def handle_pointer_event(self, event: Mapping[str, Any]) -> None:
        """Process one dict from `ipyevents`: mousedown, mousemove or mouseup.

        Events without a positive `boundingRectWidth` and `boundingRectHeight`,
        and events of other types, are ignored. A mouseup that gives a box
        without area raises ValueError after the start point is cleared.
        """
        kind = event.get("type")
        if kind not in ("mousedown", "mouseup", "mousemove"):
            return
        try:
            x = float(event["relativeX"])
            y = float(event["relativeY"])
            width = float(event["boundingRectWidth"])
            height = float(event["boundingRectHeight"])
        except (KeyError, TypeError, ValueError):
            return
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            return
        if not width > 0 or not height > 0:
            return
        if kind == "mousemove":
            self.pointer_text = self._describe_pointer(x / width, y / height)
        elif kind == "mousedown":
            self._drag_start = (x, y)
        elif self._drag_start is not None:
            start, self._drag_start = self._drag_start, None
            self.set_draft_from_corners(start[0], start[1], x, y, width, height)

    def _describe_pointer(self, fx: float, fy: float) -> str:
        if self._size is None:
            self._size = frame_size(self.video_path)
        original_width, original_height = self._size
        px = min(max(int(fx * original_width), 0), original_width - 1)
        py = min(max(int(fy * original_height), 0), original_height - 1)
        return f"x {fx:.3f}  y {fy:.3f}  pixel ({px}, {py})"

    def show(self) -> None:
        """Display the frame, sliders, buttons and pointer text. No file is written for the image.

        The frame is encoded in memory and handed to the widget. Dragging on
        the image and the four corner sliders both set the same draft.
        """
        import cv2
        import ipyevents
        import ipywidgets
        from IPython.display import display

        width, height = frame_size(self.video_path)
        picture = ipywidgets.Image(
            format="jpeg", width=DISPLAY_WIDTH, height=round(DISPLAY_WIDTH * height / width)
        )
        text = ipywidgets.HTML()
        message = ipywidgets.HTML()
        pointer = ipywidgets.HTML()
        frame_slider = ipywidgets.IntSlider(
            value=self.frame,
            min=0,
            max=self.frame_count - 1,
            description="frame",
            continuous_update=False,
            layout=ipywidgets.Layout(width="600px"),
        )
        back = ipywidgets.Button(description="-1")
        forward = ipywidgets.Button(description="+1")
        corner_sliders = {
            name: ipywidgets.FloatSlider(
                value=value, min=0.0, max=1.0, step=0.001, description=name,
                continuous_update=False, readout_format=".3f",
            )
            for name, value in (("left", 0.25), ("top", 0.25), ("right", 0.75), ("bottom", 0.75))
        }
        save_box = ipywidgets.Button(description="save box")
        save_empty = ipywidgets.Button(description="save empty")
        syncing = [False]

        def refresh() -> None:
            image, lines = self.view()
            encoded_ok, encoded = cv2.imencode(".jpg", image)
            if not encoded_ok:
                raise ValueError(f"{self.video_id}: frame {self.frame} could not be encoded")
            picture.value = encoded.tobytes()
            text.value = guidance_html(lines)
            pointer.value = guidance_html([self.pointer_text])

        def report(action) -> None:
            try:
                action()
                message.value = ""
            except ValueError as error:
                message.value = guidance_html([str(error)])
            refresh()

        def sync_sliders() -> None:
            if self.draft is None:
                return
            cx, cy, w, h = self.draft
            syncing[0] = True
            try:
                corner_sliders["left"].value = cx - w / 2
                corner_sliders["top"].value = cy - h / 2
                corner_sliders["right"].value = cx + w / 2
                corner_sliders["bottom"].value = cy + h / 2
            finally:
                syncing[0] = False

        def on_pointer(event) -> None:
            kind = event.get("type")
            if kind == "mousemove":  # text only: the frame is not decoded again
                self.handle_pointer_event(event)
                pointer.value = guidance_html([self.pointer_text])
            elif kind == "mouseup":
                report(lambda: self.handle_pointer_event(event))
                sync_sliders()
            else:
                self.handle_pointer_event(event)

        def on_corner(_change) -> None:
            if syncing[0]:
                return
            values = [corner_sliders[name].value for name in ("left", "top", "right", "bottom")]
            report(lambda: self.set_draft_from_corners(*values, 1.0, 1.0))

        def on_frame(change) -> None:
            self.goto(change["new"])
            message.value = ""
            refresh()

        frame_slider.observe(on_frame, names="value")
        for slider in corner_sliders.values():
            slider.observe(on_corner, names="value")
        back.on_click(lambda _button: setattr(frame_slider, "value", max(self.frame - 1, 0)))
        forward.on_click(
            lambda _button: setattr(
                frame_slider, "value", min(self.frame + 1, self.frame_count - 1)
            )
        )
        save_box.on_click(lambda _button: report(self.save_box))
        save_empty.on_click(lambda _button: report(self.save_empty))

        events = ipyevents.Event(
            source=picture,
            watched_events=["mousedown", "mouseup", "mousemove"],
            prevent_default_action=True,
            throttle_or_debounce="throttle",
            wait=50,
        )
        events.on_dom_event(on_pointer)
        refresh()
        display(
            ipywidgets.VBox(
                [
                    picture,
                    pointer,
                    frame_slider,
                    ipywidgets.HBox([back, forward]),
                    *corner_sliders.values(),
                    ipywidgets.HBox([save_box, save_empty]),
                    text,
                    message,
                ]
            )
        )
