"""Point labels for snatch phase boundary frames, entered by hand (issues #8, #12).

A point label is a frame number for one event of one video in one labeling pass.
Rules (issue #12):

- A new value replaces an old one only when (video_id, event, pass) are all equal.
- Pass 1 and pass 2 are separate records, kept even when the frame is the same.
- Once a video has any pass 2 record, its pass 1 records cannot be added or changed.
- A half video (completion "h") requires ready, transition and drive. catch and
  finish are ignored for it: they are never stored and are derived from the
  completion value, so they can be told apart from a missing record.

Nothing in this module writes an image file. The frame shown by
`LabelingSession.show` is encoded in memory only.
"""

from __future__ import annotations

import csv
import html
import os
import random
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wl_snatch.frames import count_frames, read_frame

EVENTS = ("ready", "transition", "drive", "catch", "finish")
PASSES = (1, 2)
_HALF_REQUIRED = EVENTS[:3]
_COLUMNS = ["video_id", "event", "frame", "pass"]


def required_events(completion: str) -> tuple[str, ...]:
    """Events that need a record: all five for "f", the first three for "h"."""
    if completion == "f":
        return EVENTS
    if completion == "h":
        return _HALF_REQUIRED
    raise ValueError(f"completion {completion!r} is not 'f' or 'h'")


def ignored_events(completion: str) -> tuple[str, ...]:
    """Events that are not labeled for this completion value."""
    required = required_events(completion)
    return tuple(event for event in EVENTS if event not in required)


@dataclass(frozen=True)
class PointLabel:
    video_id: str
    event: str
    frame: int
    pass_no: int


def _sort_key(label: PointLabel) -> tuple[str, int, int]:
    return (label.video_id, label.pass_no, EVENTS.index(label.event))


class PointLabelStore:
    """CSV file with columns video_id, event, frame, pass."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def read(self, video_id: str | None = None, pass_no: int | None = None) -> list[PointLabel]:
        """Labels in file order, optionally limited to one video and/or one pass."""
        labels = self._read_all()
        if video_id is not None:
            labels = [label for label in labels if label.video_id == video_id]
        if pass_no is not None:
            labels = [label for label in labels if label.pass_no == pass_no]
        return labels

    def add(
        self,
        video_id: str,
        event: str,
        frame: int,
        pass_no: int,
        *,
        completion: str,
        frame_count: int,
    ) -> None:
        """Store one label, replacing the one with the same (video_id, event, pass)."""
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError(f"video_id {video_id!r} is empty or not text")
        if event not in EVENTS:
            raise ValueError(f"{video_id}: event {event!r} is not one of {', '.join(EVENTS)}")
        if type(pass_no) is not int or pass_no not in PASSES:
            raise ValueError(f"{video_id}: pass {pass_no!r} is not the integer 1 or 2")
        if type(frame) is not int or not 0 <= frame < frame_count:
            raise ValueError(
                f"{video_id}: frame {frame!r} is outside 0..{frame_count - 1} "
                f"(video has {frame_count} frames)"
            )
        if event in ignored_events(completion):
            raise ValueError(
                f"{video_id}: event {event!r} is ignored for completion {completion!r}"
            )
        existing = self._read_all()
        if pass_no == 1 and any(
            label.video_id == video_id and label.pass_no == 2 for label in existing
        ):
            raise ValueError(
                f"{video_id}: pass 1 cannot be changed because pass 2 records exist "
                f"(event {event!r}, frame {frame})"
            )
        kept = [
            label
            for label in existing
            if (label.video_id, label.event, label.pass_no) != (video_id, event, pass_no)
        ]
        kept.append(PointLabel(video_id, event, frame, pass_no))
        self._write(sorted(kept, key=_sort_key))

    def missing_events(self, video_id: str, completion: str, pass_no: int) -> tuple[str, ...]:
        """Required events that have no record in this pass."""
        recorded = {label.event for label in self.read(video_id, pass_no)}
        return tuple(event for event in required_events(completion) if event not in recorded)

    def _read_all(self) -> list[PointLabel]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return []
        labels = []
        first_line: dict[tuple[str, str, int], int] = {}
        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header != _COLUMNS:
                raise ValueError(f"{self.path.name} line 1: header is {header!r}, expected {_COLUMNS!r}")
            for row in reader:
                label = self._parse_row(row, reader.line_num)
                key = (label.video_id, label.event, label.pass_no)
                if key in first_line:
                    raise ValueError(
                        f"{self.path.name} line {reader.line_num}: {label.video_id}: event "
                        f"{label.event!r} pass {label.pass_no} already recorded on "
                        f"line {first_line[key]}"
                    )
                first_line[key] = reader.line_num
                labels.append(label)
        return labels

    def _parse_row(self, row: list[str], line: int) -> PointLabel:
        where = f"{self.path.name} line {line}"
        if len(row) != len(_COLUMNS):
            raise ValueError(f"{where}: expected {len(_COLUMNS)} columns, found {len(row)}")
        video_id, event, frame_text, pass_text = row
        if not video_id.strip():
            raise ValueError(f"{where}: video_id is empty")
        if event not in EVENTS:
            raise ValueError(f"{where}: {video_id}: event {event!r} is not one of {', '.join(EVENTS)}")
        try:
            frame = int(frame_text)
            pass_no = int(pass_text)
        except ValueError:
            raise ValueError(
                f"{where}: {video_id}: frame {frame_text!r} or pass {pass_text!r} is not an integer"
            ) from None
        if frame < 0:
            raise ValueError(f"{where}: {video_id}: frame {frame} is negative")
        if pass_no not in PASSES:
            raise ValueError(f"{where}: {video_id}: pass {pass_no} is not 1 or 2")
        return PointLabel(video_id, event, frame, pass_no)

    def _write(self, labels: list[PointLabel]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                writer = csv.writer(handle)
                writer.writerow(_COLUMNS)
                for label in labels:
                    writer.writerow([label.video_id, label.event, label.frame, label.pass_no])
            os.replace(temporary, self.path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def shuffled_order(video_ids: Sequence[str], seed: int) -> list[str]:
    """Permutation of video_ids that depends only on the ids and the seed."""
    order = list(video_ids)
    random.Random(seed).shuffle(order)
    return order


def guidance_html(lines: Sequence[str]) -> str:
    """Lines joined with <br>, each escaped so that text such as a video_id is not markup."""
    return "<br>".join(html.escape(line) for line in lines)


class LabelingSession:
    """Frame navigation and recording for one video in one pass."""

    def __init__(
        self,
        video_path: Path,
        video_id: str,
        store: PointLabelStore,
        pass_no: int,
        completion: str,
    ):
        required_events(completion)
        if pass_no not in PASSES:
            raise ValueError(f"{video_id}: pass {pass_no!r} is not 1 or 2")
        self.video_path = Path(video_path)
        self.video_id = video_id
        self.store = store
        self.pass_no = pass_no
        self.completion = completion
        self.frame_count = count_frames(self.video_path)
        self.frame = 0

    def goto(self, frame: int) -> None:
        if not 0 <= frame < self.frame_count:
            raise ValueError(
                f"{self.video_id}: frame {frame} is outside 0..{self.frame_count - 1}"
            )
        self.frame = frame

    def step(self, delta: int) -> None:
        self.frame = min(max(self.frame + delta, 0), self.frame_count - 1)

    def record(self, event: str) -> None:
        self.store.add(
            self.video_id,
            event,
            self.frame,
            self.pass_no,
            completion=self.completion,
            frame_count=self.frame_count,
        )

    def visible_labels(self) -> list[PointLabel]:
        return self.store.read(self.video_id, self.pass_no)

    def view(self) -> tuple[np.ndarray, list[str]]:
        """Current frame as read (no drawing on it) and the guidance lines."""
        image = read_frame(self.video_path, self.frame)
        recorded = ", ".join(f"{label.event}={label.frame}" for label in self.visible_labels())
        missing = self.store.missing_events(self.video_id, self.completion, self.pass_no)
        lines = [
            f"{self.video_id}  pass {self.pass_no}  completion {self.completion}",
            f"frame {self.frame} / {self.frame_count}",
            f"recorded: {recorded or 'none'}",
            f"missing: {', '.join(missing) or 'none'}",
        ]
        ignored = ignored_events(self.completion)
        if ignored:
            lines.append(f"ignored: {', '.join(ignored)}")
        return image, lines

    def show(self) -> None:
        """Display slider, step buttons, event selector and record button.

        The frame is encoded in memory and handed to the widget. No file is written.
        """
        import cv2
        import ipywidgets
        from IPython.display import display

        picture = ipywidgets.Image(format="jpeg")
        text = ipywidgets.HTML()
        message = ipywidgets.HTML()
        slider = ipywidgets.IntSlider(
            value=self.frame,
            min=0,
            max=self.frame_count - 1,
            description="frame",
            continuous_update=False,
            layout=ipywidgets.Layout(width="600px"),
        )
        back = ipywidgets.Button(description="-1")
        forward = ipywidgets.Button(description="+1")
        selector = ipywidgets.Dropdown(options=list(EVENTS), description="event")
        record = ipywidgets.Button(description="record")

        def refresh() -> None:
            image, lines = self.view()
            encoded_ok, encoded = cv2.imencode(".jpg", image)
            if not encoded_ok:
                raise ValueError(f"{self.video_id}: frame {self.frame} could not be encoded")
            picture.value = encoded.tobytes()
            text.value = guidance_html(lines)

        def on_slider(change) -> None:
            self.goto(change["new"])
            refresh()

        def on_back(_button) -> None:
            self.step(-1)
            slider.value = self.frame

        def on_forward(_button) -> None:
            self.step(1)
            slider.value = self.frame

        def on_record(_button) -> None:
            try:
                self.record(selector.value)
                message.value = ""
            except ValueError as error:
                message.value = guidance_html([str(error)])
            refresh()

        slider.observe(on_slider, names="value")
        back.on_click(on_back)
        forward.on_click(on_forward)
        record.on_click(on_record)
        refresh()
        display(
            ipywidgets.VBox(
                [
                    picture,
                    slider,
                    ipywidgets.HBox([back, forward, selector, record]),
                    text,
                    message,
                ]
            )
        )
