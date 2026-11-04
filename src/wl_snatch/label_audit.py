"""Audit YOLO-format detection labels (`class cx cy w h`, normalized 0..1).

Problems found in the data are returned as LabelIssue entries; the audit
functions do not raise on bad data.

Issue kinds (closed list): missing_label, empty_label, out_of_range, bad_class,
orphan_label, duplicate_box. Every `detail` of a line-level issue starts with a
fixed tag:
- `malformed:` wrong field count or non-numeric / non-finite coordinate
  (kind out_of_range);
- `out_of_range:` value outside 0..1, w or h <= 0, or a box extending outside
  the image (kind out_of_range);
- `unreadable:` the label file could not be read or decoded (kind out_of_range);
- `bad_class:` class token is not an integer or not in the allowed set
  (kind bad_class).

duplicate_box means a line whose values repeat an earlier line of the same
file exactly (same text-parsed floats); duplicates are counted in labeled_boxes.

One validator (`check_line`) is shared by the audit, `read_boxes` and
`summarize_directory`, so box counts and statistics agree with the issue list.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
LABEL_EXTENSION = ".txt"
ISSUE_KINDS = (
    "missing_label",
    "empty_label",
    "out_of_range",
    "bad_class",
    "orphan_label",
    "duplicate_box",
)
# name -> (images dir, labels dir), relative to the data directory
PLATE_PAIRS = {
    "plate_dataset": ("plate_dataset/images", "plate_dataset/labels"),
    "plate_dataset_v2/train": ("plate_dataset_v2/images/train", "plate_dataset_v2/labels/train"),
    "plate_dataset_v2/validation": (
        "plate_dataset_v2/images/validation",
        "plate_dataset_v2/labels/validation",
    ),
}
_EPS = 1e-6

Box = tuple[int, float, float, float, float]


@dataclass(frozen=True)
class LabelIssue:
    image: str
    kind: str
    detail: str


@dataclass(frozen=True)
class AuditReport:
    images: int
    labeled_boxes: int
    issues: tuple[LabelIssue, ...]


def list_images(images_dir: Path) -> list[Path]:
    """Image files in a directory (extension compared case-insensitively), sorted by name."""
    if not images_dir.is_dir():
        return []
    return sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_files(labels_dir: Path) -> dict[str, Path]:
    """Regular files with a `.txt` extension (any case), keyed by stem. Directories are skipped."""
    if not labels_dir.is_dir():
        return {}
    return {
        p.stem: p
        for p in sorted(labels_dir.iterdir())
        if p.is_file() and p.suffix.lower() == LABEL_EXTENSION
    }


def _parse_class(token: str) -> int | None:
    try:
        value = float(token)
    except ValueError:
        return None
    if not math.isfinite(value) or value != int(value):
        return None
    return int(value)


def check_line(
    line: str, allowed_classes: frozenset[int] | None
) -> tuple[Box | None, str | None, str | None]:
    """Validate one label line. Returns (box, None, None) or (None, kind, tagged_detail).

    allowed_classes=None skips the allowed-set check (the class must still be an integer).
    """
    tokens = line.split()
    if len(tokens) != 5:
        return None, "out_of_range", f"malformed: expected 5 fields, got {len(tokens)}"
    class_id = _parse_class(tokens[0])
    if class_id is None:
        return None, "bad_class", f"bad_class: class token {tokens[0]!r} is not an integer"
    if allowed_classes is not None and class_id not in allowed_classes:
        return None, "bad_class", f"bad_class: class {class_id} not in {sorted(allowed_classes)}"
    try:
        cx, cy, w, h = (float(t) for t in tokens[1:])
    except ValueError:
        return None, "out_of_range", f"malformed: non-numeric coordinate in {tokens[1:]}"
    if not all(math.isfinite(v) for v in (cx, cy, w, h)):
        return None, "out_of_range", "malformed: non-finite coordinate"
    if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
        return None, "out_of_range", f"out_of_range: value outside 0..1 (cx={cx}, cy={cy}, w={w}, h={h})"
    if w <= 0 or h <= 0:
        return None, "out_of_range", f"out_of_range: non-positive size (w={w}, h={h})"
    if cx - w / 2 < -_EPS or cx + w / 2 > 1 + _EPS or cy - h / 2 < -_EPS or cy + h / 2 > 1 + _EPS:
        return (
            None,
            "out_of_range",
            f"out_of_range: box extends outside the image (cx={cx}, cy={cy}, w={w}, h={h})",
        )
    return (class_id, cx, cy, w, h), None, None


def audit_yolo_labels(
    images_dir: Path, labels_dir: Path, allowed_classes: frozenset[int]
) -> AuditReport:
    """Check every image has a valid label file and every label file has an image.

    Label files are matched by stem; `.txt` in any case counts. See the module
    docstring for kinds and detail tags. Lines with issues are not counted in
    labeled_boxes.
    """
    issues: list[LabelIssue] = []
    boxes = 0
    images = list_images(images_dir)
    labels = label_files(labels_dir)
    stems = {p.stem for p in images}

    for image in images:
        label = labels.get(image.stem)
        if label is None:
            issues.append(LabelIssue(image.name, "missing_label", f"no label file {image.stem}.txt"))
            continue
        try:
            text = label.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            issues.append(
                LabelIssue(image.name, "out_of_range", f"unreadable: {type(exc).__name__} reading {label.name}")
            )
            continue
        lines = [(i, ln) for i, ln in enumerate(text.splitlines(), start=1) if ln.strip()]
        if not lines:
            issues.append(LabelIssue(image.name, "empty_label", "label file has no boxes"))
            continue
        seen: dict[Box, int] = {}
        for line_no, line in lines:
            box, kind, detail = check_line(line, allowed_classes)
            if box is None:
                issues.append(LabelIssue(image.name, kind, f"{detail} (line {line_no})"))
                continue
            boxes += 1
            if box in seen:
                issues.append(
                    LabelIssue(image.name, "duplicate_box", f"line {line_no} repeats line {seen[box]}")
                )
            else:
                seen[box] = line_no

    for stem, label in labels.items():
        if stem not in stems:
            issues.append(LabelIssue(label.name, "orphan_label", "no image with this name"))

    return AuditReport(images=len(images), labeled_boxes=boxes, issues=tuple(issues))


def read_boxes(label_path: Path, allowed_classes: frozenset[int] | None = None) -> list[Box]:
    """Boxes of one label file that pass `check_line`; other lines are skipped, nothing raises."""
    try:
        text = label_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    result: list[Box] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        box, _, _ = check_line(line, allowed_classes)
        if box is not None:
            result.append(box)
    return result


_FRAME_RE = re.compile(r"^(?P<video>.+)_frame_(?P<index>\d+)\.[^.]+$")
_PERSON_RE = re.compile(r"^[A-Za-z]+\d+")


def parse_frame_name(filename: str) -> tuple[str, int] | None:
    """`c02_sn_f_01_frame_0109.jpg` -> (`c02_sn_f_01`, 109); None if the name does not match."""
    m = _FRAME_RE.match(filename)
    return (m["video"], int(m["index"])) if m else None


def person_id_of(video_id: str) -> str | None:
    """Leading letters+digits of a video id (`c02_sn_f_01` -> `c02`); None if absent."""
    m = _PERSON_RE.match(video_id)
    return m.group(0) if m else None


def describe(values: Iterable[float]) -> dict[str, float | None]:
    """min / median / max of values; all None when empty."""
    data = list(values)
    if not data:
        return {"min": None, "median": None, "max": None}
    return {"min": min(data), "median": statistics.median(data), "max": max(data)}


def filename_overlap(a: Iterable[str], b: Iterable[str]) -> list[str]:
    """Sorted names present in both collections."""
    return sorted(set(a) & set(b))


def summarize_directory(
    images_dir: Path, labels_dir: Path, allowed_classes: frozenset[int]
) -> dict:
    """Audit plus descriptive numbers for one image/label directory pair.

    Box sizes are normalized (fraction of image width/height), taken from
    the boxes that pass the same validator as the audit (so their count
    equals labeled_boxes). frames_per_video / frames_per_person count image files
    whose names match `<video_id>_frame_<index>.<ext>`; other names are listed
    under unparsed_names.
    """
    report = audit_yolo_labels(images_dir, labels_dir, allowed_classes)
    images = list_images(images_dir)
    widths: list[float] = []
    heights: list[float] = []
    per_image: list[int] = []
    labels = label_files(labels_dir)
    for image in images:
        label = labels.get(image.stem)
        valid = read_boxes(label, allowed_classes) if label is not None else []
        per_image.append(len(valid))
        widths += [b[3] for b in valid]
        heights += [b[4] for b in valid]
    per_video: dict[str, int] = {}
    per_person: dict[str, int] = {}
    unparsed: list[str] = []
    for image in images:
        parsed = parse_frame_name(image.name)
        if parsed is None:
            unparsed.append(image.name)
            continue
        per_video[parsed[0]] = per_video.get(parsed[0], 0) + 1
        person = person_id_of(parsed[0])
        if person is None:
            unparsed.append(image.name)
        else:
            per_person[person] = per_person.get(person, 0) + 1
    by_kind = {k: 0 for k in ISSUE_KINDS}
    for issue in report.issues:
        by_kind[issue.kind] = by_kind.get(issue.kind, 0) + 1
    boxes_hist: dict[str, int] = {}
    for n in per_image:
        boxes_hist[str(n)] = boxes_hist.get(str(n), 0) + 1
    return {
        "images": report.images,
        "labeled_boxes": report.labeled_boxes,
        "issues_by_kind": by_kind,
        "issues": [
            {"image": i.image, "kind": i.kind, "detail": i.detail} for i in report.issues
        ],
        "box_width_normalized": describe(widths),
        "box_height_normalized": describe(heights),
        "boxes_per_image": describe(per_image),
        "boxes_per_image_histogram": dict(sorted(boxes_hist.items(), key=lambda kv: int(kv[0]))),
        "frames_per_video": dict(sorted(per_video.items())),
        "frames_per_person": dict(sorted(per_person.items())),
        "unparsed_names": sorted(unparsed),
    }
