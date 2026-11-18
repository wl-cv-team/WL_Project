"""Export labeled frames as a YOLO training dataset, and verify an export.

YOLO training reads image files, so this module decodes the labeled frames from the
videos and writes JPEG images and label text files into `out_dir`. It is the only
place that creates training images (see data/README.md, folder `data/derived/`);
`out_dir` must be a git-ignored folder, because the images show people's faces.

Layout under `out_dir`:
    images/<folder>/<video_id>_f<frame>.jpg
    labels/<folder>/<video_id>_f<frame>.txt      one line `0 cx cy w h` (normalized, 6 decimals)
    dataset.yaml                                 relative paths, one class `plate`
    manifest.json                                counts and SHA256 of every image and label file

`dataset.yaml` has no `path` key, so Ultralytics uses the folder of the yaml file as the
dataset root. JPEG bytes can differ across OpenCV builds, so the hashes compare runs on
the same environment.

The role `train` goes to folder `train` and yaml key `train`; the role `validation`
goes to folder `validation` and yaml key `val` (`ROLE_FOLDERS`, `ROLE_YAML_KEYS`).
Ultralytics and torch are never imported here.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd

from wl_snatch.contracts import GroundTruthFrame
from wl_snatch.frames import read_frames

ROLES = ("train", "validation")
ROLE_FOLDERS = {"train": "train", "validation": "validation"}
ROLE_YAML_KEYS = {"train": "train", "validation": "val"}
_PROBES = ("images/x.jpg", "labels/x.txt", "dataset.yaml", "manifest.json")
_EPS = 1e-6  # label values are written with 6 decimals
_PIXEL_TOLERANCE = 0.5  # boxes may exceed the image by at most this many pixels (rounding)


def assert_ignored_output(out_dir: Path, repo_root: Path) -> None:
    """ValueError when `out_dir` is inside the repository and not ignored by git.

    Four probe paths under `out_dir` (an image, a label, `dataset.yaml`,
    `manifest.json`) must all be ignored; the error names the first one that is not.
    A path outside the repository is allowed.
    """
    resolved = Path(out_dir).resolve()
    root = Path(repo_root).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return
    for probe in _PROBES:
        path = (relative / probe).as_posix()
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", path], cwd=root, capture_output=True, text=True
        )
        if result.returncode == 1:
            raise ValueError(
                f"output path {path} is inside the repository and not ignored by git; "
                f"training images show faces and must not be tracked"
            )
        if result.returncode != 0:
            raise RuntimeError(f"git check-ignore failed: {result.stderr.strip()}")


@dataclass(frozen=True)
class DatasetReport:
    images_by_role: dict[str, int]
    with_plate_by_role: dict[str, int]
    without_plate_by_role: dict[str, int]
    overall_sha256: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _overall(files: Mapping[str, str]) -> str:
    """SHA256 over the lines `<relative path>:<file sha256>` sorted by path (names included)."""
    return _sha256("".join(f"{name}:{files[name]}\n" for name in sorted(files)).encode("utf-8"))


def _label_text(boxes: Sequence[tuple[float, float, float, float]], width: int, height: int,
                name: str) -> str:
    lines = []
    for x1, y1, x2, y2 in boxes:
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            raise ValueError(f"{name}: box {(x1, y1, x2, y2)} has a NaN or infinite value")
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"{name}: box {(x1, y1, x2, y2)} has zero or negative area")
        if (
            x1 < -_PIXEL_TOLERANCE
            or y1 < -_PIXEL_TOLERANCE
            or x2 > width + _PIXEL_TOLERANCE
            or y2 > height + _PIXEL_TOLERANCE
        ):
            raise ValueError(f"{name}: box {(x1, y1, x2, y2)} is outside the {width}x{height} frame")
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return "".join(lines)


def _yaml_text() -> str:
    return (
        # No `path` key: Ultralytics then uses the folder of this yaml file. A relative
        # `path` would be resolved against the working directory of the training run.
        f"train: images/{ROLE_FOLDERS['train']}\n"
        f"val: images/{ROLE_FOLDERS['validation']}\n"
        "names:\n"
        "  0: plate\n"
    )


def _check_inputs(
    frames_by_role: Mapping[str, Sequence[GroundTruthFrame]], videos: pd.DataFrame
) -> dict[str, pd.Series]:
    unknown = [role for role in frames_by_role if role not in ROLES]
    if unknown:
        raise ValueError(f"unknown roles {unknown}; use only {list(ROLES)}")
    for column in ("video_id", "person_id", "width", "height"):
        if column not in videos.columns:
            raise ValueError(f"video table is missing the column {column!r}")
    rows = {str(r["video_id"]): r for _, r in videos.iterrows()}
    seen: dict[tuple[str, int], str] = {}
    persons: dict[str, set[str]] = {}
    for role, frames in frames_by_role.items():
        for frame in frames:
            key = (frame.video_id, frame.frame_index)
            if key in seen:
                raise ValueError(
                    f"frame {frame.frame_index} of {frame.video_id} is in roles "
                    f"{seen[key]} and {role}"
                )
            seen[key] = role
            if frame.video_id not in rows:
                raise ValueError(f"{frame.video_id}: video is not in the video table")
            if frame.frame_index < 0:
                raise ValueError(f"{frame.video_id}: frame {frame.frame_index} is negative")
            persons.setdefault(str(rows[frame.video_id]["person_id"]), set()).add(role)
    for person, roles in persons.items():
        if len(roles) > 1:
            raise ValueError(f"person {person} appears in more than one role: {sorted(roles)}")
    return rows


def export_yolo_dataset(
    frames_by_role: Mapping[str, Sequence[GroundTruthFrame]],
    videos: pd.DataFrame,
    videos_root: Path,
    out_dir: Path,
    repo_root: Path,
    jpeg_quality: int = 95,
) -> Path:
    """Write the dataset under `out_dir` and return the path of `dataset.yaml`.

    Frames are decoded once per video from `<videos_root>/<video_id>.mp4`. Boxes are
    normalized with the decoded frame size, which must equal `width` and `height` in
    `videos`. Raises ValueError for a non-empty `out_dir`, unknown roles, a frame in
    two roles, a person in two roles, an unreadable frame or a size mismatch. On any
    failure everything this function created under `out_dir` is removed.
    """
    out_dir = Path(out_dir)
    assert_ignored_output(out_dir, repo_root)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"output folder {out_dir} is not empty")
    rows = _check_inputs(frames_by_role, videos)

    created_root = None if out_dir.exists() else _topmost_missing(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        return _write_dataset(frames_by_role, rows, Path(videos_root), out_dir, jpeg_quality)
    except BaseException:
        if created_root is not None:
            shutil.rmtree(created_root, ignore_errors=True)
        elif out_dir.is_dir():
            for child in out_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        raise


def _topmost_missing(path: Path) -> Path:
    top = path
    while not top.parent.exists():
        top = top.parent
    return top


def _write_frame(
    video_id: str,
    index: int,
    frame,
    truth: GroundTruthFrame,
    role: str,
    row: pd.Series,
    out_dir: Path,
    jpeg_quality: int,
) -> dict[str, str]:
    """Write the image and label of one frame; returns {relative path: SHA256}."""
    height, width = frame.shape[:2]
    if (width, height) != (int(row["width"]), int(row["height"])):
        raise ValueError(
            f"{video_id}: decoded frame is {width}x{height} but the video table "
            f"says {int(row['width'])}x{int(row['height'])}"
        )
    name = f"{video_id}_f{index}"
    label = _label_text(truth.boxes, width, height, name)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise ValueError(f"{name}: frame could not be encoded")
    written: dict[str, str] = {}
    for kind, suffix, data in (
        ("images", ".jpg", encoded.tobytes()),
        ("labels", ".txt", label.encode("ascii")),
    ):
        relative = f"{kind}/{ROLE_FOLDERS[role]}/{name}{suffix}"
        target = out_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written[relative] = _sha256(data)
    return written


def _write_dataset(
    frames_by_role: Mapping[str, Sequence[GroundTruthFrame]],
    rows: Mapping[str, pd.Series],
    videos_root: Path,
    out_dir: Path,
    jpeg_quality: int,
) -> Path:
    wanted: dict[str, dict[int, tuple[str, GroundTruthFrame]]] = {}
    for role, frames in frames_by_role.items():
        for frame in frames:
            wanted.setdefault(frame.video_id, {})[frame.frame_index] = (role, frame)

    files: dict[str, str] = {}
    stats = {role: {"images": 0, "with_plate": 0, "without_plate": 0, "ids": set()} for role in ROLES}
    for video_id in sorted(wanted):
        targets = wanted[video_id]
        row = rows[video_id]
        done: set[int] = set()
        reader = read_frames(videos_root / f"{video_id}.mp4", sorted(targets))
        while True:
            try:
                index, frame = next(reader)
            except StopIteration:
                break
            except ValueError as error:
                # raised after every reachable frame was yielded (or when the video cannot be read)
                missing = sorted(set(targets) - done)
                raise ValueError(
                    f"{video_id}: frame {missing[0]} could not be read ({error})"
                ) from error
            role, truth = targets[index]
            files.update(_write_frame(video_id, index, frame, truth, role, row, out_dir, jpeg_quality))
            stat = stats[role]
            stat["images"] += 1
            stat["with_plate" if truth.boxes else "without_plate"] += 1
            stat["ids"].add(video_id)
            done.add(index)
        missing = sorted(set(targets) - done)
        if missing:
            raise ValueError(f"{video_id}: frame {missing[0]} could not be read")

    for role in ROLES:  # make the folders exist even for an empty role
        for kind in ("images", "labels"):
            (out_dir / kind / ROLE_FOLDERS[role]).mkdir(parents=True, exist_ok=True)
    manifest = {
        "jpeg_quality": int(jpeg_quality),
        "roles": {
            role: {
                "images": stats[role]["images"],
                "with_plate": stats[role]["with_plate"],
                "without_plate": stats[role]["without_plate"],
                "video_ids": sorted(stats[role]["ids"]),
            }
            for role in ROLES
        },
        "files": dict(sorted(files.items())),
        "overall_sha256": _overall(files),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(_yaml_text(), encoding="utf-8")
    return yaml_path


def _parse_yaml(path: Path) -> dict[str, str]:
    """Read the fixed `key: value` lines of the exported dataset.yaml."""
    values: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line and not line.startswith(" ") and ":" in line:
            key, _, value = line.partition(":")
            values[key.strip()] = value.strip()
    if "names:" not in lines or lines[lines.index("names:") + 1].strip() != "0: plate":
        raise ValueError(f"{path}: names must be exactly one class, 0: plate")
    if "path" in values:
        raise ValueError(f"{path}: a `path` key is not allowed (the yaml folder is the root)")
    for key in ("train", "val"):
        if key not in values:
            raise ValueError(f"{path}: missing key {key}")
    return values


def _check_label_file(path: Path) -> int:
    """Number of boxes in the label file; ValueError (naming the file) for a bad line."""
    count = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"{path.name} line {number}: expected `0 cx cy w h`, got {line!r}")
        try:
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            raise ValueError(f"{path.name} line {number}: values are not numbers") from None
        if any(not 0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            raise ValueError(f"{path.name} line {number}: a value is outside 0..1")
        if cx - w / 2 < -_EPS or cx + w / 2 > 1 + _EPS or cy - h / 2 < -_EPS or cy + h / 2 > 1 + _EPS:
            raise ValueError(f"{path.name} line {number}: the box is outside the image")
        count += 1
    return count


def verify_exported_dataset(dataset_yaml: Path) -> DatasetReport:
    """Read-only check of an export; ValueError naming the file when something is off.

    Checks: every image has a label of the same name and the reverse; label lines are
    `0 cx cy w h` with values in 0..1 and the box inside the image; no file name is in
    both roles; every file's SHA256 and the overall hash equal `manifest.json`.
    """
    dataset_yaml = Path(dataset_yaml)
    values = _parse_yaml(dataset_yaml)
    root = dataset_yaml.parent.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{manifest_path}: manifest.json not found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed_folders = set(ROLE_FOLDERS.values())
    for kind in ("images", "labels"):
        base = root / kind
        if not base.is_dir():
            raise ValueError(f"{kind}: folder {base} is missing")
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                raise ValueError(f"{child.name}: unexpected file in {base}")
            if child.name not in allowed_folders:
                raise ValueError(f"{kind}/{child.name}: unexpected folder")

    images_by_role: dict[str, dict[str, Path]] = {}
    plates: dict[str, int] = {}
    empties: dict[str, int] = {}
    actual: dict[str, Path] = {}
    for role in ROLES:
        image_dir = root / values[ROLE_YAML_KEYS[role]]
        label_dir = root / "labels" / ROLE_FOLDERS[role]
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise ValueError(f"{role}: folder {image_dir} or {label_dir} is missing")
        for folder in (image_dir, label_dir):
            for child in folder.iterdir():
                if child.is_dir():
                    raise ValueError(f"{child.name}: unexpected folder in {folder}")
        images = {p.stem: p for p in sorted(image_dir.iterdir())}
        labels = {p.stem: p for p in sorted(label_dir.iterdir())}
        for path in list(images.values()):
            if path.suffix != ".jpg":
                raise ValueError(f"{path.name}: unexpected file in {image_dir}")
        for path in labels.values():
            if path.suffix != ".txt":
                raise ValueError(f"{path.name}: unexpected file in {label_dir}")
        for stem in sorted(set(images) - set(labels)):
            raise ValueError(f"{images[stem].name}: image has no label file")
        for stem in sorted(set(labels) - set(images)):
            raise ValueError(f"{labels[stem].name}: label has no image file")
        images_by_role[role] = images
        plates[role] = empties[role] = 0
        for stem, label_path in labels.items():
            boxes = _check_label_file(label_path)
            plates[role] += 1 if boxes else 0
            empties[role] += 0 if boxes else 1
            actual[label_path.relative_to(root).as_posix()] = label_path
            actual[images[stem].relative_to(root).as_posix()] = images[stem]
    shared = set(images_by_role["train"]) & set(images_by_role["validation"])
    if shared:
        raise ValueError(f"{sorted(shared)[0]}: file name is in both train and val")

    for role in ROLES:
        recorded_role = manifest["roles"][role]
        found = {
            "images": len(images_by_role[role]),
            "with_plate": plates[role],
            "without_plate": empties[role],
        }
        for key, count in found.items():
            if recorded_role[key] != count:
                raise ValueError(
                    f"manifest.json: {role} {key} is {recorded_role[key]} "
                    f"but the files give {count}"
                )

    recorded = manifest["files"]
    for name in sorted(set(actual) - set(recorded)):
        raise ValueError(f"{Path(name).name}: file is not listed in manifest.json")
    for name in sorted(set(recorded) - set(actual)):
        raise ValueError(f"{Path(name).name}: file listed in manifest.json is missing")
    for name in sorted(actual):
        if _sha256(actual[name].read_bytes()) != recorded[name]:
            raise ValueError(f"{Path(name).name}: SHA256 differs from manifest.json")
    overall = _overall(recorded)
    if overall != manifest["overall_sha256"]:
        raise ValueError("manifest.json: overall SHA256 does not match its file hashes")
    return DatasetReport(
        images_by_role={role: len(images_by_role[role]) for role in ROLES},
        with_plate_by_role=plates,
        without_plate_by_role=empties,
        overall_sha256=overall,
    )
