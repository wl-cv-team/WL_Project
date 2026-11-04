"""Import an archive zip of project data into the git-ignored data directory.

The zip is treated as untrusted: all entry names are validated before any file
is written, and entries are only read through zipfile streaming.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

RAW_VIDEO_FOLDER = "raw video"
RAW_VIDEO_DEST = ("raw", "videos")
ALLOWED_TOP_LEVEL = (
    RAW_VIDEO_FOLDER,
    "metadata",
    "annotations",
    "splits",
    "plate_dataset",
    "plate_dataset_v2",
)
JUNK_TOP_LEVEL = "__MACOSX"
JUNK_FILE_NAMES = (".DS_Store",)
NOT_IMPORTED_SUFFIXES = (".cache",)

_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_YAML_PATH_RE = re.compile(r"^path:.*$", re.MULTILINE)


@dataclass(frozen=True)
class ImportReport:
    files_copied: int
    videos_copied: int
    rewritten_paths: tuple[str, ...]
    skipped_existing: tuple[str, ...] = ()
    not_imported: tuple[str, ...] = ()


def _split_name(name: str) -> tuple[str, ...]:
    """Validate an entry name and return its path segments."""
    if "\x00" in name:
        raise ValueError(f"unsafe archive entry (NUL byte): {name!r}")
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or _DRIVE_RE.match(normalized):
        raise ValueError(f"unsafe archive entry (absolute path): {name!r}")
    parts = tuple(p for p in normalized.split("/") if p not in ("", "."))
    if ".." in parts:
        raise ValueError(f"unsafe archive entry (parent segment): {name!r}")
    return parts


def _destination_parts(parts: tuple[str, ...]) -> tuple[str, ...] | None:
    """Return the destination segments under data_dir, or None if not imported."""
    if not parts or parts[0] not in ALLOWED_TOP_LEVEL:
        return None
    if parts[-1] in JUNK_FILE_NAMES or parts[-1].endswith(NOT_IMPORTED_SUFFIXES):
        return None
    if parts[0] == RAW_VIDEO_FOLDER:
        return RAW_VIDEO_DEST + parts[1:]
    return parts


def import_archive(zip_path: Path, data_dir: Path) -> ImportReport:
    data_root = Path(data_dir).resolve()
    plan: list[tuple[zipfile.ZipInfo, Path, tuple[str, ...]]] = []
    not_imported: list[str] = []

    with zipfile.ZipFile(zip_path) as zf:
        # Validate every entry before anything is written.
        for info in zf.infolist():
            parts = _split_name(info.filename)
            if info.is_dir():
                continue
            dest_parts = _destination_parts(parts)
            if dest_parts is None:
                not_imported.append(info.filename)
                continue
            dest = data_root.joinpath(*dest_parts).resolve()
            if not dest.is_relative_to(data_root):
                raise ValueError(f"archive entry resolves outside data dir: {info.filename!r}")
            plan.append((info, dest, dest_parts))

        files_copied = 0
        videos_copied = 0
        rewritten: list[str] = []
        skipped: list[str] = []

        for info, dest, dest_parts in plan:
            rel = PurePosixPath(*dest_parts).as_posix()
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                out = open(dest, "xb")
            except FileExistsError:
                skipped.append(rel)
                continue
            with out, zf.open(info) as src:
                is_yaml = dest_parts[0].startswith("plate_dataset") and dest.name == "dataset.yaml"
                if is_yaml:
                    text = src.read().decode("utf-8")
                    new_text = _YAML_PATH_RE.sub("path: .", text)
                    if new_text != text:
                        rewritten.append(rel)
                    out.write(new_text.encode("utf-8"))
                else:
                    shutil.copyfileobj(src, out)
            files_copied += 1
            if dest_parts[:2] == RAW_VIDEO_DEST:
                videos_copied += 1

    return ImportReport(
        files_copied=files_copied,
        videos_copied=videos_copied,
        rewritten_paths=tuple(rewritten),
        skipped_existing=tuple(skipped),
        not_imported=tuple(not_imported),
    )
