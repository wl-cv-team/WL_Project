"""Command line entry point for wl_snatch.data_import.import_archive."""

import argparse
from pathlib import Path

from wl_snatch.data_import import import_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Import an archive zip into the data directory.")
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    report = import_archive(args.zip_path, args.data_dir)
    print(f"files_copied={report.files_copied}")
    print(f"videos_copied={report.videos_copied}")
    print(f"skipped_existing={len(report.skipped_existing)}")
    print(f"rewritten_paths={list(report.rewritten_paths)}")
    print(f"not_imported={len(report.not_imported)}")
    for name in report.not_imported:
        print(f"  not imported: {name}")


if __name__ == "__main__":
    main()
