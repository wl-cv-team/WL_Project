"""Tile images with their ground-truth boxes into contact-sheet PNGs for visual inspection."""

import argparse
from pathlib import Path

import cv2

from wl_snatch.contact_sheet import render_cell, tile_images
from wl_snatch.label_audit import PLATE_PAIRS, label_files, list_images, read_boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/label_audit"))
    parser.add_argument("--per-sheet", type=int, default=24)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--cell-width", type=int, default=270)
    parser.add_argument("--cell-height", type=int, default=480)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cell = (args.cell_width, args.cell_height)

    for pair_name, (img_rel, lbl_rel) in PLATE_PAIRS.items():
        name = pair_name.replace("/", "_")
        images = list_images(args.data_dir / img_rel)
        labels = label_files(args.data_dir / lbl_rel)
        sheets, unreadable = 0, []
        for start in range(0, len(images), args.per_sheet):
            tiles = []
            for path in images[start : start + args.per_sheet]:
                frame = cv2.imread(str(path))
                if frame is None:
                    unreadable.append(path.name)
                    continue
                label = labels.get(path.stem)
                boxes = read_boxes(label) if label is not None else []
                tiles.append(render_cell(frame, boxes, cell, path.stem))
                del frame
            if not tiles:
                continue
            out = args.out_dir / f"contact_{name}_{start // args.per_sheet + 1:02d}.png"
            cv2.imwrite(str(out), tile_images(tiles, args.columns, cell))
            sheets += 1
            print(f"wrote {out}")
        print(f"{name}: images={len(images)} sheets={sheets} unreadable={unreadable}")


if __name__ == "__main__":
    main()
