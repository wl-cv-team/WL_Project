"""Tile images with their ground-truth boxes into contact-sheet PNGs for visual inspection."""

import argparse
from pathlib import Path

import cv2

from wl_snatch.contact_sheet import draw_yolo_boxes, label_image, tile_images
from wl_snatch.label_audit import list_images, read_boxes

PAIRS = {
    "plate_dataset": ("plate_dataset/images", "plate_dataset/labels"),
    "plate_dataset_v2_train": ("plate_dataset_v2/images/train", "plate_dataset_v2/labels/train"),
    "plate_dataset_v2_validation": (
        "plate_dataset_v2/images/validation",
        "plate_dataset_v2/labels/validation",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/label_audit"))
    parser.add_argument("--per-sheet", type=int, default=24)
    parser.add_argument("--columns", type=int, default=6)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, (img_rel, lbl_rel) in PAIRS.items():
        images = list_images(args.data_dir / img_rel)
        tiles, unreadable = [], []
        for path in images:
            frame = cv2.imread(str(path))
            if frame is None:
                unreadable.append(path.name)
                continue
            boxes = read_boxes(args.data_dir / lbl_rel / f"{path.stem}.txt")
            tiles.append(label_image(draw_yolo_boxes(frame, boxes), path.stem[-28:]))
        sheets = 0
        for start in range(0, len(tiles), args.per_sheet):
            sheet = tile_images(tiles[start : start + args.per_sheet], args.columns)
            out = args.out_dir / f"contact_{name}_{start // args.per_sheet + 1:02d}.png"
            cv2.imwrite(str(out), sheet)
            sheets += 1
            print(f"wrote {out}")
        print(f"{name}: images={len(images)} sheets={sheets} unreadable={unreadable}")


if __name__ == "__main__":
    main()
