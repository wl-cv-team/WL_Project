"""Run a trained plate detector on one video and write the detections as CSV.

Needs a plate-detector weight file (`--weights`) and Ultralytics at run time. The
pretrained names (yolo26, ...) are COCO-pretrained starting points for training and
do not detect plates; the loaded model must have exactly one class named "plate".
`--model` is the architecture name and is only recorded in the config. Example:

    uv run python scripts/run_detector.py --video path/to/video.mp4 \
        --video-id v001 --model yolo26 --weights path/to/plate.pt --out detections.csv

Exits with status 1 when the video cannot be read (the CSV then holds one N/A row).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wl_snatch.config import WEIGHT_NAMES, DetectorConfig
from wl_snatch.contracts import write_detections_csv
from wl_snatch.detector import detect_frames, detect_video, load_ultralytics_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--model", required=True, choices=sorted(WEIGHT_NAMES))
    parser.add_argument("--weights", type=Path, required=True, help="trained plate weights (.pt)")
    parser.add_argument("--frames", type=int, nargs="+", help="frame numbers; default: all")
    parser.add_argument("--out", type=Path, required=True, help="CSV output path")
    args = parser.parse_args()

    config = DetectorConfig(model_name=args.model, weights_path=args.weights)
    model = load_ultralytics_model(config)
    if args.frames:
        results = detect_frames(args.video, args.video_id, args.frames, model, config)
    else:
        results = detect_video(args.video, args.video_id, model, config)
    write_detections_csv(results, args.out)
    print(f"wrote {len(results)} frames to {args.out}")
    if results and all(r.na_reason == "video_unreadable" for r in results):
        print(f"video_unreadable: {args.video} cannot be read", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
