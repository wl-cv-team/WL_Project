"""Run the plate detector on one video and write the detections as CSV.

Needs the weight file and Ultralytics at run time (weights are downloaded by
Ultralytics on first use). Example:

    uv run python scripts/run_detector.py --video path/to/video.mp4 \
        --video-id v001 --model yolo26 --out detections.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wl_snatch.config import WEIGHT_NAMES, DetectorConfig
from wl_snatch.contracts import write_detections_csv
from wl_snatch.detector import detect_frames, detect_video, load_ultralytics_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--model", required=True, choices=sorted(WEIGHT_NAMES))
    parser.add_argument("--frames", type=int, nargs="+", help="frame numbers; default: all")
    parser.add_argument("--out", type=Path, required=True, help="CSV output path")
    args = parser.parse_args()

    config = DetectorConfig(model_name=args.model)
    model = load_ultralytics_model(config)
    if args.frames:
        results = detect_frames(args.video, args.video_id, args.frames, model, config)
    else:
        results = detect_video(args.video, args.video_id, model, config)
    write_detections_csv(results, args.out)
    print(f"wrote {len(results)} frames to {args.out}")


if __name__ == "__main__":
    main()
