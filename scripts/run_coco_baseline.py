"""Run a COCO-pretrained model as it is on the labeled frames and write a text report.

Exploration only: whether the model loads and runs, which COCO classes it reports,
and the inference time. The numbers are not plate-detection performance; no label
is compared and no evaluator is called. Needs the pretrained weights (downloaded by
Ultralytics on first use) and the local videos. Example:

    uv run python scripts/run_coco_baseline.py --model yolo26
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wl_snatch.coco_baseline import (
    environment_record,
    format_report,
    load_coco_model,
    resolve_config_device,
    run_coco_baseline,
)
from wl_snatch.config import WEIGHT_NAMES, DetectorConfig
from wl_snatch.ground_truth import ROLES, load_ground_truth

NMS_IOU_NOTE = "0.7 (Ultralytics default, not read from the library)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=sorted(WEIGHT_NAMES))
    parser.add_argument("--videos-csv", type=Path, default=Path("data/metadata/videos.csv"))
    parser.add_argument("--split", type=Path, default=Path("data/splits/person_split.csv"))
    parser.add_argument("--boxes-dir", type=Path, default=Path("data/annotations/plate_boxes"))
    parser.add_argument(
        "--ranges", type=Path, default=Path("data/annotations/plate_frame_ranges.csv")
    )
    parser.add_argument("--videos-dir", type=Path, default=Path("data/raw/videos"))
    parser.add_argument("--out", type=Path, help="default: artifacts/coco_baseline/<model>.txt")
    args = parser.parse_args()
    out = args.out or Path("artifacts/coco_baseline") / f"{args.model}.txt"

    config = DetectorConfig(model_name=args.model)
    videos = pd.read_csv(args.videos_csv)
    split = pd.read_csv(args.split)
    truths = {
        role: load_ground_truth(args.boxes_dir, videos, split, role, args.ranges)[0]
        for role in ROLES
    }
    video_ids = {t.video_id for frames in truths.values() for t in frames}
    paths = {v: args.videos_dir / f"{v}.mp4" for v in video_ids}

    model = load_coco_model(config)
    run = run_coco_baseline(truths, paths, model, config)

    device, reason = resolve_config_device(config)
    record = environment_record(config, device, reason, nms_iou=NMS_IOU_NOTE)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(format_report(record, run.summaries, run.warmup_ms), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
