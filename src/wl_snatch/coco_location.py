"""Class-agnostic COCO proposals versus plate locations; not plate classification."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from wl_snatch.coco_baseline import (
    environment_record, group_frames_by_video, load_coco_model, resolve_config_device,
)
from wl_snatch.config import DetectorConfig
from wl_snatch.contracts import write_detections_csv
from wl_snatch.detector import detect_frames_labeled
from wl_snatch.eval_detection import evaluate_detections
from wl_snatch.ground_truth import load_ground_truth
from wl_snatch.split import assert_no_person_leak


def measure_location(truths, videos_dir, model, config, iou_threshold):
    """Keep all COCO classes and count every unmatched proposal as FP."""
    predictions = []
    for video_id, indices in sorted(group_frames_by_video(truths).items()):
        predictions.extend(detect_frames_labeled(
            videos_dir / f"{video_id}.mp4", video_id, indices, model, config,
        ))
    return evaluate_detections(predictions, list(truths), iou_threshold), predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iou", type=float, default=.5)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/coco_location"))
    args = parser.parse_args()
    if not 0 < args.iou <= 1:
        parser.error("--iou must be in (0, 1]")
    config = DetectorConfig(model_name="yolo26", imgsz=640, conf=.25, seed=0)
    videos = pd.read_csv("data/metadata/videos.csv")
    split = pd.read_csv("data/splits/person_split.csv")
    assert_no_person_leak(split)
    truths, _ = load_ground_truth(
        Path("data/annotations/plate_boxes"), videos, split, "validation",
        Path("data/annotations/plate_frame_ranges.csv"),
    )
    if not truths:
        parser.error("no validation labels found")
    model = load_coco_model(config)
    metrics, predictions = measure_location(
        truths, Path("data/raw/videos"), model, config, args.iou,
    )
    device, reason = resolve_config_device(config)
    report = {
        "experiment": "YOLO26 COCO class-agnostic plate-location baseline",
        "note": "Not plate-class recognition. All COCO proposals count; unmatched persons and background boxes are FP. Boundary-selected frames only.",
        "role": "validation",
        "target_frames": len(truths),
        "videos": len({t.video_id for t in truths}),
        "people": int(split.loc[split.video_id.isin({t.video_id for t in truths}), "person_id"].nunique()),
        "ground_truth_boxes": sum(len(t.boxes) for t in truths),
        "iou_threshold": args.iou,
        "environment": environment_record(config, device, reason, nms_iou="Ultralytics default; not explicitly set"),
        "metrics": asdict(metrics),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_detections_csv(predictions, args.out_dir / "predictions.csv")
    text = json.dumps(report, ensure_ascii=False, indent=2)
    (args.out_dir / "report.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
