import dataclasses
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wl_snatch import coco_baseline
from wl_snatch.coco_baseline import (
    CocoSummary,
    check_coco_names,
    environment_record,
    format_report,
    group_frames_by_video,
    run_coco_baseline,
    summarize_coco_run,
    timed,
)
from wl_snatch.config import DetectorConfig
from wl_snatch.contracts import BoxDetection, FrameDetections, GroundTruthFrame
from wl_snatch.ground_truth import load_ground_truth

CONFIG = DetectorConfig(model_name="yolo26")
COCO_LIKE = {i: f"class{i}" for i in range(80)}


def _fresh_interpreter(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_check_coco_names_accepts_80_classes_and_rejects_plate_or_other_sizes():
    assert check_coco_names(COCO_LIKE) is None
    with pytest.raises(ValueError, match="class0"):
        check_coco_names({i: f"class{i}" for i in range(79)})
    with pytest.raises(ValueError, match="class0"):
        check_coco_names({i: f"class{i}" for i in range(81)})
    with pytest.raises(ValueError, match="class1"):
        check_coco_names({i + 1: f"class{i + 1}" for i in range(80)})
    with pytest.raises(ValueError, match="plate"):
        check_coco_names({**COCO_LIKE, 5: "Plate"})
    with pytest.raises(ValueError) as excinfo:
        check_coco_names({0: "plate"})
    assert "plate" in str(excinfo.value)
    with pytest.raises(ValueError):
        check_coco_names({})


def test_check_coco_names_message_shows_only_the_first_five_names():
    with pytest.raises(ValueError) as excinfo:
        check_coco_names({i: f"class{i}" for i in range(79)})
    message = str(excinfo.value)
    assert "class4" in message
    assert "class5" not in message


def test_load_coco_model_refuses_a_weights_path_before_importing_ultralytics():
    code = (
        "import sys\n"
        "from wl_snatch.config import DetectorConfig\n"
        "from wl_snatch.coco_baseline import load_coco_model\n"
        "try:\n"
        "    load_coco_model(DetectorConfig(model_name='yolo26', weights_path='plate.pt'))\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    sys.exit(2)\n"
        "bad = [m for m in sys.modules if m == 'ultralytics' or m.startswith('ultralytics.')]\n"
        "sys.exit(1 if bad else 0)\n"
    )
    completed = _fresh_interpreter(code)
    assert completed.returncode == 0, completed.stderr


def test_coco_module_import_does_not_import_ultralytics():
    code = (
        "import sys\n"
        "import wl_snatch.coco_baseline\n"
        "bad = [m for m in sys.modules if m == 'ultralytics' or m.startswith('ultralytics.')]\n"
        "sys.exit(1 if bad else 0)\n"
    )
    completed = _fresh_interpreter(code)
    assert completed.returncode == 0, completed.stderr


def test_environment_record_contains_versions_device_and_conditions():
    config = DetectorConfig(model_name="yolo11", imgsz=640, conf=0.25, device="mps", seed=7)
    versions = {"python": "3.12.7", "ultralytics": "8.4.202", "torch": "2.15.0", "opencv": "4.11"}
    record = environment_record(
        config,
        device="cpu",
        reason="mps requested but not available; using cpu",
        versions=versions,
        nms_iou="0.7 (Ultralytics default, not read from the library)",
    )
    assert record["python"] == "3.12.7"
    assert record["ultralytics"] == "8.4.202"
    assert record["torch"] == "2.15.0"
    assert record["opencv"] == "4.11"
    assert record["requested_device"] == "mps"
    assert record["device"] == "cpu"
    assert "not available" in record["device_reason"]
    assert record["weights"] == "yolo11n.pt"
    assert record["imgsz"] == "640"
    assert record["conf"] == "0.25"
    assert record["seed"] == "7"
    assert "not read from the library" in record["nms_iou"]
    assert record["platform"] and record["machine"]
    assert all(isinstance(v, str) for v in record.values())


def test_environment_record_without_injection_uses_strings_and_does_not_import_torch():
    code = (
        "import sys\n"
        "from wl_snatch.config import DetectorConfig\n"
        "from wl_snatch.coco_baseline import environment_record\n"
        "r = environment_record(DetectorConfig(model_name='yolo26'), 'cpu', None)\n"
        "assert all(isinstance(v, str) for v in r.values()), r\n"
        "assert 'nms_iou' not in r\n"
        "assert r['device_reason'] == 'none'\n"
        "bad = [m for m in sys.modules if m in ('torch', 'ultralytics')]\n"
        "sys.exit(1 if bad else 0)\n"
    )
    completed = _fresh_interpreter(code)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_timed_records_one_time_per_call_in_order(monkeypatch: pytest.MonkeyPatch):
    clock = iter([0.0, 0.010, 1.0, 1.0025, 2.0, 2.5])
    monkeypatch.setattr(time, "perf_counter", lambda: next(clock))
    calls = []

    def model(frame):
        calls.append(frame)
        return [(0.0, 0.0, 1.0, 1.0, 0.9, "person")]

    wrapped, times = timed(model)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    for _ in range(3):
        assert wrapped(frame) == [(0.0, 0.0, 1.0, 1.0, 0.9, "person")]
    assert len(calls) == 3
    assert times == pytest.approx([10.0, 2.5, 500.0])


def _frame(index, labels=(), na=None):
    boxes = tuple(BoxDetection(index, 0.0, 0.0, 1.0, 1.0, 0.9, label) for label in labels)
    return FrameDetections("v", index, boxes, na)


def test_summarize_counts_classes_zero_box_frames_and_na_reasons():
    frames = [
        _frame(0, ["person", "person", "sports ball"]),
        _frame(1, ["person"]),
        _frame(2),
        _frame(3, na="frame_out_of_range"),
        _frame(4, na="frame_out_of_range"),
        _frame(5, na="video_unreadable"),
    ]
    summary = summarize_coco_run(frames, [])
    assert summary.class_counts == {"person": 3, "sports ball": 1}
    assert summary.processed_frames == 3
    assert summary.zero_box_frames == 1
    assert summary.na_frames == 3
    assert summary.na_reasons == {"frame_out_of_range": 2, "video_unreadable": 1}
    assert dataclasses.is_dataclass(CocoSummary)
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.na_frames = 0


def test_summarize_time_stats_separate_the_first_call_and_handle_no_calls():
    summary = summarize_coco_run([], [50.0, 10.0, 14.0, 12.0])
    assert summary.first_call_ms == 50.0
    assert summary.median_rest_ms == 12.0
    assert summary.mean_rest_ms == 12.0
    assert summary.max_rest_ms == 14.0
    none = summarize_coco_run([], [])
    assert (none.first_call_ms, none.median_rest_ms, none.mean_rest_ms, none.max_rest_ms) == (
        None,
        None,
        None,
        None,
    )
    one = summarize_coco_run([], [33.0])
    assert one.first_call_ms == 33.0
    assert one.median_rest_ms is None and one.max_rest_ms is None


def test_report_contains_settings_and_counts_and_no_plate_performance_wording():
    record = {"python": "3.12.7", "weights": "yolo26n.pt", "nms_iou": "0.7 (assumed)"}
    summaries = {
        "train": summarize_coco_run(
            [_frame(0, ["sports ball", "person"]), _frame(1, ["person"]), _frame(2)],
            [40.0, 8.0, 9.0],
        ),
        "validation": summarize_coco_run([_frame(0, na="video_unreadable")], []),
    }
    report = format_report(record, summaries)
    assert "python: 3.12.7" in report
    assert "weights: yolo26n.pt" in report
    assert "train" in report and "validation" in report
    # person (2) is listed before sports ball (1): count descending
    assert report.index("person") < report.index("sports ball")
    assert "video_unreadable" in report
    assert "not plate-detection performance" in report
    lowered = report.lower()
    for word in ("precision", "recall", "true_positive", "false_positive", "accuracy"):
        assert word not in lowered
    assert "iou" not in lowered.replace("nms_iou", "")
    assert "plate" not in lowered.replace("not plate-detection performance", "")


def test_group_frames_by_video_sorts_and_removes_duplicates():
    truths = [
        GroundTruthFrame("b01", 4, ()),
        GroundTruthFrame("a01", 3, ()),
        GroundTruthFrame("a01", 1, ()),
        GroundTruthFrame("a01", 3, ()),
    ]
    assert group_frames_by_video(truths) == {"a01": [1, 3], "b01": [4]}
    assert group_frames_by_video([]) == {}


def test_run_coco_baseline_missing_video_path_raises_naming_it(tmp_path: Path):
    truths = {"train": [GroundTruthFrame("a01", 0, ())]}
    with pytest.raises(ValueError, match="a01"):
        run_coco_baseline(truths, {}, lambda frame: [], CONFIG)


def test_coco_baseline_writes_no_image_file(tmp_path: Path, make_video):
    videos_dir = tmp_path / "videos"
    boxes_dir = tmp_path / "boxes"
    videos_dir.mkdir()
    boxes_dir.mkdir()
    for video_id in ("a01", "b01"):
        make_video(videos_dir / f"{video_id}.mp4", size=(64, 48))
    (boxes_dir / "a01_f0.txt").write_text("0 0.500000 0.500000 0.250000 0.500000\n")
    (boxes_dir / "a01_f2.txt").write_text("")
    (boxes_dir / "a01_f9.txt").write_text("0 0.500000 0.500000 0.250000 0.500000\n")
    (boxes_dir / "b01_f1.txt").write_text("0 0.500000 0.500000 0.250000 0.500000\n")
    videos = pd.DataFrame(
        [{"video_id": v, "width": 64, "height": 48, "frame_count": 10} for v in ("a01", "b01")]
    )
    split = pd.DataFrame([{"video_id": "a01", "role": "train"}, {"video_id": "b01", "role": "validation"}])
    truths = {
        role: load_ground_truth(boxes_dir, videos, split, role)[0]
        for role in ("train", "validation")
    }
    paths = {v: videos_dir / f"{v}.mp4" for v in ("a01", "b01")}

    # Fake labeled model, keyed by call order (videos ascending, frames ascending):
    # a01 frame 0, a01 frame 2, b01 frame 1. Frame 9 of a01 is beyond the 5 decoded frames.
    script = iter(
        [
            [(1.0, 1.0, 9.0, 9.0, 0.9, "person"), (2.0, 2.0, 4.0, 4.0, 0.6, "sports ball")],
            [],
            [(1.0, 1.0, 9.0, 9.0, 0.8, "person")],
        ]
    )
    summaries = run_coco_baseline(truths, paths, lambda frame: next(script), CONFIG)

    train, validation = summaries["train"], summaries["validation"]
    assert train.class_counts == {"person": 1, "sports ball": 1}
    assert train.processed_frames == 2 and train.zero_box_frames == 1
    assert train.na_reasons == {"frame_out_of_range": 1}
    assert validation.class_counts == {"person": 1}
    assert validation.processed_frames == 1
    assert train.first_call_ms is not None
    assert validation.first_call_ms is not None
    report = format_report({"weights": "yolo26n.pt"}, summaries)
    assert "plate" not in report.lower().replace("not plate-detection performance", "")
    files = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert all(name.endswith((".mp4", ".txt")) for name in files)


def test_module_exposes_no_evaluator_dependency():
    assert "eval_detection" not in vars(coco_baseline)
    assert "evaluate_detections" not in vars(coco_baseline)
