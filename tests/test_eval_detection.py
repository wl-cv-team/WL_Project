import pytest

from wl_snatch.contracts import BoxDetection, FrameDetections
from wl_snatch.eval_detection import (
    DetectionMetrics,
    GroundTruthFrame,
    evaluate_detections,
    iou,
)

BOX = (10.0, 10.0, 50.0, 50.0)


def _pred(frame_index, boxes=(), na_reason=None, video_id="v"):
    detections = tuple(
        BoxDetection(frame_index, x1, y1, x2, y2, conf, "plate")
        for (x1, y1, x2, y2, conf) in boxes
    )
    return FrameDetections(video_id, frame_index, detections, na_reason)


def _truth(frame_index, boxes=(), video_id="v"):
    return GroundTruthFrame(video_id, frame_index, tuple(boxes))


def test_perfect_prediction_gives_precision_and_recall_one():
    m = evaluate_detections(
        [_pred(0, [(*BOX, 0.9)])], [_truth(0, [BOX])], iou_threshold=0.5
    )
    assert isinstance(m, DetectionMetrics)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.false_positives_per_frame == 0.0
    assert (m.frames_evaluated, m.frames_na, m.na_reasons) == (1, 0, {})
    assert (m.true_positives, m.false_positives, m.false_negatives) == (1, 0, 0)


def test_box_below_iou_threshold_counts_as_false_positive_and_miss():
    far = (40.0, 10.0, 80.0, 50.0)  # IoU with BOX = 400 / 2800
    m = evaluate_detections(
        [_pred(0, [(*far, 0.9)])], [_truth(0, [BOX])], iou_threshold=0.5
    )
    assert (m.true_positives, m.false_positives, m.false_negatives) == (0, 1, 1)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.false_positives_per_frame == 1.0


def test_duplicate_prediction_on_same_truth_counts_one_false_positive():
    m = evaluate_detections(
        [_pred(0, [(*BOX, 0.9), (*BOX, 0.8)])], [_truth(0, [BOX])], 0.5
    )
    assert (m.true_positives, m.false_positives, m.false_negatives) == (1, 1, 0)
    assert m.precision == 0.5
    assert m.recall == 1.0


def test_box_on_frame_without_plate_is_false_positive():
    m = evaluate_detections([_pred(0, [(*BOX, 0.9)])], [_truth(0)], 0.5)
    assert (m.true_positives, m.false_positives, m.false_negatives) == (0, 1, 0)
    assert m.precision == 0.0
    assert m.recall is None
    assert m.false_positives_per_frame == 1.0


def test_na_frames_are_excluded_and_counted_by_reason():
    preds = [
        _pred(0, [(*BOX, 0.9)]),
        _pred(1, na_reason="unreadable"),
        _pred(2, na_reason="unreadable"),
        _pred(3, na_reason="other"),
    ]
    truths = [_truth(i, [BOX]) for i in range(4)]
    m = evaluate_detections(preds, truths, 0.5)
    assert m.frames_evaluated == 1
    assert m.frames_na == 3
    assert m.na_reasons == {"unreadable": 2, "other": 1}
    assert m.precision == 1.0
    assert m.recall == 1.0


def test_no_truth_boxes_anywhere_gives_recall_none():
    m = evaluate_detections([_pred(0), _pred(1)], [_truth(0), _truth(1)], 0.5)
    assert m.recall is None
    assert m.precision is None
    assert m.false_positives_per_frame == 0.0
    assert m.frames_evaluated == 2


def test_no_predictions_gives_precision_none():
    m = evaluate_detections([_pred(0)], [_truth(0, [BOX])], 0.5)
    assert m.precision is None
    assert m.recall == 0.0
    assert (m.true_positives, m.false_positives, m.false_negatives) == (0, 0, 1)


def test_predictions_for_unlabeled_frames_are_ignored():
    m = evaluate_detections(
        [_pred(0, [(*BOX, 0.9)]), _pred(5, [(*BOX, 0.9)])], [_truth(0, [BOX])], 0.5
    )
    assert m.frames_evaluated == 1
    assert (m.true_positives, m.false_positives, m.false_negatives) == (1, 0, 0)


def test_truth_frame_without_prediction_is_na_prediction_missing():
    m = evaluate_detections([_pred(0, [(*BOX, 0.9)])], [_truth(0, [BOX]), _truth(1, [BOX])], 0.5)
    assert m.frames_evaluated == 1
    assert m.frames_na == 1
    assert m.na_reasons == {"prediction_missing": 1}
    assert m.false_negatives == 0


def test_frames_are_keyed_by_video_id_and_frame_index():
    preds = [_pred(0, [(*BOX, 0.9)], video_id="a"), _pred(0, video_id="b")]
    truths = [_truth(0, [BOX], video_id="a"), _truth(0, [BOX], video_id="b")]
    m = evaluate_detections(preds, truths, 0.5)
    assert (m.true_positives, m.false_positives, m.false_negatives) == (1, 0, 1)


def test_duplicate_prediction_key_raises_naming_key():
    with pytest.raises(ValueError, match=r"\('v', 0\)"):
        evaluate_detections([_pred(0), _pred(0)], [_truth(0)], 0.5)


def test_duplicate_truth_key_raises_naming_key():
    with pytest.raises(ValueError, match=r"\('v', 0\)"):
        evaluate_detections([_pred(0)], [_truth(0), _truth(0)], 0.5)


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5, float("nan")])
def test_iou_threshold_out_of_range_raises(threshold):
    with pytest.raises(ValueError):
        evaluate_detections([], [], threshold)


def test_iou_threshold_one_is_allowed():
    m = evaluate_detections([_pred(0, [(*BOX, 0.9)])], [_truth(0, [BOX])], 1.0)
    assert m.true_positives == 1


def test_higher_confidence_prediction_matches_first():
    near = (11.0, 10.0, 51.0, 50.0)
    m = evaluate_detections(
        [_pred(0, [(*near, 0.3), (*BOX, 0.9)])], [_truth(0, [BOX])], 0.5
    )
    assert (m.true_positives, m.false_positives) == (1, 1)


def test_prediction_takes_best_iou_unmatched_truth():
    truths = [(0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 11.0)]
    m = evaluate_detections(
        [_pred(0, [(0.0, 0.0, 10.0, 10.0, 0.9), (0.0, 0.0, 10.0, 10.0, 0.8)])],
        [_truth(0, truths)],
        0.5,
    )
    assert (m.true_positives, m.false_positives, m.false_negatives) == (2, 0, 0)


def test_all_predicted_labels_count():
    other = FrameDetections(
        "v", 0, (BoxDetection(0, *BOX, 0.9, "person"),), None
    )
    m = evaluate_detections([other], [_truth(0, [BOX])], 0.5)
    assert m.true_positives == 1


def test_iou_values():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_iou_zero_area_and_inverted_boxes_give_zero():
    assert iou((5, 5, 5, 5), (5, 5, 5, 5)) == 0.0
    assert iou((10, 10, 0, 0), (0, 0, 10, 10)) == 0.0
    assert iou((0, 0, 10, 10), (0, 0, 0, 10)) == 0.0


def test_matching_order_follows_descending_confidence():
    t1 = (0.0, 0.0, 100.0, 100.0)
    t2 = (40.0, 0.0, 140.0, 100.0)
    # IoU(t1, t2) = 6000 / 14000 = 0.43, so P_high (= t1) overlaps only T1 at 0.5.
    p_low = (10.0, 0.0, 110.0, 100.0)
    # IoU(p_low, t1) = 9000 / 11000 = 0.818; IoU(p_low, t2) = 7000 / 13000 = 0.538.
    # Descending: P_high takes T1, P_low takes T2 -> TP 2.
    # Ascending: P_low takes T1 first, P_high then has only T2 (0.43) -> TP 1, FP 1, FN 1.
    m = evaluate_detections(
        [_pred(0, [(*p_low, 0.3), (*t1, 0.9)])], [_truth(0, [t1, t2])], 0.5
    )
    assert (m.true_positives, m.false_positives, m.false_negatives) == (2, 0, 0)
    assert m.precision == 1.0
    assert m.recall == 1.0


def test_bool_iou_threshold_raises():
    with pytest.raises(ValueError):
        evaluate_detections([], [], True)


def test_threshold_one_matches_identical_box_but_not_near_identical():
    box = (0.0, 0.0, 100.0, 100.0)
    near = (0.0, 0.0, 100.0, 99.9)  # IoU = 0.999
    same = evaluate_detections([_pred(0, [(*box, 0.9)])], [_truth(0, [box])], 1.0)
    assert (same.true_positives, same.false_positives, same.false_negatives) == (1, 0, 0)
    off = evaluate_detections([_pred(0, [(*near, 0.9)])], [_truth(0, [box])], 1.0)
    assert (off.true_positives, off.false_positives, off.false_negatives) == (0, 1, 1)
