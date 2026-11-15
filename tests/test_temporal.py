"""Temporal features on hand-made sequences where the right answer is obvious."""

import numpy as np
import pandas as pd
import pytest

from rascar import temporal
from rascar.temporal import logit


def _world(dets, frames=("f_prev", "f_tgt", "f_next"), context=None):
    """dets: list of (det_id, frame, cls, conf, x1, y1, x2, y2)."""
    stream = pd.DataFrame(dets, columns=["detection_id", "frame_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
    fr = pd.DataFrame({"frame_id": list(frames), "t_sec": [30.0 * i for i in range(len(frames))]})
    ctx = pd.DataFrame(context or [{"frame_id": "f_tgt", "previous": "f_prev", "next": "f_next"}])
    targets = stream[stream["frame_id"] == "f_tgt"].rename(columns={"detection_id": "id"}).reset_index(drop=True)
    return targets, stream, ctx, fr


def test_static_object_is_fully_supported():
    targets, stream, ctx, fr = _world([
        ("a", "f_prev", "cup", 0.80, 100, 100, 130, 130),
        ("b", "f_tgt", "cup", 0.60, 100, 100, 130, 130),
        ("c", "f_next", "cup", 0.70, 101, 100, 131, 130),
    ])
    out = temporal.temporal_features(targets, stream, ctx, fr).iloc[0]
    assert out["t_iou_prev"] == pytest.approx(1.0)
    assert out["t_iou_next"] > 0.9
    assert out["t_persist"] == 1.0 and out["tc_n"] == 2
    assert out["t_dt_prev"] == 30.0 and out["t_dt_next"] == 30.0
    assert out["tc_mean"] == pytest.approx((0.6 + 0.8 + 0.7) / 3)
    assert out["tc_logit"] == pytest.approx(np.mean(logit(np.array([0.6, 0.8, 0.7]))))
    # neighbours are more confident than the target -> smoothed confidence is higher than the raw one
    assert out["tc_logit"] > logit(np.array([0.6]))[0]


def test_unmatched_object_is_not_penalised():
    """No re-detection around: the smoothed confidence falls back to the box's own confidence."""
    targets, stream, ctx, fr = _world([
        ("a", "f_prev", "cup", 0.90, 500, 500, 530, 530),      # far away -> not the same object
        ("b", "f_tgt", "cup", 0.60, 100, 100, 130, 130),
    ])
    out = temporal.temporal_features(targets, stream, ctx, fr).iloc[0]
    assert out["t_iou_prev"] == 0.0 and out["t_iou_next"] == 0.0
    assert out["tc_n"] == 0 and out["t_persist"] == 0.0
    assert out["tc_mean"] == pytest.approx(0.60)
    assert out["tc_logit"] == pytest.approx(logit(np.array([0.6]))[0])
    assert np.isnan(out["t_conf_prev"]) and out["tc_gap"] == pytest.approx(0.0)


def test_other_class_never_supports():
    targets, stream, ctx, fr = _world([
        ("a", "f_prev", "tray", 0.95, 100, 100, 130, 130),     # same place, different class
        ("b", "f_tgt", "cup", 0.60, 100, 100, 130, 130),
    ])
    out = temporal.temporal_features(targets, stream, ctx, fr).iloc[0]
    assert out["t_iou_prev"] == 0.0 and out["tc_n"] == 0
    assert out["t_iou_any_prev"] == pytest.approx(1.0) and out["t_flip_prev"] == 1.0


def test_weak_overlap_below_support_threshold_is_ignored():
    # IoU = 5*30 / (2*900 - 150) ~ 0.09 < SUPPORT_IOU
    targets, stream, ctx, fr = _world([
        ("a", "f_prev", "cup", 0.95, 125, 100, 155, 130),
        ("b", "f_tgt", "cup", 0.60, 100, 100, 130, 130),
    ])
    out = temporal.temporal_features(targets, stream, ctx, fr).iloc[0]
    assert 0 < out["t_iou_prev"] < temporal.SUPPORT_IOU
    assert out["tc_n"] == 0 and out["tc_mean"] == pytest.approx(0.60)


def test_missing_neighbour_frames_give_nan_not_zero():
    targets, stream, ctx, fr = _world(
        [("b", "f_tgt", "cup", 0.60, 100, 100, 130, 130), ("c", "f_next", "cup", 0.9, 100, 100, 130, 130)],
        frames=("f_tgt", "f_next"), context=[{"frame_id": "f_tgt", "previous": None, "next": "f_next"}])
    out = temporal.temporal_features(targets, stream, ctx, fr).iloc[0]
    assert out["t_has_prev"] == 0 and out["t_has_next"] == 1 and out["t_n_sides"] == 1
    assert np.isnan(out["t_iou_prev"]) and np.isnan(out["t_dt_prev"])
    assert out["t_iou_next"] == pytest.approx(1.0)
    assert out["t_persist"] == 1.0                      # judged on the one existing side only
    assert out["tc_mean"] == pytest.approx(0.75)


def test_isolated_frame_without_any_context():
    targets, stream, ctx, fr = _world([("b", "f_tgt", "person", 0.88, 10, 10, 200, 300)], frames=("f_tgt",),
                                      context=[{"frame_id": "f_tgt", "previous": None, "next": None}])
    out = temporal.temporal_features(targets, stream, ctx, fr).iloc[0]
    assert out["t_n_sides"] == 0 and np.isnan(out["t_iou_max"]) and np.isnan(out["t_persist"])
    assert out["tc_mean"] == pytest.approx(0.88) and np.isfinite(out["tc_logit"])


def test_empty_neighbour_frame_counts_as_existing_but_unmatched():
    targets, stream, ctx, fr = _world([("b", "f_tgt", "cup", 0.60, 100, 100, 130, 130)])
    out = temporal.temporal_features(targets, stream, ctx, fr).iloc[0]
    assert out["t_has_prev"] == 1 and out["t_iou_prev"] == 0.0 and out["t_n_same_prev"] == 0.0


def test_best_match_is_chosen_among_several_candidates():
    targets, stream, ctx, fr = _world([
        ("a1", "f_prev", "nude_hands", 0.50, 108, 100, 138, 130),   # IoU ~0.58
        ("a2", "f_prev", "nude_hands", 0.90, 101, 100, 131, 130),   # IoU ~0.94 -> the match
        ("b", "f_tgt", "nude_hands", 0.70, 100, 100, 130, 130),
    ])
    out = temporal.temporal_features(targets, stream, ctx, fr).iloc[0]
    assert out["t_conf_prev"] == pytest.approx(0.90) and out["t_ncand_prev"] == 2


def test_output_is_aligned_with_target_rows_in_any_order():
    dets = [("a", "f_prev", "cup", 0.8, 100, 100, 130, 130), ("b", "f_tgt", "cup", 0.6, 100, 100, 130, 130),
            ("c", "f_tgt", "person", 0.9, 300, 100, 450, 300), ("d", "f_next", "person", 0.5, 300, 100, 450, 300)]
    targets, stream, ctx, fr = _world(dets)
    fwd = temporal.temporal_features(targets, stream, ctx, fr)
    rev = temporal.temporal_features(targets.iloc[::-1].reset_index(drop=True), stream, ctx, fr)
    pd.testing.assert_frame_equal(fwd.reset_index(drop=True), rev.iloc[::-1].reset_index(drop=True))
