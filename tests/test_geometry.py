import numpy as np
import pytest

from rascar import geometry as G


def test_iou_known_values():
    a = [[0, 0, 10, 10]]
    assert G.iou_matrix(a, [[0, 0, 10, 10]])[0, 0] == pytest.approx(1.0)
    assert G.iou_matrix(a, [[10, 10, 20, 20]])[0, 0] == 0.0            # touching corners do not overlap
    assert G.iou_matrix(a, [[5, 0, 15, 10]])[0, 0] == pytest.approx(50 / 150)
    assert G.iou_matrix(a, [[2, 2, 8, 8]])[0, 0] == pytest.approx(36 / 100)


def test_iou_shapes_and_empty_inputs():
    assert G.iou_matrix(np.zeros((0, 4)), [[0, 0, 1, 1]]).shape == (0, 1)
    assert G.iou_matrix([[0, 0, 1, 1]], np.zeros((0, 4))).shape == (1, 0)
    assert G.iou_matrix(np.zeros((3, 4)), np.ones((2, 4))).shape == (3, 2)


def test_zero_width_box_has_zero_iou_and_no_nan():
    iou = G.iou_matrix([[5, 5, 5, 20]], [[0, 0, 10, 30], [5, 5, 5, 20]])
    assert np.isfinite(iou).all() and (iou == 0).all()
    assert G.box_area([[5, 5, 5, 20], [3, 3, 1, 9]]).tolist() == [0.0, 0.0]   # inverted boxes are clipped to 0


def test_coverage_is_asymmetric():
    small, big = [[2, 2, 4, 4]], [[0, 0, 10, 10]]
    assert G.coverage_matrix(small, big)[0, 0] == pytest.approx(1.0)
    assert G.coverage_matrix(big, small)[0, 0] == pytest.approx(0.04)


def test_cxcywh_conversion_matches_frame_size():
    box = G.cxcywh_norm_to_xyxy(0.5, 0.5, 0.25, 0.5, 1280, 720)
    assert box.tolist() == [480.0, 180.0, 800.0, 540.0]


def test_matching_threshold_is_inclusive_and_strict_below():
    pred = [[0, 0, 10, 10]]
    exact = [[0, 0, 10, 6]]          # IoU = 0.60 exactly -> counts ("не ниже 0,60")
    below = [[0, 0, 10, 5.9]]        # IoU = 0.59 -> error
    assert G.match_one_to_one(pred, exact, 0.6)[0].tolist() == [0]
    assert G.match_one_to_one(pred, below, 0.6)[0].tolist() == [-1]


def test_matching_is_one_to_one_duplicates_become_errors():
    gt = [[0, 0, 10, 10]]
    pred = [[0, 0, 10, 10], [0, 0, 10, 9]]      # two predictions on one manual box
    gt_index, matched, best = G.match_one_to_one(pred, gt, 0.6)
    assert sorted(gt_index.tolist()) == [-1, 0]
    assert gt_index[0] == 0                       # the better IoU keeps the match
    assert best[1] == pytest.approx(0.9)          # the duplicate still overlaps strongly


def test_matching_maximises_pair_count_before_total_iou():
    # p0 overlaps g0 (0.905) and g1 (0.667); p1 overlaps only g0 (0.639).
    # Greedy-by-IoU would pair p0-g0 and leave p1 unmatched (1 pair);
    # the rule demands 2 pairs: p0-g1 and p1-g0.
    g0, g1 = [0, 0, 100, 100], [0, 25, 100, 125]
    p0 = [0, 5, 100, 105]
    p1 = [0, -22, 100, 78]
    iou = G.iou_matrix([p0, p1], [g0, g1])
    assert iou[0, 0] > iou[1, 0] >= 0.6 and iou[0, 1] >= 0.6 and iou[1, 1] < 0.6
    gt_index, _, _ = G.match_one_to_one([p0, p1], [g0, g1], 0.6)
    assert gt_index.tolist() == [1, 0]


def test_matching_without_manual_boxes_marks_everything_as_error():
    gt_index, matched, best = G.match_one_to_one([[0, 0, 5, 5], [1, 1, 3, 3]], np.zeros((0, 4)), 0.6)
    assert gt_index.tolist() == [-1, -1] and matched.tolist() == [0, 0] and best.tolist() == [0, 0]
