"""Box geometry and the organizers' labelling rule.

All boxes are ``xyxy`` in pixels of the 1280x720 frame unless stated otherwise.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def as_boxes(arr) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.size == 0:
        return out.reshape(0, 4)
    return out.reshape(-1, 4)


def box_area(boxes) -> np.ndarray:
    b = as_boxes(boxes)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def intersection_matrix(a, b) -> np.ndarray:
    a, b = as_boxes(a), as_boxes(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)


def iou_matrix(a, b) -> np.ndarray:
    """Pairwise IoU, shape ``(len(a), len(b))``; degenerate boxes give 0."""
    a, b = as_boxes(a), as_boxes(b)
    inter = intersection_matrix(a, b)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def coverage_matrix(a, b) -> np.ndarray:
    """Fraction of each box in ``a`` that is covered by each box in ``b``."""
    a, b = as_boxes(a), as_boxes(b)
    inter = intersection_matrix(a, b)
    area = box_area(a)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(area > 0, inter / area, 0.0)


def center_distance_matrix(a, b) -> np.ndarray:
    a, b = as_boxes(a), as_boxes(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ca = np.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2], axis=1)
    cb = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)
    return np.sqrt(((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1))


def cxcywh_norm_to_xyxy(cx, cy, w, h, width: float, height: float) -> np.ndarray:
    cx, cy, w, h = (np.asarray(v, dtype=float) for v in (cx, cy, w, h))
    return np.stack([(cx - w / 2) * width, (cy - h / 2) * height,
                     (cx + w / 2) * width, (cy + h / 2) * height], axis=-1)


def match_one_to_one(pred, gt, iou_threshold: float = 0.6):
    """Organizers' rule (TASK.txt, "Как получена метка") for one frame and one class.

    One-to-one matching between predicted and manual boxes with IoU >= threshold;
    the number of admissible pairs is maximised first, then the total IoU.

    Returns ``(gt_index, matched_iou, best_iou)`` for every prediction where
    ``gt_index`` is -1 for unmatched predictions (those are the ``error=1`` rows)
    and ``best_iou`` is the largest IoU with any manual box regardless of matching.
    """
    pred, gt = as_boxes(pred), as_boxes(gt)
    n = len(pred)
    gt_index = np.full(n, -1, dtype=int)
    matched_iou = np.zeros(n)
    if n == 0:
        return gt_index, matched_iou, np.zeros(0)
    if len(gt) == 0:
        return gt_index, matched_iou, np.zeros(n)
    iou = iou_matrix(pred, gt)
    best = iou.max(axis=1)
    big = float(min(iou.shape) + 1)  # one extra pair always beats any gain in summed IoU
    weight = np.where(iou >= iou_threshold, big + iou, 0.0)
    rows, cols = linear_sum_assignment(weight, maximize=True)
    for r, c in zip(rows, cols):
        if weight[r, c] > 0:
            gt_index[r] = c
            matched_iou[r] = iou[r, c]
    return gt_index, matched_iou, best
