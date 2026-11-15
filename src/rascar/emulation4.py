"""E4 = E2X with its probabilities conditioned on the ONE-TO-ONE rule of the label definition.

The label of the task is assigned by a one-to-one matching of predicted and manual boxes of one class inside a frame
(TASK.txt: "взаимно однозначное сопоставление ... Сюда входят дубли"). Two audited boxes that sit on the same object
can therefore not both be correct. E2X (:mod:`rascar.emulation2`, preset ``e2x_pixel_inner_hires``) scores every box on
its own: both boxes of a duplicate find the same confident public proposal.

Update (no fitted parameter):

* same-class audited boxes of a frame form groups - the connected components of the relation ``IoU >= TAU``;
* inside a group the E2X scores ``p_i`` (probabilities of error, treated as independent) are conditioned on the event
  "at most one box of the group is correct":

      Z    = prod_j p_j + sum_i (1 - p_i) prod_{j != i} p_j
      p_i' = 1 - (1 - p_i) prod_{j != i} p_j / Z

  For a pair: ``p_A' = p_A / (1 - (1 - p_A)(1 - p_B))``. Every score of a group rises, the order inside the group is
  kept; a confident pair (0.07, 0.50) becomes (0.13, 0.93), an ambiguous pair (0.47, 0.61) becomes (0.60, 0.77).

``TAU = 0.5`` was fixed before any effect on average precision was computed (research/final_mission/
HYPOTHESIS_REGISTRY.md, F1). On the labelled days distinct manual boxes of one class overlap with IoU >= 0.5 in 1 of
248 pairs (two hands) and in none of the 19 audited pairs with IoU >= 0.5 are both boxes correct. A HARD version of the
rule (lift the box with the higher score to the top of the queue) was rejected: with other training frames it lifted a
correct box. This soft version was defined after that rejection (post hoc, registry entry F1c).

Nothing is fitted here: the E2X model is used unchanged; rows outside a group keep the E2X score bit for bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import emulation2 as em2

NAME = "e4_e2x_one_to_one"
BASE = em2.E2X
TAU = 0.5
EPS = 1e-9
BOX = ["x1", "y1", "x2", "y2"]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def group_ids(table: pd.DataFrame, tau: float = TAU) -> np.ndarray:
    """Group number of every row, -1 for a box that overlaps no other box of its class and frame with IoU >= tau.

    ``table`` needs frame_id, class_name and x1..y2. Groups are connected components, so a chain A-B, B-C is one group.
    """
    n = len(table)
    gid = np.full(n, -1, dtype=int)
    if n == 0:
        return gid
    boxes = table[BOX].to_numpy(float)
    key = (table["frame_id"].astype(str) + "|" + table["class_name"].astype(str)).to_numpy()
    nxt = 0
    for k in pd.unique(key):
        idx = np.flatnonzero(key == k)
        if len(idx) < 2:
            continue
        parent = {int(i): int(i) for i in idx}

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = int(idx[a]), int(idx[b])
                if _iou(boxes[i], boxes[j]) >= tau:
                    parent[find(i)] = find(j)
        comps: dict[int, list[int]] = {}
        for i in idx:
            comps.setdefault(find(int(i)), []).append(int(i))
        for members in comps.values():
            if len(members) > 1:
                gid[members] = nxt
                nxt += 1
    return gid


def condition_one_to_one(table: pd.DataFrame, scores: np.ndarray, tau: float = TAU) -> tuple[np.ndarray, np.ndarray]:
    """(conditioned scores, flag "row belongs to a group"). Apply ONCE to the scores of the base model."""
    s = np.asarray(scores, dtype=float)
    if len(s) != len(table):
        raise ValueError("scores and table differ in length")
    if not (0.0 < tau <= 1.0):
        raise ValueError("tau must lie in (0, 1]")
    if len(s) and (not np.isfinite(s).all() or s.min() < 0.0 or s.max() > 1.0):
        raise ValueError("scores must be probabilities in [0, 1]")
    gid = group_ids(table, tau)
    out = s.copy()
    for g in np.unique(gid[gid >= 0]):
        members = np.flatnonzero(gid == g)
        p = np.clip(s[members], EPS, 1.0 - EPS)
        all_wrong = float(np.prod(p))
        only_correct = np.array([(1.0 - p[i]) * float(np.prod(np.delete(p, i))) for i in range(len(members))])
        out[members] = 1.0 - only_correct / (all_wrong + float(only_correct.sum()))
    return np.clip(out, 0.0, 1.0), gid >= 0


def predict(model: em2.EmulationModel2, table: pd.DataFrame, store: em2.ProposalStore2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(E4 scores, E2X scores, group flag) for the rows of ``table``. The table must hold COMPLETE frames: the update
    compares the audited boxes of a frame with each other."""
    if model.cfg != BASE:
        raise ValueError("E4 is defined on the E2X model")
    base = model.predict(table, store)
    new, grouped = condition_one_to_one(table.reset_index(drop=True), base)
    return new, base, grouped


def leave_one_day_out(train: pd.DataFrame, manual: pd.DataFrame, store: em2.ProposalStore2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(E4 out-of-day scores, E2X out-of-day scores, group flag). The update has no parameter fitted on any day."""
    base = em2.leave_one_day_out(train, manual, store, BASE)
    new, grouped = condition_one_to_one(train.reset_index(drop=True), base)
    return new, base, grouped
