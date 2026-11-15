"""Unit tests of rascar.emulation4 (E4 = E2X conditioned on the one-to-one rule of the label definition)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rascar import emulation2 as em2
from rascar import emulation4 as em4


def _t(rows):
    return pd.DataFrame(rows, columns=["frame_id", "class_name", "x1", "y1", "x2", "y2"])


A = ["F0", "cup", 100.0, 100.0, 140.0, 140.0]
B = ["F0", "cup", 105.0, 100.0, 145.0, 140.0]          # IoU with A = 35 / 45 = 0.78


def test_constants_and_base_model():
    assert em4.TAU == 0.5 and em4.BASE == em2.E2X and em4.NAME == "e4_e2x_one_to_one"


def test_pair_formula_and_order_inside_the_pair():
    t = _t([A, B])
    new, grouped = em4.condition_one_to_one(t, np.array([0.07, 0.50]))
    assert grouped.tolist() == [True, True]
    z = 1 - 0.93 * 0.5
    assert new[0] == pytest.approx(0.07 / z) and new[1] == pytest.approx(0.50 / z)
    assert new[0] < new[1] and (new >= np.array([0.07, 0.50])).all()
    amb, _ = em4.condition_one_to_one(t, np.array([0.474, 0.613]))
    assert amb[0] == pytest.approx(0.474 / (1 - 0.526 * 0.387)) and amb[1] < 0.80          # an ambiguous pair moves little
    both_fine, _ = em4.condition_one_to_one(t, np.array([0.06, 0.07]))
    assert 0.4 < both_fine[0] < both_fine[1] < 0.6                                            # "one of them is wrong, unknown which"
    sure, _ = em4.condition_one_to_one(t, np.array([0.99, 0.98]))
    assert sure == pytest.approx([0.99, 0.98], abs=3e-4)                                      # two near-certain errors stay


def test_it_is_the_conditional_probability_of_at_most_one_correct_box():
    rng = np.random.default_rng(1)
    c = ["F0", "cup", 113.34, 100.0, 153.34, 140.0]              # chain A-B-c (IoU(A, c) < 0.5, IoU(B, c) = 0.655)
    t = _t([A, B, c])
    assert em4.group_ids(t).tolist() == [0, 0, 0]
    p = np.array([0.2, 0.6, 0.35])
    new, _ = em4.condition_one_to_one(t, p)
    draws = rng.random((400000, 3)) < p                           # True = error
    ok = (~draws).sum(axis=1) <= 1                                # at most one correct box
    np.testing.assert_allclose(new, draws[ok].mean(axis=0), atol=4e-3)


def test_other_class_other_frame_or_small_overlap_change_nothing():
    s = np.array([0.2, 0.3])
    for other in (["F0", "tray", 105.0, 100.0, 145.0, 140.0], ["F1", "cup", 105.0, 100.0, 145.0, 140.0], ["F0", "cup", 121.0, 100.0, 161.0, 140.0]):
        new, grouped = em4.condition_one_to_one(_t([A, other]), s)     # last case: IoU = 19 / 61 = 0.31 < 0.5
        assert not grouped.any() and np.array_equal(new, s)
    exact = _t([["F0", "cup", 0.0, 0.0, 30.0, 30.0], ["F0", "cup", 10.0, 0.0, 40.0, 30.0]])       # IoU = 20 / 40 = 0.5 exactly: inclusive
    assert em4.group_ids(exact).tolist() == [0, 0]


def test_rows_outside_groups_keep_their_score_bit_for_bit_and_result_is_order_free():
    rng = np.random.default_rng(0)
    rows = []
    for f in range(30):
        for k in range(6):
            x, y = rng.uniform(0, 1200), rng.uniform(0, 650)
            w, h = rng.uniform(20, 60), rng.uniform(20, 60)
            rows.append([f"F{f}", ["cup", "nude_hands", "person"][k % 3], x, y, x + w, y + h])
        rows.append([f"F{f}", "cup", 300.0, 300.0, 340.0, 340.0]); rows.append([f"F{f}", "cup", 304.0, 300.0, 344.0, 340.0])   # one sure group per frame
    t = _t(rows)
    s = rng.uniform(0, 1, len(t))
    new, grouped = em4.condition_one_to_one(t, s)
    assert grouped.sum() >= 60 and np.array_equal(new[~grouped], s[~grouped])
    assert (new >= 0).all() and (new <= 1).all() and (new[grouped] >= s[grouped] - 1e-12).all()
    perm = rng.permutation(len(t))
    new2, grouped2 = em4.condition_one_to_one(t.iloc[perm].reset_index(drop=True), s[perm])
    assert np.allclose(new2, new[perm]) and np.array_equal(grouped2, grouped[perm])


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError):
        em4.condition_one_to_one(_t([A, B]), np.array([0.1]))
    with pytest.raises(ValueError):
        em4.condition_one_to_one(_t([A, B]), np.array([0.1, 1.2]))
    with pytest.raises(ValueError):
        em4.condition_one_to_one(_t([A, B]), np.array([0.1, 0.2]), tau=0.0)
    assert em4.group_ids(_t([])).tolist() == []
    extreme, _ = em4.condition_one_to_one(_t([A, B]), np.array([0.0, 1.0]))
    assert np.isfinite(extreme).all() and extreme[0] < 1e-6 and extreme[1] == pytest.approx(1.0)
