"""Unit tests of rascar.emulation (final-round candidate E1): adapter arithmetic, the emulated label rule,
leakage properties and serialisation. Synthetic proposal caches are written to a temporary folder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rascar import emulation as em


def _write_caches(vision_dir, frame_id: str, boxes: np.ndarray, cls_index: int, scores: np.ndarray) -> None:
    """One view per cache folder; the same proposals for every detector."""
    for folders, family in em.OPINION_CACHES.values():
        for k, folder in enumerate(folders):
            d = vision_dir / folder
            d.mkdir(parents=True, exist_ok=True)
            b = boxes if k == 0 else np.zeros((0, 4), dtype=np.float32)
            s = scores if k == 0 else np.zeros(0)
            if family == "owl":
                logits = np.full((len(b), 24), -12.0, dtype=np.float32)
                col = int(np.flatnonzero(em._OWL_GROUP == em.CLASSES[cls_index])[0])
                logits[:, col] = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
                np.savez(d / f"{frame_id}.npz", v_boxes=b.astype(np.float32), v_logits=logits, v_objectness=np.zeros(len(b), dtype=np.float32))
            else:
                probs = np.zeros((len(b), 14), dtype=np.float32)
                col = int(np.flatnonzero(em._GD_GROUP == em.CLASSES[cls_index])[0])
                probs[:, col] = s
                np.savez(d / f"{frame_id}.npz", v_boxes=b.astype(np.float32), v_probs=probs)


@pytest.fixture()
def store(tmp_path):
    return em.ProposalStore(tmp_path / "vision"), tmp_path / "vision"


def test_iou_one_to_many():
    box = np.array([0.0, 0.0, 10.0, 10.0])
    others = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    np.testing.assert_allclose(em.iou_one_to_many(box, others), [1.0, 1 / 3, 0.0], atol=1e-9)


def test_adapt_moves_edges_by_fraction_of_size():
    boxes = np.array([[100.0, 200.0, 140.0, 220.0]])            # w = 40, h = 20
    out = em.adapt(boxes, [-0.1, -0.05, 0.1, 0.05])
    np.testing.assert_allclose(out, [[96.0, 199.0, 144.0, 221.0]])
    np.testing.assert_allclose(em.adapt(boxes, [0, 0, 0, 0]), boxes)


def test_fit_adapter_recovers_a_known_convention(store):
    st, vdir = store
    rng = np.random.default_rng(0)
    rows = []
    for f in range(8):                                           # manual box = proposal grown by 10 % of its size per side
        p = np.array([[100.0 + 10 * f, 50.0, 140.0 + 10 * f, 90.0]])
        _write_caches(vdir, f"F{f}", p + rng.normal(0, 1e-3, p.shape), 1, np.array([0.8]))
        rows.append({"frame_id": f"F{f}", "class_name": "nude_hands", "x1": p[0, 0] - 4, "y1": p[0, 1] - 4, "x2": p[0, 2] + 4, "y2": p[0, 3] + 4})
    ad = em.fit_adapter(pd.DataFrame(rows), st)
    np.testing.assert_allclose(ad["nude_hands"], [-0.1, -0.1, 0.1, 0.1], atol=1e-3)
    assert ad["person"] == [0.0, 0.0, 0.0, 0.0]                  # fewer than MIN_PAIRS pairs -> identity


def test_smooth_emulates_the_threshold_rule(store):
    st, vdir = store
    # one confident proposal; audited boxes at IoU 1.0, ~0.62, ~0.52 and 0 with it
    _write_caches(vdir, "F0", np.array([[100.0, 100.0, 200.0, 200.0]]), 2, np.array([0.9]))
    t = pd.DataFrame({"id": list("abcd"), "frame_id": "F0", "class_name": "cup",
                      "x1": [100.0, 100.0, 100.0, 400.0], "y1": [100.0, 100.0, 100.0, 400.0],
                      "x2": [200.0, 162.0, 152.0, 500.0], "y2": [200.0, 200.0, 200.0, 500.0]})
    f = em.smooth_agreement(t, {c: [0, 0, 0, 0] for c in em.CLASSES}, st)
    v = f["gbf_smooth"].to_numpy()
    assert v[0] == pytest.approx(0.9, abs=1e-3)                  # passes every threshold
    assert v[1] == pytest.approx(0.9 * 3 / 5, abs=1e-3)          # IoU 0.62: passes 0.50, 0.55, 0.60
    assert v[2] == pytest.approx(0.9 * 1 / 5, abs=1e-3)          # IoU 0.52: passes 0.50 only
    assert v[3] == 0.0
    # another class at the same place gets nothing
    t2 = t.assign(class_name="tray")
    assert em.smooth_agreement(t2, {c: [0, 0, 0, 0] for c in em.CLASSES}, st).to_numpy().max() < 1e-4


def test_adapter_changes_the_agreement_in_the_expected_direction(store):
    st, vdir = store
    _write_caches(vdir, "F0", np.array([[100.0, 100.0, 140.0, 140.0]]), 1, np.array([0.8]))
    loose = pd.DataFrame({"id": ["a"], "frame_id": "F0", "class_name": "nude_hands", "x1": [94.0], "y1": [94.0], "x2": [146.0], "y2": [146.0]})
    raw = em.smooth_agreement(loose, {c: [0, 0, 0, 0] for c in em.CLASSES}, st)["gbf_smooth"].iloc[0]
    grown = em.smooth_agreement(loose, {c: [-0.1, -0.1, 0.1, 0.1] for c in em.CLASSES}, st)["gbf_smooth"].iloc[0]
    assert grown > raw                                           # a loose audited box agrees better with grown proposals


def _toy_problem(vdir, n_frames=12, seed=3):
    rng = np.random.default_rng(seed)
    rows, manual = [], []
    for f in range(n_frames):
        day = "2026-10-25" if f % 2 == 0 else "2026-10-26"
        fid = f"GENERAL_x_{day.replace('-', '')}_{f:06d}"
        props, scores = [], []
        for k in range(4):
            x, y = 100 + 150 * k, 100 + 20 * f
            p = np.array([x, y, x + 40.0, y + 30.0])
            props.append(p); scores.append(0.7)
            g = p + np.array([-3.0, -2.0, 3.0, 2.0])
            manual.append({"frame_id": fid, "class_name": "cup", "x1": g[0], "y1": g[1], "x2": g[2], "y2": g[3], "utc_day": day})
            wrong = rng.random() < 0.5
            shift = 22.0 if wrong else rng.normal(0, 1.5)
            grow = rng.uniform(-4.0, 6.0)                        # audited boxes of varying tightness
            rows.append({"id": f"{f}_{k}", "frame_id": fid, "class_name": "cup", "utc_day": day, "error": int(wrong),
                         "x1": g[0] + shift - grow, "y1": g[1] - grow, "x2": g[2] + shift + grow, "y2": g[3] + grow,
                         "tc_logit": rng.normal(1.0, 0.3), "p_tb_cut0": 0.0, "f_cov_same": 0.0, "edge_rel": 3.0})
        _write_caches(vdir, fid, np.asarray(props), 2, np.asarray(scores))
    return pd.DataFrame(rows), pd.DataFrame(manual)


def test_model_fit_predict_save_load_roundtrip(store, tmp_path):
    st, vdir = store
    train, manual = _toy_problem(vdir)
    model = em.EmulationModel().fit(train, manual, st)
    p = model.predict(train, st)
    assert p.shape == (len(train),) and np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    assert p[train.error == 1].mean() > p[train.error == 0].mean() + 0.3      # errors are ranked first
    model.save(tmp_path / "m")
    again = em.EmulationModel.load(tmp_path / "m").predict(train, st)
    np.testing.assert_allclose(again, p, atol=1e-12)


def test_validation_day_manual_boxes_never_reach_validation_scores(store):
    st, vdir = store
    train, manual = _toy_problem(vdir)
    base = em.leave_one_day_out(train, manual, st)
    # distort the manual boxes of 2026-10-26 only: predictions FOR that day must not move (they use the other day's adapter)
    broken = manual.copy()
    m = broken.utc_day == "2026-10-26"
    broken.loc[m, ["x1", "y1", "x2", "y2"]] = broken.loc[m, ["x1", "y1", "x2", "y2"]].to_numpy() + np.array([-5.0, -4.0, 5.0, 4.0])
    moved = em.leave_one_day_out(train, broken, st)
    d26 = (train.utc_day == "2026-10-26").to_numpy()
    np.testing.assert_allclose(moved[d26], base[d26], atol=1e-12)
    # ... while the adapter learnt from that day, and with it the other day's predictions, do change
    a0 = em.fit_adapter(manual[manual.utc_day == "2026-10-26"], st)["cup"]
    a1 = em.fit_adapter(broken[broken.utc_day == "2026-10-26"], st)["cup"]
    assert np.abs(np.array(a0) - np.array(a1)).max() > 0.05
    assert np.abs(moved[~d26] - base[~d26]).max() > 1e-9


def test_missing_cache_is_an_error(store):
    st, _ = store
    t = pd.DataFrame({"id": ["a"], "frame_id": "NOFRAME", "class_name": "cup", "x1": [0.0], "y1": [0.0], "x2": [5.0], "y2": [5.0]})
    with pytest.raises(FileNotFoundError):
        em.smooth_agreement(t, {c: [0, 0, 0, 0] for c in em.CLASSES}, st)
