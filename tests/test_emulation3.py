"""Unit tests of rascar.emulation3 (E3 = E2X-lite): same pools, margin and kernel as E2X, design matrix without
f_cov_same and edge_rel, serialisation that cannot be confused with E2X."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rascar import emulation as em
from rascar import emulation2 as em2
from rascar import emulation3 as em3
from test_emulation2 import _write


def _problem(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    rows, manual = [], []
    vdir, xdir = tmp_path / "vision", tmp_path / "extra"
    for f in range(12):
        day = "2026-10-25" if f % 2 == 0 else "2026-10-26"
        fid = f"GENERAL_x_{day.replace('-', '')}_{f:06d}"
        props = []
        for k in range(4):
            p = np.array([100 + 150 * k, 100 + 20 * f, 130 + 150 * k, 126 + 20 * f], dtype=float)
            props.append(p)
            g = p + np.array([-2.0, -2.0, 2.0, 2.0])
            manual.append({"frame_id": fid, "class_name": "cup", "x1": g[0], "y1": g[1], "x2": g[2], "y2": g[3], "utc_day": day})
            wrong = rng.random() < 0.5
            shift = 16.0 if wrong else rng.normal(0, 1.0)
            grow = rng.uniform(-2.0, 3.0)
            rows.append({"id": f"{f}_{k}", "frame_id": fid, "class_name": "cup", "utc_day": day, "error": int(wrong),
                         "x1": g[0] + shift - grow, "y1": g[1] - grow, "x2": g[2] + shift + grow, "y2": g[3] + grow,
                         "tc_logit": rng.normal(1.0, 0.3), "p_tb_cut0": 0.0, "f_cov_same": rng.random(), "edge_rel": rng.uniform(0, 12)})
        for tag, (folders, family) in em.OPINION_CACHES.items():
            _write(vdir / folders[0], fid, {"full": (np.zeros((0, 4)), np.zeros(0))}, 2, family)
            _write(vdir / folders[1], fid, {"t00": (np.zeros((0, 4)), np.zeros(0))}, 2, family)
            _write(xdir / em2.HIRES_FOLDER[tag], fid, {"zz": (np.asarray(props), np.full(4, 0.7))}, 2, family)
    return pd.DataFrame(rows), pd.DataFrame(manual), em2.ProposalStore2(vdir, xdir)


def test_configuration_is_e2x_except_for_the_name():
    assert em3.CFG.hires_classes == em2.E2X.hires_classes and em3.CFG.drop_border is True and em3.CFG.margin == "pixel"
    assert em3.CFG.name == em3.NAME != em2.E2X.name
    assert em3.INPUTS == ("tc_logit", "cons_smooth", "p_tb_cut0") and set(em3.REMOVED_INPUTS) == {"f_cov_same", "edge_rel"}


def test_design_has_three_columns_and_ignores_the_removed_inputs(tmp_path):
    train, manual, store = _problem(tmp_path)
    model = em3.EmulationModel3().fit(train, manual, store)
    assert model.pipeline[-1].coef_.shape == (1, 3)
    p = model.predict(train, store)
    changed = train.copy()
    changed["f_cov_same"] = 1.0 - changed["f_cov_same"]; changed["edge_rel"] = 12.0 - changed["edge_rel"]
    np.testing.assert_allclose(model.predict(changed, store), p, atol=0)
    assert p[train.error == 1].mean() > p[train.error == 0].mean() + 0.3


def test_same_adapter_and_agreement_as_e2x(tmp_path):
    train, manual, store = _problem(tmp_path)
    lite = em3.EmulationModel3().fit(train, manual, store)
    full = em2.EmulationModel2(em2.E2X).fit(train, manual, store)
    assert lite.adapter == full.adapter and lite.stats == full.stats
    np.testing.assert_allclose(lite.features(train, store)[em.SMOOTH_COLS].to_numpy(), full.features(train, store)[em.SMOOTH_COLS].to_numpy(), atol=0)


def test_save_load_roundtrip_and_no_confusion_with_e2x(tmp_path):
    train, manual, store = _problem(tmp_path)
    model = em3.EmulationModel3().fit(train, manual, store)
    p = model.predict(train, store)
    model.save(tmp_path / "m")
    again = em3.EmulationModel3.load(tmp_path / "m")
    assert again.cfg == em3.CFG
    np.testing.assert_allclose(again.predict(train, store), p, atol=1e-12)
    full = em2.EmulationModel2(em2.E2X).fit(train, manual, store)
    full.save(tmp_path / "m")
    with pytest.raises(ValueError):
        em3.EmulationModel3.load(tmp_path / "m", em2.E2X.name)


def test_leave_one_day_out_is_finite_and_uses_only_the_other_day(tmp_path):
    train, manual, store = _problem(tmp_path)
    oof = em3.leave_one_day_out(train, manual, store)
    assert np.isfinite(oof).all() and oof.min() >= 0 and oof.max() <= 1
    # labels of the held-out day flipped and its manual boxes moved -> its out-of-day predictions must not change
    day = "2026-10-26"
    tr2, man2 = train.copy(), manual.copy()
    tr2.loc[tr2.utc_day == day, "error"] = 1 - tr2.loc[tr2.utc_day == day, "error"]
    man2.loc[man2.utc_day == day, ["x1", "x2"]] += 3.0          # small enough to keep manual / proposal pairs (IoU >= 0.5)
    oof2 = em3.leave_one_day_out(tr2, man2, store)
    held = (train.utc_day == day).to_numpy()
    np.testing.assert_allclose(oof2[held], oof[held], atol=0)
