"""Unit tests of rascar.emulation2 (E2 / E2X): finer views for the small classes only, removal of proposals that
touch an interior border of their own view, the class-free pixel margin, and serialisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rascar import emulation as em
from rascar import emulation2 as em2

ZERO = {c: [0.0, 0.0, 0.0, 0.0] for c in em.CLASSES}


def _write(folder, frame_id, views: dict, cls_index, family):
    """views: {view name: (boxes, scores)}"""
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for view, (boxes, scores) in views.items():
        b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4); s = np.asarray(scores, dtype=float)
        arrays[f"{view}_boxes"] = b
        if family == "owl":
            logits = np.full((len(b), 24), -12.0, dtype=np.float32)
            col = int(np.flatnonzero(em._OWL_GROUP == em.CLASSES[cls_index])[0])
            sc = np.clip(s, 1e-6, 1 - 1e-6)
            logits[:, col] = np.log(sc / (1 - sc))
            arrays[f"{view}_logits"] = logits
            arrays[f"{view}_objectness"] = np.zeros(len(b), dtype=np.float32)
        else:
            probs = np.zeros((len(b), 14), dtype=np.float32)
            probs[:, int(np.flatnonzero(em._GD_GROUP == em.CLASSES[cls_index])[0])] = s
            arrays[f"{view}_probs"] = probs
    np.savez(folder / f"{frame_id}.npz", **arrays)


def _caches(tmp_path, frame_id, cls_index, base_views, hires_views):
    vdir, xdir = tmp_path / "vision", tmp_path / "extra"
    empty = {"t00": (np.zeros((0, 4)), np.zeros(0))}
    for tag, (folders, family) in em.OPINION_CACHES.items():
        _write(vdir / folders[0], frame_id, base_views, cls_index, family)
        _write(vdir / folders[1], frame_id, empty, cls_index, family)
        _write(xdir / em2.HIRES_FOLDER[tag], frame_id, hires_views, cls_index, family)
    return em2.ProposalStore2(vdir, xdir)


def _row(cls, box):
    return pd.DataFrame({"id": ["a"], "frame_id": "F0", "class_name": cls, "x1": [box[0]], "y1": [box[1]], "x2": [box[2]], "y2": [box[3]]})


def test_windows_cover_the_known_views():
    w = em2.WINDOWS
    assert w["full"] == (0, 0, 1280, 720) and w["left"] == (0, 0, 720, 720) and w["right"] == (560, 0, 1280, 720)
    assert w["t00"] == (0, 0, 480, 480) and w["t12"] == (800, 240, 1280, 720)
    assert w["h00"] == (0, 0, 384, 384) and w["h24"] == (896, 336, 1280, 720)


def test_small_classes_see_the_finer_views_and_large_classes_do_not(tmp_path):
    box = [100.0, 100.0, 130.0, 130.0]
    for ci, cls in enumerate(em.CLASSES):
        store = _caches(tmp_path / cls, "F0", ci, {"full": (np.zeros((0, 4)), np.zeros(0))}, {"h00": ([box], [0.8])})
        v = em2.smooth_agreement(_row(cls, box), ZERO, store, em2.E2)["gbf_smooth"].iloc[0]
        assert v == (pytest.approx(0.8, abs=1e-3) if cls in em2.E2.hires_classes else 0.0)


def test_border_touching_proposals_are_removed_only_with_drop_border(tmp_path):
    # view "left" is the crop x in [0, 720): a box ending at x = 719.5 is cut by the crop; the same box from the full
    # frame view is not an artefact
    cut = [600.0, 100.0, 719.5, 300.0]
    store = _caches(tmp_path / "a", "F0", 0, {"left": ([cut], [0.9]), "full": (np.zeros((0, 4)), np.zeros(0))}, {"h00": (np.zeros((0, 4)), np.zeros(0))})
    keep = em2.Config("k", hires_classes=(), drop_border=False)
    drop = em2.Config("d", hires_classes=(), drop_border=True)
    assert em2.smooth_agreement(_row("person", cut), ZERO, store, keep)["gbf_smooth"].iloc[0] == pytest.approx(0.9, abs=1e-3)
    assert em2.smooth_agreement(_row("person", cut), ZERO, store, drop)["gbf_smooth"].iloc[0] == 0.0
    store2 = _caches(tmp_path / "b", "F0", 0, {"full": ([cut], [0.9])}, {"h00": (np.zeros((0, 4)), np.zeros(0))})
    assert em2.smooth_agreement(_row("person", cut), ZERO, store2, drop)["gbf_smooth"].iloc[0] == pytest.approx(0.9, abs=1e-3)
    # a box touching the FRAME border (x = 0) is not removed
    edge = [0.0, 100.0, 120.0, 300.0]
    store3 = _caches(tmp_path / "c", "F0", 0, {"left": ([edge], [0.9])}, {"h00": (np.zeros((0, 4)), np.zeros(0))})
    assert em2.smooth_agreement(_row("person", edge), ZERO, store3, drop)["gbf_smooth"].iloc[0] == pytest.approx(0.9, abs=1e-3)


def test_pixel_margin_recovers_a_constant_plus_proportional_rule(tmp_path):
    # manual box = proposal grown by (2 px + 1 % of the side) on every side, for boxes of very different sizes
    manual = []
    vdir, xdir = tmp_path / "vision", tmp_path / "extra"
    sizes = [(30, 28), (26, 34), (80, 50), (140, 180), (120, 150), (28, 26), (32, 30), (150, 200)]
    classes = ["nude_hands", "cup", "tray", "person", "person", "nude_hands", "cup", "person"]
    for f, ((w, h), cls) in enumerate(zip(sizes, classes)):
        fid = f"F{f}"
        p = np.array([200.0, 200.0, 200.0 + w, 200.0 + h])
        mx, my = 2.0 + 0.01 * w, 2.0 + 0.01 * h
        manual.append({"frame_id": fid, "class_name": cls, "x1": p[0] - mx, "y1": p[1] - my, "x2": p[2] + mx, "y2": p[3] + my})
        for tag, (folders, family) in em.OPINION_CACHES.items():
            _write(vdir / folders[0], fid, {"full": ([p], [0.9])}, em.CLASSES.index(cls), family)
            _write(vdir / folders[1], fid, {"t00": (np.zeros((0, 4)), np.zeros(0))}, em.CLASSES.index(cls), family)
    store = em2.ProposalStore2(vdir, xdir)
    cfg = em2.Config("p", hires_classes=(), margin="pixel")
    ad = em2.fit_adapter(pd.DataFrame(manual), store, cfg)
    np.testing.assert_allclose(ad["x"], [2.0, 0.01], atol=1e-3)
    np.testing.assert_allclose(ad["y"], [2.0, 0.01], atol=1e-3)
    out = em2.adapt(np.array([[100.0, 100.0, 200.0, 150.0]]), "tray", ad, cfg)
    np.testing.assert_allclose(out, [[97.0, 97.5, 203.0, 152.5]], atol=1e-2)


def test_missing_extra_cache_is_an_error_only_when_needed(tmp_path):
    store = _caches(tmp_path, "F0", 0, {"full": ([[0.0, 0.0, 10.0, 10.0]], [0.5])}, {"h00": (np.zeros((0, 4)), np.zeros(0))})
    (store.extra_dir / em2.HIRES_FOLDER["gdf"] / "F0.npz").unlink()
    assert not store.has_all("F0", em2.E2) and store.has_all("F0", em2.Config("n", hires_classes=()))
    store.pool("gdf", "F0", "person", em2.E2)
    with pytest.raises(FileNotFoundError):
        store.pool("gdf", "F0", "cup", em2.E2)


@pytest.mark.parametrize("cfg", [em2.E2, em2.E2X])
def test_fit_predict_save_load_roundtrip(tmp_path, cfg):
    rng = np.random.default_rng(0)
    rows, manual = [], []
    vdir, xdir = tmp_path / "vision", tmp_path / "extra"
    for f in range(10):
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
                         "tc_logit": rng.normal(1.0, 0.3), "p_tb_cut0": 0.0, "f_cov_same": 0.0, "edge_rel": 3.0})
        for tag, (folders, family) in em.OPINION_CACHES.items():
            _write(vdir / folders[0], fid, {"full": (np.zeros((0, 4)), np.zeros(0))}, 2, family)
            _write(vdir / folders[1], fid, {"t00": (np.zeros((0, 4)), np.zeros(0))}, 2, family)
            _write(xdir / em2.HIRES_FOLDER[tag], fid, {"zz": (np.asarray(props), np.full(4, 0.7))}, 2, family)
    store = em2.ProposalStore2(vdir, xdir)
    train, manual = pd.DataFrame(rows), pd.DataFrame(manual)
    model = em2.EmulationModel2(cfg).fit(train, manual, store)
    p = model.predict(train, store)
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    assert p[train.error == 1].mean() > p[train.error == 0].mean() + 0.3
    model.save(tmp_path / "m")
    again = em2.EmulationModel2.load(tmp_path / "m", cfg.name)
    assert again.cfg == cfg
    np.testing.assert_allclose(again.predict(train, store), p, atol=1e-12)
    assert np.isfinite(em2.leave_one_day_out(train, manual, store, cfg)).all()
