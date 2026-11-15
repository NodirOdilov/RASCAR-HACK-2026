"""Second-opinion features from cached detector outputs (no torch needed: the cache is faked)."""

import numpy as np
import pandas as pd
import pytest

from rascar import vision


def _logit(p):
    return float(np.log(p / (1 - p)))


def _cache(tokens):
    """tokens: list of (box, {query_group: prob}) -> arrays shaped like the real cache."""
    boxes = np.array([t[0] for t in tokens], dtype=np.float32)
    logits = np.full((len(tokens), len(vision.TEXTS)), -12.0, dtype=np.float16)
    for i, (_, probs) in enumerate(tokens):
        for group, p in probs.items():
            for j, g in enumerate(vision.GROUP_OF):
                if g == group:
                    logits[i, j] = _logit(p)
    return {"full_boxes": boxes, "full_logits": logits, "full_objectness": np.zeros(len(tokens), dtype=np.float16)}


def test_prompt_groups_cover_the_scored_classes_only():
    assert set(vision.GROUP_OF) == {"person", "nude_hands", "cup", "tray", "other"}
    assert len(vision.TEXTS) == len(vision.GROUP_OF) == len(set(vision.TEXTS))


def test_agreeing_box_gives_high_support():
    cached = _cache([([100, 100, 130, 130], {"cup": 0.6}), ([500, 500, 600, 600], {"person": 0.9})])
    f = vision.frame_vision_features(np.array([[100, 100, 130, 130.0]]), np.array(["cup"], dtype=object), cached).iloc[0]
    assert f["ov_same"] == pytest.approx(0.6, abs=1e-3)
    assert f["ov_same_soft"] == pytest.approx(0.6, abs=1e-3)       # score * IoU with IoU = 1
    assert f["ov_iou"] == pytest.approx(1.0) and f["ov_other"] < 0.01 and f["ov_margin"] > 0


def test_no_object_of_the_claimed_class_gives_zero_support():
    cached = _cache([([100, 100, 130, 130], {"other": 0.7})])        # the model sees a look-alike, not a cup
    f = vision.frame_vision_features(np.array([[100, 100, 130, 130.0]]), np.array(["cup"], dtype=object), cached).iloc[0]
    assert f["ov_same"] < 0.01 and f["ov_same_soft"] < 0.01
    assert f["ov_other"] == pytest.approx(0.7, abs=1e-3) and f["ov_margin"] < 0


def test_soft_support_decreases_with_geometric_disagreement():
    box = np.array([[100, 100, 200, 200.0]])
    cls = np.array(["person"], dtype=object)
    exact = vision.frame_vision_features(box, cls, _cache([([100, 100, 200, 200], {"person": 0.5})])).iloc[0]
    taller = vision.frame_vision_features(box, cls, _cache([([100, 100, 200, 300], {"person": 0.5})])).iloc[0]
    assert taller["ov_iou"] == pytest.approx(0.5)                   # the independent box is twice as tall
    assert taller["ov_same_soft"] == pytest.approx(0.25, abs=1e-3)
    assert taller["ov_same_soft"] < exact["ov_same_soft"]
    assert taller["ov_same"] == pytest.approx(0.5, abs=1e-3)        # IoU 0.5 still counts for the hard variant
    far = vision.frame_vision_features(box, cls, _cache([([100, 100, 200, 420], {"person": 0.5})])).iloc[0]
    assert far["ov_same"] == 0.0                                    # IoU < 0.5: no hard support any more


def test_views_are_pooled_and_class_never_borrows_another_class_score():
    a = _cache([([100, 100, 130, 130], {"cup": 0.2})])
    b = _cache([([100, 100, 130, 130], {"cup": 0.7, "tray": 0.9})])
    cached = dict(a)
    cached.update({k.replace("full_", "left_"): v for k, v in b.items()})
    f = vision.frame_vision_features(np.array([[100, 100, 130, 130.0]]), np.array(["cup"], dtype=object), cached).iloc[0]
    assert f["ov_same"] == pytest.approx(0.7, abs=1e-3)             # best view wins, tray score is ignored


def test_uncached_frames_give_nan_and_cached_ones_are_aligned(synth_raw):
    targets = pd.DataFrame({"id": ["a", "b"], "frame_id": ["frame_cached", "frame_missing"], "class_name": ["cup", "cup"],
                            "confidence": [0.5, 0.5], "x1": [100.0, 100.0], "y1": [100.0, 100.0], "x2": [130.0, 130.0],
                            "y2": [130.0, 130.0]})
    np.savez_compressed(vision.cache_dir() / "frame_cached.npz", **_cache([([100, 100, 130, 130], {"cup": 0.6})]))
    out = vision.vision_features(targets)
    assert out.loc[0, "ov_same"] == pytest.approx(0.6, abs=1e-3) and out.loc[0, "ov_has"] == 1
    assert out.loc[1].isna().all()


def test_view_windows_stay_inside_the_frame_and_cover_it():
    for kind, n in (("base", 3), ("fine", 6)):
        windows = vision.view_windows(1280, 720, kind)
        assert len(windows) == n
        covered = np.zeros((720, 1280), dtype=bool)
        for x0, y0, x1, y1 in windows.values():
            assert 0 <= x0 < x1 <= 1280 and 0 <= y0 < y1 <= 720
            covered[y0:y1, x0:x1] = True
        assert covered.all()
    assert all(x1 - x0 == vision.FINE_TILE == y1 - y0 for x0, y0, x1, y1 in vision.view_windows(1280, 720, "fine").values())


def test_fine_tiles_are_pooled_into_separate_columns(synth_raw):
    targets = pd.DataFrame({"id": ["a", "b"], "frame_id": ["frame_both", "frame_base_only"], "class_name": ["cup", "cup"],
                            "confidence": [0.5, 0.5], "x1": [100.0, 100.0], "y1": [100.0, 100.0], "x2": [130.0, 130.0],
                            "y2": [130.0, 130.0]})
    weak = _cache([([100, 100, 130, 130], {"cup": 0.2})])
    strong = {k.replace("full_", "t00_"): v for k, v in _cache([([100, 100, 130, 130], {"cup": 0.7})]).items()}
    np.savez_compressed(vision.cache_dir("base") / "frame_both.npz", **weak)
    np.savez_compressed(vision.cache_dir("fine") / "frame_both.npz", **strong)
    np.savez_compressed(vision.cache_dir("base") / "frame_base_only.npz", **weak)
    out = vision.vision_features(targets)
    assert out.loc[0, "ov_same_soft"] == pytest.approx(0.2, abs=1e-3)       # base views only
    assert out.loc[0, "ovf_same_soft"] == pytest.approx(0.7, abs=1e-3)      # magnified tile sees the object better
    assert out.loc[1, "ov_same_soft"] == pytest.approx(0.2, abs=1e-3)
    assert np.isnan(out.loc[1, "ovf_same_soft"])                            # no silent fallback when the fine cache is missing
    assert vision.cache_dir("base") != vision.cache_dir("fine")


def test_temporal_average_of_the_second_opinion(synth_raw):
    np.savez_compressed(vision.cache_dir() / "f_prev.npz", **_cache([([100, 100, 130, 130], {"cup": 0.8})]))
    np.savez_compressed(vision.cache_dir() / "f_tgt.npz", **_cache([([100, 100, 130, 130], {"cup": 0.4})]))
    stream = pd.DataFrame({"detection_id": ["p", "t", "n"], "frame_id": ["f_prev", "f_tgt", "f_next"],
                           "class_name": ["cup"] * 3, "confidence": [0.5] * 3, "x1": [100.0] * 3, "y1": [100.0] * 3,
                           "x2": [130.0] * 3, "y2": [130.0] * 3})
    targets = stream[stream["frame_id"] == "f_tgt"].rename(columns={"detection_id": "id"}).reset_index(drop=True)
    own = vision.vision_features(targets)
    ids = pd.DataFrame({"tm_id_prev": np.array(["p"], dtype=object), "tm_id_next": np.array(["n"], dtype=object)})
    out = vision.temporal_vision_features(own, ids, stream).iloc[0]
    assert out["ovt_soft"] == pytest.approx((0.4 + 0.8) / 2, abs=1e-3)      # f_next is not cached -> skipped
    assert out["ovt_n"] == 1 and out["ovt_soft_nb"] == pytest.approx(0.8, abs=1e-3)
    none = pd.DataFrame({"tm_id_prev": np.array([None], dtype=object), "tm_id_next": np.array([None], dtype=object)})
    alone = vision.temporal_vision_features(own, none, stream).iloc[0]
    assert alone["ovt_soft"] == pytest.approx(0.4, abs=1e-3) and alone["ovt_n"] == 0
    assert alone["ovt_soft_nb"] == pytest.approx(0.4, abs=1e-3)             # unsupported boxes are not penalised
