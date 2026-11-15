"""Round-2 additions: consensus of two second opinions, fixed input clipping, the acceptance rule,
the Grounding DINO agreement feature."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rascar import models, train, vision_gdino

ROOT = Path(__file__).resolve().parents[1]
CONS = [{"name": "cons", "op": "mean_z", "columns": ["op_a", "op_b"]}]
SPEC = {"name": "cons_lr", "kind": "logreg", "params": {"C": 1.0}, "derived": CONS, "clip": {"edge_rel": [0.0, 12.0]},
        "features": {"columns": ["tc_logit", "cons", "edge_rel"]}}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def table(synth_raw):
    X, y, aux = train.load_training_table()
    rng = np.random.default_rng(0)
    X = X.copy()
    signal = -(y - y.mean())                                   # both opinions agree more with correct boxes
    X["op_a"] = 0.3 + 0.2 * signal + rng.normal(0, 0.1, len(X))
    X["op_b"] = 5.0 + 3.0 * signal + rng.normal(0, 2.0, len(X))
    return X, y, aux


# ---------------------------------------------------------------- consensus of two opinions

def test_consensus_statistics_come_from_the_fitted_rows_only(table):
    X, y, aux = table
    half = len(X) // 2
    model = models.make_model(SPEC).fit(X.iloc[:half], y[:half], aux.iloc[:half])
    (mean_a, std_a), (mean_b, std_b) = model.derived_stats_["cons"]
    assert mean_a == pytest.approx(X["op_a"].iloc[:half].mean())
    assert std_b == pytest.approx(X["op_b"].iloc[:half].std(ddof=0), abs=1e-6)
    assert mean_a != pytest.approx(X["op_a"].mean())           # not the statistics of all rows
    before = model.predict(X.iloc[:half])
    shifted = X.iloc[half:].copy()
    shifted["op_a"] += 100.0                                   # rows scored later cannot move the stored statistics
    model.predict(shifted)
    np.testing.assert_allclose(model.predict(X.iloc[:half]), before)


def test_consensus_is_invariant_to_the_scale_of_each_opinion(table):
    X, y, aux = table
    score = models.make_model(SPEC).fit(X, y, aux).predict(X)
    rescaled = X.copy()
    rescaled["op_b"] = rescaled["op_b"] * 37.0 - 4.0
    np.testing.assert_allclose(models.make_model(SPEC).fit(rescaled, y, aux).predict(rescaled), score, atol=1e-6)


def test_missing_opinion_falls_back_to_the_other_one(table):
    X, y, aux = table
    model = models.make_model(SPEC).fit(X, y, aux)
    (mean_a, std_a), _ = model.derived_stats_["cons"]
    part = X.head(6).copy()
    part.loc[part.index[:3], "op_b"] = np.nan                  # one opinion missing
    part.loc[part.index[5], ["op_a", "op_b"]] = np.nan         # both missing
    prepared = model._prepare(part, fit=False)
    np.testing.assert_allclose(prepared["cons"].iloc[:3], (part["op_a"].iloc[:3] - mean_a) / std_a)
    assert np.isnan(prepared["cons"].iloc[5])
    score = model.predict(part)
    assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
    assert "cons" not in part.columns                          # the caller's table is not modified


def test_unknown_derived_op_and_missing_source_fail_loudly(table):
    X, y, aux = table
    bad_op = {**SPEC, "derived": [{"name": "cons", "op": "median_z", "columns": ["op_a", "op_b"]}]}
    with pytest.raises(KeyError):
        models.make_model(bad_op).fit(X, y, aux)
    with pytest.raises(KeyError):
        models.make_model(SPEC).fit(X.drop(columns=["op_b"]), y, aux)


# ---------------------------------------------------------------- fixed clipping of an input

def test_clip_makes_a_degenerate_box_score_like_the_bound(table):
    X, y, aux = table
    model = models.make_model(SPEC).fit(X, y, aux)
    rows = X.head(4).copy()
    at_bound, absurd = rows.copy(), rows.copy()
    at_bound["edge_rel"] = 12.0
    absurd["edge_rel"] = [287.0, 260400.0, np.inf, 1e12]       # values the audit observed for zero-width / point boxes
    np.testing.assert_allclose(model.predict(absurd), model.predict(at_bound))
    assert absurd["edge_rel"].iloc[0] == 287.0                 # input frame untouched
    unknown = rows.copy()
    unknown["edge_rel"] = np.nan                               # NaN is not turned into a bound, it is imputed
    assert np.isfinite(model.predict(unknown)).all()


def test_clip_changes_nothing_inside_the_bounds(table):
    X, y, aux = table
    assert float(X["edge_rel"].max()) <= 12.0
    plain = {k: v for k, v in SPEC.items() if k != "clip"}
    np.testing.assert_allclose(models.make_model(SPEC).fit(X, y, aux).predict(X),
                               models.make_model(plain).fit(X, y, aux).predict(X))


def test_raw_inputs_lists_the_sources_of_derived_columns():
    assert models.raw_inputs(SPEC) == ["edge_rel", "op_a", "op_b", "tc_logit"]
    other = {"name": "b", "kind": "logreg", "features": {"columns": ["conf_logit", "tc_logit"]}}
    blend = {"name": "bl", "kind": "blend", "members": [SPEC, other]}
    assert models.raw_inputs(blend) == ["conf_logit", "edge_rel", "op_a", "op_b", "tc_logit"]


# ---------------------------------------------------------------- acceptance rule

def test_acceptance_thresholds_follow_the_reviewed_table():
    rule = load_script("accept_test")
    assert [rule.delta_threshold(k) for k in (1, 3, 4, 12, 13, 70, 71, 300)] == [0.005, 0.005, 0.010, 0.010, 0.012, 0.012, 0.015, 0.015]
    assert rule.p_threshold(3) == 0.95
    assert rule.p_threshold(42) == pytest.approx(1 - 0.05 / 42)
    for k in (0, 301):
        with pytest.raises(ValueError):
            rule.delta_threshold(k)


def _two_day_problem(seed: int = 3):
    rng = np.random.default_rng(seed)
    days = np.repeat(["d1", "d2"], [60, 90])
    frames = np.array([f"{d}_f{i // 6}" for i, d in enumerate(days)])
    y = rng.integers(0, 2, len(days))
    return y, days, frames, rng


def test_paired_day_bootstrap_separates_a_real_gain_from_a_copy():
    rule = load_script("accept_test")
    y, days, frames, rng = _two_day_problem()
    good = y + rng.normal(0, 0.6, len(y))
    weak = y + rng.normal(0, 3.0, len(y))
    gain = rule.paired_day_bootstrap(y, good, weak, days, frames, n_boot=400, seed=1)
    assert gain["mean_day"]["p_gt0"] > 0.95 and gain["mean_day"]["ci_low"] > 0
    assert set(gain["days"]) == {"d1", "d2"}
    copy = rule.paired_day_bootstrap(y, good, good.copy(), days, frames, n_boot=200, seed=1)
    assert copy["mean_day"]["p_gt0"] == 0.0 and copy["mean_day"]["se"] == 0.0
    loss = rule.paired_day_bootstrap(y, weak, good, days, frames, n_boot=400, seed=1)
    assert loss["mean_day"]["p_gt0"] < 0.05
    assert loss["mean_day"]["ci_high"] == pytest.approx(-gain["mean_day"]["ci_low"])       # same resamples, sign flipped


def test_day_rank_removes_scale_differences_between_days():
    rule = load_script("accept_test")
    days = np.array(["a"] * 4 + ["b"] * 4)
    score = np.array([0.1, 0.2, 0.3, 0.4, 10.0, 20.0, 30.0, 40.0])
    np.testing.assert_allclose(rule.day_rank(score, days), [0.25, 0.5, 0.75, 1.0] * 2)


# ---------------------------------------------------------------- Grounding DINO agreement

def test_gdino_agreement_uses_the_claimed_class_and_the_overlap():
    n_phrases = len(vision_gdino.PHRASE_LIST)
    probs = np.zeros((3, n_phrases), dtype=np.float32)
    person, hand, cup = (vision_gdino.PHRASE_LIST.index(p) for p in ("person", "hand", "cup"))
    phone = vision_gdino.PHRASE_LIST.index("phone")
    probs[0, person] = 0.8                                     # proposal 0: a person exactly on the audited box
    probs[1, cup] = 0.6                                        # proposal 1: a cup elsewhere
    probs[2, phone] = 0.7                                      # proposal 2: a phone on the audited box
    probs[2, hand] = 0.05
    cached = {"full_boxes": np.array([[10, 10, 110, 210], [400, 300, 440, 340], [10, 10, 110, 210]], dtype=np.float32),
              "full_probs": probs}
    box = np.array([10.0, 10.0, 110.0, 210.0])
    as_person = vision_gdino.agreement(box, "person", cached)
    assert as_person["gd_same_soft"] == pytest.approx(0.8, abs=1e-6) and as_person["gd_iou"] == pytest.approx(1.0)
    assert as_person["gd_other"] == pytest.approx(0.7, abs=1e-6)
    as_cup = vision_gdino.agreement(box, "cup", cached)        # the only cup proposal does not overlap the box
    assert as_cup["gd_same_soft"] == 0.0 and as_cup["gd_same"] == 0.0
    as_hand = vision_gdino.agreement(box, "nude_hands", cached)
    assert as_hand["gd_same_soft"] == pytest.approx(0.05, abs=1e-6)
    assert vision_gdino.agreement(box, "person", {}) == {c: 0.0 for c in vision_gdino.GDINO_COLS}


def test_gdino_features_are_nan_without_a_cache(synth_raw):
    X, _, _ = train.load_training_table()
    gd = [c for c in X.columns if c.startswith(("gd_", "gdf_", "gdz_"))]
    assert len(gd) == 3 * len(vision_gdino.GDINO_COLS)
    assert X[gd].isna().all().all()                            # no cache in the synthetic project -> missing, never zero
    assert isinstance(X, pd.DataFrame)


def test_every_pretrained_model_is_named_with_its_source():
    from rascar import vision
    assert vision_gdino.MODEL_URL == "https://huggingface.co/" + vision_gdino.MODEL_NAME
    assert vision.MODEL_NAME == "google/owlv2-base-patch16-ensemble"
