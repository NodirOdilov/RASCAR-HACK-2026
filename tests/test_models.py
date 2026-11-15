"""Every model kind trains on a tiny dataset, predicts valid probabilities, and survives a save/load."""

import joblib
import numpy as np
import pytest

from rascar import metrics, models, train

SPECS = [
    {"name": "baseline", "kind": "baseline"},
    {"name": "tconf", "kind": "formula", "params": {"column": "tc_logit"}},
    {"name": "lr", "kind": "logreg", "features": {"columns": ["conf_logit", "cls", "tc_nb_logit"]}, "params": {"C": 1.0}},
    {"name": "cb", "kind": "catboost", "features": {"groups": ["conf", "geom"]}, "seeds": [1],
     "params": {"iterations": 30, "depth": 2}},
    {"name": "lgbm", "kind": "lgbm", "features": {"groups": ["conf", "geom", "temporal"]}, "seeds": [1],
     "params": {"n_estimators": 30}},
    {"name": "hgb", "kind": "hgb", "features": {"groups": ["conf", "frame"]}, "seeds": [1], "params": {"max_iter": 20}},
    {"name": "resid", "kind": "lgbm_resid", "features": {"groups": ["geom"], "columns": ["tc_logit"]}, "seeds": [1],
     "params": {"prior": "tc_logit", "monotone": {"tc_logit": -1}, "n_estimators": 20}},
    {"name": "iou", "kind": "iou_reg", "features": {"groups": ["conf", "geom"]}, "seeds": [1], "params": {"iterations": 30}},
]
BLEND = {"name": "blend", "kind": "blend", "method": "rank", "weights": [0.5, 0.5], "members": [SPECS[1], SPECS[2]]}


@pytest.fixture()
def table(synth_raw):
    return train.load_training_table()


@pytest.mark.parametrize("spec", SPECS + [BLEND], ids=lambda s: s["name"])
def test_fit_predict_serialize(spec, table, tmp_path):
    X, y, aux = table
    model = models.make_model(spec).fit(X, y, aux)
    score = model.predict(X)
    assert score.shape == (len(X),) and np.isfinite(score).all()
    assert score.min() >= 0.0 and score.max() <= 1.0
    path = tmp_path / f"{spec['name']}.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict(X), score)
    again = models.make_model(spec).fit(X, y, aux).predict(X)      # fixed seeds -> repeatable
    np.testing.assert_allclose(again, score, rtol=1e-9, atol=1e-12)


def test_models_learn_the_planted_signal(table):
    """In the synthetic data low confidence and missing persistence cause errors."""
    X, y, aux = table
    assert metrics.average_precision(y, models.make_model(SPECS[0]).fit(X, y, aux).predict(X)) > y.mean() + 0.15
    assert metrics.average_precision(y, models.make_model(SPECS[1]).fit(X, y, aux).predict(X)) > y.mean() + 0.15


def test_formula_score_is_monotone_in_smoothed_confidence(table):
    X, y, aux = table
    score = models.make_model(SPECS[1]).predict(X)
    order = np.argsort(X["tc_logit"].to_numpy())
    assert (np.diff(score[order]) <= 1e-12).all()                   # higher smoothed confidence -> lower error score


def test_unknown_kind_and_missing_columns_fail_loudly(table):
    X, y, aux = table
    with pytest.raises(KeyError):
        models.make_model({"name": "x", "kind": "does_not_exist"})
    with pytest.raises(KeyError):
        models.make_model({"name": "x", "kind": "logreg", "features": {"columns": ["no_such_column"]}}).fit(X, y, aux)
    with pytest.raises(KeyError):
        models.make_model(SPECS[2]).fit(X, y, aux).predict(X.drop(columns=["tc_nb_logit"]))


def test_unexpected_class_does_not_crash_inference(table):
    """A class never seen in training (e.g. a new detector label) still gets a finite score."""
    X, y, aux = table
    odd = X.head(5).copy()
    odd["cls"] = "bottle"
    odd["class_name"] = "bottle"
    odd[["is_person", "is_nude_hands", "is_cup", "is_tray"]] = 0.0
    for spec in (SPECS[0], SPECS[1], SPECS[2], SPECS[4]):
        score = models.make_model(spec).fit(X, y, aux).predict(odd)
        assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
