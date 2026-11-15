"""Tests of rascar.final_models (round 6): reproduction of the delivered files, leakage and robustness properties."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rascar import final_models as fm

ROOT = Path(__file__).resolve().parents[1]
FZ = ROOT / "results" / "frozen_v3b"
DELIVERED = {
    "o5c_clsstd_q4": ROOT / "submissions" / "kaggle_submitted" / "2026-11-12_RASCAR_v3_03_o5c_clsstd_q4_public0.89976.csv",
    "r4a_clsstd_all": ROOT / "submissions" / "candidates_v3" / "RASCAR_v3_05_r4a_clsstd_all.csv",
}
pytestmark = pytest.mark.skipif(not (FZ / "features_train.csv").exists(), reason="frozen feature table not present")


@pytest.fixture(scope="module")
def tables():
    tr = pd.read_csv(FZ / "features_train.csv", dtype={"id": str})
    te = pd.read_csv(FZ / "features_test.csv", dtype={"id": str})
    return tr, te


@pytest.mark.parametrize("name", list(DELIVERED))
def test_reproduces_delivered_file(tables, name):
    tr, te = tables
    if not DELIVERED[name].exists():
        pytest.skip("delivered file missing")
    pred = pd.Series(fm.FinalModel(name).fit(tr).predict(te), index=te["id"])
    ref = pd.read_csv(DELIVERED[name], dtype={"id": str}).set_index("id")["score"]
    assert np.abs(pred.loc[ref.index].to_numpy() - ref.to_numpy()).max() < 1e-8


@pytest.mark.parametrize("name", list(DELIVERED))
def test_test_rows_are_scored_independently(tables, name):
    tr, te = tables
    model = fm.FinalModel(name).fit(tr)
    full = model.predict(te)
    part = te.iloc[::7]
    assert np.allclose(model.predict(part), full[::7], atol=0, rtol=0)


@pytest.mark.parametrize("name", list(DELIVERED))
def test_label_column_is_not_used_at_inference(tables, name):
    tr, te = tables
    model = fm.FinalModel(name).fit(tr)
    noisy = te.copy()
    noisy["error"] = np.random.default_rng(0).integers(0, 2, len(te))
    noisy["best_iou"] = np.random.default_rng(1).random(len(te))
    assert np.array_equal(model.predict(noisy), model.predict(te))


def test_class_statistics_come_from_training_rows_only(tables):
    tr, te = tables
    model = fm.FinalModel("o5c_clsstd_q4").fit(tr)
    shifted = te.copy()
    for col in fm.OPINIONS:
        shifted[col] = shifted[col] + 0.0          # same values
    other = te.copy()
    other.loc[other.index[1:], fm.OPINIONS] = other.loc[other.index[1:], fm.OPINIONS] * 0.5   # change all rows but the first
    assert model.predict(other.iloc[:1])[0] == model.predict(te.iloc[:1])[0]


def test_save_load_roundtrip(tables, tmp_path):
    tr, te = tables
    model = fm.FinalModel("o5c_clsstd_q4").fit(tr)
    model.save(tmp_path)
    loaded = fm.FinalModel.load("o5c_clsstd_q4", tmp_path)
    assert np.array_equal(model.predict(te), loaded.predict(te))


def test_leave_one_day_out_matches_project_value(tables):
    tr, _ = tables
    from sklearn.metrics import average_precision_score
    oof = fm.leave_one_day_out("o5c_clsstd_q4", tr)
    y = tr["error"].to_numpy(int)
    aps = [average_precision_score(y[(tr["utc_day"] == d).to_numpy()], oof[(tr["utc_day"] == d).to_numpy()])
           for d in sorted(tr["utc_day"].unique())]
    assert abs(np.mean(aps) - 0.9545218) < 1e-6


def test_write_submission_rejects_bad_input(tmp_path):
    ids = pd.Series(["a", "b", "c"])
    sample = pd.Series(["c", "b", "a"])
    fm.write_submission(ids, np.array([0.1, 0.5, 0.9]), tmp_path / "ok.csv", sample)
    out = pd.read_csv(tmp_path / "ok.csv", dtype={"id": str})
    assert out["id"].tolist() == ["c", "b", "a"] and out["score"].tolist() == [0.9, 0.5, 0.1]
    with pytest.raises(ValueError):
        fm.write_submission(ids, np.array([0.1, np.nan, 0.9]), tmp_path / "nan.csv", sample)
    with pytest.raises(ValueError):
        fm.write_submission(ids, np.array([0.1, 1.5, 0.9]), tmp_path / "range.csv", sample)
    with pytest.raises(ValueError):
        fm.write_submission(pd.Series(["a", "b", "x"]), np.array([0.1, 0.2, 0.3]), tmp_path / "ids.csv", sample)


# ---------------- round 7: P7r (frozen 03 + person-only n_inside refinement) ----------------
N_FEATS = ROOT / "results" / "agent_N"


@pytest.fixture(scope="module")
def tables_n(tables):
    tr, te = tables
    if not (N_FEATS / "n1_features_train.csv").exists():
        pytest.skip("n_inside features not present")
    out = []
    for df, part in ((tr.copy(), "train"), (te.copy(), "test")):
        f = pd.read_csv(N_FEATS / f"n1_features_{part}.csv", dtype={"id": str}).set_index("id").loc[df["id"]]
        for c in fm.N_INSIDE_COLS:
            df[c] = f[c].to_numpy(float)
        out.append(df)
    return out


def test_p7r_keeps_03_for_non_person_rows(tables_n):
    tr, te = tables_n
    p7r = fm.PersonRefinedModel().fit(tr)
    base = fm.FinalModel("o5c_clsstd_q4").fit(tr)
    other = (te["class_name"] != "person").to_numpy()
    assert np.abs(p7r.predict(te)[other] - base.predict(te)[other]).max() < 1e-12
    assert p7r.beta < 0                                  # more inside proposals -> fewer errors (pre-registered sign)


def test_p7r_reproduces_lead_file_and_roundtrip(tables_n, tmp_path):
    tr, te = tables_n
    ref_path = ROOT / "results" / "r7_lead" / "P7r_test.csv"
    if not ref_path.exists():
        pytest.skip("P7r reference file missing")
    p7r = fm.PersonRefinedModel().fit(tr)
    p7r.save(tmp_path)
    loaded = fm.PersonRefinedModel.load(tmp_path)
    pred = pd.Series(loaded.predict(te), index=te["id"])
    ref = pd.read_csv(ref_path, dtype={"id": str}).set_index("id")["score"]
    assert np.abs(pred.loc[ref.index].to_numpy() - ref.to_numpy()).max() < 1e-9
    noisy = te.copy()
    noisy["error"] = 1
    assert np.array_equal(loaded.predict(noisy), loaded.predict(te))
    assert np.array_equal(loaded.predict(te.iloc[::5]), loaded.predict(te)[::5])
