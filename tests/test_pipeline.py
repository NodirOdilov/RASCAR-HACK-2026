"""End-to-end checks: the documented numbers and the train -> save -> load -> predict -> submit path."""

import json

import numpy as np
import pandas as pd
import pytest

from rascar import audit, metrics, predict, submission, train


def test_baseline_numbers_on_real_data(real_raw):
    """Known values of the organizer baseline (1 - confidence) on the files on disk."""
    X, y, aux = train.load_training_table(real_raw)
    assert len(X) == 198 and int(y.sum()) == 109 and "table" not in set(X["class_name"])
    res = train.evaluate({"name": "baseline", "kind": "baseline"}, X, y, aux, gkf_repeats=0, n_boot=50)
    assert res["per_day"]["2026-10-25"]["ap"] == pytest.approx(0.8607, abs=5e-4)
    assert res["per_day"]["2026-10-26"]["ap"] == pytest.approx(0.7648, abs=5e-4)
    assert res["mean_day_ap"] == pytest.approx(0.8128, abs=5e-4)
    assert res["per_day"]["2026-10-25"]["prevalence"] == pytest.approx(0.5)
    assert res["delta_vs_baseline_mean_day"] == pytest.approx(0.0, abs=1e-12)
    # the training-only auxiliary target is consistent with the label
    assert ((aux["best_iou"] >= 0.6) | (y == 1)).all()
    assert (aux.loc[y == 0, "matched_iou"] >= 0.6).all()


def test_temporal_smoothing_beats_baseline_on_both_days(real_raw):
    """The headline finding must be reproducible from scratch."""
    X, y, aux = train.load_training_table(real_raw)
    res = train.evaluate({"name": "tconf", "kind": "formula", "params": {"column": "tc_logit"}}, X, y, aux,
                         gkf_repeats=0, n_boot=50)
    for day in ("2026-10-25", "2026-10-26"):
        base = metrics.average_precision(y[X["utc_day"] == day], 1 - X.loc[X["utc_day"] == day, "conf"])
        assert res["per_day"][day]["ap"] > base
    assert res["mean_day_ap"] == pytest.approx(0.8531, abs=1e-3)


@pytest.mark.parametrize("spec", [
    {"name": "baseline", "kind": "baseline"},
    {"name": "tconf", "kind": "formula", "params": {"column": "tc_logit"}},
    {"name": "lr", "kind": "logreg", "features": {"columns": ["tc_logit", "log_area"]}},
], ids=lambda s: s["name"])
def test_train_save_load_predict_submit(synth_raw, spec):
    """Tiny synthetic dataset through the real entry points; outputs land in the temp project root."""
    res = train.run(spec, gkf_repeats=1, final=True, decision="test")
    assert 0.0 <= res["mean_day_ap"] <= 1.0 and set(res["per_day"]) == {"2026-10-25", "2026-10-26"}
    from rascar import paths

    model_path = paths.models_dir() / f"{spec['name']}.joblib"
    meta = json.loads(model_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["train_days"] == ["2026-10-25", "2026-10-26"] and len(meta["model_sha256"]) == 64
    log = pd.read_csv(paths.project_root() / "EXPERIMENT_LOG.csv")
    assert list(log.columns) == train.LOG_COLUMNS and log["experiment_id"].tolist() == [spec["name"]]

    pred_a = predict.predict_test(model_path)          # standalone inference from the saved artifact
    pred_b = predict.predict_test(model_path)
    pd.testing.assert_frame_equal(pred_a, pred_b)      # deterministic
    sample = pd.read_csv(synth_raw / "sample_submission.csv", dtype={"id": str})
    sub = submission.build(pred_a, sample)
    entry = submission.write(sub, paths.submissions_dir() / "x.csv", sample=sample)
    assert entry["rows"] == len(sample) and all(entry["checks"].values())
    assert sub["id"].tolist() == sample["id"].tolist()


def test_prediction_is_independent_of_training_files_content(synth_raw, tmp_path, monkeypatch):
    """A saved model scores the test set identically even when labels and manual boxes are gone."""
    import shutil

    spec = {"name": "tconf", "kind": "formula", "params": {"column": "tc_logit"}}
    train.run(spec, gkf_repeats=0, final=True)
    from rascar import paths

    model_path = paths.models_dir() / "tconf.joblib"
    before = predict.predict_test(model_path)
    stripped = tmp_path / "stripped"
    shutil.copytree(synth_raw, stripped)
    (stripped / "human_boxes_train.csv").unlink()
    pd.read_csv(stripped / "train.csv", dtype={"id": str}).drop(columns="error").to_csv(stripped / "train.csv", index=False)
    monkeypatch.setenv("RASCAR_DATA_DIR", str(stripped))
    pd.testing.assert_frame_equal(before, predict.predict_test(model_path))


def test_audit_runs_on_synthetic_data(synth_raw):
    report = audit.run(synth_raw, hash_images=False)
    lab = report["labels"]
    assert lab["reproduction"]["agree"] == lab["reproduction"]["rows"]       # labels were built by the same rule
    assert report["identifiers_and_joins"]["sample_ids_equal_test_ids"] is True
    assert report["images"]["train_frames_missing_count"] > 0                 # synthetic set has no JPG files
    assert report["missing_expected_files"] == ["TASK.txt", "MACHINE_POOL.txt", "baseline.ipynb"]


def test_missing_image_does_not_break_tabular_features(synth_raw):
    """No JPG exists in the synthetic directory: every non-vision feature is still produced."""
    feats = train.load_test_table()
    assert feats["tc_logit"].notna().all() and feats["conf"].between(0, 1).all()
    assert feats["ov_same_soft"].isna().all()                                 # second opinion is simply absent
    assert np.isfinite(feats[["log_area", "edge_rel", "f_cov_same"]].to_numpy()).all()
