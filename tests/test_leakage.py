"""Anti-leakage guarantees of the inference path (Stage C4 of the specification)."""

import ast
import inspect
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rascar import data, features as F, models, predict, temporal, train

FORBIDDEN_PREDICTORS = {"id", "frame_id", "utc_day", "t_sec", "error", "best_iou", "matched_iou", "detection_id"}


def _calls(module) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            names.add(fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", ""))
    return names


def test_feature_code_never_calls_the_manual_box_loader():
    for module in (F, temporal, predict):
        assert "load_human_boxes" not in _calls(module), module.__name__
    # the training module may read manual boxes, but only inside load_training_table
    src = inspect.getsource(train)
    assert src.count("load_human_boxes(") == 1
    assert "load_human_boxes" not in inspect.getsource(train.load_test_table)


def test_features_do_not_depend_on_the_label_column(synth_raw):
    part = data.load_part("train")
    ref = F.load_scene_reference()
    with_label = F.build_features(part, ref)
    flipped = part.targets.copy()
    flipped["error"] = 1 - flipped["error"]                       # adversarial: invert every label
    part_flipped = data.Part(part.name, flipped, part.frames, part.context, part.stream)
    without = data.Part(part.name, part.targets.drop(columns="error"), part.frames, part.context, part.stream)
    pd.testing.assert_frame_equal(with_label, F.build_features(part_flipped, ref))
    pd.testing.assert_frame_equal(with_label, F.build_features(without, ref))
    assert "error" not in with_label.columns


def test_test_features_work_without_labels_and_without_manual_boxes(synth_raw, tmp_path, monkeypatch):
    stripped = tmp_path / "stripped"
    shutil.copytree(synth_raw, stripped)
    (stripped / "human_boxes_train.csv").unlink()                 # no manual boxes at all
    tr = pd.read_csv(stripped / "train.csv", dtype={"id": str})
    tr.drop(columns="error").to_csv(stripped / "train.csv", index=False)   # no labels at all
    monkeypatch.setenv("RASCAR_DATA_DIR", str(stripped))
    feats = train.load_test_table()
    reference = train.load_test_table(synth_raw)
    pd.testing.assert_frame_equal(feats, reference)
    assert len(feats) == len(pd.read_csv(stripped / "test.csv"))
    with pytest.raises(FileNotFoundError):
        data.load_human_boxes(stripped)


def test_real_test_features_have_no_label_derived_columns(real_raw):
    feats = train.load_test_table(real_raw)
    assert not ({"error", "best_iou", "matched_iou"} & set(feats.columns))
    assert len(feats) == 362 and feats["id"].is_unique
    numeric = feats.drop(columns=F.META_COLS + ["cls"])
    assert not np.isinf(numeric.to_numpy(dtype=float)).any()


def test_identifiers_are_never_used_as_predictors():
    for group, cols in F.GROUPS.items():
        assert not (FORBIDDEN_PREDICTORS & set(cols)), group
    for config in sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.json")):
        import json

        spec = json.loads(config.read_text(encoding="utf-8"))
        assert not (FORBIDDEN_PREDICTORS & set(models.make_model(spec).columns)), config.name


def test_features_are_invariant_to_row_order_and_ids(synth_raw):
    """Shuffling rows or renaming the random ids must not change any feature value."""
    part = data.load_part("test")
    ref = F.load_scene_reference()
    base = F.build_features(part, ref).set_index("id")
    shuffled = data.Part(part.name, part.targets.sample(frac=1.0, random_state=7).reset_index(drop=True),
                         part.frames, part.context, part.stream.sample(frac=1.0, random_state=8).reset_index(drop=True))
    again = F.build_features(shuffled, ref).set_index("id").loc[base.index]
    pd.testing.assert_frame_equal(base, again)
    rename = {i: f"zz{k:05d}" for k, i in enumerate(part.stream["detection_id"])}
    renamed = data.Part(part.name, part.targets.assign(id=part.targets["id"].map(rename)), part.frames, part.context,
                        part.stream.assign(detection_id=part.stream["detection_id"].map(rename)))
    third = F.build_features(renamed, ref)
    num = [c for c in base.columns if c not in F.META_COLS and c != "cls"]
    np.testing.assert_allclose(third[num].to_numpy(float), F.build_features(part, ref)[num].to_numpy(float), equal_nan=True)


def test_scene_statistics_exclude_the_rows_own_day(synth_raw):
    """A box is compared with other days only: wiping its own day from the reference changes nothing."""
    part = data.load_part("train")
    ref = F.load_scene_reference()
    tgt = data.attach_day(part.targets.drop(columns="error"), part.frames)
    one_day = tgt[tgt["utc_day"] == "2026-10-25"].reset_index(drop=True)
    full = F.scene_features(one_day, one_day["utc_day"].to_numpy(), ref)
    other_only = F.SceneReference(ref.dets[ref.dets["utc_day"] != "2026-10-25"].reset_index(drop=True),
                                  {d: n for d, n in ref.frames_per_day.items() if d != "2026-10-25"})
    pd.testing.assert_frame_equal(full, F.scene_features(one_day, one_day["utc_day"].to_numpy(), other_only))


def test_preprocessing_statistics_come_from_the_training_fold_only(synth_raw):
    """Changing validation rows must not change a model fitted on the training fold."""
    X, y, aux = train.load_training_table()
    spec = {"name": "t", "kind": "logreg", "features": {"columns": ["conf_logit", "tc_nb_logit", "log_area"]}}
    days = X["utc_day"].to_numpy()
    tr_idx = np.flatnonzero(days == "2026-10-25")
    model_a = models.make_model(spec).fit(X.iloc[tr_idx], y[tr_idx], aux.iloc[tr_idx])
    X2 = X.copy()
    X2.loc[days != "2026-10-25", ["conf_logit", "tc_nb_logit", "log_area"]] = 123.0     # corrupt the other day
    model_b = models.make_model(spec).fit(X2.iloc[tr_idx], y[tr_idx], aux.iloc[tr_idx])
    np.testing.assert_allclose(model_a.predict(X.iloc[tr_idx]), model_b.predict(X.iloc[tr_idx]))
