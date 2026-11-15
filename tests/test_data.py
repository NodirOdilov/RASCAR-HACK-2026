"""Schema, identifiers and join cardinalities of the real competition files."""

import numpy as np
import pandas as pd
import pytest

from rascar import audit, data, geometry


def test_required_columns_and_missing_file_error(synth_raw, tmp_path):
    for name in data.REQUIRED:
        assert set(data.REQUIRED[name]) <= set(data.read_table(name).columns)
    with pytest.raises(FileNotFoundError):
        data.read_table("train", tmp_path / "nowhere")
    bad = tmp_path / "bad"
    bad.mkdir()
    pd.DataFrame({"id": ["a"], "frame_id": ["f"]}).to_csv(bad / "test.csv", index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        data.read_table("test", bad)


def test_duplicate_ids_are_rejected(synth_raw, tmp_path):
    import shutil

    broken = tmp_path / "dup"
    shutil.copytree(synth_raw, broken)
    test = pd.read_csv(broken / "test.csv", dtype={"id": str})
    pd.concat([test, test.head(1)]).to_csv(broken / "test.csv", index=False)
    with pytest.raises(ValueError, match="not unique"):
        data.load_targets("test", broken)


def test_ids_and_template(real_raw):
    train, test = data.load_targets("train", real_raw), data.load_targets("test", real_raw)
    sample = data.load_sample_submission(real_raw)
    assert train["id"].is_unique and test["id"].is_unique
    assert set(train["id"]).isdisjoint(test["id"])
    assert set(sample["id"]) == set(test["id"]) and len(sample) == len(test) == 362
    assert "error" in train.columns and "error" not in test.columns
    assert "table" not in set(test["class_name"])


@pytest.mark.parametrize("part", ["train", "test"])
def test_join_cardinalities(real_raw, part):
    p = data.load_part(part, real_raw)
    # target -> stream is one-to-one on the random id and carries identical values
    merged = p.targets.merge(p.stream, left_on="id", right_on="detection_id", validate="one_to_one", suffixes=("", "_s"))
    assert len(merged) == len(p.targets)
    for col in ("class_name", "confidence", *data.BOX_COLS):
        assert (merged[col].to_numpy() == merged[col + "_s"].to_numpy()).all()
    # target -> frames is many-to-one and never multiplies rows
    with_day = data.attach_day(p.targets, p.frames)
    assert len(with_day) == len(p.targets) and with_day["utc_day"].notna().all()
    # every frame of the part is a target or one of its listed neighbours
    union = set(p.context["frame_id"]) | set(p.context["previous"].dropna()) | set(p.context["next"].dropna())
    assert union == set(p.frames["frame_id"]) == set(p.stream["frame_id"])
    assert set(p.targets["frame_id"]) <= set(p.context["frame_id"])


def test_boxes_are_pixels_inside_the_frame(real_raw):
    for part in ("train", "test", "machine"):
        s = data.load_stream(part, real_raw)
        b = s[data.BOX_COLS].to_numpy(float)
        assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()
        assert b.min() >= 0 and b[:, [0, 2]].max() <= data.FRAME_W and b[:, [1, 3]].max() <= data.FRAME_H
        assert s["confidence"].between(0, 1).all()


def test_frames_times_and_images(real_raw):
    for part, n_days in (("train", 2), ("test", 2), ("machine", 2)):
        f = data.load_frames(part, real_raw)
        assert f["utc_day"].nunique() == n_days
        assert str(f["ts"].dt.tz) == "UTC"
        assert (f["image_path"] == f["frame_id"] + ".jpg").all()
        assert all((real_raw / name).is_file() for name in f["image_path"].head(25))
    assert sorted(data.load_frames("test", real_raw)["utc_day"].unique()) == ["2026-10-27", "2026-10-28"]


def test_alias_mapping_follows_the_policy_file(real_raw):
    policy = data.load_policy(real_raw)
    human = data.load_human_boxes(real_raw, policy)
    assert policy["aliases"] == {"nude_hand": "nude_hands", "white_napkin": "napkin"}
    assert "nude_hand" in set(human["class_name"]) and "nude_hand" not in set(human["class_canon"])
    assert (human.loc[human["class_name"] == "nude_hand", "class_canon"] == "nude_hands").all()
    untouched = human[~human["class_name"].isin(policy["aliases"])]
    assert (untouched["class_name"] == untouched["class_canon"]).all()       # nothing else is renamed
    assert policy["correct_iou"] == 0.6 and policy["test_excluded_classes"] == ["table"]


def test_labels_are_reproduced_exactly_from_manual_boxes(real_raw):
    policy = data.load_policy(real_raw)
    train = data.load_targets("train", real_raw)
    repro = audit.reproduce_labels(train, data.load_human_boxes(real_raw, policy), policy)
    assert len(repro) == 670 and (repro["error"] == repro["repro_error"]).all()
    correct = repro[repro["error"] == 0]
    assert (correct["matched_iou"] >= 0.6).all()


def test_human_box_conversion_round_trip(real_raw):
    human = data.load_human_boxes(real_raw)
    xyxy = geometry.cxcywh_norm_to_xyxy(human["cx"], human["cy"], human["w"], human["h"], data.FRAME_W, data.FRAME_H)
    np.testing.assert_allclose(xyxy, human[data.BOX_COLS].to_numpy(float))
    assert (human["x2"] > human["x1"]).all() and (human["y2"] > human["y1"]).all()
