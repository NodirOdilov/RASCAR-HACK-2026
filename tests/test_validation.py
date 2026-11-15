import numpy as np
import pandas as pd
import pytest

from rascar import data, validation


def test_day_folds_swap_days_and_never_overlap():
    days = pd.Series(["2026-10-25"] * 5 + ["2026-10-26"] * 7)
    frames = pd.Series([f"a{i // 2}" for i in range(5)] + [f"b{i // 2}" for i in range(7)])
    folds = validation.day_folds(days)
    assert [f.name for f in folds] == ["valid_2026-10-25", "valid_2026-10-26"]
    assert len(folds[0].valid_idx) == 5 and len(folds[0].train_idx) == 7
    validation.assert_no_overlap(folds, frames, days)
    covered = np.sort(np.concatenate([f.valid_idx for f in folds]))
    assert covered.tolist() == list(range(12))             # every row is validated exactly once


def test_day_folds_need_two_days():
    with pytest.raises(ValueError):
        validation.day_folds(pd.Series(["2026-10-25"] * 4))


def test_frame_group_folds_keep_frames_together():
    frames = np.repeat([f"f{i}" for i in range(20)], 3)
    folds = validation.frame_group_folds(frames, n_splits=5, repeats=2, seed=3)
    assert len(folds) == 10
    validation.assert_no_overlap(folds, frames)
    for rep in range(2):
        covered = np.sort(np.concatenate([f.valid_idx for f in folds[rep * 5:(rep + 1) * 5]]))
        assert covered.tolist() == list(range(60))


def test_overlap_detector_actually_fires():
    bad = [validation.Fold("bad", np.array([0, 1, 2]), np.array([2, 3]))]
    with pytest.raises(AssertionError):
        validation.assert_no_overlap(bad, np.array(["a", "b", "c", "d"]))
    same_frame = [validation.Fold("bad", np.array([0]), np.array([1]))]
    with pytest.raises(AssertionError):
        validation.assert_no_overlap(same_frame, np.array(["a", "a"]))


def test_real_split_is_by_utc_day_without_table(real_raw):
    frames = data.load_frames("train", real_raw)
    train = data.non_table(data.attach_day(data.load_targets("train", real_raw), frames), data.load_policy(real_raw))
    folds = validation.day_folds(train["utc_day"])
    validation.assert_no_overlap(folds, train["frame_id"], train["utc_day"])
    assert sorted(train["utc_day"].unique()) == ["2026-10-25", "2026-10-26"]
    assert "table" not in set(train["class_name"])
    assert {len(f.valid_idx) for f in folds} == {68, 130}
    # the context frames of a validation day never belong to the training day
    ctx = data.load_context("train", real_raw)
    day = frames.set_index("frame_id")["utc_day"]
    for col in ("previous", "next"):
        ok = ctx[col].dropna()
        assert (ok.map(day).to_numpy() == ctx.loc[ok.index, "frame_id"].map(day).to_numpy()).all()
