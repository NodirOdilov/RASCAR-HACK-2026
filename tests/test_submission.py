import numpy as np
import pandas as pd
import pytest

from rascar import submission
from rascar.submission import SubmissionError


@pytest.fixture()
def sample():
    return pd.DataFrame({"id": ["c", "a", "d", "b"], "score": 0.5})


def _pred(ids=("a", "b", "c", "d"), scores=(0.1, 0.2, 0.3, 0.4)):
    return pd.DataFrame({"id": list(ids), "score": list(scores)})


def test_build_aligns_shuffled_predictions_to_template_order(sample):
    sub = submission.build(_pred(), sample)
    assert list(sub.columns) == ["id", "score"]
    assert sub["id"].tolist() == ["c", "a", "d", "b"]
    assert sub["score"].tolist() == [0.3, 0.1, 0.4, 0.2]
    shuffled = _pred().sample(frac=1.0, random_state=3)
    pd.testing.assert_frame_equal(sub, submission.build(shuffled, sample))


def test_missing_id_is_rejected(sample):
    with pytest.raises(SubmissionError, match="scores_numeric_no_nan"):
        submission.build(_pred(ids=("a", "b", "c", "x")), sample)


def test_duplicate_id_is_rejected(sample):
    with pytest.raises(SubmissionError, match="duplicated"):
        submission.build(_pred(ids=("a", "a", "c", "d")), sample)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -0.01, 1.01])
def test_bad_scores_are_rejected(sample, bad):
    with pytest.raises(SubmissionError):
        submission.build(_pred(scores=(0.1, 0.2, bad, 0.4)), sample)


def test_validate_catches_extra_missing_and_wrong_columns(sample):
    good = submission.build(_pred(), sample)
    assert all(submission.validate(good, sample).values())
    with pytest.raises(SubmissionError, match="row_count"):
        submission.validate(pd.concat([good, pd.DataFrame({"id": ["zz"], "score": [0.5]})]), sample)
    with pytest.raises(SubmissionError, match="columns"):
        submission.validate(good.rename(columns={"score": "prediction"}), sample)
    with pytest.raises(SubmissionError, match="columns"):
        submission.validate(good.assign(extra=1), sample)
    swapped = good.copy()
    swapped.loc[0, "id"] = "zz"
    with pytest.raises(SubmissionError, match="no_missing_ids"):
        submission.validate(swapped, sample)
    with pytest.raises(SubmissionError, match="not_constant"):
        submission.validate(good.assign(score=0.5), sample)


def test_write_is_deterministic_and_round_trips(sample, tmp_path):
    sub = submission.build(_pred(scores=(0.123456789012, 0.2, 0.3, 1.0)), sample)
    a = submission.write(sub, tmp_path / "a.csv", sample=sample)
    b = submission.write(sub, tmp_path / "b.csv", sample=sample)
    assert a["sha256"] == b["sha256"] and a["rows"] == 4
    text = (tmp_path / "a.csv").read_bytes()
    assert text.startswith(b"id,score\n") and b"\r" not in text
    back = pd.read_csv(tmp_path / "a.csv", dtype={"id": str})
    assert back["id"].tolist() == ["c", "a", "d", "b"]
    np.testing.assert_allclose(back["score"], sub["score"], atol=1e-10)
    manifest = (tmp_path / "submission_manifest.json").read_text(encoding="utf-8")
    assert '"a.csv"' in manifest and '"b.csv"' in manifest


def test_real_baseline_submission_matches_template(real_raw):
    sample = pd.read_csv(real_raw / "sample_submission.csv", dtype={"id": str})
    test = pd.read_csv(real_raw / "test.csv", dtype={"id": str})
    pred = test[["id"]].assign(score=1 - test["confidence"]).sample(frac=1.0, random_state=0)
    sub = submission.build(pred, sample)
    assert len(sub) == 362 and (sub["id"].to_numpy() == sample["id"].to_numpy()).all()
    assert sub["score"].between(0, 1).all()
