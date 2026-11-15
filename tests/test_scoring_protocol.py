"""Edge cases of the official-compatible scorer (prompt section 3): perfect, reversed, constant,
several classes, excluded classes, missing groups, invalid scores, ties, wrong orientation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from rascar import metrics, validation
from rascar.metrics import ScoringError

POLICY = {"test_excluded_classes": ["table"]}


def table(error, score, cls=None):
    df = pd.DataFrame({"id": [f"r{i}" for i in range(len(error))], "error": error, "score": score})
    if cls is not None:
        df["class_name"] = cls
    return df


def test_perfect_ranking_scores_one():
    assert metrics.official_ap(table([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]), POLICY) == pytest.approx(1.0)


def test_completely_reversed_ranking_has_the_known_floor():
    # positives sit at ranks 3 and 4 -> (1/3 + 2/4) / 2
    assert metrics.official_ap(table([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]), POLICY) == pytest.approx((1 / 3 + 2 / 4) / 2)


def test_constant_predictions_give_the_prevalence():
    error = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    assert metrics.official_ap(table(error, [0.5] * 10), POLICY) == pytest.approx(0.3)


def test_several_classes_are_pooled_into_one_global_ap_not_averaged():
    # class A is ranked perfectly on its own, class B too, but B's scores are all below A's scores:
    # the per-class APs are both 1.0, the official global AP is lower because one ranking is formed.
    error = [1, 0, 1, 0]
    score = [0.9, 0.8, 0.3, 0.2]
    cls = ["person", "person", "cup", "cup"]
    df = table(error, score, cls)
    per_class = [metrics.official_ap(df[df["class_name"] == c], POLICY) for c in ("person", "cup")]
    assert per_class == [pytest.approx(1.0), pytest.approx(1.0)]
    pooled = metrics.official_ap(df, POLICY)
    assert pooled == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert pooled < np.mean(per_class)


def test_excluded_class_rows_do_not_influence_the_score():
    base = table([1, 0, 1, 0], [0.9, 0.6, 0.4, 0.1], ["cup", "person", "tray", "person"])
    extra = pd.DataFrame({"id": ["t1", "t2", "t3"], "error": [1, 0, 0], "score": [0.0, 1.0, 1.0], "class_name": ["table"] * 3})
    with_tables = pd.concat([base, extra], ignore_index=True)
    assert metrics.official_ap(with_tables, POLICY) == pytest.approx(metrics.official_ap(base, POLICY))
    # without a policy the default exclusion list is still ["table"]
    assert metrics.official_ap(with_tables) == pytest.approx(metrics.official_ap(base, POLICY))
    # a policy that excludes nothing scores the table rows too
    assert metrics.official_ap(with_tables, {"test_excluded_classes": []}) != pytest.approx(metrics.official_ap(base, POLICY))


def test_group_without_positives_is_nan_locally_but_harmless_globally():
    df = table([1, 0, 0, 0], [0.9, 0.5, 0.4, 0.3], ["cup", "cup", "tray", "tray"])
    assert np.isnan(metrics.official_ap(df[df["class_name"] == "tray"], POLICY))    # AP undefined: no positive
    assert metrics.official_ap(df, POLICY) == pytest.approx(1.0)                       # global AP is defined
    days = np.array(["d1", "d1", "d2", "d2"])
    res = validation.evaluate_day_oof(df["error"], df["score"], days, np.array(["f1", "f1", "f2", "f2"]), n_boot=20)
    assert np.isnan(res["per_day"]["d2"]["ap"]) and np.isnan(res["mean_day_ap"])       # never silently treated as 0


def test_missing_and_extra_ids_are_rejected():
    truth = table([1, 0, 1], [0, 0, 0]).drop(columns="score")
    good = pd.DataFrame({"id": ["r0", "r1", "r2"], "score": [0.9, 0.1, 0.8]})
    assert metrics.official_ap_from_submission(good, truth, POLICY) == pytest.approx(1.0)
    assert metrics.official_ap_from_submission(good.iloc[::-1], truth, POLICY) == pytest.approx(1.0)   # order is arbitrary
    with pytest.raises(ScoringError, match="missing"):
        metrics.official_ap_from_submission(good.iloc[:2], truth, POLICY)
    with pytest.raises(ScoringError, match="unknown"):
        metrics.official_ap_from_submission(pd.concat([good, pd.DataFrame({"id": ["zz"], "score": [0.5]})]), truth, POLICY)
    with pytest.raises(ScoringError, match="duplicated"):
        metrics.official_ap_from_submission(pd.concat([good, good.iloc[:1]]), truth, POLICY)
    with pytest.raises(ScoringError, match="columns"):
        metrics.official_ap_from_submission(good.rename(columns={"score": "prediction"}), truth, POLICY)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -0.001, 1.001, "abc"])
def test_invalid_scores_are_rejected_not_repaired(bad):
    with pytest.raises(ScoringError):
        metrics.official_ap(table([1, 0, 1], [0.9, bad, 0.1]), POLICY)


def test_invalid_labels_are_rejected():
    with pytest.raises(ScoringError, match="labels"):
        metrics.official_ap(table([1, 2, 0], [0.9, 0.5, 0.1]), POLICY)


def test_ties_follow_sklearn_and_do_not_depend_on_row_order():
    error = np.array([1, 0, 1, 0, 1, 0])
    score = np.array([0.8, 0.8, 0.8, 0.3, 0.3, 0.1])        # two tie groups
    ap = metrics.official_ap(table(error, score), POLICY)
    # group 1 (3 rows, 2 positives): recall 2/3 at precision 2/3; group 2 (2 rows, 1 positive): recall 1 at precision 3/5
    assert ap == pytest.approx((2 / 3) * (2 / 3) + (1 / 3) * (3 / 5))
    assert ap == pytest.approx(average_precision_score(error, score))
    perm = np.array([5, 3, 1, 4, 2, 0])
    assert metrics.official_ap(table(error[perm], score[perm]), POLICY) == pytest.approx(ap)
    # sklearn evaluates a tie group at its END: every positive of the group gets the precision reached after
    # the whole group. Resolving the ties with distinct scores moves the result in the expected directions:
    lucky = score + np.array([2e-6, 0, 1e-6, 0, 1e-6, 0])        # positives first inside each group: ranks 1, 2, 4
    unlucky = score - np.array([1e-6, 0, 2e-6, 0, 1e-6, 0])      # positives last inside each group: ranks 2, 3, 5
    assert metrics.official_ap(table(error, lucky), POLICY) == pytest.approx((1 + 1 + 3 / 4) / 3)
    assert metrics.official_ap(table(error, unlucky), POLICY) == pytest.approx((1 / 2 + 2 / 3 + 3 / 5) / 3)
    assert metrics.official_ap(table(error, lucky), POLICY) > ap > metrics.official_ap(table(error, unlucky), POLICY)
    # ... and positives that stay tied with each other keep the group-end precision even when a negative precedes them
    still_tied = score - np.where(error == 1, 1e-6, 0.0)
    assert metrics.official_ap(table(error, still_tied), POLICY) == pytest.approx(ap)


def test_wrong_orientation_is_detectably_worse():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.4, 0.95, 400)
    error = (rng.uniform(size=400) > conf).astype(int)       # low confidence -> error, as in the real data
    right = metrics.official_ap(table(error, 1 - conf), POLICY)
    wrong_score = metrics.official_ap(table(error, conf), POLICY)             # submitting confidence instead of 1 - confidence
    wrong_label = metrics.official_ap(table(1 - error, 1 - conf), POLICY)     # treating "correct" as the positive class
    assert right > error.mean() + 0.1 > wrong_score
    assert wrong_label != pytest.approx(right)


def test_real_baseline_through_the_official_scorer(real_raw):
    from rascar import data

    policy = data.load_policy(real_raw)
    train = data.attach_day(data.load_targets("train", real_raw), data.load_frames("train", real_raw))
    scored = train.assign(score=1 - train["confidence"])
    pooled = metrics.official_ap(scored, policy)                  # table rows present in the input, excluded by the scorer
    assert pooled == pytest.approx(0.7892, abs=5e-4)
    with_tables = metrics.official_ap(scored, {"test_excluded_classes": []})
    assert with_tables == pytest.approx(0.6986, abs=5e-4) and with_tables < pooled - 0.05   # scoring the 472 tables would change the number
    per_day = {d: metrics.official_ap(g, policy) for d, g in scored.groupby("utc_day")}
    assert per_day["2026-10-25"] == pytest.approx(0.8607, abs=5e-4)
    assert per_day["2026-10-26"] == pytest.approx(0.7648, abs=5e-4)
