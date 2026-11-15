"""Round 6 / Agent SA - the official scoring protocol pinned by an INDEPENDENT reference implementation.

Sources: data/raw/rascar-hack-case-1/TASK.txt ("Как получена метка", "Метрика и формат отправки") and
scoring_policy.json. The ``ref_*`` functions below do not use the project package; ``rascar`` is imported
only in the tests named ``test_project_*`` that explicitly compare the project's scorer with the reference.
Real-data tests are skipped when the competition files are absent.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "rascar-hack-case-1"
POLICY = {  # the parts of scoring_policy.json the evaluator needs (checked against the file below)
    "classes": ["cup", "tray", "person", "mop", "bottle", "napkin", "nude_hands", "lunch_box", "sauce",
                "cleaning_uniform", "yellow_gloves", "table"],
    "aliases": {"nude_hand": "nude_hands", "white_napkin": "napkin"},
    "correct_iou": 0.6,
    "test_excluded_classes": ["table"],
}


# =============================================================================== reference implementation
def ref_iou(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area = lambda x: np.clip(x[:, 2] - x[:, 0], 0, None) * np.clip(x[:, 3] - x[:, 1], 0, None)  # noqa: E731
    union = area(a)[:, None] + area(b)[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def ref_match(iou, thr):
    """Max number of admissible pairs (IoU >= thr), then max total IoU; exact (weights 1000 + IoU)."""
    iou = np.asarray(iou, dtype=float)
    if iou.size == 0 or not (iou >= thr).any():
        return []
    adm = iou >= thr
    r, c = linear_sum_assignment(np.where(adm, 1000.0 + iou, 0.0), maximize=True)
    return [(int(i), int(j)) for i, j in zip(r, c) if adm[i, j]]


def greedy_iou_match(iou, thr):
    """NOT the official rule - used to show where greedy matching differs."""
    iou = np.asarray(iou, dtype=float)
    used_p, used_g, out = set(), set(), []
    for _, i, j in sorted(((iou[i, j], i, j) for i, j in zip(*np.nonzero(iou >= thr))), key=lambda t: -t[0]):
        if i not in used_p and j not in used_g:
            used_p.add(i), used_g.add(j), out.append((int(i), int(j)))
    return out


def max_total_iou_match(iou, thr):
    """NOT the official rule - total IoU without the cardinality priority."""
    iou = np.asarray(iou, dtype=float)
    adm = iou >= thr
    r, c = linear_sum_assignment(np.where(adm, iou, 0.0), maximize=True)
    return [(int(i), int(j)) for i, j in zip(r, c) if adm[i, j]]


def ref_labels(preds: pd.DataFrame, gts: pd.DataFrame, policy=POLICY, matcher=ref_match) -> np.ndarray:
    """error per prediction: per frame and canonical class, matched -> 0, everything else -> 1."""
    al, classes, thr = policy["aliases"], set(policy["classes"]), float(policy["correct_iou"])
    pc = preds["class_name"].map(lambda n: al.get(n, n)).to_numpy()
    gc = gts["class_name"].map(lambda n: al.get(n, n)).to_numpy()
    pf, gf = preds["frame_id"].to_numpy(), gts["frame_id"].to_numpy()
    pb = preds[["x1", "y1", "x2", "y2"]].to_numpy(float)
    gb = gts[["x1", "y1", "x2", "y2"]].to_numpy(float)
    err = np.ones(len(preds), dtype=int)
    for f, c in set(zip(pf, pc)):
        if c not in classes:
            continue
        pi = np.flatnonzero((pf == f) & (pc == c))
        gi = np.flatnonzero((gf == f) & (gc == c))
        for i, _ in matcher(ref_iou(pb[pi], gb[gi]), thr) if gi.size else []:
            err[pi[i]] = 0
    return err


def ref_ap(y_true, score) -> float:
    """sklearn semantics: one precision per DISTINCT threshold (tied rows enter together), no interpolation."""
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(score, dtype=float).ravel()
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    last = np.r_[ss[1:] != ss[:-1], True]
    tp = np.cumsum(ys)[last].astype(float)
    k = (np.flatnonzero(last) + 1).astype(float)
    return float(np.sum(np.diff(np.r_[0.0, tp / y.sum()]) * (tp / k)))


class SubmissionError(ValueError):
    pass


def ref_validate(sub: pd.DataFrame, sample_ids) -> pd.Series:
    if list(sub.columns) != ["id", "score"]:
        raise SubmissionError(f"columns must be ['id', 'score'], got {list(sub.columns)}")
    ids = sub["id"].astype(str)
    if not ids.is_unique:
        raise SubmissionError("duplicated ids")
    sample = set(map(str, sample_ids))
    if sample - set(ids):
        raise SubmissionError("missing ids")
    if set(ids) - sample:
        raise SubmissionError("unknown ids")
    score = pd.to_numeric(sub["score"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(score).all():
        raise SubmissionError("non-numeric, NaN or infinite score")
    if ((score < 0) | (score > 1)).any():
        raise SubmissionError("score outside [0, 1]")
    return pd.Series(score, index=ids.to_numpy())


def ref_score(sub: pd.DataFrame, truth: pd.DataFrame, sample_ids, policy=POLICY, scored_ids=None) -> float:
    """One global AP over the scored rows of one leaderboard part (excluded classes removed)."""
    s = ref_validate(sub, sample_ids)
    al = policy["aliases"]
    t = truth[~truth["class_name"].map(lambda n: al.get(n, n)).isin(policy["test_excluded_classes"])]
    if scored_ids is not None:
        t = t[t["id"].isin(scored_ids)]
    return ref_ap(t["error"].to_numpy(), s.loc[t["id"].astype(str)].to_numpy())


def _boxes(rows, kind):
    cols = ["frame_id", "class_name", "x1", "y1", "x2", "y2"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _truth(n=12, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"id{i:03d}" for i in range(n)]
    err = (rng.random(n) < 0.5).astype(int)
    err[0], err[1] = 1, 0
    return pd.DataFrame({"id": ids, "class_name": rng.choice(["person", "cup", "nude_hands"], n), "error": err})


# =============================================================================== AP semantics
def test_perfect_ranking_gives_ap_one():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.0])
    assert ref_ap(y, s) == pytest.approx(1.0) and average_precision_score(y, s) == pytest.approx(1.0)


def test_reversed_ranking_gives_the_known_floor_not_zero():
    n, p = 12, 5
    y = np.r_[np.ones(p), np.zeros(n - p)].astype(int)
    s = np.r_[np.linspace(0.0, 0.4, p), np.linspace(0.5, 1.0, n - p)]  # distinct scores, positives last
    floor = np.mean([k / (n - p + k) for k in range(1, p + 1)])
    assert ref_ap(y, s) == pytest.approx(floor) and average_precision_score(y, s) == pytest.approx(floor)
    tied_bottom = np.r_[np.zeros(p), np.linspace(0.5, 1.0, n - p)]  # positives tied -> one block -> prevalence
    assert ref_ap(y, tied_bottom) == pytest.approx(p / n) == average_precision_score(y, tied_bottom)


@pytest.mark.parametrize("c", [0.0, 0.5, 1.0])
def test_constant_scores_give_the_prevalence(c):
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    assert ref_ap(y, np.full(8, c)) == pytest.approx(3 / 8)
    assert average_precision_score(y, np.full(8, c)) == pytest.approx(3 / 8)


def test_ties_enter_as_one_block_like_sklearn():
    y, s = [1, 1, 0, 1, 0], [0.8, 0.6, 0.6, 0.6, 0.1]
    expected = 1 / 3 * 1.0 + 2 / 3 * 0.75  # block {0.6}: 3 rows, 2 positives -> precision 3/4 at recall 1
    assert ref_ap(y, s) == pytest.approx(expected) and average_precision_score(y, s) == pytest.approx(expected)
    assert expected not in (pytest.approx(1.0), pytest.approx(np.mean([1, 2 / 3, 3 / 4])))  # not a tie-break


@pytest.mark.parametrize("seed", range(6))
def test_reference_ap_equals_sklearn_on_random_and_tied_inputs(seed):
    rng = np.random.default_rng(seed)
    for decimals in (None, 2, 1, 0):
        n = int(rng.integers(2, 300))
        y = (rng.random(n) < rng.uniform(0.05, 0.95)).astype(int)
        y[0] = 1
        s = rng.random(n) if decimals is None else np.round(rng.random(n), decimals)
        assert abs(ref_ap(y, s) - average_precision_score(y, s)) < 1e-12


def test_orientation_higher_score_means_more_suspicious_and_error_is_positive():
    conf = np.array([0.95, 0.9, 0.85, 0.8, 0.4, 0.3, 0.2, 0.1])
    error = np.array([0, 0, 0, 1, 1, 1, 1, 0])  # low confidence tends to be wrong
    good, wrong = ref_ap(error, 1 - conf), ref_ap(error, conf)
    assert good > error.mean() > wrong
    assert ref_ap(1 - error, 1 - conf) != pytest.approx(good)  # the positive class matters


def test_global_ap_per_part_is_not_a_mean_of_group_aps():
    y = np.array([1, 0, 0, 0, 1, 1, 0, 1])
    s = np.array([0.9, 0.8, 0.1, 0.05, 0.4, 0.35, 0.3, 0.2])
    g = np.array([0] * 4 + [1] * 4)
    mean_groups = np.mean([ref_ap(y[g == k], s[g == k]) for k in (0, 1)])
    assert ref_ap(y, s) != pytest.approx(mean_groups)


# =============================================================================== scope and submission format
def test_excluded_class_rows_are_removed_from_scoring():
    truth = _truth()
    truth.loc[[2, 5, 7], "class_name"] = "table"
    sub = truth[["id"]].assign(score=np.linspace(0.0, 1.0, len(truth)))
    a = ref_score(sub, truth, truth["id"])
    sub2 = sub.copy()
    sub2.loc[[2, 5, 7], "score"] = [1.0, 0.0, 0.5]
    keep = truth["class_name"] != "table"
    assert ref_score(sub2, truth, truth["id"]) == a == ref_ap(truth.loc[keep, "error"], sub.loc[keep, "score"])


def test_unscored_rows_do_not_influence_the_score():
    truth = _truth(20, seed=3)
    scored = set(truth["id"].iloc[:12])
    sub = truth[["id"]].assign(score=np.linspace(0.1, 0.9, 20))
    sub2 = sub.copy()
    sub2.loc[12:, "score"] = 1.0
    assert ref_score(sub, truth, truth["id"], scored_ids=scored) == ref_score(sub2, truth, truth["id"], scored_ids=scored)


def test_duplicate_ids_are_rejected():
    truth = _truth()
    sub = pd.concat([truth[["id"]].assign(score=0.5), truth[["id"]].iloc[:1].assign(score=0.1)])
    with pytest.raises(SubmissionError, match="duplicated"):
        ref_score(sub, truth, truth["id"])


def test_missing_ids_are_rejected():
    truth = _truth()
    with pytest.raises(SubmissionError, match="missing"):
        ref_score(truth[["id"]].iloc[1:].assign(score=0.5), truth, truth["id"])


def test_unknown_ids_and_wrong_columns_are_rejected():
    truth = _truth()
    extra = pd.concat([truth[["id"]].assign(score=0.5), pd.DataFrame({"id": ["zzz"], "score": [0.5]})])
    with pytest.raises(SubmissionError, match="unknown"):
        ref_score(extra, truth, truth["id"])
    with pytest.raises(SubmissionError, match="columns"):
        ref_score(truth[["id"]].assign(prob=0.5), truth, truth["id"])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -1e-9, 1.0000001, 2.0, "abc", None])
def test_nan_inf_and_out_of_range_scores_are_rejected(bad):
    truth = _truth()
    sub = truth[["id"]].assign(score=0.5).astype({"score": object})
    sub.loc[3, "score"] = bad
    with pytest.raises(SubmissionError):
        ref_score(sub, truth, truth["id"])


def test_bounds_zero_and_one_are_accepted():
    truth = _truth()
    sub = truth[["id"]].assign(score=np.where(truth["error"] == 1, 1.0, 0.0))
    assert ref_score(sub, truth, truth["id"]) == pytest.approx(1.0)


def test_row_order_does_not_change_the_score():
    truth = _truth(40, seed=7)
    sub = truth[["id"]].assign(score=np.round(np.random.default_rng(1).random(40), 1))  # with ties
    shuffled = sub.sample(frac=1.0, random_state=5).reset_index(drop=True)
    assert ref_score(sub, truth, truth["id"]) == ref_score(shuffled, truth, truth["id"])


# =============================================================================== label rule on synthetic frames
F = "frame_A"


def test_one_to_one_matching_two_objects_two_predictions():
    gts = _boxes([(F, "person", 0, 0, 10, 10), (F, "person", 50, 0, 60, 10)], "g")
    preds = _boxes([(F, "person", 51, 0, 61, 10), (F, "person", 1, 0, 11, 10)], "p")
    assert ref_labels(preds, gts).tolist() == [0, 0]


def test_duplicate_detection_is_an_error():
    gts = _boxes([(F, "cup", 0, 0, 10, 10)], "g")
    preds = _boxes([(F, "cup", 1, 0, 11, 10), (F, "cup", 0, 0, 10, 10)], "p")
    assert ref_labels(preds, gts).tolist() == [1, 0]  # the better-IoU copy is matched, the duplicate is wrong


def test_class_mismatch_is_an_error_and_matching_is_per_frame():
    gts = _boxes([(F, "bottle", 0, 0, 10, 10), ("frame_B", "cup", 0, 0, 10, 10)], "g")
    preds = _boxes([(F, "cup", 0, 0, 10, 10)], "p")
    assert ref_labels(preds, gts).tolist() == [1]


def test_prediction_without_any_manual_box_is_an_error():
    assert ref_labels(_boxes([(F, "tray", 0, 0, 10, 10)], "p"), _boxes([], "g")).tolist() == [1]


def test_iou_exactly_060_is_correct_and_just_below_is_an_error():
    gts = _boxes([(F, "person", 0, 0, 10, 10)], "g")
    exact = _boxes([(F, "person", 2.5, 0, 12.5, 10)], "p")
    assert ref_iou(exact[["x1", "y1", "x2", "y2"]], gts[["x1", "y1", "x2", "y2"]])[0, 0] == 0.6
    assert ref_labels(exact, gts).tolist() == [0]
    below = _boxes([(F, "person", 2.5001, 0, 12.5001, 10)], "p")
    assert ref_labels(below, gts).tolist() == [1]


@pytest.mark.parametrize("pred_cls,gt_cls", [("nude_hand", "nude_hands"), ("nude_hands", "nude_hand"),
                                             ("white_napkin", "napkin"), ("napkin", "white_napkin")])
def test_aliases_are_applied_before_matching(pred_cls, gt_cls):
    gts = _boxes([(F, gt_cls, 0, 0, 10, 10)], "g")
    preds = _boxes([(F, pred_cls, 0, 0, 10, 10)], "p")
    assert ref_labels(preds, gts).tolist() == [0]
    no_alias = dict(POLICY, aliases={})
    assert ref_labels(preds, gts, no_alias).tolist() == [1]


def test_maximum_cardinality_where_greedy_by_iou_fails():
    gts = _boxes([(F, "cup", 10, 0, 20, 10), (F, "cup", 12, 0, 22, 10)], "g")
    preds = _boxes([(F, "cup", 10.5, 0, 20.5, 10), (F, "cup", 8, 0, 18, 10)], "p")
    assert ref_labels(preds, gts).tolist() == [0, 0]  # P1-G2 (0.739) + P2-G1 (0.667)
    assert ref_labels(preds, gts, matcher=greedy_iou_match).tolist() == [0, 1]  # greedy takes P1-G1 (0.905)


def test_number_of_pairs_comes_before_total_iou():
    s = 3.5 / 1.65  # shift of width-10 boxes giving IoU 0.65
    preds = _boxes([(F, "tray", k * s, 0, k * s + 10, 10) for k in range(3)], "p")
    gts = _boxes([(F, "tray", k * s, 0, k * s + 10, 10) for k in range(1, 4)], "g")
    assert ref_labels(preds, gts).tolist() == [0, 0, 0]  # 3 pairs, total IoU 1.95
    assert ref_labels(preds, gts, matcher=max_total_iou_match).tolist() == [1, 0, 0]  # 2 pairs, total 2.00


# =============================================================================== real data (train only)
def _need_real():
    if not (RAW / "train.csv").is_file():
        pytest.skip("real competition data is not available")


def test_real_policy_file_matches_the_reference_constants():
    _need_real()
    policy = json.loads((RAW / "scoring_policy.json").read_text(encoding="utf-8"))
    for key, value in POLICY.items():
        assert policy[key] == value, key
    assert policy["positive_label"] == "error" and "average_precision_score" in policy["metric"]


def test_real_label_rule_reproduces_train_error_on_every_row():
    _need_real()
    train = pd.read_csv(RAW / "train.csv", dtype={"id": str, "frame_id": str})
    hb = pd.read_csv(RAW / "human_boxes_train.csv", dtype={"frame_id": str})
    wh = pd.read_csv(RAW / "train_frames.csv", dtype={"frame_id": str}).set_index("frame_id")
    hb = hb.join(wh[["image_width", "image_height"]], on="frame_id")
    hb["x1"] = (hb["cx"] - hb["w"] / 2) * hb["image_width"]
    hb["x2"] = (hb["cx"] + hb["w"] / 2) * hb["image_width"]
    hb["y1"] = (hb["cy"] - hb["h"] / 2) * hb["image_height"]
    hb["y2"] = (hb["cy"] + hb["h"] / 2) * hb["image_height"]
    labels = ref_labels(train, hb)
    assert len(train) == 670
    assert int((labels == train["error"].to_numpy()).sum()) == 670


def test_real_baseline_day_aps_and_pooled_ap():
    _need_real()
    train = pd.read_csv(RAW / "train.csv", dtype={"id": str, "frame_id": str})
    frames = pd.read_csv(RAW / "train_frames.csv", dtype={"frame_id": str})
    frames["d"] = pd.to_datetime(frames["timestamp"], utc=True, format="mixed").dt.date
    v = train[train["class_name"] != "table"].merge(frames[["frame_id", "d"]], on="frame_id", validate="many_to_one")
    day_ap = [average_precision_score(p["error"], 1 - p["confidence"]) for _, p in v.groupby("d")]
    assert [round(a, 4) for a in day_ap] == [0.8607, 0.7648]
    assert round(float(np.mean(day_ap)), 4) == 0.8128
    assert round(ref_ap(v["error"], 1 - v["confidence"]), 4) == 0.7892


def test_real_sample_submission_covers_exactly_the_test_rows_and_no_table():
    _need_real()
    test = pd.read_csv(RAW / "test.csv", dtype={"id": str})
    sample = pd.read_csv(RAW / "sample_submission.csv", dtype={"id": str})
    assert list(sample.columns) == ["id", "score"] and sample["id"].is_unique
    assert set(sample["id"]) == set(test["id"]) and len(sample) == len(test) == 362
    assert "error" not in test.columns and not (test["class_name"] == "table").any()
    ref_validate(test[["id"]].assign(score=1 - test["confidence"]), sample["id"])  # the notebook baseline is valid


# =============================================================================== explicit comparison with the project
def test_project_official_ap_equals_the_reference():
    metrics = pytest.importorskip("rascar.metrics")
    rng = np.random.default_rng(11)
    for t in range(300):
        n = int(rng.integers(2, 200))
        y = (rng.random(n) < 0.5).astype(int)
        y[0] = 1
        s = np.round(rng.random(n), int(rng.integers(0, 4)))
        cls = rng.choice(["person", "cup", "table"], n)
        cls[0] = "person"
        frame = pd.DataFrame({"score": s, "error": y, "class_name": cls})
        keep = cls != "table"
        assert abs(metrics.official_ap(frame, POLICY) - ref_ap(y[keep], s[keep])) < 1e-12


def test_project_submission_checks_reject_the_same_defects():
    metrics = pytest.importorskip("rascar.metrics")
    truth = _truth()
    good = truth[["id"]].assign(score=np.linspace(0, 1, len(truth)))
    assert metrics.official_ap_from_submission(good, truth, POLICY) == pytest.approx(ref_score(good, truth, truth["id"]))
    bad_subs = [pd.concat([good, good.iloc[:1]]), good.iloc[1:],
                pd.concat([good, pd.DataFrame({"id": ["zzz"], "score": [0.5]})]),
                good.assign(score=np.r_[np.nan, good["score"].iloc[1:]]), good.assign(score=good["score"] + 0.5)]
    for bad in bad_subs:
        with pytest.raises(ValueError):
            metrics.official_ap_from_submission(bad, truth, POLICY)
        with pytest.raises(SubmissionError):
            ref_score(bad, truth, truth["id"])
