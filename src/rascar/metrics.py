"""Competition metric and the diagnostics built around it.

The official metric is ``sklearn.metrics.average_precision_score`` on the binary
label ``error`` (1 = erroneous box); a higher score must mean "more suspicious".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def average_precision(y_true, score) -> float:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    if y_true.size == 0 or y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, score))


class ScoringError(ValueError):
    """A prediction table that the official scorer could not accept."""


def official_ap(table: pd.DataFrame, policy: dict | None = None, score_col: str = "score",
                label_col: str = "error", class_col: str = "class_name") -> float:
    """Official-compatible metric (TASK.txt, "Метрика и формат отправки").

    One global ``sklearn.metrics.average_precision_score`` over all scored rows: no per-class,
    per-day or per-frame averaging and no weights. Positive class = ``error == 1``; a higher score
    must mean "more suspicious". Rows of the classes listed in ``test_excluded_classes`` (``table``)
    are outside the scored scope and are dropped when a class column is present. Scores must be
    finite numbers in [0, 1] ("score — конечное число от 0 до 1"); anything else raises
    :class:`ScoringError` instead of being silently repaired. Ties are handled by sklearn (all
    rows sharing a score enter the ranking together). Returns NaN when no positive row exists,
    because AP is undefined there.
    """
    if score_col not in table or label_col not in table:
        raise ScoringError(f"table needs columns {score_col!r} and {label_col!r}")
    frame = table
    if class_col in frame:
        excluded = set((policy or {}).get("test_excluded_classes", ["table"]))
        frame = frame[~frame[class_col].isin(excluded)]
    score = pd.to_numeric(frame[score_col], errors="coerce").to_numpy(dtype=float)
    if np.isnan(score).any():
        raise ScoringError("scores contain NaN or non-numeric values")
    if not np.isfinite(score).all():
        raise ScoringError("scores contain infinite values")
    if ((score < 0) | (score > 1)).any():
        raise ScoringError("scores outside [0, 1]")
    label = frame[label_col].to_numpy()
    if not np.isin(label, (0, 1)).all():
        raise ScoringError("labels must be 0 / 1")
    return average_precision(label.astype(int), score)


def official_ap_from_submission(submission: pd.DataFrame, truth: pd.DataFrame, policy: dict | None = None) -> float:
    """Score a submission (``id, score``) against a truth table (``id, error`` and optionally
    ``class_name``). Every truth id must occur exactly once in the submission; extra ids are an error."""
    if list(submission.columns) != ["id", "score"]:
        raise ScoringError(f"submission columns must be exactly ['id', 'score'], got {list(submission.columns)}")
    sub_ids = submission["id"].astype(str)
    if not sub_ids.is_unique:
        raise ScoringError("duplicated ids in the submission")
    truth_ids = truth["id"].astype(str)
    if set(truth_ids) - set(sub_ids):
        raise ScoringError(f"{len(set(truth_ids) - set(sub_ids))} ids are missing from the submission")
    if set(sub_ids) - set(truth_ids):
        raise ScoringError(f"{len(set(sub_ids) - set(truth_ids))} unknown ids in the submission")
    merged = truth.assign(id=truth_ids).merge(submission.assign(id=sub_ids), on="id", how="left", validate="one_to_one")
    return official_ap(merged, policy)


def precision_at_k(y_true, score, k: int) -> float:
    """Precision among the ``k`` highest scores; ties are broken pessimistically-neutral
    by a stable sort on the original order (reported only as a diagnostic)."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    if y_true.size == 0:
        return float("nan")
    k = int(min(k, y_true.size))
    order = np.argsort(-score, kind="stable")[:k]
    return float(y_true[order].mean())


def per_class_ap(df: pd.DataFrame, score_col: str, label_col: str = "error",
                 class_col: str = "class_name") -> dict:
    out = {}
    for cls, part in df.groupby(class_col):
        out[str(cls)] = {
            "n": int(len(part)),
            "errors": int(part[label_col].sum()),
            "prevalence": float(part[label_col].mean()),
            "ap": average_precision(part[label_col], part[score_col]),
        }
    return out


def _group_index(groups) -> list[np.ndarray]:
    groups = np.asarray(groups)
    _, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.flatnonzero(np.diff(inverse[order])) + 1
    return np.split(order, bounds)


def bootstrap_ap(y_true, score, groups, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> dict:
    """Cluster (frame-block) bootstrap of AP: whole frames are resampled."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    blocks = _group_index(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        if y_true[idx].sum() == 0:
            continue
        vals.append(average_precision_score(y_true[idx], score[idx]))
    vals = np.asarray(vals)
    if vals.size < 2:        # e.g. a group without any positive row: AP and its interval are undefined
        nan = float("nan")
        return {"ap": average_precision(y_true, score), "boot_mean": nan, "boot_std": nan, "ci_low": nan,
                "ci_high": nan, "n_boot": int(vals.size)}
    return {
        "ap": average_precision(y_true, score),
        "boot_mean": float(vals.mean()),
        "boot_std": float(vals.std(ddof=1)),
        "ci_low": float(np.quantile(vals, alpha / 2)),
        "ci_high": float(np.quantile(vals, 1 - alpha / 2)),
        "n_boot": int(len(vals)),
    }


def paired_bootstrap_delta(y_true, score_a, score_b, groups, n_boot: int = 2000,
                           seed: int = 0, alpha: float = 0.05) -> dict:
    """Frame-block bootstrap of ``AP(a) - AP(b)`` on identical resamples."""
    y_true = np.asarray(y_true).astype(int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    blocks = _group_index(groups)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        if y_true[idx].sum() == 0:
            continue
        deltas.append(average_precision_score(y_true[idx], score_a[idx])
                      - average_precision_score(y_true[idx], score_b[idx]))
    deltas = np.asarray(deltas)
    return {
        "delta": average_precision(y_true, score_a) - average_precision(y_true, score_b),
        "boot_mean": float(deltas.mean()),
        "ci_low": float(np.quantile(deltas, alpha / 2)),
        "ci_high": float(np.quantile(deltas, 1 - alpha / 2)),
        "p_a_better": float((deltas > 0).mean()),
        "n_boot": int(len(deltas)),
    }


def decile_table(y_true, score, key, bins: int = 10) -> pd.DataFrame:
    """Error prevalence by quantile bin of ``key`` (e.g. detector confidence)."""
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "score": np.asarray(score, dtype=float),
                       "key": np.asarray(key, dtype=float)})
    df["bin"] = pd.qcut(df["key"], q=bins, duplicates="drop")
    return (df.groupby("bin", observed=True)
              .agg(n=("y", "size"), errors=("y", "sum"), error_rate=("y", "mean"),
                   mean_score=("score", "mean"))
              .reset_index())


def summary(y_true, score, ks=(20, 50, 100)) -> dict:
    y_true = np.asarray(y_true).astype(int)
    out = {
        "n": int(y_true.size),
        "errors": int(y_true.sum()),
        "prevalence": float(y_true.mean()) if y_true.size else float("nan"),
        "ap": average_precision(y_true, score),
    }
    for k in ks:
        out[f"p_at_{k}"] = precision_at_k(y_true, score, k)
    return out
