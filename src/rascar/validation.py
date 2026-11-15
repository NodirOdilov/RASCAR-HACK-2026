"""Leakage-controlled validation schemes.

Primary scheme (prescribed by TASK.txt and the organizer notebook): split the
labelled rows by UTC day - fit on one day, validate on the other, then swap.
With two labelled days this gives exactly two folds; no additional independent
day exists and none is claimed.

Secondary scheme (optimistic, used only as a noise/sanity check): repeated
group K-fold where a whole frame never straddles the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics


@dataclass(frozen=True)
class Fold:
    name: str
    train_idx: np.ndarray
    valid_idx: np.ndarray


def day_folds(days: pd.Series) -> list[Fold]:
    """Leave-one-UTC-day-out folds over positional indices."""
    days = pd.Series(np.asarray(days))
    uniq = sorted(days.unique())
    if len(uniq) < 2:
        raise ValueError("at least two labelled UTC days are required for a day-level split")
    folds = []
    for day in uniq:
        valid = np.flatnonzero(days.to_numpy() == day)
        train = np.flatnonzero(days.to_numpy() != day)
        folds.append(Fold(name=f"valid_{day}", train_idx=train, valid_idx=valid))
    return folds


def frame_group_folds(frame_ids: pd.Series, n_splits: int = 5, repeats: int = 3, seed: int = 0) -> list[Fold]:
    """Repeated K-fold over frames (all rows of a frame stay together)."""
    frame_ids = np.asarray(frame_ids)
    uniq = np.unique(frame_ids)
    rng = np.random.default_rng(seed)
    folds = []
    for rep in range(repeats):
        perm = rng.permutation(uniq)
        for k, chunk in enumerate(np.array_split(perm, n_splits)):
            valid_mask = np.isin(frame_ids, chunk)
            folds.append(Fold(name=f"rep{rep}_fold{k}", train_idx=np.flatnonzero(~valid_mask),
                              valid_idx=np.flatnonzero(valid_mask)))
    return folds


def assert_no_overlap(folds: list[Fold], frame_ids, days=None) -> None:
    """Hard guarantees: no row, no frame (and for day folds no day) on both sides."""
    frame_ids = np.asarray(frame_ids)
    for fold in folds:
        if np.intersect1d(fold.train_idx, fold.valid_idx).size:
            raise AssertionError(f"{fold.name}: row overlap")
        if np.intersect1d(frame_ids[fold.train_idx], frame_ids[fold.valid_idx]).size:
            raise AssertionError(f"{fold.name}: frame overlap")
        if days is not None:
            d = np.asarray(days)
            if np.intersect1d(d[fold.train_idx], d[fold.valid_idx]).size:
                raise AssertionError(f"{fold.name}: day overlap")


def evaluate_day_oof(y, oof, days, frame_ids, classes=None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Score out-of-fold predictions produced with :func:`day_folds`.

    ``mean_day_ap`` (the organizers' "mean AP over two days") is the selection metric;
    ``pooled_ap`` concatenates both validation days and is reported for reference
    (it additionally depends on how well scores of the two fold models are calibrated
    against each other).
    """
    y = np.asarray(y).astype(int)
    oof = np.asarray(oof, dtype=float)
    days = np.asarray(days)
    frame_ids = np.asarray(frame_ids)
    out: dict = {"per_day": {}}
    for day in sorted(np.unique(days)):
        m = days == day
        entry = metrics.summary(y[m], oof[m])
        entry.update({k: v for k, v in metrics.bootstrap_ap(y[m], oof[m], frame_ids[m], n_boot=n_boot, seed=seed).items()
                      if k in ("ci_low", "ci_high", "boot_std")})
        out["per_day"][str(day)] = entry
    out["mean_day_ap"] = float(np.mean([v["ap"] for v in out["per_day"].values()]))
    pooled = metrics.summary(y, oof)
    pooled.update({k: v for k, v in metrics.bootstrap_ap(y, oof, frame_ids, n_boot=n_boot, seed=seed).items()
                   if k in ("ci_low", "ci_high", "boot_std")})
    out["pooled"] = pooled
    if classes is not None:
        df = pd.DataFrame({"error": y, "score": oof, "class_name": np.asarray(classes)})
        out["per_class"] = metrics.per_class_ap(df, "score")
    return out
