"""Final Kaggle models of round 3–6 in one place: definition, fitting, serialisation and inference.

Every model here is a logistic regression on a handful of inputs of the frozen feature table
(`results/frozen_v3b/features_{train,test}.csv`, produced from the raw data and the public-detector caches by
`research/lead/v3_export_frozen.py`). Class statistics are always computed on the TRAINING rows and shrunk to the
all-class value with a pseudo-count of 10; nothing is computed from test rows, so a test row's score depends only on
that row (no transductive step).

Models
  o5c_clsstd_q4  (03, Public 0.89976)  inputs [tc_logit, cons_cls, p_tb_cut0, f_cov_same, edge_rel∈[0,12]]
  r4a_clsstd_all (05)                  inputs [cons_cls, z_c(tc_logit), z_c(p_tb_cut0), z_c(f_cov_same), z_c(edge_rel)]
where cons_cls = mean over {ovf, gdf, gbf, oLf}_same_soft of the class z-score (x − m_c)/s_c.
Additional round-6 models may register extra input builders in ``EXTRA``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

OPINIONS = ["ovf_same_soft", "gdf_same_soft", "gbf_same_soft", "oLf_same_soft"]
GEO = ["p_tb_cut0", "f_cov_same", "edge_rel"]
PSEUDO_COUNT = 10.0
C_REG = 0.5


def class_stats(train: pd.DataFrame, col: str, k: float = PSEUDO_COUNT) -> dict[str, list[float]]:
    """Class mean / std of ``col`` on the training rows, shrunk to the all-class value (ddof 0)."""
    v = train[col].astype(float)
    mu, var = float(v.mean()), float(v.var(ddof=0))
    out = {"__all__": [mu, float(np.sqrt(var) + 1e-9)]}
    for c, g in train.groupby("class_name"):
        n = len(g)
        m = (n * g[col].mean() + k * mu) / (n + k)
        s2 = (n * g[col].var(ddof=0) + k * var) / (n + k)
        out[str(c)] = [float(m), float(np.sqrt(s2) + 1e-9)]
    return out


def class_z(df: pd.DataFrame, col: str, st: dict[str, list[float]]) -> np.ndarray:
    m = df["class_name"].map(lambda c: st.get(c, st["__all__"])[0]).to_numpy(float)
    s = df["class_name"].map(lambda c: st.get(c, st["__all__"])[1]).to_numpy(float)
    return (df[col].to_numpy(float) - m) / s


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["edge_rel"] = out["edge_rel"].clip(0, 12)
    return out


# Extra input builders for round-6 models: name -> (columns needed, function(df, stats) -> 2-D array)
EXTRA: dict[str, tuple[list[str], Callable[[pd.DataFrame, dict], np.ndarray]]] = {}


@dataclass
class FinalModel:
    name: str
    stats: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    pipeline: Pipeline | None = None

    # ----- design -------------------------------------------------------------------------------------------
    def _stat_columns(self) -> list[str]:
        cols = list(OPINIONS)
        if self.name == "r4a_clsstd_all":
            cols += ["tc_logit"] + GEO
        if self.name in EXTRA:
            cols += EXTRA[self.name][0]
        return cols

    def design(self, df: pd.DataFrame) -> np.ndarray:
        df = _prep(df)
        cons = np.mean([class_z(df, c, self.stats[c]) for c in OPINIONS], axis=0)
        if self.name == "o5c_clsstd_q4" or self.name in EXTRA:
            base = np.column_stack([df["tc_logit"].to_numpy(float), cons] + [df[c].to_numpy(float) for c in GEO])
        elif self.name == "r4a_clsstd_all":
            base = np.column_stack([cons] + [class_z(df, c, self.stats[c]) for c in ["tc_logit"] + GEO])
        else:
            raise KeyError(f"unknown model {self.name}")
        if self.name in EXTRA:
            base = np.column_stack([base, EXTRA[self.name][1](df, self.stats)])
        return base

    # ----- fit / predict --------------------------------------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "FinalModel":
        tr = _prep(train)
        self.stats = {c: class_stats(tr, c) for c in self._stat_columns()}
        self.pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      LogisticRegression(C=C_REG, max_iter=5000))
        self.pipeline.fit(self.design(tr), tr["error"].to_numpy(int))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("model is not fitted")
        return np.clip(self.pipeline.predict_proba(self.design(df))[:, 1], 0.0, 1.0)

    # ----- persistence ----------------------------------------------------------------------------------------
    def save(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, folder / f"{self.name}.joblib")
        (folder / f"{self.name}.stats.json").write_text(json.dumps(self.stats, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, name: str, folder: Path) -> "FinalModel":
        folder = Path(folder)
        stats = json.loads((folder / f"{name}.stats.json").read_text(encoding="utf-8"))
        return cls(name=name, stats=stats, pipeline=joblib.load(folder / f"{name}.joblib"))


N_INSIDE_COLS = ["ovf_n_inside", "gdf_n_inside", "gbf_n_inside", "oLf_n_inside"]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


@dataclass
class PersonRefinedModel:
    """Round 7 candidate P7r: frozen 03 + a person-only refinement.

    person rows:  sigmoid(logit_03 + beta * z_person(n_inside));  every other row keeps 03's score exactly.
    z_person = mean over the four opinions of the class z-score of `<d>_n_inside` (training-row statistics, pseudo-count
    10); beta = maximum likelihood on the training rows of class person with logit_03 as a fixed offset.
    Needs the columns of ``N_INSIDE_COLS`` (``rascar.extent.n_inside_features``) next to 03's inputs.
    """

    name: str = "p7r_person_ninside"
    base: FinalModel | None = None
    stats: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    beta: float = 0.0

    def _z(self, df: pd.DataFrame) -> np.ndarray:
        z = np.mean([class_z(df, c, self.stats[c]) for c in N_INSIDE_COLS], axis=0)
        return np.where(df["class_name"].to_numpy() == "person", z, 0.0)

    def fit(self, train: pd.DataFrame) -> "PersonRefinedModel":
        from scipy.optimize import minimize_scalar
        self.base = FinalModel("o5c_clsstd_q4").fit(train)
        self.stats = {c: class_stats(train, c) for c in N_INSIDE_COLS}
        off, z = _logit(self.base.predict(train)), self._z(train)
        m = (train["class_name"] == "person").to_numpy()
        yy = train["error"].to_numpy(int)[m]

        def nll(b: float) -> float:
            eta = off[m] + b * z[m]
            return float(np.sum(np.logaddexp(0, eta) - yy * eta))

        self.beta = float(minimize_scalar(nll, bounds=(-5, 5), method="bounded").x)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.base is None:
            raise RuntimeError("model is not fitted")
        return np.clip(1 / (1 + np.exp(-(_logit(self.base.predict(df)) + self.beta * self._z(df)))), 0.0, 1.0)

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        self.base.save(folder)
        (folder / f"{self.name}.json").write_text(json.dumps({"beta": self.beta, "stats": self.stats}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, name: str = "p7r_person_ninside") -> "PersonRefinedModel":
        folder = Path(folder)
        meta = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
        return cls(name=name, base=FinalModel.load("o5c_clsstd_q4", folder), stats=meta["stats"], beta=meta["beta"])


def leave_one_day_out(name: str, train: pd.DataFrame) -> np.ndarray:
    """Out-of-day predictions for the labelled rows (validation statistic of the project)."""
    oof = np.full(len(train), np.nan)
    for day in sorted(train["utc_day"].unique()):
        held = (train["utc_day"] == day).to_numpy()
        tr_d, va_d = train[~held].reset_index(drop=True), train[held].reset_index(drop=True)
        model = PersonRefinedModel().fit(tr_d) if name == "p7r_person_ninside" else FinalModel(name).fit(tr_d)
        oof[held] = model.predict(va_d)
    return oof


def write_submission(ids: pd.Series, scores: np.ndarray, path: Path, sample_ids: pd.Series) -> None:
    """id, score in sample_submission order; refuses anything but a complete, finite, [0, 1] score set."""
    s = pd.Series(np.asarray(scores, dtype=float), index=ids.astype(str).to_numpy())
    if s.index.duplicated().any() or set(s.index) != set(sample_ids.astype(str)):
        raise ValueError("ids do not match sample_submission")
    s = s.loc[sample_ids.astype(str).to_numpy()]
    if not np.isfinite(s.to_numpy()).all() or (s < 0).any() or (s > 1).any():
        raise ValueError("scores must be finite and inside [0, 1]")
    pd.DataFrame({"id": s.index, "score": s.to_numpy()}).to_csv(path, index=False, float_format="%.10f", lineterminator="\n")
