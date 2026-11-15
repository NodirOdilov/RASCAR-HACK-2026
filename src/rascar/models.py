"""Model zoo. Every scorer returns P(error) in [0, 1]; higher = more suspicious.

A scorer is described by a plain dict (the JSON files in ``configs/``):

    {"name": "...", "kind": "catboost", "features": {"groups": ["conf", "geom"]} | {"columns": [...]},
     "params": {...}, "seeds": [0, 1, 2]}

``kind`` is one of ``baseline``, ``logreg``, ``catboost``, ``lgbm``, ``hgb``, ``iou_reg``, ``blend``.
All preprocessing statistics (imputation medians, scaling) are fitted inside
``fit`` on the rows passed to it, i.e. on the training fold only.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features as F

CAT_COL = "cls"
ONEHOT = [f"is_{c}" for c in F.CLASSES]


def resolve_columns(spec: dict) -> list[str]:
    feat = spec.get("features", {})
    cols: list[str] = []
    if "groups" in feat:
        cols.extend(F.columns_for(feat["groups"]))
    for c in feat.get("columns", []):
        if c not in cols:
            cols.append(c)
    for c in feat.get("drop", []):
        if c in cols:
            cols.remove(c)
    return cols


def raw_inputs(spec: dict) -> list[str]:
    """Columns of the feature table that a configuration reads (a derived input is replaced by its sources)."""
    if spec.get("kind") == "blend":
        return sorted({c for member in spec["members"] for c in raw_inputs(member)})
    derived = {item["name"]: list(item["columns"]) for item in spec.get("derived", [])}
    cols: list[str] = []
    for c in resolve_columns(spec):
        cols.extend(derived.get(c, [c]))
    return sorted(set(cols))


class Scorer:
    """Common interface."""

    def __init__(self, spec: dict):
        self.spec = copy.deepcopy(spec)
        self.columns = resolve_columns(spec)
        self.params = dict(spec.get("params", {}))
        self.seeds = list(spec.get("seeds", [0]))

    def fit(self, X: pd.DataFrame, y: np.ndarray, aux: pd.DataFrame | None = None) -> "Scorer":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def _check(self, X: pd.DataFrame) -> None:
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"{self.spec.get('name')}: feature columns missing from the input: {missing}")

    def _prepare(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Clip inputs (``spec["clip"]``), add the derived inputs (``spec["derived"]``), verify that every input exists.

        ``{"edge_rel": [0.0, 12.0]}`` in ``spec["clip"]`` bounds an input by constants that do not depend on any data
        (a degenerate box would otherwise produce an ``edge_rel`` of several hundred and an arbitrary score).

        ``{"name": "cons_soft", "op": "mean_z", "columns": [a, b, ...]}`` builds the consensus of several
        second opinions: each column is standardised with the mean / std of the TRAINING rows (stored at fit
        time, re-used at predict time) and the standardised values are averaged. A missing opinion (NaN) is
        skipped, so the consensus falls back to the remaining ones.
        """
        clip = self.spec.get("clip", {})
        if clip:                                   # fixed, label-free bounds, e.g. {"edge_rel": [0.0, 12.0]}
            X = X.copy()
            for col, (low, high) in clip.items():
                if col in X.columns:
                    X[col] = X[col].astype(float).clip(lower=low, upper=high)      # NaN stays NaN
        derived = self.spec.get("derived", [])
        if derived:
            X = X.copy()
            if fit:
                self.derived_stats_ = {}
            for item in derived:
                cols = item["columns"]
                absent = [c for c in cols if c not in X.columns]
                if absent:
                    raise KeyError(f"{self.spec.get('name')}: columns of derived input {item['name']!r} are missing: {absent}")
                if item["op"] != "mean_z":
                    raise KeyError(f"unknown derived op {item['op']!r}")
                values = X[cols].astype(float)
                if fit:
                    self.derived_stats_[item["name"]] = [(float(values[c].mean()), float(values[c].std(ddof=0)) + 1e-9) for c in cols]
                stats = self.derived_stats_[item["name"]]
                zs = np.column_stack([(values[c].to_numpy() - m) / s for c, (m, s) in zip(cols, stats)])
                all_nan = np.isnan(zs).all(axis=1)
                X[item["name"]] = np.where(all_nan, np.nan, np.nanmean(np.where(all_nan[:, None], 0.0, zs), axis=1))
        self._check(X)
        return X


class ConfidenceBaseline(Scorer):
    """Organizer baseline: score = 1 - confidence. Needs no training."""

    def __init__(self, spec: dict):
        super().__init__(spec)
        self.columns = ["conf"]

    def fit(self, X, y, aux=None):
        return self

    def predict(self, X):
        X = self._prepare(X, fit=False)
        return np.clip(1.0 - X["conf"].to_numpy(dtype=float), 0.0, 1.0)


class FormulaScorer(Scorer):
    """Parameter-free score ``sigmoid(-x)`` of one logit-scale feature.

    With ``tc_logit`` (mean logit-confidence over the detection and its re-detections in
    the previous / next context frame) this is "1 - temporally smoothed confidence".
    Nothing is fitted, so there is nothing to overfit and nothing to leak.
    """

    def __init__(self, spec: dict):
        super().__init__(spec)
        self.column = self.params.get("column", "tc_logit")
        self.columns = [self.column]

    def fit(self, X, y, aux=None):
        return self

    def predict(self, X):
        X = self._prepare(X, fit=False)
        return 1.0 / (1.0 + np.exp(X[self.column].to_numpy(dtype=float)))


def _numeric(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Replace the categorical class column by its one-hot columns (for linear models)."""
    cols = []
    for c in columns:
        if c == CAT_COL:
            cols.extend(o for o in ONEHOT if o not in cols)
        elif c not in cols:
            cols.append(c)
    return X[cols].astype(float)


class LogRegScorer(Scorer):
    def fit(self, X, y, aux=None):
        X = self._prepare(X, fit=True)
        self.model_ = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=bool(self.params.get("missing_indicator", False))),
            StandardScaler(),
            LogisticRegression(C=float(self.params.get("C", 1.0)), max_iter=5000,
                               class_weight=self.params.get("class_weight")),
        )
        self.model_.fit(_numeric(X, self.columns), np.asarray(y).astype(int))
        return self

    def predict(self, X):
        X = self._prepare(X, fit=False)
        return self.model_.predict_proba(_numeric(X, self.columns))[:, 1]


class CatBoostScorer(Scorer):
    def fit(self, X, y, aux=None):
        from catboost import CatBoostClassifier

        X = self._prepare(X, fit=True)
        cat = [c for c in self.columns if c == CAT_COL]
        # one-hot instead of target statistics for the 4-level class column: no label-derived
        # encoding at all (and ~10x faster on ~200 rows than the default ordered CTR mode)
        defaults = dict(depth=3, iterations=300, learning_rate=0.03, l2_leaf_reg=5.0, verbose=0,
                        thread_count=1, allow_writing_files=False, boosting_type="Plain",
                        one_hot_max_size=8, border_count=32)
        defaults.update(self.params)
        self.models_ = []
        for seed in self.seeds:
            model = CatBoostClassifier(random_seed=int(seed), cat_features=cat, **defaults)
            model.fit(X[self.columns], np.asarray(y).astype(int))
            self.models_.append(model)
        return self

    def predict(self, X):
        X = self._prepare(X, fit=False)
        return np.mean([m.predict_proba(X[self.columns])[:, 1] for m in self.models_], axis=0)

    def feature_importance(self) -> pd.Series:
        imp = np.mean([m.get_feature_importance() for m in self.models_], axis=0)
        return pd.Series(imp, index=self.columns).sort_values(ascending=False)


def _encode_cat(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = X[columns].copy()
    if CAT_COL in out.columns:
        out[CAT_COL] = pd.Categorical(out[CAT_COL], categories=F.CLASSES).codes.astype(float)
    return out.astype(float)


class LGBMScorer(Scorer):
    def fit(self, X, y, aux=None):
        import lightgbm as lgb

        X = self._prepare(X, fit=True)
        defaults = dict(n_estimators=200, learning_rate=0.03, num_leaves=4, max_depth=3, min_child_samples=8,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=2)
        defaults.update(self.params)
        self.models_ = []
        for seed in self.seeds:
            model = lgb.LGBMClassifier(random_state=int(seed), **defaults)
            model.fit(_encode_cat(X, self.columns), np.asarray(y).astype(int))
            self.models_.append(model)
        return self

    def predict(self, X):
        X = self._prepare(X, fit=False)
        return np.mean([m.predict_proba(_encode_cat(X, self.columns))[:, 1] for m in self.models_], axis=0)


class ResidualGBMScorer(Scorer):
    """Boosting *on top of* the confidence prior.

    Step 1: one-dimensional logistic regression on ``prior`` (default ``tc_logit``) gives a
    calibrated offset that keeps the fine-grained confidence ordering.
    Step 2: small, strongly regularised LightGBM trees are boosted from that offset
    (``init_score``) and can only add corrections. Optional monotone constraints encode
    prior knowledge (higher confidence can never increase the error probability).
    """

    def fit(self, X, y, aux=None):
        import lightgbm as lgb

        X = self._prepare(X, fit=True)
        y = np.asarray(y).astype(int)
        self.prior_col = self.params.get("prior", "tc_logit")
        self.prior_ = LogisticRegression(C=1e3, max_iter=2000).fit(X[[self.prior_col]].astype(float), y)
        offset = self.prior_.decision_function(X[[self.prior_col]].astype(float))
        mono = self.params.get("monotone", {})
        defaults = dict(n_estimators=60, learning_rate=0.05, num_leaves=3, max_depth=2, min_child_samples=12,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0, min_split_gain=0.0,
                        verbose=-1, n_jobs=1)
        defaults.update({k: v for k, v in self.params.items() if k not in ("prior", "monotone")})
        if mono:
            defaults["monotone_constraints"] = [int(mono.get(c, 0)) for c in self.columns]
        self.models_ = []
        for seed in self.seeds:
            model = lgb.LGBMClassifier(random_state=int(seed), **defaults)
            model.fit(_encode_cat(X, self.columns), y, init_score=offset)
            self.models_.append(model)
        return self

    def predict(self, X):
        X = self._prepare(X, fit=False)
        offset = self.prior_.decision_function(X[[self.prior_col]].astype(float))
        raw = np.mean([m.predict(_encode_cat(X, self.columns), raw_score=True) for m in self.models_], axis=0)
        return 1.0 / (1.0 + np.exp(-(offset + raw)))


class HGBScorer(Scorer):
    def fit(self, X, y, aux=None):
        X = self._prepare(X, fit=True)
        defaults = dict(max_depth=3, max_iter=150, learning_rate=0.05, l2_regularization=5.0, min_samples_leaf=8)
        defaults.update(self.params)
        encoded = _encode_cat(X, self.columns)
        # sklearn's histogram binning cannot handle a column without a single observed value
        self.keep_ = [c for c in self.columns if encoded[c].notna().any()]
        self.models_ = []
        for seed in self.seeds:
            model = HistGradientBoostingClassifier(random_state=int(seed), **defaults)
            model.fit(encoded[self.keep_], np.asarray(y).astype(int))
            self.models_.append(model)
        return self

    def predict(self, X):
        X = self._prepare(X, fit=False)
        encoded = _encode_cat(X, self.columns)[self.keep_]
        return np.mean([m.predict_proba(encoded)[:, 1] for m in self.models_], axis=0)


class IoURegressionScorer(Scorer):
    """Regress the training-only auxiliary target "best IoU with a same-class manual box"
    and rank by ``1 - predicted IoU``. The manual boxes are used only as a training target
    (``aux['best_iou']``); prediction needs detector output only."""

    def fit(self, X, y, aux=None):
        from catboost import CatBoostRegressor

        X = self._prepare(X, fit=True)
        if aux is None or "best_iou" not in aux:
            raise ValueError("iou_reg needs aux['best_iou'] at fit time")
        target = np.clip(np.asarray(aux["best_iou"], dtype=float), 0.0, 1.0)
        cat = [c for c in self.columns if c == CAT_COL]
        defaults = dict(depth=3, iterations=300, learning_rate=0.03, l2_leaf_reg=5.0, verbose=0, thread_count=1,
                        allow_writing_files=False, loss_function="RMSE", boosting_type="Plain",
                        one_hot_max_size=8, border_count=32)
        defaults.update(self.params)
        self.models_ = []
        for seed in self.seeds:
            model = CatBoostRegressor(random_seed=int(seed), cat_features=cat, **defaults)
            model.fit(X[self.columns], target)
            self.models_.append(model)
        return self

    def predict(self, X):
        X = self._prepare(X, fit=False)
        iou = np.mean([m.predict(X[self.columns]) for m in self.models_], axis=0)
        return np.clip(1.0 - iou, 0.0, 1.0)


class IoURidgeScorer(Scorer):
    """Linear (ridge) version of :class:`IoURegressionScorer` for a handful of features."""

    def fit(self, X, y, aux=None):
        from sklearn.linear_model import Ridge

        X = self._prepare(X, fit=True)
        if aux is None or "best_iou" not in aux:
            raise ValueError("iou_ridge needs aux['best_iou'] at fit time")
        self.model_ = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                    Ridge(alpha=float(self.params.get("alpha", 10.0))))
        self.model_.fit(_numeric(X, self.columns), np.clip(np.asarray(aux["best_iou"], dtype=float), 0.0, 1.0))
        return self

    def predict(self, X):
        X = self._prepare(X, fit=False)
        return np.clip(1.0 - self.model_.predict(_numeric(X, self.columns)), 0.0, 1.0)


def _rank01(values: np.ndarray) -> np.ndarray:
    n = len(values)
    if n <= 1:
        return np.full(n, 0.5)
    return (rankdata(values, method="average") - 1.0) / (n - 1.0)


class BlendScorer(Scorer):
    """Weighted average of member scorers (``method``: ``mean`` of probabilities or ``rank``).

    Rank blending normalises ranks inside the batch that is being predicted, so it is
    only used on a full day / the full test set, never on single rows.
    """

    def __init__(self, spec: dict):
        super().__init__(spec)
        self.members = [make_model(m) for m in spec["members"]]
        weights = np.asarray(spec.get("weights", [1.0] * len(self.members)), dtype=float)
        self.weights = weights / weights.sum()
        self.method = spec.get("method", "mean")
        self.columns = sorted({c for m in self.members for c in m.columns})

    def fit(self, X, y, aux=None):
        for member in self.members:
            member.fit(X, y, aux)
        return self

    def member_predictions(self, X) -> np.ndarray:
        return np.vstack([m.predict(X) for m in self.members])

    def predict(self, X):
        if len(X) == 0:                                   # nothing to rank (audit finding N8)
            return np.empty(0, dtype=float)
        preds = self.member_predictions(X)
        if self.method == "rank":
            preds = np.vstack([_rank01(p) for p in preds])
        elif self.method == "logit":
            p = np.clip(preds, 1e-4, 1 - 1e-4)
            z = (np.log(p / (1 - p)) * self.weights[:, None]).sum(axis=0)
            return 1.0 / (1.0 + np.exp(-z))
        return np.clip((preds * self.weights[:, None]).sum(axis=0), 0.0, 1.0)


KINDS = {
    "baseline": ConfidenceBaseline,
    "formula": FormulaScorer,
    "lgbm_resid": ResidualGBMScorer,
    "logreg": LogRegScorer,
    "catboost": CatBoostScorer,
    "lgbm": LGBMScorer,
    "hgb": HGBScorer,
    "iou_reg": IoURegressionScorer,
    "iou_ridge": IoURidgeScorer,
    "blend": BlendScorer,
}


def make_model(spec: dict) -> Scorer:
    kind = spec.get("kind")
    if kind not in KINDS:
        raise KeyError(f"unknown model kind {kind!r}; known: {sorted(KINDS)}")
    return KINDS[kind](spec)
