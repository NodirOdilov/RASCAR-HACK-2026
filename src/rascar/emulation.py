"""Label-rule emulation on public-detector proposals (final round, candidate E1 ``e1_smooth_pooled``).

Idea
----
A detection is labelled correct iff a manual box of its class overlaps it with IoU >= 0.60. The manual boxes of the
test frames are unknown, but four public open-vocabulary detectors propose thousands of scored boxes per frame. Their
boxes follow a tighter convention than the manual ones (hands / cups: 6-9 % smaller on every side; persons ~2 %).

1. *Convention adapter* - four numbers per class, learnt from the MANUAL BOXES of the training frames: the median
   offset (G - P) / size(P) between a manual box G and the proposal P that represents it (arg-max score x IoU),
   pooled over the four detectors. ``A(P) = P + e_class * [w, h, w, h]``.
2. *Emulated label rule* - for an audited box D of class c and one detector:
   ``pi_t = max{ score_c(P) : IoU(D, A(P)) >= t }`` (0 if empty), ``smooth = mean_t pi_t`` over
   t in {0.50, 0.55, 0.60, 0.65, 0.70}: "is there a confident proposal that would MATCH the audited box".
3. *Model* - exactly the form of model 03 (``rascar.final_models``): logistic regression (C = 0.5) on
   [tc_logit, consensus, p_tb_cut0, f_cov_same, edge_rel clipped to 0..12]; the consensus is the mean over the four
   detectors of the class z-score of ``smooth`` (class statistics from the training rows, pseudo-count 10).

Leakage: the adapter reads manual boxes only (never ``error``); in validation the rows of a held-out day use the
adapter fitted on the other day's manual boxes (:func:`leave_one_day_out`). Test rows never contribute to any statistic.

Public weights behind the cached proposals (all Apache-2.0, used zero-shot):
  ovf  google/owlv2-base-patch16-ensemble      https://huggingface.co/google/owlv2-base-patch16-ensemble
  gdf  IDEA-Research/grounding-dino-tiny       https://huggingface.co/IDEA-Research/grounding-dino-tiny
  gbf  IDEA-Research/grounding-dino-base       https://huggingface.co/IDEA-Research/grounding-dino-base
  oLf  google/owlv2-large-patch14-ensemble     https://huggingface.co/google/owlv2-large-patch14-ensemble
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from . import paths
from .final_models import C_REG, GEO, class_stats, class_z

NAME = "e1_smooth_pooled"
CLASSES = ["person", "nude_hands", "cup", "tray"]
BOX = ["x1", "y1", "x2", "y2"]
FRAME_W, FRAME_H = 1280, 720
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)
MIN_PAIR_IOU = 0.5       # a manual box enters the adapter only if its representative proposal overlaps it this much
MIN_PAIRS = 5            # classes with fewer pairs keep the identity adapter
# opinion tag -> (cache folders under data/interim/vision, family)
OPINION_CACHES: dict[str, tuple[tuple[str, ...], str]] = {
    "ovf": (("owlv2", "owlv2_fine"), "owl"),
    "gdf": (("gdino", "gdino_fine"), "gd"),
    "gbf": (("gdinob", "gdinob_fine"), "gd"),
    "oLf": (("owlL", "owlL_fine"), "owl"),
}
SMOOTH_COLS = [f"{tag}_smooth" for tag in OPINION_CACHES]
# prompt-column layout of the caches (rascar.vision.QUERIES / rascar.vision_gdino.PHRASES)
_OWL_GROUP = np.array(["person"] * 2 + ["nude_hands"] * 2 + ["cup"] * 4 + ["tray"] * 2 + ["other"] * 14, dtype=object)
_GD_GROUP = np.array(["person"] + ["nude_hands"] + ["cup"] * 3 + ["tray"] + ["other"] * 8, dtype=object)


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-12)


class ProposalStore:
    """Cached proposals of the public detectors: ``get(tag, frame_id) -> (boxes [N, 4] px, class score [N, 4])``."""

    def __init__(self, vision_dir: Path | None = None):
        self.vision_dir = Path(vision_dir) if vision_dir else paths.interim_dir() / "vision"
        self._cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def get(self, tag: str, frame_id: str) -> tuple[np.ndarray, np.ndarray]:
        key = (tag, frame_id)
        if key in self._cache:
            return self._cache[key]
        folders, family = OPINION_CACHES[tag]
        boxes, probs = [], []
        for folder in folders:
            path = self.vision_dir / folder / f"{frame_id}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"no cached proposals of {folder} for frame {frame_id}: {path}")
            with np.load(path) as z:
                for view in sorted({k.rsplit("_", 1)[0] for k in z.files}):
                    boxes.append(z[f"{view}_boxes"].astype(np.float64))
                    if family == "owl":
                        probs.append(1.0 / (1.0 + np.exp(-z[f"{view}_logits"].astype(np.float64))))
                    else:
                        probs.append(z[f"{view}_probs"].astype(np.float64))
        b = np.concatenate(boxes); p = np.concatenate(probs)
        ok = ((b[:, 2] - b[:, 0]) > 0.5) & ((b[:, 3] - b[:, 1]) > 0.5)      # degenerate proposals dropped
        b, p = b[ok], p[ok]
        group = _OWL_GROUP if family == "owl" else _GD_GROUP
        score = np.stack([p[:, group == c].max(axis=1) for c in CLASSES], axis=1)
        self._cache[key] = (b, score)
        return self._cache[key]

    def has(self, frame_id: str) -> bool:
        return all((self.vision_dir / folder / f"{frame_id}.npz").is_file()
                   for folders, _ in OPINION_CACHES.values() for folder in folders)


def load_manual_boxes(raw: Path | None = None) -> pd.DataFrame:
    """Manual boxes of the scored non-table classes on the TRAIN frames, in pixels (training annotations)."""
    raw = Path(raw) if raw else paths.raw_dir()
    hb = pd.read_csv(raw / "human_boxes_train.csv")
    hb["class_name"] = hb["class_name"].replace({"nude_hand": "nude_hands"})
    hb = hb[hb["class_name"].isin(CLASSES)].copy()
    hb["x1"] = (hb["cx"] - hb["w"] / 2) * FRAME_W; hb["x2"] = (hb["cx"] + hb["w"] / 2) * FRAME_W
    hb["y1"] = (hb["cy"] - hb["h"] / 2) * FRAME_H; hb["y2"] = (hb["cy"] + hb["h"] / 2) * FRAME_H
    hb["utc_day"] = hb["frame_id"].str.extract(r"_(\d{4})(\d{2})(\d{2})_").agg("-".join, axis=1)
    return hb[["frame_id", "class_name", *BOX, "utc_day"]].reset_index(drop=True)


def fit_adapter(manual: pd.DataFrame, store: ProposalStore) -> dict[str, list[float]]:
    """Class-wise convention adapter pooled over the four detectors. Frames without cached proposals are skipped."""
    offsets: dict[str, list[list[float]]] = {c: [] for c in CLASSES}
    for tag in OPINION_CACHES:
        for row in manual.itertuples(index=False):
            if not store.has(row.frame_id):
                continue
            boxes, score = store.get(tag, row.frame_id)
            g = np.array([row.x1, row.y1, row.x2, row.y2], dtype=float)
            iou = iou_one_to_many(g, boxes)
            j = int((score[:, CLASSES.index(row.class_name)] * iou).argmax())
            if iou[j] < MIN_PAIR_IOU:
                continue
            p = boxes[j]; pw, ph = p[2] - p[0], p[3] - p[1]
            offsets[row.class_name].append([(g[0] - p[0]) / pw, (g[1] - p[1]) / ph, (g[2] - p[2]) / pw, (g[3] - p[3]) / ph])
    out: dict[str, list[float]] = {}
    for c in CLASSES:
        v = np.asarray(offsets[c], dtype=float).reshape(-1, 4)
        out[c] = np.median(v, axis=0).tolist() if len(v) >= MIN_PAIRS else [0.0, 0.0, 0.0, 0.0]
    return out


def adapt(boxes: np.ndarray, e) -> np.ndarray:
    w = (boxes[:, 2] - boxes[:, 0])[:, None]; h = (boxes[:, 3] - boxes[:, 1])[:, None]
    return boxes + np.asarray(e, dtype=float)[None, :] * np.concatenate([w, h, w, h], axis=1)


def smooth_agreement(targets: pd.DataFrame, adapter: dict[str, list[float]], store: ProposalStore) -> pd.DataFrame:
    """``<tag>_smooth`` for every audited box (columns id, frame_id, class_name, x1, y1, x2, y2), index preserved."""
    out = np.zeros((len(targets), len(OPINION_CACHES)))
    for i, row in enumerate(targets.itertuples(index=False)):
        d = np.array([row.x1, row.y1, row.x2, row.y2], dtype=float)
        ci = CLASSES.index(row.class_name)
        for k, tag in enumerate(OPINION_CACHES):
            boxes, score = store.get(tag, row.frame_id)
            iou = iou_one_to_many(d, adapt(boxes, adapter[row.class_name]))
            s = score[:, ci]
            vals = []
            for t in THRESHOLDS:
                m = iou >= t
                vals.append(float(s[m].max()) if m.any() else 0.0)
            out[i, k] = float(np.mean(vals))
    return pd.DataFrame(out, columns=SMOOTH_COLS, index=targets.index)


def attach_boxes(table: pd.DataFrame, part: str, raw: Path | None = None) -> pd.DataFrame:
    """Add x1..y2 of the audited boxes (raw ``train.csv`` / ``test.csv``) to a feature table keyed by ``id``."""
    raw = Path(raw) if raw else paths.raw_dir()
    boxes = pd.read_csv(raw / f"{part}.csv", dtype={"id": str})[["id", *BOX]]
    out = table.drop(columns=[c for c in BOX if c in table.columns]).merge(boxes, on="id", how="left", validate="one_to_one")
    if out[BOX].isna().any().any():
        raise ValueError("a row of the feature table has no box in the raw table")
    return out


@dataclass
class EmulationModel:
    """Candidate E1. ``fit`` needs the 03 inputs, the audited boxes (x1..y2), ``error`` and the manual boxes."""

    name: str = NAME
    adapter: dict[str, list[float]] = field(default_factory=dict)
    stats: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    pipeline: Pipeline | None = None

    def features(self, table: pd.DataFrame, store: ProposalStore) -> pd.DataFrame:
        return pd.concat([table.reset_index(drop=True), smooth_agreement(table.reset_index(drop=True), self.adapter, store)], axis=1)

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        cons = np.mean([class_z(df, c, self.stats[c]) for c in SMOOTH_COLS], axis=0)
        return np.column_stack([df["tc_logit"].to_numpy(float), cons, df["p_tb_cut0"].to_numpy(float),
                                df["f_cov_same"].to_numpy(float), df["edge_rel"].clip(0, 12).to_numpy(float)])

    def fit(self, train: pd.DataFrame, manual: pd.DataFrame, store: ProposalStore) -> "EmulationModel":
        self.adapter = fit_adapter(manual, store)
        tr = self.features(train, store)
        self.stats = {c: class_stats(tr, c) for c in SMOOTH_COLS}
        self.pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      LogisticRegression(C=C_REG, max_iter=5000))
        self.pipeline.fit(self._design(tr), tr["error"].to_numpy(int))
        return self

    def predict(self, table: pd.DataFrame, store: ProposalStore) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("model is not fitted")
        return np.clip(self.pipeline.predict_proba(self._design(self.features(table, store)))[:, 1], 0.0, 1.0)

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, folder / f"{self.name}.joblib")
        meta = {"name": self.name, "adapter": self.adapter, "stats": self.stats, "thresholds": list(THRESHOLDS),
                "opinion_caches": {k: list(v[0]) for k, v in OPINION_CACHES.items()}, "inputs": ["tc_logit", "cons_smooth", *GEO]}
        (folder / f"{self.name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, name: str = NAME) -> "EmulationModel":
        folder = Path(folder)
        meta = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
        return cls(name=name, adapter=meta["adapter"], stats=meta["stats"], pipeline=joblib.load(folder / f"{name}.joblib"))


def leave_one_day_out(train: pd.DataFrame, manual: pd.DataFrame, store: ProposalStore) -> np.ndarray:
    """Out-of-day predictions: adapter, class statistics, scaler and regression all come from the other day."""
    oof = np.full(len(train), np.nan)
    for day in sorted(train["utc_day"].unique()):
        held = (train["utc_day"] == day).to_numpy()
        model = EmulationModel().fit(train[~held].reset_index(drop=True), manual[manual["utc_day"] != day], store)
        oof[held] = model.predict(train[held].reset_index(drop=True), store)
    return oof
