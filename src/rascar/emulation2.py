"""Successors of E1 (:mod:`rascar.emulation`): the same model with a better description of the proposals.

Three independent switches, each justified without the label ``error`` (research/e2_breakthrough/):

``hires_classes``  for audited boxes of these classes the proposal pool of every public detector is the union of E1's
                   nine views and fifteen 384 px tiles (x0 in {0, 224, 448, 672, 896}, y0 in {0, 168, 336}); the tiles
                   localise hands (~27 px) and cups (~26 x 33 px) better and truncate large objects, so persons and
                   trays keep E1's views.
``drop_border``    proposals whose edge lies within 2 px of an INTERIOR border of their own view are removed: a crop
                   cuts large objects and the detector then boxes the visible part, ending exactly at the crop border
                   (among confident proposals near manual persons 40 % of the partial matches are such boxes, 6 % of
                   the good matches).
``margin``         "class": E1's adapter (median offset per class and side, 16 numbers);
                   "pixel": a class-free rule in pixels, margin = a + b * side per side, one line for horizontal and
                   one for vertical sides (4 numbers) - it predicts manual boxes out of day better than the class table.

Presets:  E2  = finer views for hands and cups only (E1 otherwise unchanged)
          E2X = finer views + border removal + pixel margin
Kernel, consensus, model form and every constant are those of E1.

Extra caches (same array format as data/interim/vision, written by
research/final_breakthrough/agent_G/g01_cache_gpu.py --views hires): ``<extra_dir>/<model>_hires/<frame>.npz`` with
model in {owlv2, gdino, gdinob, owlL}. Public weights: the same four checkpoints as E1 (URLs in rascar.emulation).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from . import emulation as em
from . import paths
from .final_models import C_REG, class_stats, class_z

CLASSES, BOX, SMOOTH_COLS, THRESHOLDS = em.CLASSES, em.BOX, em.SMOOTH_COLS, em.THRESHOLDS
FRAME_W, FRAME_H = em.FRAME_W, em.FRAME_H
HIRES_FOLDER = {"ovf": "owlv2_hires", "gdf": "gdino_hires", "gbf": "gdinob_hires", "oLf": "owlL_hires"}
BORDER_PX = 2.0
FINE_TILE, HIRES_TILE = 480, 384


def view_windows() -> dict[str, tuple[int, int, int, int]]:
    """Crop window (x0, y0, x1, y1) of every cached view, keyed by the view name used in the cache files."""
    w, h = FRAME_W, FRAME_H
    win = {"full": (0, 0, w, h), "left": (0, 0, h, h), "right": (w - h, 0, w, h)}
    for r, y0 in enumerate([0, h - FINE_TILE]):
        for c, x0 in enumerate([0, (w - FINE_TILE) // 2, w - FINE_TILE]):
            win[f"t{r}{c}"] = (x0, y0, x0 + FINE_TILE, y0 + FINE_TILE)
    for r, y0 in enumerate([0, 168, 336]):
        for c, x0 in enumerate([0, 224, 448, 672, 896]):
            win[f"h{r}{c}"] = (x0, y0, x0 + HIRES_TILE, y0 + HIRES_TILE)
    return win


WINDOWS = view_windows()


@dataclass(frozen=True)
class Config:
    name: str
    hires_classes: tuple[str, ...] = ("nude_hands", "cup")
    drop_border: bool = False
    margin: str = "class"


E2 = Config("e2_smooth_hires_small")
E2X = Config("e2x_pixel_inner_hires", drop_border=True, margin="pixel")
PRESETS = {c.name: c for c in (E2, E2X)}
NAME = E2.name


def default_extra_dir() -> Path:
    return paths.project_root() / "results" / "final_breakthrough" / "agent_G" / "cache"


class ProposalStore2:
    """Cached proposals with a flag "touches an interior border of its own view".

    ``pool(tag, frame, cls, cfg) -> (boxes [n, 4] px, class score [n, 4])`` for an audited box of class ``cls``.
    """

    def __init__(self, vision_dir: Path | None = None, extra_dir: Path | None = None):
        self.vision_dir = Path(vision_dir) if vision_dir else paths.interim_dir() / "vision"
        self.extra_dir = Path(extra_dir) if extra_dir else default_extra_dir()
        self._cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _paths(self, tag: str, frame_id: str, part: str) -> list[Path]:
        if part == "base":
            return [self.vision_dir / folder / f"{frame_id}.npz" for folder in em.OPINION_CACHES[tag][0]]
        return [self.extra_dir / HIRES_FOLDER[tag] / f"{frame_id}.npz"]

    def _load(self, tag: str, frame_id: str, part: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (tag, frame_id, part)
        if key in self._cache:
            return self._cache[key]
        family = em.OPINION_CACHES[tag][1]
        boxes, probs, touch = [], [], []
        for path in self._paths(tag, frame_id, part):
            if not path.is_file():
                raise FileNotFoundError(f"no cached proposals ({part}) of {tag} for frame {frame_id}: {path}")
            with np.load(path) as z:
                for view in sorted({k.rsplit("_", 1)[0] for k in z.files}):
                    b = z[f"{view}_boxes"].astype(np.float64)
                    x0, y0, x1, y1 = WINDOWS.get(view, (0, 0, FRAME_W, FRAME_H))
                    t = np.zeros(len(b), dtype=bool)
                    if x0 > 0:
                        t |= b[:, 0] <= x0 + BORDER_PX
                    if x1 < FRAME_W:
                        t |= b[:, 2] >= x1 - BORDER_PX
                    if y0 > 0:
                        t |= b[:, 1] <= y0 + BORDER_PX
                    if y1 < FRAME_H:
                        t |= b[:, 3] >= y1 - BORDER_PX
                    boxes.append(b); touch.append(t)
                    if family == "owl":
                        probs.append(1.0 / (1.0 + np.exp(-z[f"{view}_logits"].astype(np.float64))))
                    else:
                        probs.append(z[f"{view}_probs"].astype(np.float64))
        b = np.concatenate(boxes); p = np.concatenate(probs); t = np.concatenate(touch)
        ok = ((b[:, 2] - b[:, 0]) > 0.5) & ((b[:, 3] - b[:, 1]) > 0.5)      # degenerate proposals dropped (as in E1)
        b, p, t = b[ok], p[ok], t[ok]
        group = em._OWL_GROUP if family == "owl" else em._GD_GROUP
        score = np.stack([p[:, group == c].max(axis=1) for c in CLASSES], axis=1)
        self._cache[key] = (b, score, t)
        return self._cache[key]

    def pool(self, tag: str, frame_id: str, cls: str, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
        b, s, t = self._load(tag, frame_id, "base")
        if cls in cfg.hires_classes:
            hb, hs, ht = self._load(tag, frame_id, "hires")
            b, s, t = np.concatenate([b, hb]), np.concatenate([s, hs]), np.concatenate([t, ht])
        if cfg.drop_border:
            b, s = b[~t], s[~t]
        return b, s

    def has_base(self, frame_id: str) -> bool:
        return all(p.is_file() for tag in em.OPINION_CACHES for p in self._paths(tag, frame_id, "base"))

    def has_all(self, frame_id: str, cfg: Config) -> bool:
        need_hires = bool(cfg.hires_classes)
        return self.has_base(frame_id) and (not need_hires or all(p.is_file() for tag in HIRES_FOLDER for p in self._paths(tag, frame_id, "hires")))


# ----------------------------------------------------------------------------------------------------------------------
# margin models
# ----------------------------------------------------------------------------------------------------------------------
def _pairs(manual: pd.DataFrame, store: ProposalStore2, cfg: Config) -> pd.DataFrame:
    """Manual box / representative proposal pairs: class, proposal width and height, relative offsets (G - P) / size."""
    rows = []
    for tag in em.OPINION_CACHES:
        for row in manual.itertuples(index=False):
            if not store.has_base(row.frame_id):
                continue
            boxes, score = store.pool(tag, row.frame_id, row.class_name, cfg)
            if len(boxes) == 0:
                continue
            g = np.array([row.x1, row.y1, row.x2, row.y2], dtype=float)
            iou = em.iou_one_to_many(g, boxes)
            j = int((score[:, CLASSES.index(row.class_name)] * iou).argmax())
            if iou[j] < em.MIN_PAIR_IOU:
                continue
            p = boxes[j]; pw, ph = p[2] - p[0], p[3] - p[1]
            rows.append((row.class_name, pw, ph, (g[0] - p[0]) / pw, (g[1] - p[1]) / ph, (g[2] - p[2]) / pw, (g[3] - p[3]) / ph))
    return pd.DataFrame(rows, columns=["cls", "w", "h", "o0", "o1", "o2", "o3"])


def fit_adapter(manual: pd.DataFrame, store: ProposalStore2, cfg: Config) -> dict:
    pairs = _pairs(manual, store, cfg)
    if cfg.margin == "class":
        out = {}
        for c in CLASSES:
            v = pairs.loc[pairs.cls == c, ["o0", "o1", "o2", "o3"]].to_numpy(float)
            out[c] = np.median(v, axis=0).tolist() if len(v) >= em.MIN_PAIRS else [0.0, 0.0, 0.0, 0.0]
        return out
    if cfg.margin == "pixel":
        # margin in pixels = a + b * side; class-balanced weights; iteratively re-weighted least absolute deviations
        w, h = pairs.w.to_numpy(float), pairs.h.to_numpy(float)
        wgt = (1.0 / pairs.cls.map(pairs.cls.value_counts())).to_numpy(float)
        out = {}
        for key, side, px in (("x", w, np.concatenate([-pairs.o0.to_numpy() * w, pairs.o2.to_numpy() * w])),
                              ("y", h, np.concatenate([-pairs.o1.to_numpy() * h, pairs.o3.to_numpy() * h]))):
            s2, w2 = np.concatenate([side, side]), np.concatenate([wgt, wgt])
            a, b = 2.0, 0.02
            for _ in range(30):
                r = px - (a + b * s2)
                ww = w2 / np.maximum(np.abs(r), 0.5)
                design = np.stack([np.ones_like(s2), s2], axis=1)
                a, b = np.linalg.lstsq(design * ww[:, None] ** 0.5, px * ww ** 0.5, rcond=None)[0]
            out[key] = [float(a), float(b)]
        return out
    raise KeyError(cfg.margin)


def adapt(boxes: np.ndarray, cls: str, adapter: dict, cfg: Config) -> np.ndarray:
    if cfg.margin == "class":
        return em.adapt(boxes, adapter[cls])
    w = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0); h = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    mx = adapter["x"][0] + adapter["x"][1] * w               # margin in pixels on the left and on the right
    my = adapter["y"][0] + adapter["y"][1] * h
    return boxes + np.stack([-mx, -my, mx, my], axis=1)


def smooth_agreement(targets: pd.DataFrame, adapter: dict, store: ProposalStore2, cfg: Config) -> pd.DataFrame:
    out = np.zeros((len(targets), len(em.OPINION_CACHES)))
    for i, row in enumerate(targets.itertuples(index=False)):
        d = np.array([row.x1, row.y1, row.x2, row.y2], dtype=float)
        ci = CLASSES.index(row.class_name)
        for k, tag in enumerate(em.OPINION_CACHES):
            boxes, score = store.pool(tag, row.frame_id, row.class_name, cfg)
            if len(boxes) == 0:
                continue
            iou = em.iou_one_to_many(d, adapt(boxes, row.class_name, adapter, cfg))
            s = score[:, ci]
            vals = []
            for t in THRESHOLDS:
                m = iou >= t
                vals.append(float(s[m].max()) if m.any() else 0.0)
            out[i, k] = float(np.mean(vals))
    return pd.DataFrame(out, columns=SMOOTH_COLS, index=targets.index)


@dataclass
class EmulationModel2:
    cfg: Config = E2
    adapter: dict = field(default_factory=dict)
    stats: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    pipeline: Pipeline | None = None

    @property
    def name(self) -> str:
        return self.cfg.name

    def features(self, table: pd.DataFrame, store: ProposalStore2) -> pd.DataFrame:
        t = table.reset_index(drop=True)
        return pd.concat([t, smooth_agreement(t, self.adapter, store, self.cfg)], axis=1)

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        cons = np.mean([class_z(df, c, self.stats[c]) for c in SMOOTH_COLS], axis=0)
        return np.column_stack([df["tc_logit"].to_numpy(float), cons, df["p_tb_cut0"].to_numpy(float),
                                df["f_cov_same"].to_numpy(float), df["edge_rel"].clip(0, 12).to_numpy(float)])

    def fit(self, train: pd.DataFrame, manual: pd.DataFrame, store: ProposalStore2) -> "EmulationModel2":
        self.adapter = fit_adapter(manual, store, self.cfg)
        tr = self.features(train, store)
        self.stats = {c: class_stats(tr, c) for c in SMOOTH_COLS}
        self.pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      LogisticRegression(C=C_REG, max_iter=5000))
        self.pipeline.fit(self._design(tr), tr["error"].to_numpy(int))
        return self

    def predict(self, table: pd.DataFrame, store: ProposalStore2) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("model is not fitted")
        return np.clip(self.pipeline.predict_proba(self._design(self.features(table, store)))[:, 1], 0.0, 1.0)

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, folder / f"{self.name}.joblib")
        meta = {"name": self.name, "config": asdict(self.cfg), "adapter": self.adapter, "stats": self.stats,
                "thresholds": list(THRESHOLDS), "border_px": BORDER_PX, "hires_folders": HIRES_FOLDER,
                "opinion_caches": {k: list(v[0]) for k, v in em.OPINION_CACHES.items()},
                "inputs": ["tc_logit", "cons_smooth", "p_tb_cut0", "f_cov_same", "edge_rel"]}
        (folder / f"{self.name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, name: str = NAME) -> "EmulationModel2":
        folder = Path(folder)
        meta = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
        c = meta["config"]
        cfg = Config(c["name"], tuple(c["hires_classes"]), bool(c["drop_border"]), c["margin"])
        return cls(cfg=cfg, adapter=meta["adapter"], stats=meta["stats"], pipeline=joblib.load(folder / f"{name}.joblib"))


def leave_one_day_out(train: pd.DataFrame, manual: pd.DataFrame, store: ProposalStore2, cfg: Config = E2) -> np.ndarray:
    """Out-of-day predictions: adapter, class statistics, scaler and regression all come from the other day."""
    oof = np.full(len(train), np.nan)
    for day in sorted(train["utc_day"].unique()):
        held = (train["utc_day"] == day).to_numpy()
        model = EmulationModel2(cfg).fit(train[~held].reset_index(drop=True), manual[manual["utc_day"] != day], store)
        oof[held] = model.predict(train[held].reset_index(drop=True), store)
    return oof
