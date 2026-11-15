"""Temporal-consistency features from the ``previous`` / ``next`` context frames.

Only detector output is used (stream tables), never labels or manual boxes. A
target detection is compared with the detections of the neighbouring frames that
the organizers list in ``*_context.csv`` (typical gap 30 s) - identically for the
train and the test part.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import geometry
from .data import BOX_COLS

SIDES = ("prev", "next")
MATCH_IOU = 0.1      # minimal IoU to call a neighbour detection "the same object"
PERSIST_IOU = 0.5    # IoU that counts as a stable re-detection
MAX_CDIST = 20.0     # cap for normalised centre distances


@dataclass
class FrameDets:
    det_id: np.ndarray
    cls: np.ndarray
    conf: np.ndarray
    box: np.ndarray


EMPTY = FrameDets(np.zeros(0, dtype=object), np.zeros(0, dtype=object), np.zeros(0), np.zeros((0, 4)))


def frame_index(stream: pd.DataFrame, id_col: str = "detection_id") -> dict[str, FrameDets]:
    """frame_id -> detections of that frame (numpy views for fast pairwise geometry)."""
    index: dict[str, FrameDets] = {}
    for frame_id, grp in stream.groupby("frame_id", sort=False):
        index[frame_id] = FrameDets(
            det_id=grp[id_col].to_numpy(dtype=object),
            cls=grp["class_name"].to_numpy(dtype=object),
            conf=grp["confidence"].to_numpy(dtype=float),
            box=grp[BOX_COLS].to_numpy(dtype=float),
        )
    return index


def _nan(n: int) -> np.ndarray:
    return np.full(n, np.nan)


def side_features(tb: np.ndarray, tc: np.ndarray, tconf: np.ndarray, nb: FrameDets | None) -> dict[str, np.ndarray]:
    """Compare ``n`` target boxes with the detections of one neighbouring frame."""
    n = len(tb)
    if nb is None:  # the neighbouring frame does not exist
        out = {k: _nan(n) for k in ("iou", "conf", "larea", "shift", "ncand", "cdist", "iou_any", "flip", "n_same", "dw", "dh")}
        out["mid"] = np.full(n, None, dtype=object)
        return out
    area = np.clip(geometry.box_area(tb), 1e-6, None)
    size = np.sqrt(area)
    out = {
        "iou": np.zeros(n), "conf": _nan(n), "larea": _nan(n), "shift": _nan(n), "ncand": np.zeros(n),
        "cdist": np.full(n, MAX_CDIST), "iou_any": np.zeros(n), "flip": np.zeros(n), "n_same": np.zeros(n),
        "dw": _nan(n), "dh": _nan(n), "mid": np.full(n, None, dtype=object),
    }
    if len(nb.box) == 0:
        return out
    iou = geometry.iou_matrix(tb, nb.box)
    same = tc[:, None] == nb.cls[None, :]
    out["n_same"] = same.sum(axis=1).astype(float)
    iou_same = np.where(same, iou, -1.0)
    j = iou_same.argmax(axis=1)
    best = np.clip(iou_same.max(axis=1), 0.0, None)
    out["iou"] = best
    out["ncand"] = (iou_same > MATCH_IOU).sum(axis=1).astype(float)
    matched = best > MATCH_IOU
    nb_area = np.clip(geometry.box_area(nb.box), 1e-6, None)
    cdist = geometry.center_distance_matrix(tb, nb.box) / size[:, None]
    cdist_same = np.where(same, cdist, np.inf)
    out["cdist"] = np.minimum(cdist_same.min(axis=1), MAX_CDIST)
    rows = np.arange(n)
    out["conf"] = np.where(matched, nb.conf[j], np.nan)
    supported = best > SUPPORT_IOU
    out["mid"] = np.where(supported, nb.det_id[j], None).astype(object)
    out["larea"] = np.where(matched, np.log(nb_area[j] / area), np.nan)
    out["shift"] = np.where(matched, cdist[rows, j], np.nan)
    tw, th = tb[:, 2] - tb[:, 0], tb[:, 3] - tb[:, 1]
    nw, nh = nb.box[j, 2] - nb.box[j, 0], nb.box[j, 3] - nb.box[j, 1]
    out["dw"] = np.where(matched, np.log(np.clip(nw, 1e-6, None) / np.clip(tw, 1e-6, None)), np.nan)
    out["dh"] = np.where(matched, np.log(np.clip(nh, 1e-6, None) / np.clip(th, 1e-6, None)), np.nan)
    k = iou.argmax(axis=1)
    out["iou_any"] = iou.max(axis=1)
    out["flip"] = ((nb.cls[k] != tc) & (out["iou_any"] >= PERSIST_IOU) & (out["iou_any"] > best)).astype(float)
    return out


_NANFUNC = {"max": np.nanmax, "min": np.nanmin, "mean": np.nanmean}


def _nanagg(kind: str, *cols) -> np.ndarray:
    """NaN-aware aggregate over the two sides; NaN only when every side is NaN."""
    stacked = np.vstack([np.asarray(c, dtype=float) for c in cols])
    if stacked.shape[1] == 0:
        return np.zeros(0)
    all_nan = np.isnan(stacked).all(axis=0)
    filled = np.where(all_nan[None, :], 0.0, stacked)
    out = _NANFUNC[kind](filled, axis=0)
    return np.where(all_nan, np.nan, out)


def temporal_features(targets: pd.DataFrame, stream: pd.DataFrame, context: pd.DataFrame,
                      frames: pd.DataFrame, index: dict[str, FrameDets] | None = None) -> pd.DataFrame:
    """One row per target detection (same order and index as ``targets``)."""
    index = index if index is not None else frame_index(stream)
    t_sec = frames.set_index("frame_id")["t_sec"].to_dict()
    ctx = context.set_index("frame_id")
    neighbour = {side: ctx["previous" if side == "prev" else "next"].to_dict() for side in SIDES}

    cols: dict[str, np.ndarray] = {}
    n_total = len(targets)
    names = ["iou", "conf", "larea", "shift", "ncand", "cdist", "iou_any", "flip", "n_same", "dw", "dh"]
    match_ids = {side: np.full(n_total, None, dtype=object) for side in SIDES}
    for side in SIDES:
        for name in names:
            cols[f"t_{name}_{side}"] = _nan(n_total)
        cols[f"t_has_{side}"] = np.zeros(n_total)
        cols[f"t_dt_{side}"] = _nan(n_total)

    position = np.arange(n_total)
    frame_ids = targets["frame_id"].to_numpy(dtype=object)
    boxes = targets[BOX_COLS].to_numpy(dtype=float)
    classes = targets["class_name"].to_numpy(dtype=object)
    confs = targets["confidence"].to_numpy(dtype=float)
    for frame_id in pd.unique(frame_ids):
        sel = position[frame_ids == frame_id]
        for side in SIDES:
            nf = neighbour[side].get(frame_id)
            exists = isinstance(nf, str) and nf in t_sec
            nb = index.get(nf, EMPTY) if exists else None
            feats = side_features(boxes[sel], classes[sel], confs[sel], nb)
            for name in names:
                cols[f"t_{name}_{side}"][sel] = feats[name]
            match_ids[side][sel] = feats["mid"]
            cols[f"t_has_{side}"][sel] = float(exists)
            if exists:
                cols[f"t_dt_{side}"][sel] = abs(t_sec[nf] - t_sec[frame_id])

    out = pd.DataFrame(cols, index=targets.index)
    iou_p, iou_n = out["t_iou_prev"].to_numpy(), out["t_iou_next"].to_numpy()
    out["t_n_sides"] = out["t_has_prev"] + out["t_has_next"]
    out["t_iou_max"] = _nanagg("max", iou_p, iou_n)
    out["t_iou_min"] = _nanagg("min", iou_p, iou_n)
    out["t_iou_mean"] = _nanagg("mean", iou_p, iou_n)
    # share of the existing neighbour frames where the box is re-detected with IoU >= PERSIST_IOU
    out["t_persist"] = _nanagg("mean", np.where(np.isnan(iou_p), np.nan, (iou_p >= PERSIST_IOU).astype(float)),
                               np.where(np.isnan(iou_n), np.nan, (iou_n >= PERSIST_IOU).astype(float)))
    out["t_conf_nb"] = _nanagg("mean", out["t_conf_prev"], out["t_conf_next"])
    out["t_conf_delta"] = confs - out["t_conf_nb"].to_numpy()
    out["t_shift_max"] = _nanagg("max", out["t_shift_prev"], out["t_shift_next"])
    out["t_larea_abs"] = _nanagg("max", np.abs(out["t_larea_prev"].to_numpy()), np.abs(out["t_larea_next"].to_numpy()))
    out["t_dwh_abs"] = _nanagg("max", np.abs(out["t_dw_prev"].to_numpy()), np.abs(out["t_dh_prev"].to_numpy()),
                               np.abs(out["t_dw_next"].to_numpy()), np.abs(out["t_dh_next"].to_numpy()))
    out["t_cdist_min"] = _nanagg("min", out["t_cdist_prev"], out["t_cdist_next"])
    out["t_iou_any_max"] = _nanagg("max", out["t_iou_any_prev"], out["t_iou_any_next"])
    out["t_flip_any"] = _nanagg("max", out["t_flip_prev"], out["t_flip_next"])
    out["t_n_same_nb"] = _nanagg("mean", out["t_n_same_prev"], out["t_n_same_next"])

    # ---- temporally smoothed confidence ("tc_*") -------------------------------------
    # A neighbour detection supports the target when it is the same class and overlaps it
    # with IoU > SUPPORT_IOU. The smoothed confidence is the mean over the target itself and
    # its (0..2) supporters; a missing supporter is NOT penalised (people move between frames).
    sup_p = np.where(iou_p > SUPPORT_IOU, out["t_conf_prev"].to_numpy(), np.nan)
    sup_n = np.where(iou_n > SUPPORT_IOU, out["t_conf_next"].to_numpy(), np.nan)
    out["tc_prev"] = sup_p
    out["tc_next"] = sup_n
    out["tc_n"] = (~np.isnan(sup_p)).astype(float) + (~np.isnan(sup_n)).astype(float)
    out["tc_mean"] = _nanagg("mean", confs, sup_p, sup_n)
    out["tc_logit"] = _nanagg("mean", logit(confs), logit(sup_p), logit(sup_n))
    nb_logit = _nanagg("mean", logit(sup_p), logit(sup_n))
    out["tc_nb_logit"] = np.where(np.isnan(nb_logit), logit(confs), nb_logit)
    out["tc_gap"] = logit(confs) - out["tc_nb_logit"].to_numpy()
    for side in SIDES:   # ids of the supporting detections (object columns, removed before modelling)
        out[f"tm_id_{side}"] = match_ids[side]
    return out


MATCH_ID_COLS = ["tm_id_prev", "tm_id_next"]


SUPPORT_IOU = 0.3


def logit(p, eps: float = 0.01) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    clipped = np.clip(p, eps, 1 - eps)
    return np.where(np.isnan(p), np.nan, np.log(clipped / (1 - clipped)))
