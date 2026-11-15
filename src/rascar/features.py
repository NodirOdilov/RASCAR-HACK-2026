"""Inference-time feature construction.

Every feature is a function of detector output only (stream tables, frame times,
the label-free machine pool). ``train.csv:error`` and ``human_boxes_train.csv`` are
never read here: :func:`build_features` drops an ``error`` column on entry and the
module does not import the manual-box loader (see ``tests/test_leakage.py``).

Feature groups (ablation levels of the report):
    B0  conf      detector confidence
    B1  geom      class + box geometry (+ label-free class-relative percentiles)
    B2  frame     relations to the other detections of the same frame
    B3  temporal  agreement with the previous / next context frame (30 s)
    B4  scene     recurrence of the same box on *other* days (machine pool)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, geometry, temporal
from .data import BOX_COLS, FRAME_H, FRAME_W

CLASSES = ["person", "nude_hands", "cup", "tray"]  # detector classes inside the scored scope
META_COLS = ["id", "frame_id", "class_name", "utc_day", "t_sec"]
MAX_CDIST = 20.0


# --------------------------------------------------------------------------------------
# label-free reference built from the machine pool (+ the unlabeled train stream)
# --------------------------------------------------------------------------------------
@dataclass
class SceneReference:
    """Detector output on the reference days; contains no labels and no manual boxes."""

    dets: pd.DataFrame            # class_name, confidence, x1..y2, utc_day
    frames_per_day: dict[str, int]

    @property
    def days(self) -> list[str]:
        return sorted(self.frames_per_day)


def load_scene_reference(raw: Path | None = None, include_train_stream: bool = True) -> SceneReference:
    frames = [data.load_frames("machine", raw)]
    dets = [data.load_stream("machine", raw)]
    if include_train_stream:
        frames.append(data.load_frames("train", raw))
        dets.append(data.load_stream("train", raw))
    frames_all = pd.concat(frames, ignore_index=True)
    dets_all = data.attach_day(pd.concat(dets, ignore_index=True), frames_all)
    return SceneReference(dets=dets_all[["class_name", "confidence", *BOX_COLS, "utc_day"]].reset_index(drop=True),
                          frames_per_day=frames_all["utc_day"].value_counts().to_dict())


# --------------------------------------------------------------------------------------
# B0 + B1: confidence, class, geometry
# --------------------------------------------------------------------------------------
def _logit(p: np.ndarray, eps: float = 0.01) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def base_features(targets: pd.DataFrame) -> pd.DataFrame:
    b = targets[BOX_COLS].to_numpy(float)
    conf = targets["confidence"].to_numpy(float)
    w = np.clip(b[:, 2] - b[:, 0], 1e-3, None)
    h = np.clip(b[:, 3] - b[:, 1], 1e-3, None)
    out = pd.DataFrame(index=targets.index)
    out["conf"] = conf
    out["conf_logit"] = _logit(conf)
    out["cls"] = targets["class_name"].to_numpy()
    for c in CLASSES:
        out[f"is_{c}"] = (targets["class_name"].to_numpy() == c).astype(float)
    out["w"] = w
    out["h"] = h
    out["log_area"] = np.log(w * h)
    out["log_aspect"] = np.log(w / h)
    out["size"] = np.sqrt(w * h)
    out["cx"] = (b[:, 0] + b[:, 2]) / 2 / FRAME_W
    out["cy"] = (b[:, 1] + b[:, 3]) / 2 / FRAME_H
    out["y1n"] = b[:, 1] / FRAME_H
    out["y2n"] = b[:, 3] / FRAME_H
    edge = np.minimum.reduce([b[:, 0], b[:, 1], FRAME_W - b[:, 2], FRAME_H - b[:, 3]])
    out["edge_dist"] = np.clip(edge, 0, None)
    out["edge_rel"] = np.clip(edge, 0, None) / np.sqrt(w * h)
    out["touch_border"] = (edge <= 1.0).astype(float)
    out["zero_area"] = ((b[:, 2] - b[:, 0] <= 0) | (b[:, 3] - b[:, 1] <= 0)).astype(float)
    return out


def _ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float))
    if ref.size == 0:
        return np.full(len(values), 0.5)
    lo = np.searchsorted(ref, values, side="left")
    hi = np.searchsorted(ref, values, side="right")
    return (lo + hi) / 2.0 / ref.size


def class_relative_features(targets: pd.DataFrame, ref: SceneReference) -> pd.DataFrame:
    """Percentile of confidence / size / aspect inside the detector's own class
    distribution on the reference stream (label-free, identical for train and test)."""
    out = pd.DataFrame({"conf_pct": 0.5, "size_pct": 0.5, "aspect_pct": 0.5}, index=targets.index)
    b = targets[BOX_COLS].to_numpy(float)
    w = np.clip(b[:, 2] - b[:, 0], 1e-3, None)
    h = np.clip(b[:, 3] - b[:, 1], 1e-3, None)
    cls = targets["class_name"].to_numpy()
    rb = ref.dets[BOX_COLS].to_numpy(float)
    rw = np.clip(rb[:, 2] - rb[:, 0], 1e-3, None)
    rh = np.clip(rb[:, 3] - rb[:, 1], 1e-3, None)
    rcls = ref.dets["class_name"].to_numpy()
    rconf = ref.dets["confidence"].to_numpy(float)
    conf = targets["confidence"].to_numpy(float)
    vals = out.to_numpy(copy=True)
    for c in np.unique(cls):
        m, rm = cls == c, rcls == c
        vals[m, 0] = _ecdf(rconf[rm], conf[m])
        vals[m, 1] = _ecdf(np.sqrt(rw[rm] * rh[rm]), np.sqrt(w[m] * h[m]))
        vals[m, 2] = _ecdf(np.log(rw[rm] / rh[rm]), np.log(w[m] / h[m]))
    return pd.DataFrame(vals, columns=out.columns, index=targets.index)


# --------------------------------------------------------------------------------------
# B2: relations inside the frame
# --------------------------------------------------------------------------------------
def _masked_max(values: np.ndarray, mask: np.ndarray, empty: float = 0.0) -> np.ndarray:
    if values.shape[1] == 0:
        return np.full(values.shape[0], empty)
    v = np.where(mask, values, -np.inf).max(axis=1)
    return np.where(np.isfinite(v), v, empty)


def _masked_min(values: np.ndarray, mask: np.ndarray, empty: float) -> np.ndarray:
    if values.shape[1] == 0:
        return np.full(values.shape[0], empty)
    v = np.where(mask, values, np.inf).min(axis=1)
    return np.where(np.isfinite(v), v, empty)


def frame_features(targets: pd.DataFrame, index: dict[str, temporal.FrameDets]) -> pd.DataFrame:
    names = ["f_n_nontable", "f_n_same", "f_n_person", "f_n_hands", "f_n_cup", "f_n_tray",
             "f_iou_same", "f_cov_same", "f_contains_same", "f_iou_diff", "f_person_cov", "f_host_conf",
             "f_n_person_host", "f_table_cov", "f_table_iou", "f_same_cdist", "f_n_same_near", "f_conf_rank",
             "f_conf_gap", "f_pt_bottom_rel", "f_pt_cut", "f_person_iou", "f_n_inside"]
    cols = {name: np.full(len(targets), np.nan) for name in names}
    position = np.arange(len(targets))
    frame_ids = targets["frame_id"].to_numpy(dtype=object)
    ids = targets["id"].to_numpy(dtype=object)
    boxes = targets[BOX_COLS].to_numpy(float)
    classes = targets["class_name"].to_numpy(dtype=object)
    confs = targets["confidence"].to_numpy(float)

    for frame_id in pd.unique(frame_ids):
        sel = position[frame_ids == frame_id]
        fr = index.get(frame_id, temporal.EMPTY)
        tb, tc, tconf = boxes[sel], classes[sel], confs[sel]
        n = len(sel)
        area = np.clip(geometry.box_area(tb), 1e-6, None)
        size = np.sqrt(area)
        other = ids[sel][:, None] != fr.det_id[None, :]
        iou = geometry.iou_matrix(tb, fr.box)
        cov = geometry.coverage_matrix(tb, fr.box)            # share of the target inside the other box
        cov_rev = geometry.coverage_matrix(fr.box, tb).T      # share of the other box inside the target
        cdist = geometry.center_distance_matrix(tb, fr.box) / size[:, None]
        same = (tc[:, None] == fr.cls[None, :]) & other
        is_table = np.broadcast_to(fr.cls[None, :] == "table", other.shape) & other
        is_person = np.broadcast_to(fr.cls[None, :] == "person", other.shape) & other
        diff = np.broadcast_to(fr.cls[None, :] != "table", other.shape) & other & ~same

        cols["f_n_nontable"][sel] = float((fr.cls != "table").sum())
        cols["f_n_same"][sel] = same.sum(axis=1) + 1.0
        cols["f_n_person"][sel] = float((fr.cls == "person").sum())
        cols["f_n_hands"][sel] = float((fr.cls == "nude_hands").sum())
        cols["f_n_cup"][sel] = float((fr.cls == "cup").sum())
        cols["f_n_tray"][sel] = float((fr.cls == "tray").sum())
        cols["f_iou_same"][sel] = _masked_max(iou, same)
        cols["f_cov_same"][sel] = _masked_max(cov, same)
        cols["f_contains_same"][sel] = _masked_max(cov_rev, same)
        cols["f_iou_diff"][sel] = _masked_max(iou, diff)
        person_cov = _masked_max(cov, is_person)
        cols["f_person_cov"][sel] = person_cov
        cols["f_person_iou"][sel] = _masked_max(iou, is_person)
        if fr.box.shape[0]:
            host = np.where(is_person, cov, -1.0).argmax(axis=1)
            cols["f_host_conf"][sel] = np.where(person_cov >= 0.5, fr.conf[host], np.nan)
            cols["f_n_person_host"][sel] = (is_person & (cov >= 0.5)).sum(axis=1)
            cols["f_n_inside"][sel] = (diff & (cov_rev >= 0.7)).sum(axis=1)
        else:
            cols["f_n_person_host"][sel] = 0.0
            cols["f_n_inside"][sel] = 0.0
        cols["f_table_cov"][sel] = _masked_max(cov, is_table)
        cols["f_table_iou"][sel] = _masked_max(iou, is_table)
        cols["f_same_cdist"][sel] = np.minimum(_masked_min(cdist, same, MAX_CDIST), MAX_CDIST)
        cols["f_n_same_near"][sel] = (same & (cdist < 3.0)).sum(axis=1)
        higher = same & (fr.conf[None, :] > tconf[:, None]) if fr.box.shape[0] else same
        cols["f_conf_rank"][sel] = higher.sum(axis=1) / (same.sum(axis=1) + 1.0)
        best_same_conf = _masked_max(np.broadcast_to(fr.conf[None, :], other.shape), same, np.nan) if fr.box.shape[0] else np.full(n, np.nan)
        cols["f_conf_gap"][sel] = tconf - best_same_conf
        # geometry of a box relative to the table it intersects most (meaningful for persons:
        # the manual person box covers the body hidden behind the table, the detector may cut it)
        if fr.box.shape[0] and is_table.any():
            inter = geometry.intersection_matrix(tb, fr.box)
            inter = np.where(is_table, inter, -1.0)
            j = inter.argmax(axis=1)
            has = inter.max(axis=1) > 0
            height = np.clip(tb[:, 3] - tb[:, 1], 1e-6, None)
            rel = (tb[:, 3] - fr.box[j, 1]) / height      # how far the bottom edge reaches below the table top
            cut = (fr.box[j, 3] - tb[:, 3]) / height      # > 0: the box ends above the table bottom ("cut by the table")
            cols["f_pt_bottom_rel"][sel] = np.where(has, rel, np.nan)
            cols["f_pt_cut"][sel] = np.where(has, cut, np.nan)
    return pd.DataFrame(cols, index=targets.index)


# --------------------------------------------------------------------------------------
# person-specific geometry: is the box cut at the table edge?
# --------------------------------------------------------------------------------------
def person_table_features(targets: pd.DataFrame, index: dict[str, temporal.FrameDets]) -> pd.DataFrame:
    """The manual ``person`` box covers the whole person, including the body hidden behind
    the table (TASK.txt). A detector box that stops at the table edge therefore tends to
    have IoU < 0.60. For every person box we look for the table in front of the person
    (its top edge lies in the lower part of the box, horizontal overlap > 30 %):

    p_tb_cut    (table.y2 - box.y2) / box.h   > 0: the table continues below the box bottom
    p_tb_below  (box.y2 - table.y1) / box.h   how far the box reaches below the table top
    p_tb_depth  (box.y2 - table.y1) / table.h share of the table depth covered by the box
    p_tb_has    1 if such a table exists
    All are NaN / 0 for non-person boxes.
    """
    n = len(targets)
    cut, below, depth, has = (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.zeros(n))
    frame_ids = targets["frame_id"].to_numpy(dtype=object)
    classes = targets["class_name"].to_numpy(dtype=object)
    boxes = targets[BOX_COLS].to_numpy(float)
    for i in np.flatnonzero(classes == "person"):
        fr = index.get(frame_ids[i], temporal.EMPTY)
        tables = fr.box[fr.cls == "table"]
        if len(tables) == 0:
            continue
        x1, y1, x2, y2 = boxes[i]
        w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
        hov = (np.minimum(tables[:, 2], x2) - np.maximum(tables[:, 0], x1)) / w
        cand = (hov > 0.3) & (tables[:, 1] > y1 + 0.25 * h) & (tables[:, 1] < y2 + 0.3 * h)
        if not cand.any():
            continue
        j = int(np.argmax(np.where(cand, hov, -1.0)))
        has[i] = 1.0
        cut[i] = (tables[j, 3] - y2) / h
        below[i] = (y2 - tables[j, 1]) / h
        depth[i] = (y2 - tables[j, 1]) / max(tables[j, 3] - tables[j, 1], 1e-6)
    return pd.DataFrame({"p_tb_cut": cut, "p_tb_below": below, "p_tb_depth": depth, "p_tb_has": has}, index=targets.index)


# --------------------------------------------------------------------------------------
# B4: scene statistics from other days (label-free)
# --------------------------------------------------------------------------------------
SCENE_NAMES = ["sc_recur50", "sc_recur30", "sc_recur70", "sc_iou_top", "sc_dens", "sc_conf_local",
               "sc_larea_dev", "sc_lh_dev", "sc_n_local"]


def scene_features(targets: pd.DataFrame, days: np.ndarray, ref: SceneReference,
                   top_k: int = 10, local_radius: float = 80.0) -> pd.DataFrame:
    """Compare each box with same-class detector boxes of the reference days,
    always excluding the box's own UTC day (so train and test rows are treated alike).

    sc_recur50/30/70  share of reference frames holding a same-class box with IoU >= .5/.3/.7
    sc_iou_top        mean of the ``top_k`` largest IoUs with same-class reference boxes
    sc_dens           same-class reference boxes per frame whose centre is within half a box size
    sc_conf_local     mean confidence of same-class reference boxes with IoU >= .5
    sc_larea_dev      log-area minus the median log-area of same-class reference boxes within 80 px
    sc_lh_dev         log-height minus the median log-height of those boxes
    sc_n_local        number of those neighbours per reference frame
    """
    n = len(targets)
    acc = {name: np.zeros(n) for name in SCENE_NAMES}
    cnt = {name: np.zeros(n) for name in SCENE_NAMES}
    boxes = targets[BOX_COLS].to_numpy(float)
    classes = targets["class_name"].to_numpy(dtype=object)
    days = np.asarray(days, dtype=object)
    ref_cls = ref.dets["class_name"].to_numpy(dtype=object)
    ref_day = ref.dets["utc_day"].to_numpy(dtype=object)
    ref_box = ref.dets[BOX_COLS].to_numpy(float)
    ref_conf = ref.dets["confidence"].to_numpy(float)
    ref_larea = np.log(np.clip(geometry.box_area(ref_box), 1e-6, None))
    ref_lh = np.log(np.clip(ref_box[:, 3] - ref_box[:, 1], 1e-6, None))

    def put(name: str, sel: np.ndarray, values: np.ndarray) -> None:
        ok = ~np.isnan(values)
        acc[name][sel[ok]] += values[ok]
        cnt[name][sel[ok]] += 1

    for cls in np.unique(classes):
        for day in ref.days:
            sel = np.flatnonzero((classes == cls) & (days != day))
            if sel.size == 0:
                continue
            rsel = np.flatnonzero((ref_cls == cls) & (ref_day == day))
            n_frames = float(ref.frames_per_day[day])
            tb = boxes[sel]
            if rsel.size == 0:
                for name in ("sc_recur50", "sc_recur30", "sc_recur70", "sc_iou_top", "sc_dens", "sc_n_local"):
                    put(name, sel, np.zeros(sel.size))
                continue
            rb = ref_box[rsel]
            iou = geometry.iou_matrix(tb, rb)
            put("sc_recur50", sel, (iou >= 0.5).sum(axis=1) / n_frames)
            put("sc_recur30", sel, (iou >= 0.3).sum(axis=1) / n_frames)
            put("sc_recur70", sel, (iou >= 0.7).sum(axis=1) / n_frames)
            k = min(top_k, iou.shape[1])
            put("sc_iou_top", sel, np.sort(iou, axis=1)[:, -k:].mean(axis=1))
            dist = geometry.center_distance_matrix(tb, rb)
            size = np.sqrt(np.clip(geometry.box_area(tb), 1e-6, None))
            put("sc_dens", sel, (dist <= np.maximum(15.0, 0.5 * size)[:, None]).sum(axis=1) / n_frames)
            strong = iou >= 0.5
            with np.errstate(invalid="ignore", divide="ignore"):
                conf_local = np.where(strong.any(axis=1), (strong * ref_conf[rsel][None, :]).sum(axis=1) / strong.sum(axis=1), np.nan)
            put("sc_conf_local", sel, conf_local)
            near = dist <= local_radius
            n_near = near.sum(axis=1)
            put("sc_n_local", sel, n_near / n_frames)
            med_area = np.full(sel.size, np.nan)
            med_h = np.full(sel.size, np.nan)
            for i in np.flatnonzero(n_near >= 5):
                med_area[i] = np.median(ref_larea[rsel][near[i]])
                med_h[i] = np.median(ref_lh[rsel][near[i]])
            put("sc_larea_dev", sel, np.log(np.clip(geometry.box_area(tb), 1e-6, None)) - med_area)
            put("sc_lh_dev", sel, np.log(np.clip(tb[:, 3] - tb[:, 1], 1e-6, None)) - med_h)

    out = {}
    for name in SCENE_NAMES:
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.where(cnt[name] > 0, acc[name] / cnt[name], np.nan)
    df = pd.DataFrame(out, index=targets.index)
    df["sc_log_recur50"] = np.log1p(1000.0 * df["sc_recur50"])
    df["sc_log_dens"] = np.log1p(1000.0 * df["sc_dens"])
    return df


# --------------------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------------------
def build_features(part: data.Part, ref: SceneReference | None = None,
                   targets: pd.DataFrame | None = None, use_vision: bool = True) -> pd.DataFrame:
    """Feature table for the target detections of ``part`` (one row per ``id``).

    ``targets`` may restrict / replace the rows (it needs ``id`` and the detection
    columns). A label column, if present, is dropped before anything is computed.
    With ``use_vision`` the cached second-opinion features (``rascar.vision``) are attached
    when the cache exists; rows of frames that were never processed get NaN.
    """
    tgt = (targets if targets is not None else part.targets).copy()
    tgt = tgt.drop(columns=[c for c in ("error",) if c in tgt.columns]).reset_index(drop=True)
    if not tgt["id"].is_unique:
        raise ValueError("duplicated detection ids in the target table")
    tgt = data.attach_day(tgt, part.frames)
    if tgt["utc_day"].isna().any():
        raise ValueError("target detections reference frames that are missing from the frames table")
    index = temporal.frame_index(part.stream)

    blocks = [tgt[META_COLS], base_features(tgt)]
    blocks.append(frame_features(tgt, index))
    blocks.append(person_table_features(tgt, index))
    temporal_block = temporal.temporal_features(tgt, part.stream, part.context, part.frames, index)
    match_ids = temporal_block[temporal.MATCH_ID_COLS]
    blocks.append(temporal_block.drop(columns=temporal.MATCH_ID_COLS))
    if ref is not None:
        blocks.append(class_relative_features(tgt, ref))
        blocks.append(scene_features(tgt, tgt["utc_day"].to_numpy(), ref))
    feats = pd.concat(blocks, axis=1)
    feats["conf_x_person"] = feats["conf_logit"] * feats["is_person"]
    feats["conf_x_hands"] = feats["conf_logit"] * feats["is_nude_hands"]
    feats["conf_x_cup"] = feats["conf_logit"] * feats["is_cup"]
    feats["conf_x_tray"] = feats["conf_logit"] * feats["is_tray"]
    if ref is not None:
        # person-only size deficit against the local same-class boxes of other days (0 elsewhere)
        person = feats["is_person"].to_numpy() > 0
        feats["p_larea_dev"] = np.where(person, feats["sc_larea_dev"].fillna(0.0), 0.0)
        feats["p_lh_dev"] = np.where(person, feats["sc_lh_dev"].fillna(0.0), 0.0)
    feats["p_tb_cut0"] = np.where(feats["p_tb_has"].to_numpy() > 0, feats["p_tb_cut"].fillna(0.0), 0.0)
    if use_vision:
        from . import vision  # numpy-only at import time; torch is needed only to (re)build the cache

        from . import vision_gdino, vision_zoom

        own = vision.vision_features(tgt)
        feats = pd.concat([feats, own, vision.temporal_vision_features(own, match_ids, part.stream),
                           vision_zoom.zoom_features(tgt), vision_gdino.gdino_features(tgt)], axis=1)
        # second opinion pooled over every OWLv2 view (9 frame views + 2 zoomed views); NaN unless both caches exist
        feats["ova_same_soft"] = np.where(feats["ovf_same_soft"].notna() & feats["ovz_same_soft"].notna(),
                                          np.fmax(feats["ovf_same_soft"], feats["ovz_same_soft"]), np.nan)
    if feats.shape[0] != tgt.shape[0] or not (feats["id"].to_numpy() == tgt["id"].to_numpy()).all():
        raise AssertionError("feature rows are not aligned with the target rows")
    return feats


GROUPS: dict[str, list[str]] = {
    "conf": ["conf"],
    "geom": ["cls", "w", "h", "log_area", "log_aspect", "cx", "cy", "y2n", "edge_dist", "edge_rel", "touch_border",
             "conf_pct", "size_pct", "aspect_pct"],
    "frame": ["f_n_nontable", "f_n_same", "f_n_person", "f_n_hands", "f_n_cup", "f_n_tray", "f_iou_same", "f_cov_same",
              "f_contains_same", "f_iou_diff", "f_person_cov", "f_host_conf", "f_n_person_host", "f_table_cov",
              "f_table_iou", "f_same_cdist", "f_n_same_near", "f_conf_rank", "f_conf_gap", "f_pt_bottom_rel",
              "f_pt_cut", "f_person_iou", "f_n_inside"],
    "temporal": ["t_iou_prev", "t_iou_next", "t_iou_max", "t_iou_min", "t_iou_mean", "t_persist", "t_conf_nb",
                 "t_conf_delta", "t_shift_max", "t_larea_abs", "t_dwh_abs", "t_cdist_min", "t_iou_any_max", "t_flip_any",
                 "t_n_same_nb", "t_n_sides", "t_ncand_prev", "t_ncand_next", "tc_logit", "tc_n", "tc_gap"],
    "scene": ["sc_recur50", "sc_recur30", "sc_recur70", "sc_iou_top", "sc_dens", "sc_conf_local", "sc_larea_dev",
              "sc_lh_dev", "sc_n_local"],
    "person": ["p_tb_cut", "p_tb_below", "p_tb_depth", "p_tb_has", "p_larea_dev", "p_lh_dev"],
    # the single validated temporal signal: confidence averaged with the +-30 s re-detections
    "tconf": ["tc_logit"],
    # independent second opinion of a public open-vocabulary detector (rascar.vision)
    "vision": ["ov_same_logit", "ov_same_soft", "ov_iou", "ov_other", "ov_margin", "ov_obj"],
    "vision_t": ["ovt_soft", "ovt_same", "ovt_n"],
    "vision_f": ["ovf_same_soft", "ovf_other"],
    "vision_z": ["ovz_same_soft", "ovz_other"],
    "vision_a": ["ova_same_soft"],
    "gdino": ["gd_same_soft", "gd_other"],
}


def columns_for(groups: list[str]) -> list[str]:
    cols: list[str] = []
    for g in groups:
        if g not in GROUPS:
            raise KeyError(f"unknown feature group {g!r}; known: {sorted(GROUPS)}")
        cols.extend(c for c in GROUPS[g] if c not in cols)
    return cols
