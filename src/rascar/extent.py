"""Amodal-extent evidence from the cached public-detector proposals (round 6 Agent N definitions, round 7 product code).

n_inside(box, class) = number of confident proposals of the claimed class (score ≥ 0.30) that lie inside the audited
box (≥ 80 % of the proposal's area), counted after greedy NMS (suppress IoU > 0.5). Detectors box the VISIBLE extent,
the manual convention boxes the WHOLE person, so a visible-extent person proposal inside an audited person box is
evidence that the box is the amodal (correct) one. Opinions as in model 03: ovf (owlv2 + owlv2_fine),
gdf (gdino + gdino_fine), gbf (gdinob + gdinob_fine), oLf (owlL + owlL_fine). Label-free; one row at a time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import geometry, paths, vision, vision_gdino

OPINIONS = {"ovf": ("owl", "owlv2", "owlv2_fine"), "gdf": ("gd", "gdino", "gdino_fine"),
            "gbf": ("gd", "gdinob", "gdinob_fine"), "oLf": ("owl", "owlL", "owlL_fine")}
CLASSES = ["person", "nude_hands", "cup", "tray"]
SCORE_MIN, COVER_MIN, NMS_IOU = 0.30, 0.80, 0.5


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def proposals(kind: str, cached: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """All proposals of the base + fine views and their per-class score (max over the class's prompt columns)."""
    if kind == "owl":
        views = sorted({k.split("_")[0] for k in cached})
        boxes = np.concatenate([cached[f"{v}_boxes"] for v in views]).astype(np.float64)
        prob = 1.0 / (1.0 + np.exp(-np.concatenate([cached[f"{v}_logits"] for v in views]).astype(np.float64)))
        groups = np.asarray(vision.GROUP_OF, dtype=object)
    else:
        views = sorted({k.rsplit("_", 1)[0] for k in cached if k.endswith("_boxes")})
        boxes = np.concatenate([cached[f"{v}_boxes"] for v in views]).astype(np.float64)
        prob = np.concatenate([cached[f"{v}_probs"] for v in views]).astype(np.float64)
        groups = np.asarray(vision_gdino.GROUP_OF, dtype=object)
    return boxes, {c: prob[:, groups == c].max(axis=1) for c in CLASSES}


def _nms_count(boxes: np.ndarray, scores: np.ndarray, thr: float = NMS_IOU) -> int:
    keep: list[int] = []
    for j in np.argsort(-scores, kind="stable"):
        if not keep or geometry.iou_matrix(boxes[j][None, :], boxes[keep])[0].max() <= thr:
            keep.append(int(j))
    return len(keep)


def n_inside(box: np.ndarray, props: np.ndarray, scores: np.ndarray) -> float:
    area_p = geometry.box_area(props)
    ok = area_p > 0
    p, s, area_p = props[ok], scores[ok], area_p[ok]
    inter = geometry.intersection_matrix(np.asarray(box, float)[None, :], p)[0]
    cand = (s >= SCORE_MIN) & (inter / area_p >= COVER_MIN)
    return float(_nms_count(p[cand], s[cand])) if cand.any() else 0.0


def n_inside_features(targets: pd.DataFrame, vision_dir: Path | None = None) -> pd.DataFrame:
    """Columns <d>_n_inside for every row of ``targets`` (needs frame_id, class_name, x1, y1, x2, y2); NaN if a
    frame's caches are missing."""
    vdir = Path(vision_dir) if vision_dir else paths.interim_dir() / "vision"
    out = pd.DataFrame(np.nan, index=targets.index, columns=[f"{d}_n_inside" for d in OPINIONS])
    for frame_id, grp in targets.groupby("frame_id", sort=True):
        boxes = grp[["x1", "y1", "x2", "y2"]].to_numpy(np.float64)
        for d, (kind, base, fine) in OPINIONS.items():
            cb, cf = _load(vdir / base / f"{frame_id}.npz"), _load(vdir / fine / f"{frame_id}.npz")
            if cb is None or cf is None:
                continue
            props, score = proposals(kind, {**cb, **cf})
            for pos, (idx, c) in enumerate(zip(grp.index, grp["class_name"])):
                out.loc[idx, f"{d}_n_inside"] = n_inside(boxes[pos], props, score[c])
    return out
