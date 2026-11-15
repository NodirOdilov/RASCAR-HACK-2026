"""Second opinion at adaptive magnification: one zoomed view per audited detection.

The frame-level views of :mod:`rascar.vision` magnify the scene at most 2x. Small objects
(hands, cups) still cover only 3-4 patches of the public detector there. Here every audited
box gets its own square crop, centred on the box and sized relative to it, which the
processor upscales to 960 px:

    z1  side = clamp(3 x max(w, h), 192 px, 720 px)   strong magnification, little context
    z2  side = clamp(6 x max(w, h), 320 px, 720 px)   moderate magnification, more context

The crop geometry depends only on the detector box, never on a label, and train and test
detections are processed identically. Only tokens that overlap the audited box are cached.

Run:  python -m rascar.vision_zoom --parts train,test      (needs torch + transformers; resumable)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, geometry, paths, vision
from .data import BOX_COLS, FRAME_H, FRAME_W

ZOOMS: dict[str, tuple[float, int]] = {"z1": (3.0, 192), "z2": (6.0, 320)}
KEEP_IOU = 0.05
ZOOM_COLS = ["ovz_same_soft", "ovz1_same_soft", "ovz2_same_soft", "ovz_same", "ovz_other", "ovz_iou"]


def cache_dir() -> Path:
    path = paths.interim_dir() / "vision" / "owlv2_zoom"
    path.mkdir(parents=True, exist_ok=True)
    return path


def zoom_window(box, k: float, s_min: int, width: int = FRAME_W, height: int = FRAME_H) -> tuple[int, int, int, int]:
    """Square crop window centred on the box, shifted to stay inside the frame."""
    x1, y1, x2, y2 = (float(v) for v in box)
    side = int(round(min(max(k * max(x2 - x1, y2 - y1), s_min), min(width, height))))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x0 = int(round(min(max(cx - side / 2.0, 0.0), width - side)))
    y0 = int(round(min(max(cy - side / 2.0, 0.0), height - side)))
    return x0, y0, x0 + side, y0 + side


def process(targets: pd.DataFrame, raw: Path | None = None, threads: int | None = None, log_every: int = 20) -> int:
    """Run the public detector on the zoomed crops of every listed detection that is not cached yet."""
    from PIL import Image

    raw = Path(raw) if raw else paths.raw_dir()
    todo = targets[[not (cache_dir() / f"{i}.npz").is_file() for i in targets["id"]]]
    if todo.empty:
        return 0
    owl = vision.Owl(threads)
    started, done = time.time(), 0
    for frame_id, grp in todo.groupby("frame_id", sort=True):
        with Image.open(raw / f"{frame_id}.jpg") as im:
            image = im.convert("RGB")
        for row in grp.itertuples(index=False):
            box = np.array([[row.x1, row.y1, row.x2, row.y2]], dtype=float)
            store: dict[str, np.ndarray] = {}
            for name, (k, s_min) in ZOOMS.items():
                x0, y0, x1, y1 = zoom_window(box[0], k, s_min, *image.size)
                res = owl.run(image.crop((x0, y0, x1, y1)))
                tok = res["boxes"] + np.array([x0, y0, x0, y0], dtype=np.float32)
                keep = geometry.iou_matrix(box, tok.astype(float))[0] >= KEEP_IOU
                store[f"{name}_boxes"] = tok[keep]
                store[f"{name}_logits"] = res["logits"][keep]
                store[f"{name}_objectness"] = res["objectness"][keep]
                store[f"{name}_window"] = np.array([x0, y0, x1, y1], dtype=np.int32)
            tmp = cache_dir() / f"{row.id}.tmp.npz"
            np.savez_compressed(tmp, **store)
            tmp.replace(cache_dir() / f"{row.id}.npz")
            done += 1
            if done % log_every == 0 or done == len(todo):
                rate = (time.time() - started) / done
                print(f"owlv2[zoom] {done}/{len(todo)} detections, {rate:.1f} s each, eta {rate * (len(todo) - done) / 60:.1f} min",
                      flush=True)
    return done


def _one(box: np.ndarray, cls: str, cached: dict[str, np.ndarray]) -> dict[str, float]:
    groups = np.asarray(vision.GROUP_OF, dtype=object)
    out = {c: 0.0 for c in ZOOM_COLS}
    for i, name in enumerate(ZOOMS, 1):
        tok = cached[f"{name}_boxes"].astype(float)
        if len(tok) == 0:
            continue
        prob = 1.0 / (1.0 + np.exp(-cached[f"{name}_logits"].astype(float)))
        iou = geometry.iou_matrix(box[None, :], tok)[0]
        cols = groups == cls
        same = prob[:, cols].max(axis=1) if cols.any() else np.zeros(len(tok))
        soft = float((same * iou).max())
        out[f"ovz{i}_same_soft"] = soft
        out["ovz_same_soft"] = max(out["ovz_same_soft"], soft)
        close = iou >= 0.5
        if close.any():
            out["ovz_same"] = max(out["ovz_same"], float(same[close].max()))
            out["ovz_other"] = max(out["ovz_other"], float(prob[:, groups == "other"].max(axis=1)[close].max()))
        strong = same >= 0.05
        if strong.any():
            out["ovz_iou"] = max(out["ovz_iou"], float(iou[strong].max()))
    return out


def zoom_features(targets: pd.DataFrame) -> pd.DataFrame:
    """One row per target detection (aligned with ``targets``); NaN where the detection is not cached."""
    result = pd.DataFrame(np.nan, index=targets.index, columns=ZOOM_COLS)
    boxes = targets[BOX_COLS].to_numpy(float)
    classes = targets["class_name"].to_numpy(dtype=object)
    for pos, (idx, det_id) in enumerate(zip(targets.index, targets["id"])):
        path = cache_dir() / f"{det_id}.npz"
        if not path.is_file():
            continue
        with np.load(path) as npz:
            cached = {k: npz[k] for k in npz.files}
        result.loc[idx, ZOOM_COLS] = list(_one(boxes[pos], classes[pos], cached).values())
    return result


def scored_targets(part: str, raw: Path | None = None) -> pd.DataFrame:
    return data.non_table(data.load_targets(part, raw)).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache zoomed second-opinion views for every scored detection")
    parser.add_argument("--parts", default="train,test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--raw", type=Path, default=None)
    args = parser.parse_args(argv)
    targets = pd.concat([scored_targets(p.strip(), args.raw) for p in args.parts.split(",")], ignore_index=True)
    if args.limit:
        targets = targets.head(args.limit)
    done = process(targets, args.raw, threads=args.threads or None)
    cached = sum((cache_dir() / f"{i}.npz").is_file() for i in targets["id"])
    print(f"owlv2[zoom] cache: {cached}/{len(targets)} detections available ({done} computed now) in {cache_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
