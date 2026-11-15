"""Optional stage C3: an independent "second opinion" from a public pretrained open-vocabulary detector.

Model : OWLv2 base, patch 16, ensemble checkpoint  (Apache-2.0)
Source: https://huggingface.co/google/owlv2-base-patch16-ensemble

The audited detector's box is checked against what a completely independent model sees at
the same place: does it also find the claimed class there, how well do the boxes agree, and
does it rather see something else (phone, bottle, jacket ...)?  Everything is automatic and
label-free; test frames are processed exactly like train frames (TASK.txt allows using test
images for prediction and forbids only manual labelling).

Run:  python -m rascar.vision --parts train,test          (needs torch + transformers)

Raw model outputs are cached per frame under ``data/interim/vision/owlv2/`` so the tabular
pipeline can be re-run without torch. Nothing is uploaded anywhere; the only network access
is the one-time anonymous download of the public weights into ``models/hf_cache``.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, geometry, paths
from .data import BOX_COLS

MODEL_NAME = "google/owlv2-base-patch16-ensemble"
MODEL_URL = "https://huggingface.co/google/owlv2-base-patch16-ensemble"

# Fixed a priori (never tuned on labels): prompts for the claimed class and for typical look-alikes.
QUERIES: dict[str, list[str]] = {
    "person": ["a photo of a person", "a photo of a human"],
    "nude_hands": ["a photo of a hand", "a photo of a human hand"],
    "cup": ["a photo of a cup", "a photo of a mug", "a photo of a glass", "a photo of a paper cup"],
    "tray": ["a photo of a tray", "a photo of a wooden serving board"],
    "other": ["a photo of a phone", "a photo of a bottle", "a photo of a laptop", "a photo of a jacket",
              "a photo of a bag", "a photo of sunglasses", "a photo of a wallet", "a photo of a plate",
              "a photo of a teddy bear", "a photo of a napkin", "a photo of a glove", "a photo of a computer mouse",
              "a photo of a book", "a photo of a chair"],
}
TEXTS: list[str] = [q for group in QUERIES.values() for q in group]
GROUP_OF: list[str] = [g for g, group in QUERIES.items() for _ in group]


FINE_TILE = 480                            # optional extra views: 3 x 2 overlapping tiles, upscaled 2x by the processor


def cache_dir(kind: str = "base") -> Path:
    path = paths.interim_dir() / "vision" / ("owlv2" if kind == "base" else "owlv2_fine")
    path.mkdir(parents=True, exist_ok=True)
    return path


def view_windows(width: int, height: int, kind: str = "base") -> dict[str, tuple[int, int, int, int]]:
    """Crop windows (x0, y0, x1, y1). ``base``: full frame + two square tiles; ``fine``: six 480 px tiles."""
    if kind == "base":
        return {"full": (0, 0, width, height), "left": (0, 0, height, height), "right": (width - height, 0, width, height)}
    xs = [0, (width - FINE_TILE) // 2, width - FINE_TILE]
    ys = [0, height - FINE_TILE]
    return {f"t{r}{c}": (x, y, x + FINE_TILE, y + FINE_TILE) for r, y in enumerate(ys) for c, x in enumerate(xs)}


class Owl:
    """Thin wrapper around the public checkpoint (CPU inference)."""

    def __init__(self, threads: int | None = None):
        os.environ.setdefault("HF_HOME", str(paths.models_dir() / "hf_cache"))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.processor = Owlv2Processor.from_pretrained(MODEL_NAME)
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL_NAME).eval()

    def run(self, image) -> dict[str, np.ndarray]:
        """Raw per-token predictions for one PIL image: boxes in pixels of that image."""
        torch = self.torch
        inputs = self.processor(text=[TEXTS], images=image, return_tensors="pt")
        with torch.inference_mode():
            out = self.model(**inputs)
        side = float(max(image.size))                    # the processor pads to a square (bottom/right)
        cxcywh = out.pred_boxes[0].float().numpy() * side
        boxes = np.stack([cxcywh[:, 0] - cxcywh[:, 2] / 2, cxcywh[:, 1] - cxcywh[:, 3] / 2,
                          cxcywh[:, 0] + cxcywh[:, 2] / 2, cxcywh[:, 1] + cxcywh[:, 3] / 2], axis=1)
        return {"boxes": boxes.astype(np.float32),
                "logits": out.logits[0].float().numpy().astype(np.float16),
                "objectness": out.objectness_logits[0].float().numpy().astype(np.float16)}


def process_frames(frame_ids: list[str], raw: Path | None = None, kind: str = "base", threads: int | None = None,
                   log_every: int = 5) -> int:
    """Run the detector on every listed frame that is not cached yet (resumable)."""
    from PIL import Image

    raw = Path(raw) if raw else paths.raw_dir()
    todo = [f for f in frame_ids if not (cache_dir(kind) / f"{f}.npz").is_file()]
    if not todo:
        return 0
    owl = Owl(threads)
    started = time.time()
    for k, frame_id in enumerate(todo, 1):
        with Image.open(raw / f"{frame_id}.jpg") as im:
            image = im.convert("RGB")
        store: dict[str, np.ndarray] = {}
        for view, (x0, y0, x1, y1) in view_windows(*image.size, kind=kind).items():
            res = owl.run(image.crop((x0, y0, x1, y1)))
            res["boxes"] = res["boxes"] + np.array([x0, y0, x0, y0], dtype=np.float32)
            for key, value in res.items():
                store[f"{view}_{key}"] = value
        tmp = cache_dir(kind) / f"{frame_id}.tmp.npz"
        np.savez_compressed(tmp, **store)
        tmp.replace(cache_dir(kind) / f"{frame_id}.npz")
        if k % log_every == 0 or k == len(todo):
            rate = (time.time() - started) / k
            print(f"owlv2[{kind}] {k}/{len(todo)} frames, {rate:.1f} s/frame, eta {rate * (len(todo) - k) / 60:.1f} min", flush=True)
    return len(todo)


# --------------------------------------------------------------------------------------
# features from the cached raw outputs (numpy only)
# --------------------------------------------------------------------------------------
VISION_COLS = ["ov_same", "ov_same_soft", "ov_iou", "ov_other", "ov_margin", "ov_obj", "ov_best_iou_any",
               "ov_same_logit", "ov_has"]
FINE_COLS = ["ovf_same_soft", "ovf_same", "ovf_other"]     # base + fine views pooled (NaN without the fine cache)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def frame_vision_features(boxes: np.ndarray, classes: np.ndarray, cached: dict[str, np.ndarray],
                          match_iou: float = 0.5, min_score: float = 0.05) -> pd.DataFrame:
    """Features of the audited boxes of one frame.

    ov_same       highest second-opinion score for the claimed class among its boxes with IoU >= 0.5
    ov_same_soft  max over all its boxes of score(claimed class) * IoU
    ov_iou        IoU with its best box of the claimed class (score >= 0.05), a geometry agreement measure
    ov_other      highest score of any look-alike prompt among boxes with IoU >= 0.5
    ov_margin     logit(ov_same) - logit(ov_other)
    ov_obj        class-agnostic objectness of the best-overlapping box
    """
    n = len(boxes)
    out = {c: np.zeros(n) for c in VISION_COLS}
    views = sorted({k.split("_")[0] for k in cached})
    tok_boxes = np.concatenate([cached[f"{v}_boxes"] for v in views]).astype(np.float64)
    prob = _sigmoid(np.concatenate([cached[f"{v}_logits"] for v in views]))
    obj = _sigmoid(np.concatenate([cached[f"{v}_objectness"] for v in views]))
    groups = np.asarray(GROUP_OF, dtype=object)
    iou = geometry.iou_matrix(boxes, tok_boxes)
    other = prob[:, groups == "other"].max(axis=1)
    for i in range(n):
        cols = groups == classes[i]
        same = prob[:, cols].max(axis=1) if cols.any() else np.zeros(len(prob))
        close = iou[i] >= match_iou
        out["ov_same"][i] = same[close].max() if close.any() else 0.0
        out["ov_same_soft"][i] = float((same * iou[i]).max())
        cand = same >= min_score
        out["ov_iou"][i] = iou[i][cand].max() if cand.any() else 0.0
        out["ov_other"][i] = other[close].max() if close.any() else 0.0
        j = int(iou[i].argmax())
        out["ov_obj"][i] = obj[j]
        out["ov_best_iou_any"][i] = iou[i, j]
    clip = lambda p: np.clip(p, 1e-3, 1 - 1e-3)
    out["ov_same_logit"] = np.log(clip(out["ov_same"]) / (1 - clip(out["ov_same"])))
    out["ov_margin"] = out["ov_same_logit"] - np.log(clip(out["ov_other"]) / (1 - clip(out["ov_other"])))
    out["ov_has"] = np.ones(n)
    return pd.DataFrame(out)


def vision_features(targets: pd.DataFrame) -> pd.DataFrame:
    """One row per target detection (aligned with ``targets``); NaN where a frame is not cached.

    ``ov_*`` use the base views only. ``ovf_*`` pool the base and the fine views and exist
    only for frames that have both caches.
    """
    result = pd.DataFrame(np.nan, index=targets.index, columns=VISION_COLS + FINE_COLS)
    for frame_id, grp in targets.groupby("frame_id", sort=False):
        path = cache_dir() / f"{frame_id}.npz"
        if not path.is_file():
            continue
        with np.load(path) as npz:
            cached = {k: npz[k] for k in npz.files}
        boxes, classes = grp[BOX_COLS].to_numpy(float), grp["class_name"].to_numpy(dtype=object)
        feats = frame_vision_features(boxes, classes, cached)
        result.loc[grp.index, VISION_COLS] = feats.to_numpy()
        fine_path = cache_dir("fine") / f"{frame_id}.npz"
        if fine_path.is_file():
            with np.load(fine_path) as npz:
                cached.update({k: npz[k] for k in npz.files})
            both = frame_vision_features(boxes, classes, cached)
            result.loc[grp.index, FINE_COLS] = both[["ov_same_soft", "ov_same", "ov_other"]].to_numpy()
    return result


TEMPORAL_VISION_COLS = ["ovt_soft", "ovt_same", "ovt_n", "ovt_soft_nb"]


def temporal_vision_features(own: pd.DataFrame, match_ids: pd.DataFrame, stream: pd.DataFrame) -> pd.DataFrame:
    """Second opinion averaged over time, exactly like the smoothed confidence ``tc_*``.

    ``match_ids`` holds, per target, the id of the same-class detection that re-detects it in
    the previous / next context frame (IoU > 0.3) or None. The supporter is scored by the
    independent detector *in its own frame with its own box*; the target's value is then
    averaged with those of its supporters. Missing supporters / uncached frames are skipped.

    ovt_soft     mean of ov_same_soft over the target and its supporters
    ovt_same     the same for ov_same
    ovt_n        number of supporters that contributed (0..2)
    ovt_soft_nb  mean over the supporters only (the target's own value when there is none)
    """
    wanted = pd.unique(match_ids.to_numpy(dtype=object).ravel())
    wanted = [w for w in wanted if isinstance(w, str)]
    lookup_soft: dict[str, float] = {}
    lookup_same: dict[str, float] = {}
    if wanted:
        nb = stream[stream["detection_id"].isin(wanted)].reset_index(drop=True)
        feats = vision_features(nb)
        lookup_soft = dict(zip(nb["detection_id"], feats["ov_same_soft"]))
        lookup_same = dict(zip(nb["detection_id"], feats["ov_same"]))
    out = {}
    for name, lookup, own_col in (("soft", lookup_soft, "ov_same_soft"), ("same", lookup_same, "ov_same")):
        own_v = own[own_col].to_numpy(dtype=float)
        sides = [np.array([lookup.get(i, np.nan) if isinstance(i, str) else np.nan for i in match_ids[c]], dtype=float)
                 for c in match_ids.columns]
        stacked = np.vstack([own_v, *sides])
        cnt = (~np.isnan(stacked)).sum(axis=0)
        mean = np.where(cnt > 0, np.nansum(stacked, axis=0) / np.maximum(cnt, 1), np.nan)
        out[f"ovt_{name}"] = np.where(np.isnan(own_v), np.nan, mean)
        if name == "soft":
            nb_cnt = (~np.isnan(np.vstack(sides))).sum(axis=0)
            nb_mean = np.nansum(np.vstack(sides), axis=0) / np.maximum(nb_cnt, 1)
            out["ovt_n"] = np.where(np.isnan(own_v), np.nan, nb_cnt.astype(float))
            out["ovt_soft_nb"] = np.where(nb_cnt > 0, nb_mean, own_v)
    return pd.DataFrame(out, index=own.index)[TEMPORAL_VISION_COLS]


def target_frames(part: str, raw: Path | None = None, context: bool = False) -> list[str]:
    """Frames holding at least one scored-scope (non-table) target detection."""
    targets = data.non_table(data.load_targets(part, raw))
    frames = sorted(targets["frame_id"].unique())
    if context:
        ctx = data.load_context(part, raw).set_index("frame_id").loc[frames]
        frames = sorted(set(frames) | set(ctx["previous"].dropna()) | set(ctx["next"].dropna()))
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache OWLv2 second-opinion outputs for the target frames")
    parser.add_argument("--parts", default="train,test")
    parser.add_argument("--context", action="store_true", help="also process previous/next context frames")
    parser.add_argument("--fine", action="store_true", help="compute the six extra 480 px tiles instead of the base views")
    parser.add_argument("--limit", type=int, default=0, help="process at most N frames (smoke test)")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--raw", type=Path, default=None)
    args = parser.parse_args(argv)
    frames: list[str] = []
    for part in args.parts.split(","):
        frames.extend(target_frames(part.strip(), args.raw, args.context))
    if args.limit:
        frames = frames[: args.limit]
    kind = "fine" if args.fine else "base"
    done = process_frames(frames, args.raw, kind=kind, threads=args.threads or None)
    cached = sum((cache_dir(kind) / f"{f}.npz").is_file() for f in frames)
    print(f"owlv2[{kind}] cache: {cached}/{len(frames)} frames available ({done} computed now) in {cache_dir(kind)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
