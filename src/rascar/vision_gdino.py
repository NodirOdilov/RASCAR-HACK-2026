"""Third opinion: a second public open-vocabulary detector with a different architecture.

Model : Grounding DINO tiny (Swin-T backbone, DETR-style decoder, BERT text encoder; Apache-2.0)
Source: https://huggingface.co/IDEA-Research/grounding-dino-tiny

Used zero-shot exactly like the OWLv2 second opinion (:mod:`rascar.vision`): the audited box is
compared with every box the model proposes, and the agreement is
``max(score for the claimed class x IoU with the audited box)``. Phrases are fixed a priori and
never tuned on labels; train and test frames are processed identically; nothing is uploaded.

Run:  python -m rascar.vision_gdino --parts train,test --mode base|fine|zoom
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, geometry, paths, vision, vision_zoom
from .data import BOX_COLS

MODEL_NAME = "IDEA-Research/grounding-dino-tiny"
MODEL_URL = "https://huggingface.co/IDEA-Research/grounding-dino-tiny"

PHRASES: dict[str, list[str]] = {
    "person": ["person"],
    "nude_hands": ["hand"],
    "cup": ["cup", "mug", "glass"],
    "tray": ["tray"],
    "other": ["phone", "bottle", "laptop", "jacket", "bag", "plate", "napkin", "chair"],
}
PHRASE_LIST: list[str] = [p for group in PHRASES.values() for p in group]
GROUP_OF: list[str] = [g for g, group in PHRASES.items() for _ in group]
CAPTION = " . ".join(PHRASE_LIST) + " ."
MODES = ("base", "fine", "zoom")
GDINO_COLS = ["gd_same_soft", "gd_same", "gd_other", "gd_iou"]


def cache_dir(mode: str) -> Path:
    path = paths.interim_dir() / "vision" / {"base": "gdino", "fine": "gdino_fine", "zoom": "gdino_zoom"}[mode]
    path.mkdir(parents=True, exist_ok=True)
    return path


class Gdino:
    def __init__(self, threads: int | None = None):
        os.environ.setdefault("HF_HOME", str(paths.models_dir() / "hf_cache"))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        import torch
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.processor = GroundingDinoProcessor.from_pretrained(MODEL_NAME)
        self.model = GroundingDinoForObjectDetection.from_pretrained(MODEL_NAME).eval()
        enc = self.processor.tokenizer(CAPTION, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        self.token_index: list[list[int]] = []
        cursor = 0
        for phrase in PHRASE_LIST:                       # character span of every phrase inside the caption
            start = CAPTION.index(phrase, cursor)
            end = start + len(phrase)
            cursor = end
            self.token_index.append([i for i, (a, b) in enumerate(offsets) if a < end and b > start and b > a])
        if any(len(t) == 0 for t in self.token_index):
            raise RuntimeError("a phrase has no token in the caption")

    def run(self, image) -> dict[str, np.ndarray]:
        """Per-query predictions for one PIL image: boxes in pixels, one probability per phrase."""
        torch = self.torch
        inputs = self.processor(images=image, text=CAPTION, return_tensors="pt")
        with torch.inference_mode():
            out = self.model(**inputs)
        prob = out.logits[0].sigmoid().float().numpy()           # (queries, text tokens)
        phrase_prob = np.stack([prob[:, idx].max(axis=1) for idx in self.token_index], axis=1)
        width, height = image.size
        cxcywh = out.pred_boxes[0].float().numpy() * np.array([width, height, width, height], dtype=np.float32)
        boxes = np.stack([cxcywh[:, 0] - cxcywh[:, 2] / 2, cxcywh[:, 1] - cxcywh[:, 3] / 2,
                          cxcywh[:, 0] + cxcywh[:, 2] / 2, cxcywh[:, 1] + cxcywh[:, 3] / 2], axis=1)
        return {"boxes": boxes.astype(np.float32), "probs": phrase_prob.astype(np.float16)}


def process_frames(frame_ids: list[str], mode: str, raw: Path | None = None, threads: int | None = None) -> int:
    from PIL import Image

    raw = Path(raw) if raw else paths.raw_dir()
    todo = [f for f in frame_ids if not (cache_dir(mode) / f"{f}.npz").is_file()]
    if not todo:
        return 0
    det = Gdino(threads)
    started = time.time()
    for k, frame_id in enumerate(todo, 1):
        with Image.open(raw / f"{frame_id}.jpg") as im:
            image = im.convert("RGB")
        store: dict[str, np.ndarray] = {}
        for view, (x0, y0, x1, y1) in vision.view_windows(*image.size, kind=mode).items():
            res = det.run(image.crop((x0, y0, x1, y1)))
            store[f"{view}_boxes"] = res["boxes"] + np.array([x0, y0, x0, y0], dtype=np.float32)
            store[f"{view}_probs"] = res["probs"]
        tmp = cache_dir(mode) / f"{frame_id}.tmp.npz"
        np.savez_compressed(tmp, **store)
        tmp.replace(cache_dir(mode) / f"{frame_id}.npz")
        if k % 5 == 0 or k == len(todo):
            rate = (time.time() - started) / k
            print(f"gdino[{mode}] {k}/{len(todo)} frames, {rate:.1f} s/frame, eta {rate * (len(todo) - k) / 60:.1f} min", flush=True)
    return len(todo)


def process_zoom(targets: pd.DataFrame, raw: Path | None = None, threads: int | None = None) -> int:
    from PIL import Image

    raw = Path(raw) if raw else paths.raw_dir()
    todo = targets[[not (cache_dir("zoom") / f"{i}.npz").is_file() for i in targets["id"]]]
    if todo.empty:
        return 0
    det = Gdino(threads)
    started, done = time.time(), 0
    for frame_id, grp in todo.groupby("frame_id", sort=True):
        with Image.open(raw / f"{frame_id}.jpg") as im:
            image = im.convert("RGB")
        for row in grp.itertuples(index=False):
            box = np.array([[row.x1, row.y1, row.x2, row.y2]], dtype=float)
            store: dict[str, np.ndarray] = {}
            for name, (k, s_min) in vision_zoom.ZOOMS.items():
                x0, y0, x1, y1 = vision_zoom.zoom_window(box[0], k, s_min, *image.size)
                res = det.run(image.crop((x0, y0, x1, y1)))
                tok = res["boxes"] + np.array([x0, y0, x0, y0], dtype=np.float32)
                keep = geometry.iou_matrix(box, tok.astype(float))[0] >= vision_zoom.KEEP_IOU
                store[f"{name}_boxes"] = tok[keep]
                store[f"{name}_probs"] = res["probs"][keep]
            tmp = cache_dir("zoom") / f"{row.id}.tmp.npz"
            np.savez_compressed(tmp, **store)
            tmp.replace(cache_dir("zoom") / f"{row.id}.npz")
            done += 1
            if done % 20 == 0 or done == len(todo):
                rate = (time.time() - started) / done
                print(f"gdino[zoom] {done}/{len(todo)} detections, {rate:.1f} s each, eta {rate * (len(todo) - done) / 60:.1f} min",
                      flush=True)
    return done


def agreement(box: np.ndarray, cls: str, cached: dict[str, np.ndarray]) -> dict[str, float]:
    """Agreement of one audited box with the proposals stored in ``cached`` (keys ``<view>_boxes`` / ``<view>_probs``)."""
    groups = np.asarray(GROUP_OF, dtype=object)
    views = sorted({k.rsplit("_", 1)[0] for k in cached if k.endswith("_boxes")})
    out = {c: 0.0 for c in GDINO_COLS}
    if not views:
        return out
    tok = np.concatenate([cached[f"{v}_boxes"] for v in views]).astype(float)
    if len(tok) == 0:
        return out
    prob = np.concatenate([cached[f"{v}_probs"] for v in views]).astype(float)
    iou = geometry.iou_matrix(box[None, :], tok)[0]
    cols = groups == cls
    same = prob[:, cols].max(axis=1) if cols.any() else np.zeros(len(tok))
    out["gd_same_soft"] = float((same * iou).max())
    close = iou >= 0.5
    if close.any():
        out["gd_same"] = float(same[close].max())
        out["gd_other"] = float(prob[:, groups == "other"].max(axis=1)[close].max())
    strong = same >= 0.1
    if strong.any():
        out["gd_iou"] = float(iou[strong].max())
    return out


def gdino_features(targets: pd.DataFrame) -> pd.DataFrame:
    """Columns ``gd_*`` (frame views: base, and base + fine when cached) and ``gdz_*`` (zoom views).
    NaN where the corresponding cache is missing."""
    names = GDINO_COLS + [c.replace("gd_", "gdf_") for c in GDINO_COLS] + [c.replace("gd_", "gdz_") for c in GDINO_COLS]
    result = pd.DataFrame(np.nan, index=targets.index, columns=names)
    boxes = targets[BOX_COLS].to_numpy(float)
    classes = targets["class_name"].to_numpy(dtype=object)
    frame_cache: dict[tuple[str, str], dict | None] = {}

    def load(mode: str, key: str):
        if (mode, key) not in frame_cache:
            path = cache_dir(mode) / f"{key}.npz"
            if path.is_file():
                with np.load(path) as npz:
                    frame_cache[(mode, key)] = {k: npz[k] for k in npz.files}
            else:
                frame_cache[(mode, key)] = None
        return frame_cache[(mode, key)]

    for pos, (idx, det_id, frame_id) in enumerate(zip(targets.index, targets["id"], targets["frame_id"])):
        base = load("base", frame_id)
        if base is not None:
            result.loc[idx, GDINO_COLS] = list(agreement(boxes[pos], classes[pos], base).values())
            fine = load("fine", frame_id)
            if fine is not None:
                both = {**base, **fine}
                result.loc[idx, [c.replace("gd_", "gdf_") for c in GDINO_COLS]] = list(agreement(boxes[pos], classes[pos], both).values())
        zoom_path = cache_dir("zoom") / f"{det_id}.npz"
        if zoom_path.is_file():
            with np.load(zoom_path) as npz:
                cached = {k: npz[k] for k in npz.files}
            result.loc[idx, [c.replace("gd_", "gdz_") for c in GDINO_COLS]] = list(agreement(boxes[pos], classes[pos], cached).values())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache Grounding DINO third-opinion outputs")
    parser.add_argument("--parts", default="train,test")
    parser.add_argument("--mode", choices=MODES, default="base")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--raw", type=Path, default=None)
    args = parser.parse_args(argv)
    parts = [p.strip() for p in args.parts.split(",")]
    if args.mode == "zoom":
        targets = pd.concat([vision_zoom.scored_targets(p, args.raw) for p in parts], ignore_index=True)
        if args.limit:
            targets = targets.head(args.limit)
        done = process_zoom(targets, args.raw, threads=args.threads or None)
        cached = sum((cache_dir("zoom") / f"{i}.npz").is_file() for i in targets["id"])
        print(f"gdino[zoom] cache: {cached}/{len(targets)} detections ({done} computed now)")
    else:
        frames: list[str] = []
        for p in parts:
            frames.extend(vision.target_frames(p, args.raw))
        if args.limit:
            frames = frames[: args.limit]
        done = process_frames(frames, args.mode, args.raw, threads=args.threads or None)
        cached = sum((cache_dir(args.mode) / f"{f}.npz").is_file() for f in frames)
        print(f"gdino[{args.mode}] cache: {cached}/{len(frames)} frames ({done} computed now)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
