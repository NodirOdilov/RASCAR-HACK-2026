"""g01 (Agent G, final_breakthrough, hypothesis H6): extra proposal views of the four public open-vocabulary detectors.

Copied and adapted from research/lead/g1_detectors_gpu.py (the original is not edited). Same fixed prompts / phrases,
same pre-processing and the same array format as the existing caches in data/interim/vision:
  `<view>_boxes`  float32 [N, 4]  x1, y1, x2, y2 in FULL-FRAME pixels
  `<view>_probs`  float16 [N, 14] Grounding DINO (one probability per phrase of rascar.vision_gdino.PHRASE_LIST)
  `<view>_logits` float16 [N, 24] + `<view>_objectness` float16 [N]   OWLv2 (queries of rascar.vision.TEXTS)

View sets
  flip   the existing 9 windows (full, left, right, t00 .. t12) mirrored horizontally before inference; boxes are
         mirrored back (x' = window_width - x) and shifted to full-frame coordinates.  Views Ffull, Fleft, ..., Ft12.
  hires  15 tiles of 384 x 384 px, x0 in {0, 224, 448, 672, 896}, y0 in {0, 168, 336}.     Views h00 .. h24 (row, col).

Caches: results/final_breakthrough/agent_G/cache/<model>_<viewset>/<frame>.npz - resumable (cached frames are skipped).
Train and test target frames are processed identically, by code only; nothing is displayed, uploaded or downloaded.

Usage (GPU environment, one model at a time):
  .venv_gpu/Scripts/python.exe -I -X utf8 research/final_breakthrough/agent_G/g01_cache_gpu.py --model gdinob --views hires
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

# Portable project root (the original working copy hard-coded the author's local folder here): RASCAR_ROOT if set,
# otherwise three levels above this file (research/final_breakthrough/agent_G/ -> project root).
ROOT = Path(os.environ.get("RASCAR_ROOT") or Path(__file__).resolve().parents[3]).resolve()
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import torch  # noqa: E402

from rascar import paths, vision, vision_gdino  # noqa: E402

torch.set_num_threads(2)
MODELS = {
    "gdinob": "IDEA-Research/grounding-dino-base",
    "gdino": "IDEA-Research/grounding-dino-tiny",
    "owlL": "google/owlv2-large-patch14-ensemble",
    "owlv2": "google/owlv2-base-patch16-ensemble",
}
OLD = ROOT / "data" / "interim" / "vision"                    # existing caches (read-only, self-test only)
OUT = ROOT / "results" / "final_breakthrough" / "agent_G"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HIRES_TILE, HIRES_X, HIRES_Y = 384, (0, 224, 448, 672, 896), (0, 168, 336)


class GdinoGpu:
    def __init__(self, name: str):
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

        self.processor = GroundingDinoProcessor.from_pretrained(name)
        self.model = GroundingDinoForObjectDetection.from_pretrained(name).eval().to(DEV)
        enc = self.processor.tokenizer(vision_gdino.CAPTION, return_offsets_mapping=True)
        self.token_index, cursor = [], 0
        for phrase in vision_gdino.PHRASE_LIST:
            start = vision_gdino.CAPTION.index(phrase, cursor)
            end = start + len(phrase)
            cursor = end
            self.token_index.append([i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start and b > a])
        assert all(self.token_index)

    @torch.inference_mode()
    def run(self, image) -> dict[str, np.ndarray]:
        inputs = self.processor(images=image, text=vision_gdino.CAPTION, return_tensors="pt").to(DEV)
        out = self.model(**inputs)
        prob = out.logits[0].sigmoid().float().cpu().numpy()
        phrase_prob = np.stack([prob[:, idx].max(axis=1) for idx in self.token_index], axis=1)
        width, height = image.size
        c = out.pred_boxes[0].float().cpu().numpy() * np.array([width, height, width, height], dtype=np.float32)
        boxes = np.stack([c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2, c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2], axis=1)
        return {"boxes": boxes.astype(np.float32), "probs": phrase_prob.astype(np.float16)}


class OwlGpu:
    def __init__(self, name: str):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.processor = Owlv2Processor.from_pretrained(name)
        self.model = Owlv2ForObjectDetection.from_pretrained(name, torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).eval().to(DEV)

    @torch.inference_mode()
    def run(self, image) -> dict[str, np.ndarray]:
        inputs = self.processor(text=[vision.TEXTS], images=image, return_tensors="pt").to(DEV)
        if DEV == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half()
        out = self.model(**inputs)
        side = float(max(image.size))                                   # the processor pads to a square (bottom / right)
        c = out.pred_boxes[0].float().cpu().numpy() * side
        boxes = np.stack([c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2, c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2], axis=1)
        return {"boxes": boxes.astype(np.float32), "logits": out.logits[0].float().cpu().numpy().astype(np.float16),
                "objectness": out.objectness_logits[0].float().cpu().numpy().astype(np.float16)}


def windows(viewset: str, width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    if viewset == "flip":
        w = {**vision.view_windows(width, height, "base"), **vision.view_windows(width, height, "fine")}
        return {f"F{k}": v for k, v in w.items()}
    assert (width, height) == (1280, 720), (width, height)
    return {f"h{r}{c}": (x, y, x + HIRES_TILE, y + HIRES_TILE) for r, y in enumerate(HIRES_Y) for c, x in enumerate(HIRES_X)}


def run_view(det, image, window, flip: bool) -> dict[str, np.ndarray]:
    """Inference on one crop; boxes returned in full-frame pixels (mirrored back when the crop was flipped)."""
    from PIL import Image

    x0, y0, x1, y1 = window
    crop = image.crop((x0, y0, x1, y1))
    if flip:
        crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
    res = det.run(crop)
    b = res["boxes"]
    if flip:
        w = np.float32(x1 - x0)
        b = np.stack([w - b[:, 2], b[:, 1], w - b[:, 0], b[:, 3]], axis=1)
    res["boxes"] = (b + np.array([x0, y0, x0, y0], dtype=np.float32)).astype(np.float32)
    return res


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64); b = b.astype(np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-12)


def _scores(res: dict, kind: str) -> np.ndarray:
    """Highest probability of a scored-class query per proposal (GD: phrases 0..5, OWL: queries 0..9)."""
    if kind == "gd":
        return res["probs"][:, :6].astype(np.float64).max(axis=1)
    return (1.0 / (1.0 + np.exp(-res["logits"][:, :10].astype(np.float64)))).max(axis=1)


def selftest(det, model: str, frame_id: str, image) -> dict:
    """(1) the unflipped full view must reproduce the existing cache; (2) the mirrored full view and the hires tiles,
    mapped back, must land on the same objects as the cached confident proposals (checks the coordinate mapping)."""
    kind = "gd" if model.startswith("gdino") else "owl"
    with np.load(OLD / model / f"{frame_id}.npz") as z:
        old = {k[5:]: z[k] for k in z.files if k.startswith("full_")}
    with np.load(OLD / f"{model}_fine" / f"{frame_id}.npz") as z:
        fine = {v: {"boxes": z[f"{v}_boxes"], ("probs" if kind == "gd" else "logits"): z[f"{v}_probs" if kind == "gd" else f"{v}_logits"]}
                for v in sorted({k.rsplit("_", 1)[0] for k in z.files})}
    full_w = (0, 0, *image.size)
    new = run_view(det, image, full_w, flip=False)
    s_old, s_new = _scores(old, kind), _scores(new, kind)
    rec = {"frame": frame_id, "n": int(len(s_old))}
    if new["boxes"].shape == old["boxes"].shape:
        rec["same_order_max_box_diff_px"] = float(np.abs(new["boxes"] - old["boxes"]).max())
        rec["same_order_max_score_diff"] = float(np.abs(s_new - s_old).max())
    top = np.argsort(-s_old)[:30]
    rec["top30_old_score_min"] = float(s_old[top].min())
    m = _iou(old["boxes"][top], new["boxes"][s_new >= 0.5 * s_old[top].min()])
    rec["rerun_top30_best_iou_median"] = float(np.median(m.max(axis=1)))
    fl = run_view(det, image, full_w, flip=True)
    s_fl = _scores(fl, kind)
    m = _iou(old["boxes"][top], fl["boxes"][s_fl >= 0.5 * s_old[top].min()])
    rec["flip_top30_best_iou_median"] = float(np.median(m.max(axis=1)))
    # negative control: without mirroring back the same statistic must collapse
    wrong = fl["boxes"].copy(); wrong[:, [0, 2]] = image.size[0] - fl["boxes"][:, [2, 0]]
    rec["flip_NOT_mirrored_back_control"] = float(np.median(_iou(old["boxes"][top], wrong[s_fl >= 0.5 * s_old[top].min()]).max(axis=1)))
    # hires tile h11 = (224, 168, 608, 552) against the cached 480 px tiles: confident cached proposals fully inside the tile
    hw = windows("hires", *image.size)["h11"]
    hi = run_view(det, image, hw, flip=False)
    s_hi = _scores(hi, kind)
    fb_boxes = np.concatenate([v["boxes"] for v in fine.values()])
    fb_s = np.concatenate([_scores(v, kind) for v in fine.values()])
    inside = (fb_boxes[:, 0] >= hw[0] + 8) & (fb_boxes[:, 1] >= hw[1] + 8) & (fb_boxes[:, 2] <= hw[2] - 8) & (fb_boxes[:, 3] <= hw[3] - 8)
    idx = np.flatnonzero(inside)
    idx = idx[np.argsort(-fb_s[idx])[:15]]
    if len(idx):
        m = _iou(fb_boxes[idx], hi["boxes"][s_hi >= 0.25 * fb_s[idx].min()]) if (s_hi >= 0.25 * fb_s[idx].min()).any() else np.zeros((len(idx), 1))
        rec["hires_h11_top15_inside_best_iou_median"] = float(np.median(m.max(axis=1)))
        rec["hires_h11_top15_old_score_min"] = float(fb_s[idx].min())
    return rec


def main() -> int:
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--views", choices=["flip", "hires"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--part", choices=["all", "train", "test"], default="all",
                        help="frame subset of this call (scheduling only: every frame goes through the same code path)")
    parser.add_argument("--max-minutes", type=float, default=0.0, help="stop (resumable) after this many minutes of inference")
    parser.add_argument("--no-selftest", action="store_true")
    args = parser.parse_args()
    f_train, f_test = vision.target_frames("train"), vision.target_frames("test")
    frames = f_train + f_test
    assert len(frames) == 121 and len(set(frames)) == 121, len(frames)
    selftest_frame = f_train[0]
    if args.part != "all":
        frames = f_train if args.part == "train" else f_test
    if args.limit:
        frames = frames[: args.limit]
    raw = paths.raw_dir()
    out_dir = OUT / "cache" / f"{args.model}_{args.views}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    todo = [f for f in frames if not (out_dir / f"{f}.npz").is_file()]
    print(f"{args.model} = {MODELS[args.model]} [{args.views}] on {DEV}; {len(frames)} frames; {len(todo)} to do; torch {torch.__version__}", flush=True)
    if not todo:
        return 0
    t_load = time.time()
    det = GdinoGpu(MODELS[args.model]) if args.model.startswith("gdino") else OwlGpu(MODELS[args.model])
    print(f"  model loaded in {time.time() - t_load:.1f} s; GPU allocated {torch.cuda.memory_allocated() / 2**20:.0f} MiB" if DEV == "cuda" else "  CPU", flush=True)
    if not args.no_selftest:
        with Image.open(raw / f"{selftest_frame}.jpg") as im:         # always a TRAIN frame
            st = selftest(det, args.model, selftest_frame, im.convert("RGB"))
        print("  selftest " + json.dumps(st), flush=True)
        (log_dir / f"g01_selftest_{args.model}.json").write_text(json.dumps(st, indent=1))
    started = time.time()
    per_view: dict[str, list[float]] = {}
    min_free = float("inf")
    timing_rows = []
    for n, frame_id in enumerate(todo, 1):
        t_frame = time.time()
        with Image.open(raw / f"{frame_id}.jpg") as im:
            image = im.convert("RGB")
        store = {}
        for view, win in windows(args.views, *image.size).items():
            t0 = time.time()
            res = run_view(det, image, win, flip=args.views == "flip")
            per_view.setdefault(view, []).append(time.time() - t0)
            for key, value in res.items():
                store[f"{view}_{key}"] = value
        tmp = out_dir / f"{frame_id}.tmp.npz"
        np.savez_compressed(tmp, **store)
        tmp.replace(out_dir / f"{frame_id}.npz")
        if DEV == "cuda":
            min_free = min(min_free, torch.cuda.mem_get_info()[0] / 2**20)
        timing_rows.append((frame_id, time.time() - t_frame))
        elapsed = time.time() - started
        if n % 10 == 0 or n == len(todo):
            rate = elapsed / n
            mem = torch.cuda.max_memory_allocated() / 2**20 if DEV == "cuda" else 0
            res_mem = torch.cuda.max_memory_reserved() / 2**20 if DEV == "cuda" else 0
            v_mean = float(np.mean([np.mean(v) for v in per_view.values()]))
            print(f"  {n}/{len(todo)} frames, {rate:.2f} s/frame, {v_mean:.3f} s/view (inference), eta {rate * (len(todo) - n) / 60:.1f} min, "
                  f"peak GPU allocated {mem:.0f} MiB, reserved {res_mem:.0f} MiB, min device free {min_free:.0f} MiB", flush=True)
        if args.max_minutes and elapsed > args.max_minutes * 60 and n < len(todo):
            print(f"  STOPPED by --max-minutes after {n} frames ({elapsed / 60:.1f} min); resumable", flush=True)
            break
    elapsed = time.time() - started
    summary = {
        "model": args.model, "hf_name": MODELS[args.model], "views": args.views, "part": args.part, "device": DEV,
        "minutes_model_load_and_selftest": (started - t_load) / 60,
        "frames_done_now": len(timing_rows), "frames_cached": sum((out_dir / f"{f}.npz").is_file() for f in frames), "frames_total": len(frames),
        "minutes_inference_loop": elapsed / 60, "seconds_per_frame": elapsed / max(len(timing_rows), 1),
        "seconds_per_view": {v: float(np.mean(t)) for v, t in per_view.items()},
        "seconds_per_view_mean": float(np.mean([np.mean(v) for v in per_view.values()])) if per_view else None,
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if DEV == "cuda" else 0,
        "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if DEV == "cuda" else 0,
        "min_device_free_mib": min_free if DEV == "cuda" else None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    stamp = time.strftime("%H%M%S")
    (log_dir / f"g01_summary_{args.model}_{args.views}_{stamp}.json").write_text(json.dumps(summary, indent=1))
    print("  summary " + json.dumps({k: v for k, v in summary.items() if k != "seconds_per_view"}), flush=True)
    print("  seconds per view " + json.dumps({k: round(v, 3) for k, v in summary["seconds_per_view"].items()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
