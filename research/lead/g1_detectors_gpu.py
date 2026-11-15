"""More independent public detectors on the GPU (round 3, hypothesis G1).

Hypothesis (written before the first result): the label asks "is there an annotated object of this class at this
place"; every public open-vocabulary detector answers a noisy version of that question and the detectors err
differently. Round 2 showed it for two detectors (OWLv2 base + Grounding DINO tiny). Larger models of both families
should give a less noisy consensus. Expected sign: higher consensus -> lower error probability.

Models (public weights, Apache-2.0, used zero-shot with the SAME fixed prompts / phrases as their smaller siblings):
  gdinob  IDEA-Research/grounding-dino-base           https://huggingface.co/IDEA-Research/grounding-dino-base
  owlL    google/owlv2-large-patch14-ensemble         https://huggingface.co/google/owlv2-large-patch14-ensemble
Train and test target frames are processed identically, by code only; nothing is uploaded.

Usage (GPU environment):  .venv_gpu/Scripts/python.exe -I research/lead/g1_detectors_gpu.py --model gdinob|owlL
Caches: data/interim/vision/<model>/<frame>.npz (3 frame views) and data/interim/vision/<model>_fine/ (6 tiles),
in the same array format as the caches of the smaller models.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from rascar import paths, vision, vision_gdino

os.environ.setdefault("HF_HOME", str(paths.models_dir() / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import torch  # noqa: E402

MODELS = {"gdinob": "IDEA-Research/grounding-dino-base", "owlL": "google/owlv2-large-patch14-ensemble"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


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


def main() -> int:
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    frames = vision.target_frames("train") + vision.target_frames("test")
    if args.limit:
        frames = frames[: args.limit]
    raw = paths.raw_dir()
    dirs = {"base": paths.interim_dir() / "vision" / args.model, "fine": paths.interim_dir() / "vision" / f"{args.model}_fine"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    todo = [(f, k) for k in ("base", "fine") for f in frames if not (dirs[k] / f"{f}.npz").is_file()]
    print(f"{args.model} = {MODELS[args.model]} on {DEV}; {len(frames)} frames; {len(todo)} frame/view-set jobs to do", flush=True)
    if not todo:
        return 0
    det = GdinoGpu(MODELS[args.model]) if args.model.startswith("gdino") else OwlGpu(MODELS[args.model])
    started = time.time()
    for n, (frame_id, kind) in enumerate(todo, 1):
        with Image.open(raw / f"{frame_id}.jpg") as im:
            image = im.convert("RGB")
        store = {}
        for view, (x0, y0, x1, y1) in vision.view_windows(*image.size, kind=kind).items():
            res = det.run(image.crop((x0, y0, x1, y1)))
            res["boxes"] = res["boxes"] + np.array([x0, y0, x0, y0], dtype=np.float32)
            for key, value in res.items():
                store[f"{view}_{key}"] = value
        tmp = dirs[kind] / f"{frame_id}.tmp.npz"
        np.savez_compressed(tmp, **store)
        tmp.replace(dirs[kind] / f"{frame_id}.npz")
        if n % 20 == 0 or n == len(todo):
            rate = (time.time() - started) / n
            mem = torch.cuda.max_memory_allocated() / 2**20 if DEV == "cuda" else 0
            print(f"  {n}/{len(todo)} jobs, {rate:.1f} s each, eta {rate * (len(todo) - n) / 60:.1f} min, peak GPU memory {mem:.0f} MiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
