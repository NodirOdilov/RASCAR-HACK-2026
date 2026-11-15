"""Shared fixtures: the real competition data (skipped when absent) and a tiny synthetic
dataset with the same schema, written to a temporary directory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rascar import paths

REAL_RAW = paths.project_root() / "data" / "raw" / paths.DATASET_DIRNAME


@pytest.fixture(scope="session")
def real_raw() -> Path:
    if not (REAL_RAW / "train.csv").is_file():
        pytest.skip("real competition data is not available")
    return REAL_RAW


def _frame_id(day: str, slot: int) -> str:
    seconds = slot * 30
    return f"GENERAL_synth_1_5_{day}_{seconds // 3600:02d}{seconds % 3600 // 60:02d}{seconds % 60:02d}"


def _timestamp(day: str, slot: int) -> str:
    seconds = slot * 30
    return (f"{day[:4]}-{day[4:6]}-{day[6:]} {seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
            ".949000+00:00")


def make_synthetic(root: Path, seed: int = 0, n_targets_per_day: int = 6) -> Path:
    """Write a miniature dataset. Rule used to create labels: low-confidence boxes and boxes that
    flicker between frames are errors (plus noise), so models have something to learn."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    classes = ["person", "nude_hands", "cup", "tray"]
    frames, context, stream, human = {"train": [], "test": [], "machine": []}, {"train": [], "test": []}, \
        {"train": [], "test": [], "machine": []}, []
    targets = {"train": [], "test": []}
    det_counter = [0]

    def new_id() -> str:
        det_counter[0] += 1
        return f"{det_counter[0]:016x}"

    def add_frame(part: str, day: str, slot: int, size=(1280, 720)) -> str:
        fid = _frame_id(day, slot)
        frames[part].append({"frame_id": fid, "timestamp": _timestamp(day, slot), "location_id": "synth_1", "camera_id": 5,
                             "image_path": fid + ".jpg", "image_width": size[0], "image_height": size[1],
                             "image_exists": True, "day": int(day)})
        return fid

    def add_det(part: str, fid: str, cls: str, conf: float, box) -> str:
        did = new_id()
        stream[part].append({"detection_id": did, "frame_id": fid, "class_name": cls, "confidence": round(float(conf), 2),
                             "x1": round(float(box[0]), 1), "y1": round(float(box[1]), 1),
                             "x2": round(float(box[2]), 1), "y2": round(float(box[3]), 1)})
        return did

    for part, days in (("train", ["20261025", "20261026"]), ("test", ["20261027", "20261028"])):
        for day in days:
            for k in range(n_targets_per_day):
                slot = 600 + k * 60
                prev_f, tgt_f, next_f = (add_frame(part, day, slot - 1), add_frame(part, day, slot), add_frame(part, day, slot + 1))
                first = part == "test" and k == 0
                context[part].append({"frame_id": tgt_f, "previous": None if first else prev_f, "next": next_f})
                for f in (prev_f, tgt_f, next_f):
                    add_det(part, f, "table", 0.95, (400, 300, 600, 420))
                for _ in range(int(rng.integers(3, 7))):
                    cls = classes[int(rng.integers(0, 4))]
                    w, h = (150, 170) if cls == "person" else (32, 28)
                    x1, y1 = rng.uniform(50, 1000), rng.uniform(150, 450)
                    box = np.array([x1, y1, x1 + w, y1 + h])
                    quality = rng.uniform(0, 1)                      # latent "this is a good detection"
                    conf = float(np.clip(0.45 + 0.45 * quality + rng.normal(0, 0.05), 0.41, 0.97))
                    did = add_det(part, tgt_f, cls, conf, box)
                    for f in (prev_f, next_f):                       # good objects persist with similar confidence
                        if rng.uniform() < 0.4 + 0.55 * quality:
                            jitter = rng.normal(0, 2.0, 4)
                            add_det(part, f, cls, float(np.clip(conf + rng.normal(0, 0.05), 0.41, 0.97)), box + jitter)
                    row = dict(stream[part][-1]) if False else {"id": did, "frame_id": tgt_f, "class_name": cls,
                                                                "confidence": round(conf, 2), "x1": round(box[0], 1),
                                                                "y1": round(box[1], 1), "x2": round(box[2], 1), "y2": round(box[3], 1)}
                    if part == "train":
                        correct = quality + rng.normal(0, 0.15) > 0.5
                        row["error"] = int(not correct)
                        if correct:                                  # manual box that matches with IoU >= 0.6
                            b = np.array([row["x1"], row["y1"], row["x2"], row["y2"]]) + rng.normal(0, 1.0, 4)
                            human.append({"frame_id": tgt_f, "class_name": "nude_hand" if cls == "nude_hands" else cls,
                                          "cx": (b[0] + b[2]) / 2 / 1280, "cy": (b[1] + b[3]) / 2 / 720,
                                          "w": (b[2] - b[0]) / 1280, "h": (b[3] - b[1]) / 720,
                                          "evaluated_class": True, "source": "annotator"})
                    targets[part].append(row)
                if part == "train":                                  # tables are targets in train, matched by a manual box
                    for det in [d for d in stream[part] if d["frame_id"] == tgt_f and d["class_name"] == "table"]:
                        targets[part].append({"id": det["detection_id"], "frame_id": tgt_f, "class_name": "table",
                                              "confidence": det["confidence"], "x1": det["x1"], "y1": det["y1"],
                                              "x2": det["x2"], "y2": det["y2"], "error": 0})
                        human.append({"frame_id": tgt_f, "class_name": "table", "cx": 500 / 1280, "cy": 360 / 720,
                                      "w": 200 / 1280, "h": 120 / 720, "evaluated_class": True, "source": "annotator"})
    for day in ("20261025", "20261026"):
        for slot in range(100, 160):
            fid = add_frame("machine", day, slot, size=(640, 360))
            add_det("machine", fid, "table", 0.95, (400, 300, 600, 420))
            for cls in classes:
                w, h = (150, 170) if cls == "person" else (32, 28)
                x1, y1 = rng.uniform(50, 1000), rng.uniform(150, 450)
                add_det("machine", fid, cls, float(rng.uniform(0.41, 0.95)), (x1, y1, x1 + w, y1 + h))

    pd.DataFrame(targets["train"]).to_csv(root / "train.csv", index=False)
    test = pd.DataFrame(targets["test"])
    test.to_csv(root / "test.csv", index=False)
    sample = test[["id"]].sample(frac=1.0, random_state=1).assign(score=0.5)
    sample.to_csv(root / "sample_submission.csv", index=False)
    for part in ("train", "test"):
        pd.DataFrame(frames[part]).to_csv(root / f"{part}_frames.csv", index=False)
        pd.DataFrame(context[part]).to_csv(root / f"{part}_context.csv", index=False)
        pd.DataFrame(stream[part]).to_csv(root / f"{part}_stream_predictions.csv", index=False)
    pd.DataFrame(frames["machine"]).drop(columns="day").to_csv(root / "machine_frames.csv", index=False)
    pd.DataFrame(stream["machine"]).to_csv(root / "machine_predictions.csv", index=False)
    pd.DataFrame(human).to_csv(root / "human_boxes_train.csv", index=False)
    policy = {"classes": ["cup", "tray", "person", "nude_hands", "table", "bottle", "napkin"],
              "aliases": {"nude_hand": "nude_hands", "white_napkin": "napkin"}, "correct_iou": 0.6,
              "positive_label": "error", "metric": "sklearn average_precision_score", "test_excluded_classes": ["table"]}
    (root / "scoring_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    return root


@pytest.fixture()
def synth_raw(tmp_path, monkeypatch) -> Path:
    """Synthetic raw directory + isolated project root, so tests never touch real outputs."""
    raw = make_synthetic(tmp_path / "raw")
    monkeypatch.setenv("RASCAR_DATA_DIR", str(raw))
    monkeypatch.setenv("RASCAR_ROOT", str(tmp_path / "project"))
    (tmp_path / "project").mkdir()
    return raw
