"""Loading and light validation of the raw competition tables.

All readers treat the files as untrusted data: they are parsed with pandas only,
never executed. Identifier columns are always read as strings.

Tables that carry labels or manual annotation (``train.csv:error`` and
``human_boxes_train.csv``) are only reachable through :func:`load_targets` with
``part="train"`` and :func:`load_human_boxes`; nothing on the inference path
calls them (enforced by ``tests/test_leakage.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

BOX_COLS = ["x1", "y1", "x2", "y2"]
DET_COLS = ["frame_id", "class_name", "confidence", *BOX_COLS]
FRAME_W, FRAME_H = 1280, 720  # coordinate system of every prediction table (TASK.txt)

REQUIRED = {
    "train": ["id", *DET_COLS, "error"],
    "test": ["id", *DET_COLS],
    "sample_submission": ["id", "score"],
    "train_frames": ["frame_id", "timestamp", "image_path", "image_width", "image_height"],
    "test_frames": ["frame_id", "timestamp", "image_path", "image_width", "image_height"],
    "machine_frames": ["frame_id", "timestamp", "image_path", "image_width", "image_height"],
    "train_context": ["frame_id", "previous", "next"],
    "test_context": ["frame_id", "previous", "next"],
    "train_stream_predictions": ["detection_id", *DET_COLS],
    "test_stream_predictions": ["detection_id", *DET_COLS],
    "machine_predictions": ["detection_id", *DET_COLS],
    "human_boxes_train": ["frame_id", "class_name", "cx", "cy", "w", "h", "evaluated_class", "source"],
}


def _dir(raw: Path | None) -> Path:
    return Path(raw) if raw is not None else paths.raw_dir()


def read_table(name: str, raw: Path | None = None) -> pd.DataFrame:
    """Read ``<name>.csv`` and verify that the required columns are present."""
    path = _dir(raw) / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required competition file is missing: {path}")
    df = pd.read_csv(path, dtype={"id": str, "detection_id": str, "frame_id": str,
                                  "previous": str, "next": str, "class_name": str})
    missing = [c for c in REQUIRED.get(name, []) if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}")
    return df


def load_policy(raw: Path | None = None) -> dict:
    with open(_dir(raw) / "scoring_policy.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def canonical_class(names: pd.Series, policy: dict) -> pd.Series:
    """Apply the alias table from ``scoring_policy.json`` (e.g. nude_hand -> nude_hands)."""
    aliases = policy.get("aliases", {})
    return names.map(lambda n: aliases.get(n, n))


def _with_time(frames: pd.DataFrame) -> pd.DataFrame:
    out = frames.copy()
    out["ts"] = pd.to_datetime(out["timestamp"], utc=True, format="mixed")
    out["utc_day"] = out["ts"].dt.strftime("%Y-%m-%d")
    out["t_sec"] = (out["ts"] - pd.Timestamp("2026-10-25", tz="UTC")).dt.total_seconds()
    return out


def load_frames(part: str, raw: Path | None = None) -> pd.DataFrame:
    """``part`` is ``train``, ``test`` or ``machine``. Adds parsed UTC time columns."""
    frames = _with_time(read_table(f"{part}_frames", raw))
    if not frames["frame_id"].is_unique:
        raise ValueError(f"{part}_frames.csv: frame_id is not unique")
    return frames


def load_targets(part: str, raw: Path | None = None) -> pd.DataFrame:
    """Target detections (``train.csv`` with ``error`` / ``test.csv`` without)."""
    if part not in ("train", "test"):
        raise ValueError(part)
    df = read_table(part, raw)
    if not df["id"].is_unique:
        raise ValueError(f"{part}.csv: id is not unique")
    if part == "test" and "error" in df.columns:
        raise ValueError("test.csv unexpectedly contains an 'error' column")
    return df


def load_stream(part: str, raw: Path | None = None) -> pd.DataFrame:
    """Detector output on every frame of the part (targets + unlabeled context)."""
    name = "machine_predictions" if part == "machine" else f"{part}_stream_predictions"
    df = read_table(name, raw)
    if not df["detection_id"].is_unique:
        raise ValueError(f"{name}.csv: detection_id is not unique")
    return df


def load_context(part: str, raw: Path | None = None) -> pd.DataFrame:
    df = read_table(f"{part}_context", raw)
    if not df["frame_id"].is_unique:
        raise ValueError(f"{part}_context.csv: frame_id is not unique")
    return df


def load_sample_submission(raw: Path | None = None) -> pd.DataFrame:
    return read_table("sample_submission", raw)


def load_human_boxes(raw: Path | None = None, policy: dict | None = None) -> pd.DataFrame:
    """TRAIN-ONLY manual annotation. Adds pixel xyxy and the canonical class name.

    FORBIDDEN AT INFERENCE: used only to audit labels and to build training-time
    auxiliary targets.
    """
    hb = read_table("human_boxes_train", raw)
    policy = policy or load_policy(raw)
    hb["class_canon"] = canonical_class(hb["class_name"], policy)
    hb["x1"] = (hb["cx"] - hb["w"] / 2) * FRAME_W
    hb["x2"] = (hb["cx"] + hb["w"] / 2) * FRAME_W
    hb["y1"] = (hb["cy"] - hb["h"] / 2) * FRAME_H
    hb["y2"] = (hb["cy"] + hb["h"] / 2) * FRAME_H
    return hb


@dataclass
class Part:
    """Everything that is legitimately available for one part at inference time."""

    name: str
    targets: pd.DataFrame      # id + detection columns (+ error for train)
    frames: pd.DataFrame       # frame_id, ts, utc_day, t_sec, ...
    context: pd.DataFrame      # frame_id, previous, next
    stream: pd.DataFrame       # detector output on targets and context frames


def load_part(part: str, raw: Path | None = None) -> Part:
    return Part(
        name=part,
        targets=load_targets(part, raw),
        frames=load_frames(part, raw),
        context=load_context(part, raw),
        stream=load_stream(part, raw),
    )


def attach_day(df: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Add ``utc_day`` and ``t_sec`` of the frame to a detection table (many-to-one)."""
    return df.merge(frames[["frame_id", "utc_day", "t_sec"]], on="frame_id", how="left", validate="many_to_one")


def non_table(df: pd.DataFrame, policy: dict | None = None) -> pd.DataFrame:
    """Rows inside the scored scope of the test (``test_excluded_classes`` removed)."""
    excluded = set((policy or {}).get("test_excluded_classes", ["table"]))
    return df[~df["class_name"].isin(excluded)]


def finite_boxes(df: pd.DataFrame) -> np.ndarray:
    return df[BOX_COLS].to_numpy(dtype=float)
