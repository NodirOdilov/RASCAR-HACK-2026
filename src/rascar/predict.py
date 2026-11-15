"""Standalone inference: saved model -> scores for every test row. No training, no labels.

Run:  python -m rascar.predict --model models/<name>.joblib --output results/predictions/<name>.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import paths, train


def predict_test(model_path: Path, raw: Path | None = None) -> pd.DataFrame:
    model = joblib.load(model_path)
    feats = train.load_test_table(raw)          # reads test.csv + stream/context/frames + machine pool only
    score = np.asarray(model.predict(feats), dtype=float)
    if not np.isfinite(score).all():
        raise ValueError("model produced non-finite scores")
    return pd.DataFrame({"id": feats["id"].to_numpy(), "score": np.clip(score, 0.0, 1.0)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the test set with a saved model")
    parser.add_argument("--model", type=Path, required=True, help="models/<name>.joblib produced by rascar.train")
    parser.add_argument("--output", type=Path, default=None, help="CSV with id,score (default results/predictions/<name>.csv)")
    parser.add_argument("--raw", type=Path, default=None)
    args = parser.parse_args(argv)
    pred = predict_test(args.model, args.raw)
    out = args.output or (paths.results_dir() / "predictions" / f"{Path(args.model).stem}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, index=False, encoding="utf-8", lineterminator="\n", float_format="%.10f")
    meta = Path(args.model).with_suffix(".meta.json")
    name = json.loads(meta.read_text(encoding="utf-8"))["name"] if meta.is_file() else Path(args.model).stem
    print(f"predictions: {out} rows={len(pred)} model={name} score_range=[{pred['score'].min():.4f}, {pred['score'].max():.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
