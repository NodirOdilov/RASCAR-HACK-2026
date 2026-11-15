"""Frozen feature table for round 3 (read-only input for the parallel agents).

Writes results/frozen_v3/features_{train,test}.csv: every project feature, plus the opinions of the larger public
detectors where their caches exist (gb_/gbf_same_soft = Grounding DINO base 3 / 9 views; oL_/oLf_same_soft = OWLv2
large 3 / 9 views), plus, for TRAIN only, the label `error` and the training-only target `best_iou`.
Test rows carry no label (none exists). Also writes the two uploaded files' scores for reference.

Usage:  python research/lead/v3_export_frozen.py
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_consensus_eval as g2  # noqa: E402

from rascar import data, paths, train  # noqa: E402

import os
out = paths.results_dir() / os.environ.get("FROZEN_TAG", "frozen_v3")
out.mkdir(parents=True, exist_ok=True)
X, y, aux = train.load_training_table()
T = train.load_test_table()
tr_t = data.load_part("train").targets.set_index("id").loc[X["id"]].reset_index()
te_t = data.load_part("test").targets.set_index("id").loc[T["id"]].reset_index()
X = pd.concat([X.reset_index(drop=True), g2.opinions(tr_t).reset_index(drop=True)], axis=1)
T = pd.concat([T.reset_index(drop=True), g2.opinions(te_t).reset_index(drop=True)], axis=1)
X["error"] = y
X["best_iou"] = aux["best_iou"].to_numpy()
sub = paths.submissions_dir() / "kaggle_submitted"
for tag, pat in (("score_A_vf_blend_geo_gbm_public07198", "*FINAL_vf_blend*"), ("score_B_c9_lr_geo_public08049", "*FALLBACK_c9*")):
    s = pd.read_csv(next(sub.glob(pat + ".csv")), dtype={"id": str}).set_index("id")["score"]
    T[tag] = T["id"].map(s)
X.to_csv(out / "features_train.csv", index=False)
T.to_csv(out / "features_test.csv", index=False)
info = {"train_rows": len(X), "test_rows": len(T), "columns": list(T.columns),
        "opinion_columns_complete": {c: bool(X[c].notna().all() and T[c].notna().all()) for c in ["gb_same_soft", "gbf_same_soft", "oL_same_soft", "oLf_same_soft"]},
        "sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (out / "features_train.csv", out / "features_test.csv")}}
(out / "README.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in info.items() if k != "columns"}, indent=1))
