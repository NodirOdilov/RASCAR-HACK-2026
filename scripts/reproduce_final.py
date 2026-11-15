"""One-command reproduction of the final Kaggle files (prompt.txt §14).

Usage:
  python scripts/reproduce_final.py                      # from the frozen feature table (fast, ~10 s)
  python scripts/reproduce_final.py --rebuild-features   # also rebuild the feature table from raw data + caches
Options: --out results/repro_r6  --models o5c_clsstd_q4,r4a_clsstd_all

Steps
  1. (optional) rebuild results/frozen_v3b-equivalent features from data/raw + data/interim/vision with the same code
     (rascar.train + research/lead/g2_consensus_eval.opinions) and compare every model input with the frozen table;
  2. for every model: leave-one-UTC-day-out AP (validation), fit on all 198 labelled rows, serialise
     (<out>/models/<name>.joblib + .stats.json), reload, predict the 362 test rows, write <out>/<name>.csv;
  3. compare with the delivered files (max |Δscore|, SHA-256) and write <out>/REPRO_REPORT.json.
The GPU detector caches are inputs; regenerating them is documented in REPRODUCE.md (hours on the RTX 3050).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rascar import final_models as fm  # noqa: E402

DELIVERED = {
    "o5c_clsstd_q4": ROOT / "submissions" / "kaggle_submitted" / "2026-11-12_RASCAR_v3_03_o5c_clsstd_q4_public0.89976.csv",
    "r4a_clsstd_all": ROOT / "submissions" / "candidates_v3" / "RASCAR_v3_05_r4a_clsstd_all.csv",
    "p7r_person_ninside": ROOT / "results" / "r7_lead" / "P7r_test.csv",
}
INPUTS = ["class_name", "utc_day", "tc_logit", "p_tb_cut0", "f_cov_same", "edge_rel"] + fm.OPINIONS
# The uploaded 03 file was written by Agent O with 8 decimals and re-written with 10 by build_candidates_v3.py;
# reproducing that two-step rounding makes the regenerated file byte-identical.
PRE_ROUND = {"o5c_clsstd_q4": 8}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rebuild_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    sys.path.insert(0, str(ROOT / "research" / "lead"))
    import g2_consensus_eval as g2  # noqa: E402
    from rascar import data, train  # noqa: E402
    X, y, aux = train.load_training_table()
    T = train.load_test_table()
    tr_t = data.load_part("train").targets.set_index("id").loc[X["id"]].reset_index()
    te_t = data.load_part("test").targets.set_index("id").loc[T["id"]].reset_index()
    X = pd.concat([X.reset_index(drop=True), g2.opinions(tr_t).reset_index(drop=True)], axis=1)
    T = pd.concat([T.reset_index(drop=True), g2.opinions(te_t).reset_index(drop=True)], axis=1)
    X["error"] = y
    return X, T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "repro_r6"))
    ap.add_argument("--models", default="o5c_clsstd_q4,r4a_clsstd_all")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frozen_tr = pd.read_csv(ROOT / "results" / "frozen_v3b" / "features_train.csv", dtype={"id": str})
    frozen_te = pd.read_csv(ROOT / "results" / "frozen_v3b" / "features_test.csv", dtype={"id": str})
    report: dict = {"python": sys.version.split()[0], "rebuild_features": args.rebuild_features, "models": {}}
    tr, te = frozen_tr, frozen_te
    if args.rebuild_features:
        X, T = rebuild_features()
        diffs = {}
        for name, a, b in (("train", X, frozen_tr), ("test", T, frozen_te)):
            a = a.set_index("id").loc[b["id"]]
            for col in INPUTS[2:]:
                va, vb = a[col].to_numpy(float), b[col].to_numpy(float)
                same_nan = bool((np.isnan(va) == np.isnan(vb)).all())
                d = float(np.nanmax(np.abs(va - vb))) if np.isfinite(va).any() else 0.0
                diffs[f"{name}:{col}"] = {"max_abs_diff": d, "same_nan_pattern": same_nan}
        report["feature_rebuild"] = diffs
        report["feature_rebuild_ok"] = all(v["max_abs_diff"] < 1e-9 and v["same_nan_pattern"] for v in diffs.values())
        tr = X.merge(frozen_tr[["id"]], on="id")
        tr = tr.set_index("id").loc[frozen_tr["id"]].reset_index()
        te = T.set_index("id").loc[frozen_te["id"]].reset_index()
    sample = pd.read_csv(ROOT / "data" / "raw" / "rascar-hack-case-1" / "sample_submission.csv", dtype={"id": str})
    if "p7r_person_ninside" in args.models:          # amodal-extent inputs from the detector caches (rascar.extent)
        from rascar import extent  # noqa: E402
        raw_dir = ROOT / "data" / "raw" / "rascar-hack-case-1"
        for part, df in (("train", tr), ("test", te)):
            boxes = pd.read_csv(raw_dir / f"{part}.csv", dtype={"id": str})[["id", "x1", "y1", "x2", "y2"]]
            t = df[["id", "frame_id", "class_name"]].merge(boxes, on="id", how="left", validate="one_to_one")
            feats = extent.n_inside_features(t)
            for c in feats.columns:
                df[c] = feats[c].to_numpy()
    y = tr["error"].to_numpy(int)
    for name in [m for m in args.models.split(",") if m]:
        oof = fm.leave_one_day_out(name, tr)
        days = sorted(tr["utc_day"].unique())
        day_ap = {d: float(average_precision_score(y[(tr["utc_day"] == d).to_numpy()], oof[(tr["utc_day"] == d).to_numpy()])) for d in days}
        if name == "p7r_person_ninside":
            model = fm.PersonRefinedModel().fit(tr)
            model.save(out / "models")
            loaded = fm.PersonRefinedModel.load(out / "models")
        else:
            model = fm.FinalModel(name).fit(tr)
            model.save(out / "models")
            loaded = fm.FinalModel.load(name, out / "models")
        scores = loaded.predict(te)
        if name in PRE_ROUND:
            scores = np.array([float(f"{v:.{PRE_ROUND[name]}f}") for v in scores])
        path = out / f"{name}.csv"
        fm.write_submission(te["id"], scores, path, sample["id"])
        entry = {"lodo_ap": day_ap, "mean_of_day_ap": float(np.mean(list(day_ap.values()))), "csv": str(path), "sha256": sha(path)}
        if name in DELIVERED and DELIVERED[name].exists():
            ref = pd.read_csv(DELIVERED[name], dtype={"id": str}).set_index("id")["score"]
            new = pd.read_csv(path, dtype={"id": str}).set_index("id")["score"].loc[ref.index]
            entry.update({"delivered": str(DELIVERED[name]), "delivered_sha256": sha(DELIVERED[name]),
                          "max_abs_diff_vs_delivered": float(np.abs(new.to_numpy() - ref.to_numpy()).max()),
                          "byte_identical": sha(path) == sha(DELIVERED[name])})
        report["models"][name] = entry
    report["seconds"] = round(time.time() - t0, 1)
    (out / "REPRO_REPORT.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "feature_rebuild"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
