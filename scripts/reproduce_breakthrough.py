"""One-command reproduction of the final-round candidate E1 (``rascar.emulation``, model ``e1_smooth_pooled``).

Usage (from the project root):
  python scripts/reproduce_breakthrough.py                      # frozen 03 inputs + detector caches (~3 min)
  python scripts/reproduce_breakthrough.py --rebuild-features   # also rebuild the 03 inputs from the raw tables
Options: --out results/final_breakthrough/repro_e1   --transplant   (also write the within-class transplant file)

Pipeline: raw CSVs -> 03 inputs (tc_logit, p_tb_cut0, f_cov_same, edge_rel; `scripts/reproduce_final.rebuild_features`)
-> cached public-detector proposals (data/interim/vision) -> convention adapter (manual boxes of the train frames)
-> emulated label-rule agreement -> logistic regression (03's form) -> serialise -> reload -> predict -> CSV.
Validation printed and stored: leave-one-UTC-day-out AP (adapter, statistics and model fitted on the other day).
The script NEVER writes into `submissions/`; it compares its CSV with a delivered file if `--compare` is given.
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
sys.path.insert(0, str(ROOT / "scripts"))
from rascar import emulation as em  # noqa: E402
from rascar import final_models as fm  # noqa: E402

INPUTS = ["tc_logit", "p_tb_cut0", "f_cov_same", "edge_rel"]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "final_breakthrough" / "repro_e1"))
    ap.add_argument("--transplant", action="store_true")
    ap.add_argument("--compare", default="", help="path of a delivered CSV to compare with")
    args = ap.parse_args()
    out = Path(args.out)
    if "submissions" in out.resolve().parts:
        raise SystemExit("refusing to write into submissions/")
    (out / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frozen_tr = pd.read_csv(ROOT / "results" / "frozen_v3b" / "features_train.csv", dtype={"id": str})
    frozen_te = pd.read_csv(ROOT / "results" / "frozen_v3b" / "features_test.csv", dtype={"id": str})
    report: dict = {"model": em.NAME, "python": sys.version.split()[0], "rebuild_features": args.rebuild_features}
    tr, te = frozen_tr, frozen_te
    if args.rebuild_features:
        import reproduce_final as rf  # noqa: E402
        X, T = rf.rebuild_features()
        X["id"] = X["id"].astype(str); T["id"] = T["id"].astype(str)
        tr = X.set_index("id").loc[frozen_tr["id"]].reset_index()
        te = T.set_index("id").loc[frozen_te["id"]].reset_index()
        diffs = {}
        for name, a, b in (("train", tr, frozen_tr), ("test", te, frozen_te)):
            for col in INPUTS:
                diffs[f"{name}:{col}"] = float(np.nanmax(np.abs(a[col].to_numpy(float) - b[col].to_numpy(float))))
        report["feature_rebuild_max_abs_diff"] = diffs
        report["feature_rebuild_ok"] = bool(max(diffs.values()) < 1e-9)
    keep = ["id", "frame_id", "class_name", "utc_day", *INPUTS]
    tr = em.attach_boxes(tr[keep + ["error"]], "train")
    te = em.attach_boxes(te[[c for c in keep if c in te.columns]], "test")
    store = em.ProposalStore()
    manual = em.load_manual_boxes()
    y = tr["error"].to_numpy(int)

    oof = em.leave_one_day_out(tr, manual, store)
    days = sorted(tr["utc_day"].unique())
    day_ap = {d: float(average_precision_score(y[(tr["utc_day"] == d).to_numpy()], oof[(tr["utc_day"] == d).to_numpy()])) for d in days}
    report["lodo_ap"] = day_ap
    report["mean_of_day_ap"] = float(np.mean(list(day_ap.values())))
    pd.DataFrame({"id": tr["id"], "oof": oof}).to_csv(out / f"{em.NAME}_oof.csv", index=False, lineterminator="\n")

    model = em.EmulationModel().fit(tr, manual, store)
    model.save(out / "models")
    loaded = em.EmulationModel.load(out / "models")
    scores = loaded.predict(te, store)
    sample = pd.read_csv(ROOT / "data" / "raw" / "rascar-hack-case-1" / "sample_submission.csv", dtype={"id": str})
    path = out / f"{em.NAME}.csv"
    fm.write_submission(te["id"], scores, path, sample["id"])
    report.update({"csv": str(path), "sha256": sha(path), "adapter": model.adapter,
                   "coef": model.pipeline[-1].coef_[0].tolist(), "intercept": float(model.pipeline[-1].intercept_[0]),
                   "n_train": int(len(tr)), "n_test": int(len(te))})
    if args.transplant:
        # 03's test scores re-assigned inside each class in E1's order (cross-class interleaving identical to 03)
        base = fm.FinalModel("o5c_clsstd_q4").fit(frozen_tr).predict(frozen_te)
        base = np.array([float(f"{v:.8f}") for v in base])       # the uploaded 03 file carries 8 decimals
        tp = base.copy()
        cls = te["class_name"].to_numpy()
        for c in em.CLASSES:
            m = np.flatnonzero(cls == c)
            rank = np.argsort(np.argsort(scores[m], kind="stable"), kind="stable")
            tp[m] = np.sort(base[m], kind="stable")[rank]
        tpath = out / f"{em.NAME}_transplant.csv"
        fm.write_submission(te["id"], tp, tpath, sample["id"])
        report.update({"transplant_csv": str(tpath), "transplant_sha256": sha(tpath)})
    if args.compare:
        ref = pd.read_csv(args.compare, dtype={"id": str}).set_index("id")["score"]
        new = pd.read_csv(path, dtype={"id": str}).set_index("id")["score"].loc[ref.index]
        report.update({"compared_with": args.compare, "compared_sha256": sha(Path(args.compare)),
                       "max_abs_diff": float(np.abs(new.to_numpy() - ref.to_numpy()).max()),
                       "byte_identical": sha(path) == sha(Path(args.compare))})
    report["seconds"] = round(time.time() - t0, 1)
    (out / "REPRO_REPORT.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
