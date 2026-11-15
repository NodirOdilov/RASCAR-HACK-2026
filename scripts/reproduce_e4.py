"""One-command reproduction of E4 (``rascar.emulation4``, model ``e4_e2x_one_to_one``): the E2X model whose scores are
conditioned on the one-to-one rule of the label definition (same-class audited boxes of a frame with IoU >= 0.5 form a
group; inside a group the error probabilities are conditioned on "at most one box is correct"; no fitted parameter).

Usage (from the project root):
  python scripts/reproduce_e4.py --rebuild-features [--out results/final_mission/repro_e4] [--compare <csv>]

Pipeline: exactly the E2X pipeline of scripts/reproduce_e2.py (raw CSVs -> 03 inputs -> cached proposals of the four
public detectors -> pixel margin -> emulated label rule -> logistic regression) -> E2X scores -> one-to-one update -> CSV.
Checked on the way: the E2X scores of this run equal the delivered E2X file (max abs difference reported), rows
outside a group keep the E2X score. Printed and stored: strict leave-one-UTC-day-out AP of E2X and of E4.
The script never writes into ``submissions/``.
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
from rascar import emulation2 as em2  # noqa: E402
from rascar import emulation4 as em4  # noqa: E402
from rascar import final_models as fm  # noqa: E402

INPUTS = ["tc_logit", "p_tb_cut0", "f_cov_same", "edge_rel"]
E2X_CSV = ROOT / "submissions" / "EVOLYUTSIYA_CASE1_E2X_OPTIONAL.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "final_mission" / "repro_e4"))
    ap.add_argument("--compare", default="", help="path of a delivered CSV to compare with")
    ap.add_argument("--extra-dir", default="", help="folder holding <model>_hires caches (default: rascar.emulation2.default_extra_dir())")
    args = ap.parse_args()
    cfg = em4.BASE
    out = Path(args.out)
    if "submissions" in out.resolve().parts:
        raise SystemExit("refusing to write into submissions/")
    (out / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frozen_tr = pd.read_csv(ROOT / "results" / "frozen_v3b" / "features_train.csv", dtype={"id": str})
    frozen_te = pd.read_csv(ROOT / "results" / "frozen_v3b" / "features_test.csv", dtype={"id": str})
    report: dict = {"model": em4.NAME, "base_model": cfg.name, "rule": {"kind": "conditioning on at most one correct box per group", "tau": em4.TAU, "fitted_parameters": 0},
                    "python": sys.version.split()[0], "rebuild_features": args.rebuild_features}
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
    store = em2.ProposalStore2(extra_dir=Path(args.extra_dir) if args.extra_dir else None)
    missing = [f for f in sorted(set(tr["frame_id"]) | set(te["frame_id"])) if not store.has_all(f, cfg)]
    if missing:
        raise SystemExit(f"{len(missing)} frames lack a proposal cache, e.g. {missing[:3]}")
    manual = em.load_manual_boxes()
    y = tr["error"].to_numpy(int)

    oof, base_oof, loser_tr = em4.leave_one_day_out(tr, manual, store)
    days = sorted(tr["utc_day"].unique())
    for key, s in (("lodo_ap", oof), ("lodo_ap_e2x", base_oof)):
        d = {day: float(average_precision_score(y[(tr["utc_day"] == day).to_numpy()], s[(tr["utc_day"] == day).to_numpy()])) for day in days}
        report[key] = d
        report["mean_of_day_ap" + ("" if key == "lodo_ap" else "_e2x")] = float(np.mean(list(d.values())))
    report["train_rows_in_groups"] = {"n": int(loser_tr.sum()), "errors_among_them": int(y[loser_tr].sum()),
                                      "by_class": tr.loc[loser_tr, "class_name"].value_counts().to_dict()}
    pd.DataFrame({"id": tr["id"], "oof": oof, "oof_e2x": base_oof, "in_group": loser_tr.astype(int)}).to_csv(out / (em4.NAME + "_oof.csv"), index=False, lineterminator=chr(10))

    model = em2.EmulationModel2(cfg).fit(tr, manual, store)
    model.save(out / "models")
    loaded = em2.EmulationModel2.load(out / "models", cfg.name)
    scores, base, loser = em4.predict(loaded, te, store)
    if E2X_CSV.is_file():
        ref = pd.read_csv(E2X_CSV, dtype={"id": str}).set_index("id")["score"].loc[te["id"]].to_numpy()
        report["base_scores_vs_delivered_e2x_max_abs_diff"] = float(np.abs(base - ref).max())
        report["rows_outside_groups_equal_e2x_file"] = bool(np.abs(scores[~loser] - ref[~loser]).max() < 1e-9)
    sample = pd.read_csv(ROOT / "data" / "raw" / "rascar-hack-case-1" / "sample_submission.csv", dtype={"id": str})
    path = out / (em4.NAME + ".csv")
    fm.write_submission(te["id"], scores, path, sample["id"])
    pd.DataFrame({"id": te["id"], "class_name": te["class_name"], "e2x": base, "e4": scores, "in_group": loser.astype(int)}).to_csv(out / (em4.NAME + "_test_detail.csv"), index=False, lineterminator=chr(10))
    report.update({"csv": str(path), "sha256": sha(path), "test_rows_in_groups": {"n": int(loser.sum()), "by_class": te.loc[loser, "class_name"].value_counts().to_dict()},
                   "adapter": model.adapter, "coef": model.pipeline[-1].coef_[0].tolist(), "intercept": float(model.pipeline[-1].intercept_[0]),
                   "n_train": int(len(tr)), "n_test": int(len(te)), "extra_dir": str(store.extra_dir)})
    if args.compare:
        refc = pd.read_csv(args.compare, dtype={"id": str}).set_index("id")["score"]
        new = pd.read_csv(path, dtype={"id": str}).set_index("id")["score"].loc[refc.index]
        report.update({"compared_with": args.compare, "compared_sha256": sha(Path(args.compare)),
                       "max_abs_diff": float(np.abs(new.to_numpy() - refc.to_numpy()).max()),
                       "byte_identical": sha(path) == sha(Path(args.compare))})
    report["seconds"] = round(time.time() - t0, 1)
    (out / "REPRO_REPORT.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
