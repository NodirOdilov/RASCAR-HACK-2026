"""Round 3, hypothesis G1: does a consensus of more public detectors rank wrong boxes better?

Pre-specified before any number of this script existed (2026-11-12 ~16:00 MSK):
  * new opinions: Grounding DINO base (`gbf_same_soft`, 9 views) and OWLv2 large (`oLf_same_soft`, 9 views), computed
    with exactly the same agreement function, prompts and views as their smaller siblings;
  * candidate model form = the form of `c9_lr_geo` (logistic regression, C = 0.5, inputs tc_logit, cons_soft, p_tb_cut0,
    f_cov_same, edge_rel clipped to [0, 12]); only the list of opinions inside cons_soft changes;
  * primary candidate: k4 = all four opinions; secondary: k3g = the three without OWLv2 large (in case it cannot be run),
    k3o = the three without Grounding DINO base;
  * reference: `c9_lr_geo`, the file with the best verified Public score (0.80490).
Everything is leave-one-UTC-day-out on the 198 labelled rows; the z-score statistics of the consensus come from the
training day (rascar.models.Scorer._prepare). Test rows are only given features, never labels.

Usage:  python research/lead/g2_consensus_eval.py            (CPU environment; reads the caches written on the GPU)
Writes results/research/g2_consensus.csv, results/lb_discrepancy/g2_features_{train,test}.csv
"""

from __future__ import annotations

import copy
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from rascar import data, metrics, paths, train, vision, vision_gdino

warnings.filterwarnings("ignore")
CLASSES = ["person", "nude_hands", "cup", "tray"]
GEO = ["p_tb_cut0", "f_cov_same", "edge_rel"]
VDIR = paths.interim_dir() / "vision"


def load(path):
    if not path.is_file():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def opinions(targets: pd.DataFrame) -> pd.DataFrame:
    """gb / gbf = Grounding DINO base (3 / 9 views); oL / oLf = OWLv2 large (3 / 9 views). NaN where a cache is missing."""
    out = pd.DataFrame(np.nan, index=targets.index, columns=["gb_same_soft", "gbf_same_soft", "oL_same_soft", "oLf_same_soft"])
    for frame_id, grp in targets.groupby("frame_id", sort=False):
        boxes, classes = grp[data.BOX_COLS].to_numpy(float), grp["class_name"].to_numpy(dtype=object)
        gb, gbf = load(VDIR / "gdinob" / f"{frame_id}.npz"), load(VDIR / "gdinob_fine" / f"{frame_id}.npz")
        if gb is not None:
            out.loc[grp.index, "gb_same_soft"] = [vision_gdino.agreement(b, c, gb)["gd_same_soft"] for b, c in zip(boxes, classes)]
            if gbf is not None:
                both = {**gb, **gbf}
                out.loc[grp.index, "gbf_same_soft"] = [vision_gdino.agreement(b, c, both)["gd_same_soft"] for b, c in zip(boxes, classes)]
        ol, olf = load(VDIR / "owlL" / f"{frame_id}.npz"), load(VDIR / "owlL_fine" / f"{frame_id}.npz")
        if ol is not None:
            out.loc[grp.index, "oL_same_soft"] = vision.frame_vision_features(boxes, classes, ol)["ov_same_soft"].to_numpy()
            if olf is not None:
                out.loc[grp.index, "oLf_same_soft"] = vision.frame_vision_features(boxes, classes, {**ol, **olf})["ov_same_soft"].to_numpy()
    return out


def spec(name: str, ops: list[str], geo: bool = True) -> dict:
    cols = ["tc_logit", "cons_soft"] + (GEO if geo else [])
    return {"name": name, "kind": "logreg", "params": {"C": 0.5 if geo else 1.0}, "clip": {"edge_rel": [0.0, 12.0]},
            "derived": [{"name": "cons_soft", "op": "mean_z", "columns": ops}], "features": {"columns": cols}}


def main() -> int:
    X, y, aux = train.load_training_table()
    T = train.load_test_table()
    tr_t = data.load_part("train").targets.set_index("id").loc[X["id"]].reset_index()
    te_t = data.load_part("test").targets.set_index("id").loc[T["id"]].reset_index()
    Xo, To = opinions(tr_t), opinions(te_t)
    X = pd.concat([X.reset_index(drop=True), Xo.reset_index(drop=True)], axis=1)
    T = pd.concat([T.reset_index(drop=True), To.reset_index(drop=True)], axis=1)
    have = [c for c in Xo.columns if X[c].notna().all() and T[c].notna().all()]
    print("opinions available for every train and test row:", have)
    out_dir = paths.results_dir() / "lb_discrepancy"
    X[["id", "frame_id", "class_name", "utc_day"] + list(Xo.columns)].to_csv(out_dir / "g2_features_train.csv", index=False)
    T[["id", "frame_id", "class_name"] + list(To.columns)].to_csv(out_dir / "g2_features_test.csv", index=False)
    dayv, cls = X["utc_day"].to_numpy(), X["class_name"].to_numpy()
    days = sorted(np.unique(dayv))

    def dayrank(s):
        o = np.empty(len(s))
        for d in days:
            m = dayv == d
            o[m] = rankdata(s[m]) / m.sum()
        return o

    def row(name, score, note=""):
        a = [metrics.average_precision(y[dayv == d], score[dayv == d]) for d in days]
        r = dayrank(score)
        hi = (X["conf"] >= 0.85).to_numpy()
        return {"model": name, "ap_day1": a[0], "ap_day2": a[1], "mean_day": float(np.mean(a)), "pooled_equal": metrics.average_precision(y, r),
                **{f"ap_{c}": metrics.average_precision(y[cls == c], r[cls == c]) for c in CLASSES},
                "ap_conf_ge_085": metrics.average_precision(y[hi], r[hi]), "note": note}

    ops_old = ["ovf_same_soft", "gdf_same_soft"]
    cand = {"c9_lr_geo (reference, public 0.8049)": spec("c9_lr_geo", ops_old)}
    if "gbf_same_soft" in have:
        cand["k3g_lr_geo: + Grounding DINO base"] = spec("k3g_lr_geo", ops_old + ["gbf_same_soft"])
        cand["2 inputs: tc_logit + Grounding DINO base alone"] = spec("x_gb_only", ["gbf_same_soft"], geo=False)
        cand["2 inputs: tc_logit + cons(OWL-b, GDINO-t, GDINO-b)"] = spec("k3g_lr_soft", ops_old + ["gbf_same_soft"], geo=False)
    if "oLf_same_soft" in have:
        cand["k3o_lr_geo: + OWLv2 large"] = spec("k3o_lr_geo", ops_old + ["oLf_same_soft"])
        cand["2 inputs: tc_logit + OWLv2 large alone"] = spec("x_oL_only", ["oLf_same_soft"], geo=False)
    if "gbf_same_soft" in have and "oLf_same_soft" in have:
        cand["k4_lr_geo: all four opinions (pre-specified primary)"] = spec("k4_lr_geo", ops_old + ["gbf_same_soft", "oLf_same_soft"])
        cand["2 inputs: tc_logit + cons(all four)"] = spec("k4_lr_soft", ops_old + ["gbf_same_soft", "oLf_same_soft"], geo=False)
        cand["k2L_lr_geo: the two large models only"] = spec("k2L_lr_geo", ["gbf_same_soft", "oLf_same_soft"])
    cand["2 inputs: tc_logit + cons(OWL-b, GDINO-t) (c9_lr_soft)"] = spec("c9_lr_soft", ops_old, geo=False)
    rows, oofs = [row("baseline 1 - confidence", 1 - X["conf"].to_numpy())], {}
    for name, sp in cand.items():
        oof = train.day_oof(sp, X, y, aux)
        oofs[sp["name"]] = oof
        rows.append(row(name, oof))
    table = pd.DataFrame(rows)
    table.round(4).to_csv(paths.results_dir() / "research" / "g2_consensus.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_colwidth", 62):
        print(table.drop(columns=["note"]).round(4).to_string(index=False))
    pd.DataFrame({"id": X["id"], "utc_day": dayv, "class_name": cls, "error": y, **oofs}).to_csv(out_dir / "g2_oof.csv", index=False)

    print("\nAUC of each opinion for 'correct box' (1 - error), all rows / inside each class:")
    for c in ["conf", "tc_logit", "ovf_same_soft", "gdf_same_soft"] + have:
        line = f"  {c:15s} all {roc_auc_score(1 - y, X[c]):.3f}"
        for k in CLASSES:
            m = cls == k
            line += f"  {k} {roc_auc_score(1 - y[m], X.loc[m, c]):.3f}"
        print(line)
    print("\nSpearman between opinions (train rows):")
    print(X[["ovf_same_soft", "gdf_same_soft"] + have].corr(method="spearman").round(2).to_string())
    print("\nmean opinion by class, train vs test:")
    both = pd.concat([X.assign(part="train"), T.assign(part="test")])
    print(both.groupby(["class_name", "part"])[["ovf_same_soft", "gdf_same_soft"] + have].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
