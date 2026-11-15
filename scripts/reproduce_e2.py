"""One-command reproduction of candidate E2 (``rascar.emulation2``; default model ``e2_smooth_hires_small``,
``--config e2x_pixel_inner_hires`` for the research variant E2X).

Usage (from the project root):
  python scripts/reproduce_e2.py --rebuild-features [--out results/e2_breakthrough/repro_e2] [--compare <csv>]

Pipeline: raw CSVs -> 03 inputs (tc_logit, p_tb_cut0, f_cov_same, edge_rel; scripts/reproduce_final.rebuild_features)
-> cached proposals of the four public detectors (E1's nine views in data/interim/vision; for hands and cups also the
fifteen 384 px tiles in results/final_breakthrough/agent_G/cache/<model>_hires) -> convention adapter from the manual
boxes of the train frames -> emulated label rule -> logistic regression (03's form) -> save -> reload -> predict -> CSV.
Printed and stored: strict leave-one-UTC-day-out AP (adapter, statistics and model fitted on the other day).
The script never writes into ``submissions/``.

Regenerating the extra caches (GPU, ~10-17 min per model for the 121 frames):
  .venv_gpu/Scripts/python.exe -I -X utf8 research/final_breakthrough/agent_G/g01_cache_gpu.py --model <gdinob|gdino|owlL|owlv2> --views hires
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
from rascar import final_models as fm  # noqa: E402

INPUTS = ["tc_logit", "p_tb_cut0", "f_cov_same", "edge_rel"]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "e2_breakthrough" / "repro_e2"))
    ap.add_argument("--compare", default="", help="path of a delivered CSV to compare with")
    ap.add_argument("--extra-dir", default="", help="folder holding <model>_hires caches (default: rascar.emulation2.default_extra_dir())")
    ap.add_argument("--config", default=em2.E2.name, choices=sorted(em2.PRESETS))
    args = ap.parse_args()
    cfg = em2.PRESETS[args.config]
    out = Path(args.out)
    if "submissions" in out.resolve().parts:
        raise SystemExit("refusing to write into submissions/")
    (out / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frozen_tr = pd.read_csv(ROOT / "results" / "frozen_v3b" / "features_train.csv", dtype={"id": str})
    frozen_te = pd.read_csv(ROOT / "results" / "frozen_v3b" / "features_test.csv", dtype={"id": str})
    report: dict = {"model": cfg.name, "config": {"hires_classes": list(cfg.hires_classes), "drop_border": cfg.drop_border, "margin": cfg.margin},
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

    oof = em2.leave_one_day_out(tr, manual, store, cfg)
    days = sorted(tr["utc_day"].unique())
    day_ap = {d: float(average_precision_score(y[(tr["utc_day"] == d).to_numpy()], oof[(tr["utc_day"] == d).to_numpy()])) for d in days}
    report["lodo_ap"] = day_ap
    report["mean_of_day_ap"] = float(np.mean(list(day_ap.values())))
    pd.DataFrame({"id": tr["id"], "oof": oof}).to_csv(out / (cfg.name + "_oof.csv"), index=False, lineterminator=chr(10))

    model = em2.EmulationModel2(cfg).fit(tr, manual, store)
    model.save(out / "models")
    loaded = em2.EmulationModel2.load(out / "models", cfg.name)
    scores = loaded.predict(te, store)
    sample = pd.read_csv(ROOT / "data" / "raw" / "rascar-hack-case-1" / "sample_submission.csv", dtype={"id": str})
    path = out / (cfg.name + ".csv")
    fm.write_submission(te["id"], scores, path, sample["id"])
    report.update({"csv": str(path), "sha256": sha(path), "adapter": model.adapter,
                   "coef": model.pipeline[-1].coef_[0].tolist(), "intercept": float(model.pipeline[-1].intercept_[0]),
                   "n_train": int(len(tr)), "n_test": int(len(te)), "extra_dir": str(store.extra_dir)})
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
