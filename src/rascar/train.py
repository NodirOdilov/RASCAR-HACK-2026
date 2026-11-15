"""Cross-validated evaluation and final fit of one model configuration.

Run:  python -m rascar.train --config configs/<name>.json

1. builds the training table (features from detector output; labels only as target),
2. evaluates the configuration with leave-one-UTC-day-out folds (selection metric: mean of
   the two day APs) and, as an optimistic secondary check, repeated frame-group K-fold,
3. appends one row to ``EXPERIMENT_LOG.csv`` and stores the out-of-fold predictions,
4. refits on all scored-scope training rows and saves ``models/<name>.joblib`` (+ meta).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import SEED, __version__, audit, data, features as F, metrics, models, paths, validation

LOG_COLUMNS = [
    "experiment_id", "git_commit_if_any", "features", "split", "algorithm", "params", "seed", "train_rows",
    "valid_rows", "positive_rate", "validation_AP", "P_at_50", "per_class_AP", "wall_seconds", "memory",
    "decision", "artifact_path", "ap_day_1", "ap_day_2", "pooled_oof_AP", "pooled_ci95", "frame_gkf_AP",
    "delta_vs_baseline_mean_day", "p_better_than_baseline", "timestamp_utc",
]


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def load_training_table(raw: Path | None = None):
    """Features, labels and training-only auxiliary targets for the scored scope (no ``table``)."""
    policy = data.load_policy(raw)
    part = data.load_part("train", raw)
    ref = F.load_scene_reference(raw)
    feats = F.build_features(part, ref)
    labels = part.targets[["id", "error"]]
    human = data.load_human_boxes(raw, policy)                       # TRAIN-ONLY
    repro = audit.reproduce_labels(part.targets, human, policy)[["id", "best_iou", "matched_iou"]]
    table = feats.merge(labels, on="id", validate="one_to_one").merge(repro, on="id", validate="one_to_one")
    table = data.non_table(table, policy).reset_index(drop=True)
    y = table["error"].to_numpy(dtype=int)
    aux = table[["best_iou", "matched_iou"]].copy()
    X = table.drop(columns=["error", "best_iou", "matched_iou"])
    return X, y, aux


def load_test_table(raw: Path | None = None) -> pd.DataFrame:
    """Inference-time features of every test row. Reads no label and no manual box."""
    part = data.load_part("test", raw)
    ref = F.load_scene_reference(raw)
    return F.build_features(part, ref)


# --------------------------------------------------------------------------------------
# cross-validation
# --------------------------------------------------------------------------------------
def _fit_predict(spec: dict, X: pd.DataFrame, y: np.ndarray, aux: pd.DataFrame, fold: validation.Fold) -> np.ndarray:
    model = models.make_model(spec)
    model.fit(X.iloc[fold.train_idx], y[fold.train_idx], aux.iloc[fold.train_idx])
    return model.predict(X.iloc[fold.valid_idx])


def day_oof(spec: dict, X: pd.DataFrame, y: np.ndarray, aux: pd.DataFrame) -> np.ndarray:
    folds = validation.day_folds(X["utc_day"])
    validation.assert_no_overlap(folds, X["frame_id"], X["utc_day"])
    oof = np.full(len(X), np.nan)
    for fold in folds:
        oof[fold.valid_idx] = _fit_predict(spec, X, y, aux, fold)
    if np.isnan(oof).any():
        raise AssertionError("some rows did not receive an out-of-fold prediction")
    return oof


def frame_gkf_ap(spec: dict, X: pd.DataFrame, y: np.ndarray, aux: pd.DataFrame, repeats: int = 3,
                 n_splits: int = 5, seed: int = SEED) -> dict:
    """Mean pooled-OOF AP over repeated frame-group K-fold (optimistic: same-day rows on both sides)."""
    folds = validation.frame_group_folds(X["frame_id"], n_splits=n_splits, repeats=repeats, seed=seed)
    validation.assert_no_overlap(folds, X["frame_id"])
    aps, oofs = [], []
    for rep in range(repeats):
        oof = np.full(len(X), np.nan)
        for fold in folds[rep * n_splits:(rep + 1) * n_splits]:
            oof[fold.valid_idx] = _fit_predict(spec, X, y, aux, fold)
        aps.append(metrics.average_precision(y, oof))
        oofs.append(oof)
    return {"mean": float(np.mean(aps)), "std": float(np.std(aps)), "per_repeat": [float(a) for a in aps],
            "oof_mean": np.mean(oofs, axis=0)}


def evaluate(spec: dict, X: pd.DataFrame, y: np.ndarray, aux: pd.DataFrame, gkf_repeats: int = 3,
             n_boot: int = 2000) -> dict:
    started = time.time()
    oof = day_oof(spec, X, y, aux)
    res = validation.evaluate_day_oof(y, oof, X["utc_day"], X["frame_id"], X["class_name"], n_boot=n_boot, seed=SEED)
    res["oof"] = oof
    base = 1.0 - X["conf"].to_numpy()
    days = X["utc_day"].to_numpy()
    res["baseline_mean_day_ap"] = float(np.mean([metrics.average_precision(y[days == d], base[days == d])
                                                 for d in sorted(np.unique(days))]))
    res["delta_vs_baseline_mean_day"] = res["mean_day_ap"] - res["baseline_mean_day_ap"]
    res["paired_vs_baseline_pooled"] = metrics.paired_bootstrap_delta(y, oof, base, X["frame_id"], n_boot=n_boot, seed=SEED)
    res["paired_vs_baseline_per_day"] = {
        str(d): metrics.paired_bootstrap_delta(y[days == d], oof[days == d], base[days == d],
                                               X["frame_id"].to_numpy()[days == d], n_boot=n_boot, seed=SEED)
        for d in sorted(np.unique(days))}
    if gkf_repeats > 0:
        gkf = frame_gkf_ap(spec, X, y, aux, repeats=gkf_repeats)
        res["frame_gkf"] = {k: v for k, v in gkf.items() if k != "oof_mean"}
    res["wall_seconds"] = round(time.time() - started, 2)
    return res


# --------------------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------------------
def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.project_root(), capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - git is optional
        return ""


def peak_memory_mb() -> float:
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return round(counters.PeakWorkingSetSize / 1e6, 1)
    except Exception:  # noqa: BLE001 - non-Windows or restricted environment
        pass
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3, 1)
    except Exception:  # noqa: BLE001
        return float("nan")


def append_log(row: dict) -> Path:
    path = paths.project_root() / "EXPERIMENT_LOG.csv"
    new = not path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)
    return path


def log_row(spec: dict, res: dict, X: pd.DataFrame, y: np.ndarray, decision: str, artifact: str) -> dict:
    days = sorted(res["per_day"])
    model = models.make_model(spec)
    return {
        "experiment_id": spec["name"],
        "git_commit_if_any": git_commit(),
        "features": json.dumps(spec.get("features", {"columns": model.columns}), ensure_ascii=False)
        if spec.get("kind") != "blend" else "blend:" + "+".join(m["name"] for m in spec["members"]),
        "split": "leave-one-UTC-day-out (2 folds: " + ", ".join(days) + "), no table",
        "algorithm": spec["kind"],
        "params": json.dumps({k: v for k, v in spec.items() if k in ("params", "weights", "method")}, ensure_ascii=False),
        "seed": json.dumps(spec.get("seeds", [0])),
        "train_rows": json.dumps({d: int((X["utc_day"] != d).sum()) for d in days}),
        "valid_rows": json.dumps({d: int(res["per_day"][d]["n"]) for d in days}),
        "positive_rate": round(float(y.mean()), 4),
        "validation_AP": round(res["mean_day_ap"], 4),
        "P_at_50": round(res["pooled"]["p_at_50"], 4),
        "per_class_AP": json.dumps({k: round(v["ap"], 4) for k, v in res["per_class"].items()}),
        "wall_seconds": res["wall_seconds"],
        "memory": f"{peak_memory_mb()} MB peak RSS",
        "decision": decision,
        "artifact_path": artifact,
        "ap_day_1": round(res["per_day"][days[0]]["ap"], 4),
        "ap_day_2": round(res["per_day"][days[1]]["ap"], 4),
        "pooled_oof_AP": round(res["pooled"]["ap"], 4),
        "pooled_ci95": f"[{res['pooled']['ci_low']:.4f}, {res['pooled']['ci_high']:.4f}]",
        "frame_gkf_AP": round(res["frame_gkf"]["mean"], 4) if "frame_gkf" in res else "",
        "delta_vs_baseline_mean_day": round(res["delta_vs_baseline_mean_day"], 4),
        "p_better_than_baseline": round(res["paired_vs_baseline_pooled"]["p_a_better"], 3),
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def environment() -> dict:
    import catboost
    import lightgbm
    import scipy
    import sklearn

    return {"python": platform.python_version(), "platform": platform.platform(), "rascar": __version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__, "catboost": catboost.__version__,
            "joblib": joblib.__version__}


def fit_final(spec: dict, X: pd.DataFrame, y: np.ndarray, aux: pd.DataFrame, res: dict | None = None) -> Path:
    model = models.make_model(spec)
    model.fit(X, y, aux)
    out = paths.models_dir() / f"{spec['name']}.joblib"
    joblib.dump(model, out)
    raw = paths.raw_dir()
    meta = {
        "name": spec["name"], "spec": spec, "feature_columns": model.columns,
        "train_rows": int(len(X)), "train_errors": int(y.sum()), "train_days": sorted(X["utc_day"].unique().tolist()),
        "model_sha256": sha256_file(out),
        "raw_sha256": {name: sha256_file(raw / name) for name in
                       ("train.csv", "train_stream_predictions.csv", "train_context.csv", "train_frames.csv",
                        "machine_predictions.csv", "machine_frames.csv", "human_boxes_train.csv", "scoring_policy.json")},
        "environment": environment(), "seed": SEED,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
    }
    if res is not None:
        meta["cv"] = {"mean_day_ap": res["mean_day_ap"], "per_day": {d: v["ap"] for d, v in res["per_day"].items()},
                      "pooled_ap": res["pooled"]["ap"], "pooled_ci95": [res["pooled"]["ci_low"], res["pooled"]["ci_high"]],
                      "baseline_mean_day_ap": res["baseline_mean_day_ap"]}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def save_cv(spec: dict, res: dict, X: pd.DataFrame, y: np.ndarray) -> Path:
    oof_dir = paths.results_dir() / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof = X[["id", "frame_id", "class_name", "utc_day", "conf"]].copy()
    oof["error"] = y
    oof["oof_score"] = res["oof"]
    oof_path = oof_dir / f"{spec['name']}.csv"
    oof.to_csv(oof_path, index=False, encoding="utf-8", lineterminator="\n")
    cv_dir = paths.results_dir() / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in res.items() if k != "oof"}
    (cv_dir / f"{spec['name']}.json").write_text(json.dumps(audit._jsonable(slim), ensure_ascii=False, indent=2), encoding="utf-8")
    return oof_path


def run(spec: dict, gkf_repeats: int = 3, final: bool = True, decision: str = "", raw: Path | None = None,
        table=None, log: bool = True) -> dict:
    X, y, aux = table if table is not None else load_training_table(raw)
    res = evaluate(spec, X, y, aux, gkf_repeats=gkf_repeats)
    oof_path = save_cv(spec, res, X, y)
    artifact = str(oof_path.relative_to(paths.project_root()))
    if final:
        model_path = fit_final(spec, X, y, aux, res)
        artifact = str(model_path.relative_to(paths.project_root()))
    if log:
        append_log(log_row(spec, res, X, y, decision, artifact))
    return res


def describe(name: str, res: dict) -> str:
    days = sorted(res["per_day"])
    parts = [f"{name}: mean-day AP {res['mean_day_ap']:.4f} ("
             + ", ".join(f"{d[-5:]} {res['per_day'][d]['ap']:.4f}" for d in days) + ")",
             f"pooled {res['pooled']['ap']:.4f} [{res['pooled']['ci_low']:.3f}, {res['pooled']['ci_high']:.3f}]",
             f"P@50 {res['pooled']['p_at_50']:.3f}",
             f"delta vs 1-conf {res['delta_vs_baseline_mean_day']:+.4f} (P(better)={res['paired_vs_baseline_pooled']['p_a_better']:.2f})"]
    if "frame_gkf" in res:
        parts.append(f"frame-GKF {res['frame_gkf']['mean']:.4f}")
    return " | ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate (leave-one-day-out) and fit one model configuration")
    parser.add_argument("--config", type=Path, required=True, help="JSON model configuration (see configs/)")
    parser.add_argument("--gkf-repeats", type=int, default=3, help="repeats of the secondary frame-group K-fold (0 = skip)")
    parser.add_argument("--no-final", action="store_true", help="do not refit on all rows / do not save a model")
    parser.add_argument("--decision", default="", help="free-text decision recorded in EXPERIMENT_LOG.csv")
    parser.add_argument("--raw", type=Path, default=None)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.config).read_text(encoding="utf-8"))
    res = run(spec, gkf_repeats=args.gkf_repeats, final=not args.no_final, decision=args.decision, raw=args.raw)
    print(describe(spec["name"], res))
    if not args.no_final:
        print(f"model saved: {paths.models_dir() / (spec['name'] + '.joblib')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
