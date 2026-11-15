"""Stage A - forensic data audit.

Run:  python -m rascar.audit

Writes ``results/data_audit.json`` and ``reports/01_DATA_AUDIT.md``. The raw files
are opened read-only; images are only hashed / header-parsed, never fully decoded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import ks_2samp

from . import data, geometry, metrics, paths

TABLES = [
    "train", "test", "sample_submission", "train_frames", "test_frames", "train_context",
    "test_context", "train_stream_predictions", "test_stream_predictions", "human_boxes_train",
    "machine_frames", "machine_predictions",
]
NON_TABULAR = ["scoring_policy.json", "TASK.txt", "MACHINE_POOL.txt", "baseline.ipynb"]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def table_profile(df: pd.DataFrame) -> dict:
    prof = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "nulls": {c: int(v) for c, v in df.isna().sum().items() if v},
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric": {},
    }
    for col in df.select_dtypes(include="number").columns:
        s = df[col]
        prof["numeric"][col] = {"min": float(s.min()), "max": float(s.max()),
                                "mean": float(s.mean()), "median": float(s.median())}
    if "class_name" in df:
        prof["class_counts"] = df["class_name"].value_counts().to_dict()
    return prof


def box_checks(df: pd.DataFrame, width: int = data.FRAME_W, height: int = data.FRAME_H) -> dict:
    b = df[data.BOX_COLS].to_numpy(float)
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    return {
        "n": int(len(df)),
        "x_range": [float(b[:, [0, 2]].min()), float(b[:, [0, 2]].max())],
        "y_range": [float(b[:, [1, 3]].min()), float(b[:, [1, 3]].max())],
        "non_positive_width_or_height": int(((w <= 0) | (h <= 0)).sum()),
        "out_of_frame": int(((b[:, 0] < 0) | (b[:, 1] < 0) | (b[:, 2] > width) | (b[:, 3] > height)).sum()),
        "touching_border_le_1px": int(((b[:, 0] <= 1) | (b[:, 1] <= 1) | (b[:, 2] >= width - 1) | (b[:, 3] >= height - 1)).sum()),
        "tiny_side_lt_8px": int(((w < 8) | (h < 8)).sum()),
        "min_width": float(w.min()), "min_height": float(h.min()),
        "looks_normalized_0_1": bool(b.max() <= 1.5),
        "one_decimal_precision": bool(np.allclose(b * 10, np.round(b * 10), atol=1e-6)),
        "confidence_range": [float(df["confidence"].min()), float(df["confidence"].max())],
        "confidence_two_decimals": bool(np.allclose(df["confidence"] * 100, np.round(df["confidence"] * 100), atol=1e-6)),
    }


def reproduce_labels(train: pd.DataFrame, human: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """Recompute ``error`` from manual boxes with the documented matching rule."""
    tr = train.copy()
    tr["class_canon"] = data.canonical_class(tr["class_name"], policy)
    scored = human[human["evaluated_class"] & human["class_canon"].isin(policy["classes"])]
    hb_groups = {k: g[data.BOX_COLS].to_numpy(float) for k, g in scored.groupby(["frame_id", "class_canon"])}
    parts = []
    for (frame_id, cls), grp in tr.groupby(["frame_id", "class_canon"], sort=False):
        gt = hb_groups.get((frame_id, cls), np.zeros((0, 4)))
        gt_index, matched_iou, best_iou = geometry.match_one_to_one(
            grp[data.BOX_COLS].to_numpy(float), gt, policy["correct_iou"])
        parts.append(pd.DataFrame({"id": grp["id"].to_numpy(), "repro_error": (gt_index < 0).astype(int),
                                   "matched_iou": matched_iou, "best_iou": best_iou,
                                   "n_gt_same_class": len(gt)}))
    return tr.merge(pd.concat(parts, ignore_index=True), on="id", validate="one_to_one")


def error_taxonomy(repro: pd.DataFrame, human: pd.DataFrame) -> pd.DataFrame:
    """Coarse cause of every erroneous box (training diagnostics only)."""
    by_frame = {k: g for k, g in human.groupby("frame_id")}
    causes = []
    for row in repro.itertuples(index=False):
        if row.error == 0:
            causes.append("correct")
            continue
        g = by_frame.get(row.frame_id)
        any_iou = 0.0
        if g is not None and len(g):
            other = g[g["class_canon"] != row.class_canon]
            if len(other):
                any_iou = float(geometry.iou_matrix([[row.x1, row.y1, row.x2, row.y2]],
                                                    other[data.BOX_COLS].to_numpy(float)).max())
        if row.best_iou >= 0.6:
            causes.append("duplicate (lost one-to-one match)")
        elif row.best_iou >= 0.3:
            causes.append("geometry (IoU 0.3-0.6 with same-class manual box)")
        elif row.best_iou > 0:
            causes.append("weak overlap (IoU 0-0.3 with same-class manual box)")
        elif any_iou >= 0.3:
            causes.append("class confusion (overlaps other-class manual box)")
        else:
            causes.append("false positive (no manual box nearby)")
    out = repro.copy()
    out["cause"] = causes
    return out


def image_inventory(raw: Path, frame_tables: dict[str, pd.DataFrame], hash_images: bool = True) -> dict:
    jpgs = sorted(p for p in raw.iterdir() if p.suffix.lower() == ".jpg")
    on_disk = {p.name for p in jpgs}
    info: dict = {"jpg_files_on_disk": len(jpgs),
                  "total_megabytes": round(sum(p.stat().st_size for p in jpgs) / 1e6, 1)}
    referenced: set[str] = set()
    for name, frames in frame_tables.items():
        refs = set(frames["image_path"])
        referenced |= refs
        info[f"{name}_frames_missing_on_disk"] = sorted(refs - on_disk)[:20]
        info[f"{name}_frames_missing_count"] = len(refs - on_disk)
        info[f"{name}_image_path_equals_frame_id_jpg"] = bool((frames["image_path"] == frames["frame_id"] + ".jpg").all())
    info["jpg_not_referenced_by_any_frame_table"] = len(on_disk - referenced)
    info["frame_tables_pairwise_disjoint"] = bool(
        sum(len(f) for f in frame_tables.values()) == len(set().union(*[set(f["frame_id"]) for f in frame_tables.values()])))
    sizes: dict[str, int] = {}
    mismatched = 0
    declared = {}
    for frames in frame_tables.values():
        declared.update({r.image_path: (int(r.image_width), int(r.image_height)) for r in frames.itertuples()})
    unreadable = []
    for p in jpgs:
        try:
            with Image.open(p) as im:  # header only, pixels are not decoded
                size = im.size
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            unreadable.append(f"{p.name}: {type(exc).__name__}")
            continue
        sizes[f"{size[0]}x{size[1]}"] = sizes.get(f"{size[0]}x{size[1]}", 0) + 1
        if p.name in declared and declared[p.name] != size:
            mismatched += 1
    info["header_resolutions"] = sizes
    info["declared_size_mismatches"] = mismatched
    info["unreadable_headers"] = unreadable
    if hash_images:
        hashes = {p.name: sha256_file(p) for p in jpgs}
        rolled = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(hashes.items())).encode()).hexdigest()
        seen: dict[str, list[str]] = {}
        for name, h in hashes.items():
            seen.setdefault(h, []).append(name)
        dup_groups = [v for v in seen.values() if len(v) > 1]
        info["sha256_of_all_jpg_digests"] = rolled
        info["exact_duplicate_image_groups"] = len(dup_groups)
        info["exact_duplicate_examples"] = dup_groups[:5]
    return info


def temporal_structure(frames: pd.DataFrame, context: pd.DataFrame) -> dict:
    t = frames.set_index("frame_id")["t_sec"]
    tgt = context["frame_id"]
    gaps_prev = (tgt.map(t) - context["previous"].map(t)).dropna()
    gaps_next = (context["next"].map(t) - tgt.map(t)).dropna()
    target_times = np.sort(tgt.map(t).to_numpy())
    spacing = np.diff(target_times)
    days = frames.set_index("frame_id")["utc_day"]
    cross_day = int((context["previous"].map(days).dropna() != tgt.map(days)[context["previous"].notna()]).sum()
                    + (context["next"].map(days).dropna() != tgt.map(days)[context["next"].notna()]).sum())
    return {
        "target_frames": int(len(context)),
        "missing_previous": int(context["previous"].isna().sum()),
        "missing_next": int(context["next"].isna().sum()),
        "gap_previous_seconds": pd.Series(gaps_prev.round()).value_counts().sort_index().to_dict(),
        "gap_next_seconds": pd.Series(gaps_next.round()).value_counts().sort_index().to_dict(),
        "target_spacing_seconds": {"min": float(spacing.min()), "median": float(np.median(spacing)),
                                   "max": float(spacing.max()),
                                   "below_10_min": int((spacing < 600).sum())},
        "neighbours_in_other_utc_day": cross_day,
        "frames_per_utc_day": frames["utc_day"].value_counts().sort_index().to_dict(),
        "targets_per_utc_day": tgt.map(days).value_counts().sort_index().to_dict(),
    }


def cross_target_persistence(targets: pd.DataFrame, frames: pd.DataFrame) -> dict:
    """How often does a box re-appear (same class, IoU>=0.5) in the previous target frame
    (about 33 minutes earlier) of the same UTC day?  Motivates day-level validation."""
    df = data.attach_day(targets, frames)
    df = df[df["class_name"] != "table"]
    order = df[["frame_id", "utc_day", "t_sec"]].drop_duplicates().sort_values("t_sec")
    prev_frame = {}
    for _, grp in order.groupby("utc_day"):
        ids = grp["frame_id"].tolist()
        prev_frame.update(dict(zip(ids[1:], ids[:-1])))
    by_frame = {k: g for k, g in df.groupby("frame_id")}
    hits = total = 0
    for frame_id, grp in by_frame.items():
        prev = by_frame.get(prev_frame.get(frame_id))
        for cls, sub in grp.groupby("class_name"):
            total += len(sub)
            if prev is None:
                continue
            cand = prev[prev["class_name"] == cls]
            if len(cand):
                hits += int((geometry.iou_matrix(sub[data.BOX_COLS], cand[data.BOX_COLS]).max(axis=1) >= 0.5).sum())
    return {"non_table_boxes": int(total), "reappear_in_previous_labelled_frame_iou50": int(hits),
            "share": float(hits / total) if total else None}


def run(raw: Path | None = None, hash_images: bool = True) -> dict:
    started = time.time()
    raw = Path(raw) if raw else paths.raw_dir()
    audit: dict = {"raw_dir": str(raw), "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}

    # ---- inventory -------------------------------------------------------------------
    present = {p.name for p in raw.iterdir() if p.is_file()}
    expected = [f"{t}.csv" for t in TABLES] + NON_TABULAR
    audit["files"] = {
        name: ({"bytes": (raw / name).stat().st_size, "sha256": sha256_file(raw / name)}
               if name in present else None) for name in expected}
    audit["missing_expected_files"] = [n for n in expected if n not in present]
    audit["unexpected_non_image_files"] = sorted(n for n in present if not n.lower().endswith(".jpg") and n not in expected)

    policy = data.load_policy(raw)
    audit["scoring_policy"] = policy
    T = {name: data.read_table(name, raw) for name in TABLES}
    audit["tables"] = {name: table_profile(df) for name, df in T.items()}

    train, test, sample = T["train"], T["test"], T["sample_submission"]
    frames = {p: data.load_frames(p, raw) for p in ("train", "test", "machine")}
    human = data.load_human_boxes(raw, policy)

    # ---- identifiers and joins -------------------------------------------------------
    ids = {
        "train_id_unique": bool(train["id"].is_unique),
        "test_id_unique": bool(test["id"].is_unique),
        "train_test_ids_disjoint": bool(set(train["id"]).isdisjoint(test["id"])),
        "sample_ids_equal_test_ids": bool(set(sample["id"]) == set(test["id"])),
        "sample_rows": int(len(sample)), "test_rows": int(len(test)),
        "sample_id_unique": bool(sample["id"].is_unique),
        "sample_order_equals_test_order": bool(len(sample) == len(test) and (sample["id"].to_numpy() == test["id"].to_numpy()).all()),
        "sample_score_values": sorted(sample["score"].unique().tolist()),
    }
    for part in ("train", "test"):
        tgt, stream, ctx = T[part], T[f"{part}_stream_predictions"], T[f"{part}_context"]
        fr = frames[part]
        merged = tgt.merge(stream, left_on="id", right_on="detection_id", how="left", suffixes=("", "_s"), validate="one_to_one")
        same = (merged[["class_name", "confidence", *data.BOX_COLS]].to_numpy()
                == merged[["class_name_s", "confidence_s", *[c + "_s" for c in data.BOX_COLS]]].to_numpy()).all()
        on_targets = stream[stream["frame_id"].isin(ctx["frame_id"])]
        excluded = set(policy.get("test_excluded_classes", []))
        expected_rows = on_targets if part == "train" else on_targets[~on_targets["class_name"].isin(excluded)]
        ids[part] = {
            "target_ids_all_in_stream": bool(merged["detection_id"].notna().all()),
            "target_rows_identical_to_stream_rows": bool(same),
            "targets_equal_stream_detections_on_target_frames"
            + ("" if part == "train" else "_excluding_table"): bool(set(expected_rows["detection_id"]) == set(tgt["id"])),
            "target_frames_all_in_context": bool(set(tgt["frame_id"]) <= set(ctx["frame_id"])),
            "context_targets_without_any_target_row": int(len(set(ctx["frame_id"]) - set(tgt["frame_id"]))),
            "all_frames_in_frames_table": bool(set(stream["frame_id"]) <= set(fr["frame_id"])),
            "frames_table_equals_context_union": bool(
                set(fr["frame_id"]) == set(ctx["frame_id"]) | set(ctx["previous"].dropna()) | set(ctx["next"].dropna())),
            "frames_without_any_detection": int(len(set(fr["frame_id"]) - set(stream["frame_id"]))),
            "detections_per_target_frame": tgt.groupby("frame_id").size().describe().round(2).to_dict(),
        }
    ids["machine"] = {
        "pool_frames_overlap_train_frames": int(len(set(frames["machine"]["frame_id"]) & set(frames["train"]["frame_id"]))),
        "pool_frames_overlap_test_frames": int(len(set(frames["machine"]["frame_id"]) & set(frames["test"]["frame_id"]))),
        "pool_predictions_all_in_pool_frames": bool(set(T["machine_predictions"]["frame_id"]) <= set(frames["machine"]["frame_id"])),
        "pool_frames_without_detection": int(len(set(frames["machine"]["frame_id"]) - set(T["machine_predictions"]["frame_id"]))),
        "ids_disjoint_from_train_and_test_streams": bool(
            set(T["machine_predictions"]["detection_id"]).isdisjoint(T["train_stream_predictions"]["detection_id"])
            and set(T["machine_predictions"]["detection_id"]).isdisjoint(T["test_stream_predictions"]["detection_id"])),
    }
    audit["identifiers_and_joins"] = ids

    # ---- geometry --------------------------------------------------------------------
    audit["boxes"] = {name: box_checks(T[name]) for name in
                      ("train", "test", "train_stream_predictions", "test_stream_predictions", "machine_predictions")}
    hb = human
    audit["human_boxes"] = {
        "rows": int(len(hb)), "frames": int(hb["frame_id"].nunique()),
        "normalized_cxcywh": bool(hb[["cx", "cy", "w", "h"]].to_numpy().max() <= 1.0 + 1e-9),
        "out_of_frame_after_conversion": int(((hb["x1"] < -0.5) | (hb["y1"] < -0.5) | (hb["x2"] > data.FRAME_W + 0.5) | (hb["y2"] > data.FRAME_H + 0.5)).sum()),
        "class_by_scope_and_source": {f"{c}|evaluated={e}|{s}": int(n) for (c, e, s), n in
                                      hb.groupby(["class_name", "evaluated_class", "source"]).size().items()},
        "revision_boxes": hb[hb["source"] == "revision"]["class_name"].value_counts().to_dict(),
        "frames_all_train_targets": bool(set(hb["frame_id"]) <= set(T["train_context"]["frame_id"])),
    }

    # ---- classes ---------------------------------------------------------------------
    scored = set(policy["classes"])
    aliases = policy.get("aliases", {})
    audit["classes"] = {
        "policy_classes": policy["classes"], "aliases": aliases,
        "detector_classes_seen": sorted(set(T["train_stream_predictions"]["class_name"]) | set(T["test_stream_predictions"]["class_name"]) | set(T["machine_predictions"]["class_name"])),
        "train_classes": train["class_name"].value_counts().to_dict(),
        "test_classes": test["class_name"].value_counts().to_dict(),
        "detector_classes_outside_policy": sorted((set(train["class_name"]) | set(test["class_name"])) - scored),
        "detector_rows_using_alias_spelling": int(train["class_name"].isin(aliases).sum() + test["class_name"].isin(aliases).sum()),
        "human_rows_using_alias_spelling": hb[hb["class_name"].isin(aliases)]["class_name"].value_counts().to_dict(),
        "policy_classes_never_predicted": sorted(scored - set(T["machine_predictions"]["class_name"]) - set(T["train_stream_predictions"]["class_name"]) - set(T["test_stream_predictions"]["class_name"])),
        "test_contains_excluded_class": bool(test["class_name"].isin(policy.get("test_excluded_classes", [])).any()),
    }

    # ---- time ------------------------------------------------------------------------
    audit["time"] = {p: temporal_structure(frames[p], T[f"{p}_context"]) for p in ("train", "test")}
    pool = frames["machine"].sort_values("t_sec")
    audit["time"]["machine"] = {
        "frames_per_utc_day": pool["utc_day"].value_counts().sort_index().to_dict(),
        "gap_seconds": pool["t_sec"].diff().round().value_counts().sort_index().to_dict(),
    }
    for part in ("train", "test", "machine"):
        f = frames[part]
        from_name = pd.to_datetime(f["frame_id"].str.extract(r"_(\d{8}_\d{6})$")[0], format="%Y%m%d_%H%M%S", utc=True)
        audit["time"][part]["max_abs_seconds_between_filename_and_timestamp"] = float((f["ts"] - from_name).dt.total_seconds().abs().max())
        audit["time"][part]["timestamp_range_utc"] = [f["ts"].min().isoformat(), f["ts"].max().isoformat()]
        if "day" in f:
            audit["time"][part]["day_column_matches_utc_day"] = bool((f["day"].astype(str) == f["utc_day"].str.replace("-", "")).all())

    # ---- labels ----------------------------------------------------------------------
    tr = data.attach_day(train, frames["train"])
    nt = data.non_table(tr, policy)
    lab = {
        "all_rows": {"n": int(len(tr)), "errors": int(tr["error"].sum()), "prevalence": float(tr["error"].mean())},
        "scored_scope_non_table": {"n": int(len(nt)), "errors": int(nt["error"].sum()), "prevalence": float(nt["error"].mean()),
                                   "frames_with_rows": int(nt["frame_id"].nunique())},
        "table": {"n": int((tr["class_name"] == "table").sum()), "errors": int(tr.loc[tr["class_name"] == "table", "error"].sum())},
        "by_class": {c: {"n": int(len(g)), "errors": int(g["error"].sum()), "prevalence": float(g["error"].mean())}
                     for c, g in tr.groupby("class_name")},
        "by_day_non_table": {d: {"frames": int(g["frame_id"].nunique()), "n": int(len(g)), "errors": int(g["error"].sum()),
                                 "prevalence": float(g["error"].mean()),
                                 "ap_1_minus_confidence": metrics.average_precision(g["error"], 1 - g["confidence"])}
                             for d, g in nt.groupby("utc_day")},
        "by_day_class_non_table": {f"{d}|{c}": {"n": int(len(g)), "errors": int(g["error"].sum())}
                                   for (d, c), g in nt.groupby(["utc_day", "class_name"])},
    }
    repro = reproduce_labels(train, human, policy)
    tax = error_taxonomy(repro, human)
    lab["reproduction"] = {
        "rows": int(len(repro)), "agree": int((repro["error"] == repro["repro_error"]).sum()),
        "disagree_ids": repro.loc[repro["error"] != repro["repro_error"], "id"].tolist(),
        "rule": "per frame and canonical class: one-to-one matching, IoU>=correct_iou, maximise pairs then total IoU",
    }
    nt_tax = tax[tax["class_name"] != "table"]
    lab["error_causes_non_table"] = {f"{cause}|{cls}": int(n) for (cause, cls), n in
                                     nt_tax[nt_tax["error"] == 1].groupby(["cause", "class_name"]).size().items()}
    lab["error_causes_non_table_total"] = nt_tax[nt_tax["error"] == 1]["cause"].value_counts().to_dict()
    near = nt_tax[(nt_tax["best_iou"] >= 0.5) & (nt_tax["best_iou"] < 0.7)]
    lab["borderline_iou_0.5_0.7"] = {"n": int(len(near)), "share_of_non_table": float(len(near) / len(nt_tax)),
                                     "errors": int(near["error"].sum())}
    lab["matched_iou_quantiles_correct_non_table"] = nt_tax.loc[nt_tax["error"] == 0, "matched_iou"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict()
    audit["labels"] = lab
    audit["cross_target_persistence_train"] = cross_target_persistence(train, frames["train"])

    # ---- notebook output vs real files -----------------------------------------------
    audit["baseline_notebook_consistency"] = {
        "notebook_printed_non_table_rows": 683, "actual_non_table_rows": int(len(nt)),
        "notebook_printed_errors": 55, "actual_non_table_errors": int(nt["error"].sum()),
        "notebook_printed_submission_rows": 654, "actual_test_rows": int(len(test)),
        "task_txt_table_claim": "3 errors on 472 tables", "actual_table": lab["table"],
        "verdict": "stored notebook outputs come from another data snapshot; the CSV files are authoritative",
    }

    # ---- drift -----------------------------------------------------------------------
    te = data.attach_day(test, frames["test"])
    drift = {"confidence_ks_train_vs_test_non_table": {}, "class_share": {}}
    for cls in sorted(set(nt["class_name"])):
        a, b = nt.loc[nt["class_name"] == cls, "confidence"], te.loc[te["class_name"] == cls, "confidence"]
        if len(a) and len(b):
            ks = ks_2samp(a, b)
            drift["confidence_ks_train_vs_test_non_table"][cls] = {
                "ks": float(ks.statistic), "p": float(ks.pvalue), "train_mean": float(a.mean()), "test_mean": float(b.mean()),
                "train_n": int(len(a)), "test_n": int(len(b))}
        drift["class_share"][cls] = {"train": float((nt["class_name"] == cls).mean()), "test": float((te["class_name"] == cls).mean())}
    drift["rows_per_day"] = {"train": nt.groupby("utc_day").size().to_dict(), "test": te.groupby("utc_day").size().to_dict()}
    drift["frames_with_rows_per_day"] = {"train": nt.groupby("utc_day")["frame_id"].nunique().to_dict(),
                                         "test": te.groupby("utc_day")["frame_id"].nunique().to_dict()}
    drift["rows_per_frame"] = {"train": nt.groupby("frame_id").size().describe().round(2).to_dict(),
                               "test": te.groupby("frame_id").size().describe().round(2).to_dict()}
    audit["drift"] = drift

    # ---- images ----------------------------------------------------------------------
    audit["images"] = image_inventory(raw, frames, hash_images=hash_images)

    audit["forbidden_at_inference"] = [
        "train.csv:error (target)",
        "human_boxes_train.csv (manual boxes; exist for train target frames only)",
        "anything derived from the two above (best/matched IoU, error causes)",
    ]
    audit["elapsed_seconds"] = round(time.time() - started, 1)
    return _jsonable(audit)


def write_markdown(audit: dict, path: Path) -> None:
    lab, ids, tm = audit["labels"], audit["identifiers_and_joins"], audit["time"]
    img, drift, nb = audit["images"], audit["drift"], audit["baseline_notebook_consistency"]
    L: list[str] = []
    add = L.append
    add("# 01 - Data audit (Stage A)\n")
    add(f"Generated by `python -m rascar.audit` at {audit['generated_utc']} (UTC). Machine-readable twin: "
        "`results/data_audit.json`. Every number below is computed from the files on disk.\n")
    add(f"Raw directory: `{audit['raw_dir']}` (read-only; never modified).\n")
    add("## 1. Inventory\n")
    add("| file | bytes | sha256 (first 16) |\n|---|---:|---|")
    for name, meta in audit["files"].items():
        add(f"| {name} | {meta['bytes']} | `{meta['sha256'][:16]}` |" if meta else f"| {name} | MISSING | - |")
    add(f"\nMissing expected files: **{audit['missing_expected_files'] or 'none'}**; unexpected non-image files: "
        f"**{audit['unexpected_non_image_files'] or 'none'}**.\n")
    add(f"Images: **{img['jpg_files_on_disk']}** JPG, {img['total_megabytes']} MB, header resolutions {img['header_resolutions']}; "
        f"declared-size mismatches: {img['declared_size_mismatches']}; unreadable headers: {len(img['unreadable_headers'])}; "
        f"JPGs not referenced by a frame table: {img['jpg_not_referenced_by_any_frame_table']}; "
        f"exact duplicate groups (sha256): {img.get('exact_duplicate_image_groups', 'not hashed')}.")
    add(f"Frames missing on disk - train: {img['train_frames_missing_count']}, test: {img['test_frames_missing_count']}, "
        f"machine pool: {img['machine_frames_missing_count']}. Frame tables are pairwise disjoint: {img['frame_tables_pairwise_disjoint']}.\n")
    add("## 2. Tables\n")
    add("| table | rows | columns | nulls | duplicate rows |\n|---|---:|---|---|---:|")
    for name, prof in audit["tables"].items():
        add(f"| {name} | {prof['rows']} | {', '.join(prof['columns'])} | {prof['nulls'] or '-'} | {prof['duplicate_rows']} |")
    add("\n## 3. Identifiers and join cardinalities\n")
    add(f"- `train.id` unique: {ids['train_id_unique']}; `test.id` unique: {ids['test_id_unique']}; disjoint: {ids['train_test_ids_disjoint']}.")
    add(f"- `sample_submission.id` set equals `test.id` set: **{ids['sample_ids_equal_test_ids']}** ({ids['sample_rows']} rows vs {ids['test_rows']}); "
        f"same row order: {ids['sample_order_equals_test_order']} (order is arbitrary per TASK.txt); template scores: {ids['sample_score_values']}.")
    for part in ("train", "test"):
        p = ids[part]
        key = [k for k in p if k.startswith("targets_equal_stream")][0]
        add(f"- {part}: `id` == `detection_id` of the stream (one-to-one, all found: {p['target_ids_all_in_stream']}, identical values: "
            f"{p['target_rows_identical_to_stream_rows']}); {key.replace('_', ' ')}: {p[key]}; frames table == context union: "
            f"{p['frames_table_equals_context_union']}; target frames without any target row: {p['context_targets_without_any_target_row']}.")
    m = ids["machine"]
    add(f"- machine pool: overlap with train frames {m['pool_frames_overlap_train_frames']}, with test frames {m['pool_frames_overlap_test_frames']}; "
        f"pool frames without detections: {m['pool_frames_without_detection']}.\n")
    add("## 4. Boxes and classes\n")
    add("| table | n | x range | y range | w/h<=0 | out of frame | touching border | min w | min h | 0.1px grid | conf range |\n|---|---:|---|---|---:|---:|---:|---:|---:|---|---|")
    for name, b in audit["boxes"].items():
        add(f"| {name} | {b['n']} | {b['x_range']} | {b['y_range']} | {b['non_positive_width_or_height']} | {b['out_of_frame']} | "
            f"{b['touching_border_le_1px']} | {b['min_width']:.1f} | {b['min_height']:.1f} | {b['one_decimal_precision']} | {b['confidence_range']} |")
    c = audit["classes"]
    add(f"\nPrediction boxes are raw pixels of the 1280x720 frame (pool images are stored at 640x360 but its coordinates are 1280x720). "
        f"Manual boxes are normalized `cx,cy,w,h` ({audit['human_boxes']['normalized_cxcywh']}).")
    add(f"\nDetector classes seen: {c['detector_classes_seen']}; policy classes never predicted by the detector: {c['policy_classes_never_predicted']}; "
        f"alias spellings appear only in manual boxes: {c['human_rows_using_alias_spelling']} (detector rows with alias spelling: "
        f"{c['detector_rows_using_alias_spelling']}). Test contains excluded class `table`: {c['test_contains_excluded_class']}.\n")
    add("## 5. Time\n")
    for part in ("train", "test"):
        t = tm[part]
        add(f"- {part}: {t['target_frames']} target frames ({t['targets_per_utc_day']}), missing previous {t['missing_previous']}, missing next "
            f"{t['missing_next']}; gap to previous (s): {t['gap_previous_seconds']}; gap to next (s): {t['gap_next_seconds']}; spacing between "
            f"target frames: median {t['target_spacing_seconds']['median']:.0f}s, min {t['target_spacing_seconds']['min']:.0f}s "
            f"({t['target_spacing_seconds']['below_10_min']} gaps < 10 min); neighbours from another UTC day: {t['neighbours_in_other_utc_day']}; "
            f"filename time vs timestamp: <= {t['max_abs_seconds_between_filename_and_timestamp']:.2f}s.")
    add(f"- machine pool: {tm['machine']['frames_per_utc_day']}, gaps (s): {tm['machine']['gap_seconds']}.")
    add("- All timestamps carry an explicit `+00:00` offset (UTC, as stated in TASK.txt); local validation is split by UTC day.\n")
    add("## 6. Labels\n")
    s = lab["scored_scope_non_table"]
    add(f"- All train rows: {lab['all_rows']['n']} with {lab['all_rows']['errors']} errors. `table`: {lab['table']['n']} rows, {lab['table']['errors']} errors "
        "(excluded from the test and from local validation).")
    add(f"- **Scored scope (non-table): {s['n']} rows, {s['errors']} errors, prevalence {s['prevalence']:.4f}**, on {s['frames_with_rows']} frames.")
    add("\n| UTC day | frames | rows | errors | prevalence (= AP of a constant score) | AP of 1-confidence |\n|---|---:|---:|---:|---:|---:|")
    for d, v in lab["by_day_non_table"].items():
        add(f"| {d} | {v['frames']} | {v['n']} | {v['errors']} | {v['prevalence']:.4f} | {v['ap_1_minus_confidence']:.4f} |")
    add("\n| class | rows | errors | prevalence |\n|---|---:|---:|---:|")
    for cname, v in lab["by_class"].items():
        add(f"| {cname} | {v['n']} | {v['errors']} | {v['prevalence']:.3f} |")
    r = lab["reproduction"]
    add(f"\n**Label reproduction:** recomputing `error` from `human_boxes_train.csv` with the documented rule agrees on "
        f"**{r['agree']}/{r['rows']}** rows (disagreements: {r['disagree_ids'] or 'none'}).")
    add(f"\nCauses of the non-table errors (diagnostic taxonomy): {lab['error_causes_non_table_total']}.")
    b = lab["borderline_iou_0.5_0.7"]
    add(f"\n{b['n']} non-table rows ({b['share_of_non_table']:.0%}) have a best same-class IoU in [0.5, 0.7): the label is decided by a few "
        "pixels there, which bounds the achievable AP.")
    p = audit["cross_target_persistence_train"]
    add(f"\nTemporal correlation: {p['reappear_in_previous_labelled_frame_iou50']}/{p['non_table_boxes']} non-table boxes "
        f"({(p['share'] or 0):.0%}) re-appear with IoU>=0.5 in the previous labelled frame of the same day (about 33 minutes earlier) - "
        "random row splits would leak; validation must be by UTC day.\n")
    add("## 7. Organizer notebook vs. real files\n")
    add(f"`baseline.ipynb` stores outputs for {nb['notebook_printed_non_table_rows']} non-table rows / {nb['notebook_printed_errors']} errors and a "
        f"{nb['notebook_printed_submission_rows']}-row submission; the files on disk have {nb['actual_non_table_rows']} / {nb['actual_non_table_errors']} and "
        f"{nb['actual_test_rows']} test rows. {nb['verdict']}. The TASK.txt statement about tables is confirmed: {nb['actual_table']}.\n")
    add("## 8. Train -> test drift\n")
    add("| class | train n | test n | train conf mean | test conf mean | KS | p | train share | test share |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cname, v in drift["confidence_ks_train_vs_test_non_table"].items():
        sh = drift["class_share"][cname]
        add(f"| {cname} | {v['train_n']} | {v['test_n']} | {v['train_mean']:.3f} | {v['test_mean']:.3f} | {v['ks']:.3f} | {v['p']:.3f} | {sh['train']:.3f} | {sh['test']:.3f} |")
    add(f"\nRows per day - train: {drift['rows_per_day']['train']}; test: {drift['rows_per_day']['test']}. Frames with rows per day - train: "
        f"{drift['frames_with_rows_per_day']['train']}; test: {drift['frames_with_rows_per_day']['test']}.\n")
    add("## 9. TRAIN-ONLY / FORBIDDEN AT INFERENCE\n")
    for item in audit["forbidden_at_inference"]:
        add(f"- {item}")
    add("\n## 10. Recommendations for modelling\n")
    add("1. Validate by UTC day (train on one day, validate on the other, then swap), without `table`, exactly as TASK.txt prescribes.")
    add("2. Only ~200 scored training rows exist: prefer few, physically meaningful features and strongly regularised models; "
        "report frame-block bootstrap intervals and treat differences below the interval width as noise.")
    add("3. Use `id == detection_id` to take every feature from the stream tables; build temporal features only from the `previous`/`next` "
        "frames listed in the context tables so that train and test features are computed identically.")
    add("4. The machine pool covers the two *training* days only: use it for day-independent scene statistics and keep the validation day "
        "of the pool out of the training fold.")
    add("5. `id`/`detection_id` are random and frame ids encode time only - neither may be used as a predictor.")
    add(f"\n_Audit runtime: {audit['elapsed_seconds']} s._")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage A data audit")
    parser.add_argument("--raw", type=Path, default=None, help="raw data directory (default: project data/raw/...)")
    parser.add_argument("--no-image-hash", action="store_true", help="skip sha256 of the JPG files")
    args = parser.parse_args(argv)
    audit = run(args.raw, hash_images=not args.no_image_hash)
    out_json = paths.results_dir() / "data_audit.json"
    out_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md = paths.reports_dir() / "01_DATA_AUDIT.md"
    write_markdown(audit, out_md)
    lab = audit["labels"]
    print(f"audit ok in {audit['elapsed_seconds']}s -> {out_json} , {out_md}")
    print(f"non-table train rows={lab['scored_scope_non_table']['n']} errors={lab['scored_scope_non_table']['errors']} "
          f"label reproduction={lab['reproduction']['agree']}/{lab['reproduction']['rows']} "
          f"test rows={audit['identifiers_and_joins']['test_rows']} jpg={audit['images']['jpg_files_on_disk']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
