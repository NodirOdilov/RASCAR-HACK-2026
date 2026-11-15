"""Build and validate Kaggle submission files.

Run:  python -m rascar.submission --input results/predictions/<name>.csv --output submissions/<name>.csv

Rules enforced (TASK.txt, "Метрика и формат отправки"): exactly two columns ``id,score``;
every id of ``sample_submission.csv`` exactly once; no extra ids; finite scores in [0, 1];
higher score = more suspicious detection. Nothing is ever uploaded by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, paths


class SubmissionError(ValueError):
    pass


def validate(sub: pd.DataFrame, sample: pd.DataFrame) -> dict:
    """Raise :class:`SubmissionError` on any violation; return the list of passed checks."""
    checks: dict[str, bool] = {}

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks[name] = bool(ok)
        if not ok:
            raise SubmissionError(f"{name} FAILED {detail}".strip())

    check("columns_exactly_id_score", list(sub.columns) == ["id", "score"], f"got {list(sub.columns)}")
    check("row_count_equals_sample", len(sub) == len(sample), f"{len(sub)} vs {len(sample)}")
    ids = sub["id"].astype(str)
    check("ids_unique", ids.is_unique, f"{int(ids.duplicated().sum())} duplicates")
    sample_ids = set(sample["id"].astype(str))
    check("no_missing_ids", sample_ids <= set(ids), f"{len(sample_ids - set(ids))} missing")
    check("no_extra_ids", set(ids) <= sample_ids, f"{len(set(ids) - sample_ids)} extra")
    score = pd.to_numeric(sub["score"], errors="coerce").to_numpy(dtype=float)
    check("scores_numeric_no_nan", not np.isnan(score).any(), f"{int(np.isnan(score).sum())} NaN")
    check("scores_finite", bool(np.isfinite(score).all()))
    check("scores_within_0_1", bool(((score >= 0) & (score <= 1)).all()), f"range [{score.min()}, {score.max()}]")
    check("scores_not_constant", bool(np.unique(score).size > 1))
    return checks


def build(pred: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Align predictions (``id, score``) to the official template order."""
    if not {"id", "score"} <= set(pred.columns):
        raise SubmissionError(f"prediction file must contain id and score, got {list(pred.columns)}")
    pred = pred[["id", "score"]].copy()
    pred["id"] = pred["id"].astype(str)
    if not pred["id"].is_unique:
        raise SubmissionError("prediction file has duplicated ids")
    sub = sample[["id"]].astype({"id": str}).merge(pred, on="id", how="left", validate="one_to_one")
    validate(sub, sample)
    return sub


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def write(sub: pd.DataFrame, output: Path, meta: dict | None = None, sample: pd.DataFrame | None = None) -> dict:
    """Atomically write the CSV (UTF-8, LF, full float precision) plus its manifest entry."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    sub.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n", float_format="%.10f")
    tmp.replace(output)
    reread = pd.read_csv(output, dtype={"id": str})
    checks = validate(reread, sample if sample is not None else data.load_sample_submission())
    entry = {
        "file": output.name,
        "rows": int(len(reread)),
        "sha256": sha256_file(output),
        "score_min": float(reread["score"].min()),
        "score_max": float(reread["score"].max()),
        "score_mean": float(reread["score"].mean()),
        "unique_scores": int(reread["score"].nunique()),
        "checks": checks,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
    }
    entry.update(meta or {})
    manifest_path = output.parent / "submission_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[output.name] = entry
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a validated Kaggle submission from a prediction file")
    parser.add_argument("--input", type=Path, required=True, help="CSV with columns id,score")
    parser.add_argument("--output", type=Path, required=True, help="destination CSV inside submissions/")
    parser.add_argument("--note", default="", help="free-text provenance note stored in the manifest")
    parser.add_argument("--raw", type=Path, default=None)
    args = parser.parse_args(argv)
    sample = data.load_sample_submission(args.raw)
    pred = pd.read_csv(args.input, dtype={"id": str})
    sub = build(pred, sample)
    entry = write(sub, args.output, {"source_predictions": str(args.input), "note": args.note}, sample)
    print(f"submission ok: {args.output} rows={entry['rows']} sha256={entry['sha256']} "
          f"score_range=[{entry['score_min']:.4f}, {entry['score_max']:.4f}] unique={entry['unique_scores']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
