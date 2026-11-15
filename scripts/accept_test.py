"""Apply the acceptance rule in force (reports/VALIDATION_PROTOCOL.md, section 5.2) to a candidate.

Usage:  python scripts/accept_test.py --candidate <config> --incumbent <config> --k <variants tried> [--hypothesis "..."]

The rule is the one recommended by the independent validation review
(results/agent_A/VALIDATION_PROTOCOL_agentA.md, section 7):

  statistic  D = mean of day APs(candidate) - mean of day APs(incumbent), paired bootstrap with whole frames
             resampled inside each day (4000 resamples);
  1  D > 0 on each of the two days;
  2  D >= delta(K): 0.005 (K <= 3), 0.010 (K <= 12), 0.012 (K <= 70), 0.015 (K <= 300);
  3  P(D > 0) >= 0.95 for K <= 3, otherwise >= 1 - 0.05 / K;
  4  pooled AP difference positive after ranking both models' scores inside each day (equal footing);
  5  frame-group 5-fold, 10 repeats on common partitions: mean difference positive (sanity check).

Condition 6 (mechanism stated in advance, train/test parity, no label path) is judged by a reviewer and recorded in
the hypothesis text. One row per application is appended to results/research/acceptance_tests.csv. The legacy rule of
section 4 is evaluated as well and stored for comparison; it no longer decides anything.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from rascar import metrics, paths, train

ROOT = paths.project_root()
N_BOOT = 4000
GKF_REPEATS = 10
SEED = 11


def spec_of(name: str) -> dict:
    return json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))


def delta_threshold(k: int) -> float:
    """Smallest gain in the mean of day APs that the best of K information-free candidates reaches in < 5 % of searches."""
    if k < 1:
        raise ValueError("K counts the scored variants of the idea, the candidate included; it cannot be below 1")
    for limit, delta in ((3, 0.005), (12, 0.010), (70, 0.012), (300, 0.015)):
        if k <= limit:
            return delta
    raise ValueError("the reviewed thresholds cover K <= 300; a larger search needs a new calibration")


def p_threshold(k: int) -> float:
    return 0.95 if k <= 3 else 1.0 - 0.05 / k


def day_rank(score: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Scores replaced by their rank fraction inside each day: removes scale differences between the two fold models."""
    out = np.empty(len(score), dtype=float)
    for day in np.unique(days):
        mask = days == day
        out[mask] = rankdata(score[mask], method="average") / mask.sum()
    return out


def paired_day_bootstrap(y, cand, inc, days, frames, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Paired bootstrap of the AP difference; whole frames are resampled inside each day.

    Returns the per-day differences, the difference of the mean of day APs and the difference of the pooled AP on
    equal footing (both models ranked inside each day), all on identical resamples.
    """
    y = np.asarray(y).astype(int)
    day_list = sorted(np.unique(days))
    blocks = {d: [np.flatnonzero((days == d) & (frames == f)) for f in pd.unique(frames[days == d])] for d in day_list}
    cand_r, inc_r = day_rank(cand, days), day_rank(inc, days)
    rng = np.random.default_rng(seed)
    per_day = np.empty((n_boot, len(day_list)))
    pooled = np.empty(n_boot)
    for i in range(n_boot):
        picked = []
        for j, d in enumerate(day_list):
            bl = blocks[d]
            while True:
                idx = np.concatenate([bl[k] for k in rng.integers(0, len(bl), len(bl))])
                if y[idx].sum() > 0:
                    break
            per_day[i, j] = metrics.average_precision(y[idx], cand[idx]) - metrics.average_precision(y[idx], inc[idx])
            picked.append(idx)
        idx = np.concatenate(picked)
        pooled[i] = metrics.average_precision(y[idx], cand_r[idx]) - metrics.average_precision(y[idx], inc_r[idx])
    mean_day = per_day.mean(axis=1)

    def stats(values: np.ndarray) -> dict:
        return {"se": float(values.std(ddof=1)), "p_gt0": float((values > 0).mean()),
                "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975))}

    return {"mean_day": stats(mean_day), "pooled_equal": stats(pooled),
            "days": {str(d): stats(per_day[:, j]) for j, d in enumerate(day_list)}}


def compare(candidate: str, incumbent: str, k: int, table=None, gkf_repeats: int = GKF_REPEATS,
            n_boot: int = N_BOOT) -> dict:
    X, y, aux = table if table is not None else train.load_training_table()
    days, frames = X["utc_day"].to_numpy(), X["frame_id"].to_numpy()
    d1, d2 = sorted(np.unique(days))
    out = {"candidate": candidate, "incumbent": incumbent, "k": int(k),
           "delta_threshold": delta_threshold(k), "p_threshold": round(p_threshold(k), 5)}
    scores, gkf = {}, {}
    for role, name in (("cand", candidate), ("inc", incumbent)):
        spec = spec_of(name)
        oof = train.day_oof(spec, X, y, aux)
        scores[role] = oof
        out[f"{role}_day1"] = metrics.average_precision(y[days == d1], oof[days == d1])
        out[f"{role}_day2"] = metrics.average_precision(y[days == d2], oof[days == d2])
        out[f"{role}_mean_day"] = (out[f"{role}_day1"] + out[f"{role}_day2"]) / 2
        out[f"{role}_pooled_stored"] = metrics.average_precision(y, oof)
        out[f"{role}_pooled_equal"] = metrics.average_precision(y, day_rank(oof, days))
        gkf[role] = np.asarray(train.frame_gkf_ap(spec, X, y, aux, repeats=gkf_repeats)["per_repeat"])
        out[f"{role}_gkf"] = float(gkf[role].mean())
    boot = paired_day_bootstrap(y, scores["cand"], scores["inc"], days, frames, n_boot=n_boot)
    out["d_day1"] = out["cand_day1"] - out["inc_day1"]
    out["d_day2"] = out["cand_day2"] - out["inc_day2"]
    out["d_mean_day"] = out["cand_mean_day"] - out["inc_mean_day"]
    out["se_mean_day"] = boot["mean_day"]["se"]
    out["p_gt0_mean_day"] = boot["mean_day"]["p_gt0"]
    out["ci_low_mean_day"], out["ci_high_mean_day"] = boot["mean_day"]["ci_low"], boot["mean_day"]["ci_high"]
    out["p_gt0_day1"], out["p_gt0_day2"] = boot["days"][str(d1)]["p_gt0"], boot["days"][str(d2)]["p_gt0"]
    out["d_pooled_equal"] = out["cand_pooled_equal"] - out["inc_pooled_equal"]
    out["p_gt0_pooled_equal"] = boot["pooled_equal"]["p_gt0"]
    out["d_gkf"] = float((gkf["cand"] - gkf["inc"]).mean())
    out["gkf_share_repeats_better"] = float((gkf["cand"] > gkf["inc"]).mean())
    out["c1_both_days"] = bool(out["d_day1"] > 0 and out["d_day2"] > 0)
    out["c2_effect_size"] = bool(out["d_mean_day"] >= out["delta_threshold"])
    out["c3_bootstrap"] = bool(out["p_gt0_mean_day"] >= p_threshold(k))
    out["c4_pooled_equal"] = bool(out["d_pooled_equal"] > 0)
    out["c5_gkf"] = bool(out["d_gkf"] > 0)
    passed = all(out[c] for c in ("c1_both_days", "c2_effect_size", "c3_bootstrap", "c4_pooled_equal", "c5_gkf"))
    out["quantitative_verdict"] = "PASS" if passed else "FAIL"
    out["positive_but_unproven"] = bool(not passed and out["d_mean_day"] > 0 and out["d_pooled_equal"] > 0)
    # the rule of section 4 (superseded) on the stored pooled scores, for comparison only
    legacy = metrics.paired_bootstrap_delta(y, scores["cand"], scores["inc"], frames, n_boot=3000, seed=SEED)
    out["legacy_p_pooled_stored"] = legacy["p_a_better"]
    out["legacy_verdict"] = "PASS" if (out["c1_both_days"] and out["cand_pooled_stored"] > out["inc_pooled_stored"]
                                       and out["d_gkf"] > 0 and legacy["p_a_better"] >= 0.80) else "FAIL"
    return out


def record(row: dict, hypothesis: str = "") -> None:
    path = paths.results_dir() / "research" / "acceptance_tests.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**row, "hypothesis": hypothesis, "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
    frame = pd.DataFrame([row]).round(7)
    if path.is_file():
        old = pd.read_csv(path)
        if list(old.columns) != list(frame.columns):          # rows written under the superseded rule keep their file
            path.replace(path.with_name("acceptance_tests_rule_s4.csv"))
        else:                                                 # a deterministic re-run replaces its own earlier row
            same = (old["candidate"] == row["candidate"]) & (old["incumbent"] == row["incumbent"]) & (old["k"] == row["k"])
            frame = pd.concat([old[~same], frame], ignore_index=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def describe(r: dict) -> str:
    return (f"{r['candidate']} vs {r['incumbent']} (K = {r['k']}: delta >= {r['delta_threshold']}, P >= {r['p_threshold']}):\n"
            f"  25 Oct {r['cand_day1']:.4f} vs {r['inc_day1']:.4f} ({r['d_day1']:+.4f}, P {r['p_gt0_day1']:.3f}); "
            f"26 Oct {r['cand_day2']:.4f} vs {r['inc_day2']:.4f} ({r['d_day2']:+.4f}, P {r['p_gt0_day2']:.3f})\n"
            f"  mean of days {r['cand_mean_day']:.4f} vs {r['inc_mean_day']:.4f}: {r['d_mean_day']:+.4f} "
            f"(SE {r['se_mean_day']:.4f}, 95% CI [{r['ci_low_mean_day']:+.4f}, {r['ci_high_mean_day']:+.4f}], "
            f"P(D>0) {r['p_gt0_mean_day']:.4f})\n"
            f"  pooled on equal footing {r['cand_pooled_equal']:.4f} vs {r['inc_pooled_equal']:.4f}: "
            f"{r['d_pooled_equal']:+.4f} (P {r['p_gt0_pooled_equal']:.3f}); stored pooled "
            f"{r['cand_pooled_stored']:.4f} vs {r['inc_pooled_stored']:.4f}\n"
            f"  frame-GKF x{GKF_REPEATS} {r['cand_gkf']:.4f} vs {r['inc_gkf']:.4f}: {r['d_gkf']:+.4f} "
            f"(better in {r['gkf_share_repeats_better']:.0%} of repeats)\n"
            f"  c1 both days {r['c1_both_days']} | c2 effect size {r['c2_effect_size']} | c3 bootstrap {r['c3_bootstrap']} | "
            f"c4 pooled {r['c4_pooled_equal']} | c5 GKF {r['c5_gkf']}  =>  {r['quantitative_verdict']}"
            f"{' (positive but unproven)' if r['positive_but_unproven'] else ''}; legacy rule: {r['legacy_verdict']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--k", type=int, required=True, help="number of scored variants of the idea, discarded ones included")
    parser.add_argument("--hypothesis", default="")
    args = parser.parse_args()
    row = compare(args.candidate, args.incumbent, args.k)
    record(row, args.hypothesis)
    print(describe(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
