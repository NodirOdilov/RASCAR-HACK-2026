# Model comparison — final package (2026-11-14)

> **Repository note (2026-11-15).** This file is unchanged from 14 November 2026, i.e. written **before** E4 was
> uploaded to Kaggle (hence "E2 / E4: not uploaded" below). The team later uploaded E4 and E2; Kaggle Public AP:
> **E4 0.95778**, E2 0.93571. E4 is the team's final model. Current summary: [`README.md`](../README.md), section 11.
> Paths under `research/`, `results/` and `final_hackathon_2026/` refer to the team's working copy and are not part of
> this repository.

Written for the team captain, the jury's technical reviewers and the auditors. One protocol for every model: strict
leave-one-UTC-day-out on the 198 labelled rows (everything fitted comes from the other day). Kaggle Public numbers are
the captain's reports and were not used to fit, weight or choose anything; Private is unknown. Details and sources:
`research/final_ascension/MODEL_COMPARISON.md`, `research/final_mission/HYPOTHESIS_REGISTRY.md`.

## 1. The models

| model | what it is | file | Kaggle Public |
| --- | --- | --- | ---: |
| baseline | sort by 1 − detector confidence | — | not uploaded |
| 03 | consensus of four public detectors (score × IoU), logistic regression on five inputs | `submissions/EVOLYUTSIYA_CASE1_RANK1_CANDIDATE.csv` | 0.89976 |
| 05 | an earlier variant of 03 (`r4a_clsstd_all`) | `submissions/EVOLYUTSIYA_CASE1_ROBUST_FALLBACK.csv` | 0.89326 |
| E1 | 03 with proposals moved to the annotators' convention and the label rule as kernel | `submissions/EVOLYUTSIYA_CASE1_E1_CANDIDATE.csv` | 0.92964 |
| E2 | E1 + finer views for hands and cups | `submissions/EVOLYUTSIYA_CASE1_E2_CANDIDATE.csv` | not uploaded |
| **E2X** | E2 + proposals cut by their view removed + class-free pixel margin | `submissions/EVOLYUTSIYA_CASE1_E2X_OPTIONAL.csv` | **0.93284** |
| E3 | E2X without the inputs `f_cov_same` and `edge_rel` | `submissions/EVOLYUTSIYA_CASE1_E3_OPTIONAL.csv` | 0.89516 (vetoed) |
| E4 | E2X with its probabilities conditioned on the one-to-one rule of the label (optional; improvement not demonstrated) | `submissions/EVOLYUTSIYA_CASE1_E4_OPTIONAL.csv` | not uploaded |

## 2. Labelled days

| model | 25 Oct | 26 Oct | mean | mis-ordered pairs of 5281 | leave-one-frame-out vs E2X | grouped 5-fold vs E2X |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| baseline | 0.8607 | 0.7648 | 0.8128 | — | — | — |
| 03 | 0.9775 | 0.9316 | 0.9545 | 453 | — | — |
| E1 | 0.9846 | 0.9462 | 0.9654 | 348 | — | −0.0027 (lower in 40 of 40 repetitions; red team Q) |
| E2 | 0.9854 | 0.9479 | 0.9666 | 339 | — | — |
| **E2X** | 0.9866 | 0.9538 | 0.9702 | 311 | reference | reference |
| E3 | 0.9915 | 0.9547 | 0.9731 | 299 | +0.0001 mean of days, −0.0024 pooled | −0.0005 (200 repetitions; red team Q) |
| E4 | 0.9866 | 0.9619 | 0.9742 | — | +0.0034 mean of days, +0.0043 pooled | +0.0035, higher in 100 of 100 repetitions (stability on the same rows, not significance; red team Z: 120 of 120, 25 Oct −0.0002, 26 Oct +0.0072) |

Review queue on the labelled days (strict out-of-day scores; `results/final_mission/m04_product_stats.csv`):

| | baseline | 03 | E1 | E2X |
| --- | ---: | ---: | ---: | ---: |
| top 30 % of the queue: share of errors among the reviewed boxes | 0.81 | 1.00 | 1.00 | 1.00 |
| top 50 %: share of errors among the reviewed boxes | 0.76 | 0.87 | 0.90 | 0.91 |
| top 50 %: share of the day's errors found | 0.69 | 0.79 | 0.82 | 0.83 |
| share of the queue needed to see 90 % of the errors | 0.69 | 0.55 | 0.55 | 0.52 |

## 3. What the comparisons say

1. **03 → E1** is the one large, externally confirmed step: +0.011 locally, +0.030 on Public. Mechanism: the public
   detectors box objects tightly, annotators leave a margin; the comparison works only after the proposals are moved
   to the annotators' convention and the label rule itself is used as the kernel.
2. **E1 → E2X** is small and consistent: +0.0048 locally, +0.0027 in grouped folds, +0.0032 on Public.
3. **E3 is the cautionary case.** Locally +0.0029 (best of eight input subsets, 78 % from one row, −0.0005 in grouped
   folds); on Public −0.038. The veto fixed before the upload removed it. A local gain of this size on 198 rows is not
   evidence.
4. **E4** is the same kind of small local gain (+0.0040). It is positive in every protocol (100 of 100 grouped-fold
   repetitions — stability under refitting on the same rows, not significance) and it comes from the label's own
   one-to-one rule with no quantity fitted by optimisation (the threshold 0.5 was chosen after the pair labels of the
   labelled days were seen). It is NOT a demonstrated improvement: it is the third variant, defined after two
   registered variants failed (post hoc; eight variants of the idea were scored that day); one group of four trays
   (frame 26 Oct 11:43:00) carries 73 % of the gain and two groups 94 %; without the tray groups +0.0012 / +0.0009 /
   +0.0009 remain; trays form no group on the test; the refit bootstrap interval is −0.0005 … +0.0147; the idea was
   prompted by the Public result of E3. Ruling of the independent audit (`INDEPENDENT_AUDIT.md`): rejected as
   `FINAL_BREAKTHROUGH`, approved as an optional second selection with the veto "Public AP below 0.92964 → the second
   slot stays E1".

## 4. Rejected in the last two missions (measured)

| idea | result | decision |
| --- | --- | --- |
| hard one-to-one rule (lift the second box of a pair to the top) | strict +0.0068, leave-one-frame-out pooled −0.0010; a correct box lifted in 55 of 100 grouped-fold repetitions (81 of 120 and 133 of 200 with the red team's seeds) | rejected |
| E3 (drop two inputs) | Public 0.89516 | rejected by veto |
| score^0.5 in the kernel | 0.9657 | rejected |
| table-conditional extension of person proposals, COCO / pose opinions, fusion of proposals | failed their label-free stage | rejected without scoring |
| rank blends E2X + 03, E2X + E1 | −0.0004 … −0.0073 | rejected |
| public full-body person detector (A3) | not executed: download refused by the permission system | open |
