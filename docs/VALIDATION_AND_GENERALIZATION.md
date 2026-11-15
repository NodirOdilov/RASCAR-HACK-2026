# Validation and generalization — final package (2026-11-14)

> **Repository note (2026-11-15).** This file is unchanged from 14 November 2026, i.e. written **before** E4 was
> uploaded to Kaggle. The team later uploaded E4 and E2; Kaggle Public AP: **E4 0.95778**, E2 0.93571 (Private is
> still unknown). Current summary: [`README.md`](../README.md), section 11. Paths under `research/`, `results/` and
> `final_hackathon_2026/` refer to the team's working copy and are not part of this repository.

Written for the team captain, the jury's technical reviewers and the auditors. What the evidence in this project can
and cannot say about the hidden Private part. No expected Private score is given: it cannot be derived.
Details: `research/final_ascension/GENERALIZATION_REVIEW.md`, `DISTRIBUTION_SHIFT_REPORT.md`,
`research/final_mission/HYPOTHESIS_REGISTRY.md`.

## 1. Kinds of statements in this project

| kind | example | status |
| --- | --- | --- |
| confirmed measurement | the label rule reproduces 670 of 670 training labels; E2X regenerates byte-identically | checked by independent code |
| statistical estimate | E2X − E1 = +0.0048 by days, +0.0027 in grouped folds | 198 rows, two days: resolves about 0.02 on its own |
| research hypothesis | the test holds fewer errors than the labelled days (about 49 % against 55 %) | label-free extrapolation, not measured |
| Public observation | E2X 0.93284, E1 0.92964, 03 0.89976, E3 0.89516 | captain's reports; one number per file |
| unknown | every Private score; the scored subset; the Public / Private split | not stated anywhere in this project |

## 2. How validation is done

- **Strict day split** (recommended by TASK.txt): train on one labelled day, score the other; the annotation margin,
  the class statistics and the regression all come from the training day. Headline = mean of the two day APs
  (Kaggle computes one pooled AP per part — both are reported where they differ).
- **Robustness:** frame-grouped 5-fold with many repetitions, leave-one-frame-out, paired frame bootstrap, class-shift
  scenarios, nested selection, placebo controls.
- **Order of work:** mechanism, test and rejection condition are written before scoring; where possible an idea must
  first pass a stage that reads manual boxes only, not `error`.
- **Public scores** are used only as vetoes fixed before an upload. No weight, threshold or input was fitted to them.
- **Independent recomputation:** four internal audits (agents J, K, Q, Z) with their own code reproduced the models
  with a maximum difference of 0.0 and found no leakage.

## 3. What the labelled days cannot settle

- More than 1 000 variants have been scored on the same 198 rows (count from the team's logs; not recomputed by an
  audit). The best of 50 useless inputs gains +0.006 on
  average on these rows. A single gain below about 0.01 can be selection.
- **Demonstrated on 14 Nov:** E3 gained +0.0029 locally and lost 0.038 on Public. The removed inputs looked
  unnecessary on two days and were needed on the test days.
- Ten repetitions of grouped folds can have the wrong sign (E3: +0.0001 with one seed, −0.0005 over 200
  repetitions). Results of this kind are now reported with at least 100 repetitions.
- **E4 (14 Nov) shows the other limit:** +0.0040 by days and higher than E2X in 100 of 100 grouped-fold repetitions —
  and still not demonstrated. One group of four trays in one frame carries 73 % of the gain (two groups 94 %), the
  refit bootstrap interval includes zero (−0.0005 … +0.0147), and it is the third of eight variants scored that day.
  "Higher in every repetition" measures stability on the same rows, not significance.

## 4. Labelled days against the test (label-free)

| | labelled days | test |
| --- | ---: | ---: |
| rows confirmed by a confident public proposal | 62 % | 72 % |
| share of errors | 55 % (measured) | about 49 % (extrapolated from the confirmed / unconfirmed cells) |
| same-class pairs of audited boxes with IoU ≥ 0.5 | 19 (6 hands, 6 trays, 5 persons, 2 cups) | 26 (17 hands, 8 cups, 1 person) |
| target frames without a previous / next context frame | 1 / 0 of 84 | 8 / 10 of 128 |

Average precision falls with the share of errors at equal ranking quality, so part of the distance between the local
0.970 and the Public 0.933 is a base-rate effect; how much is unknown. The class positions in the queue are the least
known part of the test (E3 changed mainly those).

## 5. What is well supported, and what is not

Supported: the mechanism of E1 / E2X (annotation margin + label rule as kernel) — measured without labels, positive
on both days and in grouped folds, visible for each of the four detectors, followed by +0.030 and +0.003 on Public;
the ranking of "no object" errors (49 rows cost 0.002 in total); reproducibility of the pipelines.

Not supported: any ordering of E2X, E1 and E4 on Private; any gain smaller than about 0.01 as a fact; behaviour on
another camera, other annotators or new classes; a measured saving of operator time.

## 6. Consequences for the final selection

- E2X is selected in every case: it is best locally and on Public.
- The second file is chosen by rules written before the relevant Public score is known (`FINAL_SUBMISSION_MANIFEST.md`,
  `KAGGLE_SELECTION_INSTRUCTIONS_UZ.md`).
- E4 may replace E1 as the second file only after ONE upload with Public AP ≥ 0.92964 (veto fixed before the upload);
  otherwise the pair is E2X + E1. Whether Kaggle counts the better of two selected files is assumed, not verified.
- E2X, E1 and E4 are one family: any two of them are one bet with a small hedge.
