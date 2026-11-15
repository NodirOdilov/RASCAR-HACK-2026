"""E3 = E2X-lite: the E2X model (:mod:`rascar.emulation2`, preset ``e2x_pixel_inner_hires``) without the two inputs
whose coefficients change sign between the labelled days.

E2X is a logistic regression on five inputs: ``tc_logit``, the consensus of the emulated label rule, ``p_tb_cut0``,
``f_cov_same`` and ``edge_rel``. Fitted on each labelled day alone, the standardised coefficients of ``f_cov_same``
(-0.02 / +0.53) and ``edge_rel`` (-0.04 / +0.29) change sign, those of the other three do not
(research/final_ascension/, results/final_ascension/a08_coefs.csv). ``edge_rel`` separates classes, not boxes inside a
class (round 5); ``f_cov_same`` was built for duplicates and the labelled days hold one duplicate error among 109.
E3 keeps [tc_logit, consensus, p_tb_cut0]. Proposal pools, border removal, pixel margin, kernel, consensus and
regularisation are exactly those of E2X; no existing module is changed.

Status (research/final_ascension/FINAL_ASCENSION_VERDICT.md): an OPTIONAL variant, not a proven improvement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import emulation as em
from . import emulation2 as em2
from .final_models import class_z

NAME = "e3_e2x_lite"
INPUTS = ("tc_logit", "cons_smooth", "p_tb_cut0")
REMOVED_INPUTS = ("f_cov_same", "edge_rel")
CFG = em2.Config(NAME, hires_classes=em2.E2X.hires_classes, drop_border=em2.E2X.drop_border, margin=em2.E2X.margin)


@dataclass
class EmulationModel3(em2.EmulationModel2):
    """E2X with the design matrix [tc_logit, consensus, p_tb_cut0]."""

    cfg: em2.Config = CFG

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        cons = np.mean([class_z(df, c, self.stats[c]) for c in em.SMOOTH_COLS], axis=0)
        return np.column_stack([df["tc_logit"].to_numpy(float), cons, df["p_tb_cut0"].to_numpy(float)])

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, folder / f"{self.name}.joblib")
        meta = {"name": self.name, "config": asdict(self.cfg), "adapter": self.adapter, "stats": self.stats,
                "thresholds": list(em.THRESHOLDS), "border_px": em2.BORDER_PX, "hires_folders": em2.HIRES_FOLDER,
                "opinion_caches": {k: list(v[0]) for k, v in em.OPINION_CACHES.items()},
                "inputs": list(INPUTS), "removed_inputs_of_e2x": list(REMOVED_INPUTS)}
        (folder / f"{self.name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, name: str = NAME) -> "EmulationModel3":
        folder = Path(folder)
        meta = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
        if list(meta.get("inputs", [])) != list(INPUTS):
            raise ValueError(f"{name} was not saved by EmulationModel3 (inputs {meta.get('inputs')})")
        c = meta["config"]
        cfg = em2.Config(c["name"], tuple(c["hires_classes"]), bool(c["drop_border"]), c["margin"])
        return cls(cfg=cfg, adapter=meta["adapter"], stats=meta["stats"], pipeline=joblib.load(folder / f"{name}.joblib"))


def leave_one_day_out(train: pd.DataFrame, manual: pd.DataFrame, store: em2.ProposalStore2, cfg: em2.Config = CFG) -> np.ndarray:
    """Out-of-day predictions: margin model, class statistics, scaler and regression all come from the other day."""
    oof = np.full(len(train), np.nan)
    for day in sorted(train["utc_day"].unique()):
        held = (train["utc_day"] == day).to_numpy()
        model = EmulationModel3(cfg).fit(train[~held].reset_index(drop=True), manual[manual["utc_day"] != day], store)
        oof[held] = model.predict(train[held].reset_index(drop=True), store)
    return oof
