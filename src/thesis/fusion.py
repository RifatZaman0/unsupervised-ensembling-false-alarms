"""Score calibration and fusion: `fuse(S_train, S_test, calib, agg)` (Master Plan § Detectors,
calibration and fusion).

Each member's calibrator is fitted on that member's training scores only and then applied
unchanged to training and test scores. 3 calibrations x 2 aggregations = 6 fusion configs;
min-max + mean is the primary one, declared before any result.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

CALIBRATIONS = ("minmax", "zscore", "percentile")
AGGREGATIONS = ("mean", "max")
FUSION_CONFIGS = tuple((c, a) for c in CALIBRATIONS for a in AGGREGATIONS)
PRIMARY = ("minmax", "mean")
# reader-facing names (figures, LaTeX, markdown); the codes above stay in CSVs
LABELS = {"minmax": "min-max", "zscore": "z-score", "percentile": "train-percentile",
          "mean": "mean", "max": "max"}


@dataclass(frozen=True)
class Calibrator:
    """Maps one detector's raw scores to a common scale, from training scores only.

    minmax:     (s - min) / (max - min), not clipped, so drifted scores above 1 stay visible
    zscore:     (s - mean) / std
    percentile: share of training scores <= s, in [0, 1]
    A constant training score vector gets scale 1 so the output stays finite; under zscore it
    gets scale 0 instead, and every score maps to 0, so that member adds nothing to the fusion
    (provisional, DECISIONS.md 2026-10-25).
    """
    kind: str
    loc: float = 0.0
    scale: float = 1.0
    ref: np.ndarray | None = None  # sorted training scores, percentile only

    @classmethod
    def fit(cls, s_train: np.ndarray, kind: str) -> "Calibrator":
        s_train = np.asarray(s_train, dtype=np.float64)
        if kind == "minmax":
            lo, span = s_train.min(), np.ptp(s_train)
            return cls(kind, lo, span if span > 0 else 1.0)
        if kind == "zscore":
            # constant = range exactly 0; the float std of a constant vector can be ~1e-16
            # (e.g. 50 x 3.2), which would blow the member up instead of setting it to 0
            sd = s_train.std() if np.ptp(s_train) > 0 else 0.0
            return cls(kind, s_train.mean(), sd)
        if kind == "percentile":
            return cls(kind, ref=np.sort(s_train))
        raise ValueError(f"unknown calibration {kind!r}")

    def transform(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        if self.kind == "percentile":
            return np.searchsorted(self.ref, s, side="right") / len(self.ref)
        if self.scale == 0:  # zscore of a constant member
            return np.zeros_like(s)
        return (s - self.loc) / self.scale


def fuse(S_train: np.ndarray, S_test: np.ndarray, calib: str = PRIMARY[0],
         agg: str = PRIMARY[1], members: Sequence[str] | None = None,
         dataset: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fuse member scores (columns = members) into one score per row.

    Returns (fused training scores, fused test scores). Training scores are fused too because
    threshold mode takes its quantile thresholds from them. `members` (detector names, one per
    column) and `dataset` only label the warning for a zero-variance member under zscore.
    """
    S_train, S_test = np.atleast_2d(S_train), np.atleast_2d(S_test)
    if S_train.shape[1] != S_test.shape[1]:
        raise ValueError("train and test have different numbers of members")
    if agg not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {agg!r}")
    cals = [Calibrator.fit(S_train[:, j], calib) for j in range(S_train.shape[1])]
    for j, c in enumerate(cals):
        if c.kind == "zscore" and c.scale == 0:
            log.warning("zero-variance training scores under zscore: detector %s on dataset %s "
                        "set to 0", members[j] if members else f"#{j}", dataset or "?")
    C_train = np.column_stack([c.transform(S_train[:, j]) for j, c in enumerate(cals)])
    C_test = np.column_stack([c.transform(S_test[:, j]) for j, c in enumerate(cals)])
    reduce = np.mean if agg == "mean" else np.max
    return reduce(C_train, axis=1), reduce(C_test, axis=1)
