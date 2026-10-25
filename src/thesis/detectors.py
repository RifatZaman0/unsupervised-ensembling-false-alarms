"""Detector pool and `score()` (Master Plan § Detectors, calibration and fusion).

`score(name, X_fit, X_score, seed)` fits one detector on X_fit (no labels) and returns
(training scores, scores for X_score); higher = more anomalous for every detector.
Training scores are the ones calibrators and thresholds are fitted on, so they must be
computed the way new rows would be: KNN and LOF use PyOD's leave-one-out training scores
(a row is not its own neighbour), ECOD uses the train-only version below.

Every detector scores new rows independently of each other (no batch effects); see
tests/test_detectors.py. PyOD's own ECOD is batch-dependent and is not used for scoring.
"""
from __future__ import annotations

import numpy as np
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from scipy.stats import skew

from thesis import config

DETECTORS = ("IForest", "PCA", "ECOD", "HBOS", "KNN", "LOF")
SEEDED = ("IForest",)  # detectors whose result depends on the seed
# KNN / LOF are fitted on at most this many training rows (Master Plan: 30,000-row sample on
# fraud data). ADBench training splits are all smaller (largest: 29,720 rows), so it never applies there.
MAX_FIT_ROWS = 30_000
# Library defaults, recorded so the hyperparameter table and the cache key come from one place.
PARAMS = {
    "IForest": {"n_estimators": 100, "max_samples": "auto"},
    "PCA": {"standardization": True, "min_variance_ratio": 1e-10},
    "ECOD": {},
    "HBOS": {"n_bins": 10, "alpha": 0.1, "tol": 0.5},
    "KNN": {"n_neighbors": 5, "method": "largest"},
    "LOF": {"n_neighbors": 20, "novelty": True},
}


class TrainOnlyECOD:
    """ECOD (Li et al. 2022) whose tail probabilities come from the training rows only.

    PyOD's ECOD.decision_function appends the new rows to the training rows before building
    the ECDFs and the skewness, so a row's score depends on which other rows are scored with it.
    Here the ECDFs and the skewness signs are fixed at fit time; a new row x gets
    F_left(x) = #{train <= x} / n and F_right(x) = #{train >= x} / n, floored at 1/n (the smallest
    value a training row can have) so rows beyond the training range stay finite.
    On the training rows this equals PyOD's fit-time scores (up to float rounding).
    """

    def fit(self, X: np.ndarray) -> "TrainOnlyECOD":
        self.sorted_ = np.sort(X, axis=0)
        self.n_ = len(X)
        # a constant column has skewness NaN; PyOD maps it to 0 as well
        self.skew_sign_ = np.sign(np.nan_to_num(skew(X, axis=0), nan=0.0))
        self.decision_scores_ = self.decision_function(X)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        n, d = self.n_, self.sorted_.shape[1]
        le = np.empty(X.shape)
        ge = np.empty(X.shape)
        for j in range(d):
            col = self.sorted_[:, j]
            le[:, j] = np.searchsorted(col, X[:, j], side="right")
            ge[:, j] = n - np.searchsorted(col, X[:, j], side="left")
        u_l = -np.log(np.maximum(le, 1) / n)
        u_r = -np.log(np.maximum(ge, 1) / n)
        s = self.skew_sign_
        u_skew = u_l * -np.sign(s - 1) + u_r * np.sign(s + 1)
        return np.maximum(u_skew, np.maximum(u_l, u_r)).sum(axis=1)


class NonFiniteScores(ValueError):
    """A detector returned inf or NaN scores."""


def params() -> dict:
    """PARAMS under the current protocol; min_variance_ratio None = no PCA component drop."""
    if config.PROTOCOL.pca_drop_zero_variance:
        return PARAMS
    return {**PARAMS, "PCA": {**PARAMS["PCA"], "min_variance_ratio": None}}


class RankSafePCA:
    """PyOD PCA with library defaults, minus principal directions of (numerically) zero variance.

    PyOD's weighted PCA divides each distance by the component's explained-variance ratio, so a
    zero-variance direction (from constant or collinear features, e.g. in cardio) makes every
    score infinite. Components whose ratio is <= min_variance_ratio are dropped by refitting with
    n_components = the number of remaining ones. Deviation from defaults; reported in the
    hyperparameter table. min_variance_ratio=None keeps every component (plain PyOD PCA).
    """

    def __init__(self, standardization: bool = True, min_variance_ratio: float | None = 1e-10):
        self.standardization, self.min_variance_ratio = standardization, min_variance_ratio

    def fit(self, X: np.ndarray) -> "RankSafePCA":
        det = PCA(standardization=self.standardization).fit(X)
        if self.min_variance_ratio is None:
            self.n_dropped_ = 0
            self.detector_, self.decision_scores_ = det, det.decision_scores_
            return self
        keep = int((det.detector_.explained_variance_ratio_ > self.min_variance_ratio).sum())
        self.n_dropped_ = det.n_components_ - keep
        if self.n_dropped_:
            det = PCA(n_components=keep, standardization=self.standardization).fit(X)
        self.detector_ = det
        self.decision_scores_ = det.decision_scores_
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.detector_.decision_function(X)


def make_detector(name: str, seed: int | None = None):
    p = params()[name]
    if name == "IForest":
        return IForest(**p, random_state=seed)
    if name == "PCA":
        return RankSafePCA(**p)
    if name == "ECOD":
        return TrainOnlyECOD()
    if name == "HBOS":
        return HBOS(**p)
    if name == "KNN":
        return KNN(**p)
    if name == "LOF":
        return LOF(**p)
    raise ValueError(f"unknown detector {name!r}")


def score(name: str, X_fit: np.ndarray, X_score: np.ndarray,
          seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fit `name` on X_fit (unlabelled) and return (scores of X_fit, scores of X_score).

    For KNN / LOF on more than MAX_FIT_ROWS rows, the detector is fitted on a seeded sample;
    sampled rows keep their leave-one-out scores and the remaining training rows are scored
    like new rows, so every training row gets a score that did not see the row itself.
    """
    if name not in DETECTORS:
        raise ValueError(f"unknown detector {name!r}")
    if name in SEEDED and seed is None:
        raise ValueError(f"{name} needs a seed")
    det = make_detector(name, seed)
    if name in ("KNN", "LOF") and len(X_fit) > MAX_FIT_ROWS:
        rng = np.random.default_rng(0 if seed is None else seed)
        sample = np.sort(rng.choice(len(X_fit), MAX_FIT_ROWS, replace=False))
        det.fit(X_fit[sample])
        rest = np.setdiff1d(np.arange(len(X_fit)), sample)
        s_fit = np.empty(len(X_fit))
        s_fit[sample] = det.decision_scores_
        s_fit[rest] = det.decision_function(X_fit[rest])
    else:
        det.fit(X_fit)
        s_fit = np.asarray(det.decision_scores_, dtype=np.float64)
    s_score = np.asarray(det.decision_function(X_score), dtype=np.float64)
    if not (np.isfinite(s_fit).all() and np.isfinite(s_score).all()):
        raise NonFiniteScores(f"{name}: non-finite scores")
    return s_fit, s_score
