"""Label-free member weights from training top-K overlaps (spectral weighting, SW).

Background: if ensemble members make their mistakes independently given the true
class, the off-diagonal part of the covariance matrix of their binary decisions is
rank one, and its leading eigenvector is proportional to each member's balanced
accuracy minus chance, i.e. its Youden index (Dawid and Skene 1979; Parisi et al.,
PNAS 2014). At a fixed alert budget every member flags exactly K rows, so

    n * Cov(top-K_i, top-K_j) = |T_i & T_j| - K^2 / n

and the Youden index equals TP@K * (1/n_anom + 1/n_norm) - K/n_norm, an increasing
linear function of TP@K. Under the independence assumption the eigenvector therefore
ranks the members by FP@K, without labels.

Rules this module keeps:
- it never receives labels or test rows (only training scores);
- ties at the K-th score are split in expectation (fractional top-K membership),
  so the row order never matters;
- K is the repository's K rule (metrics.budget_k over n_train), as for E2's training overlap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from thesis import metrics

KRule = Callable[[float, int], int]


def topk_membership(scores: np.ndarray, k: int) -> np.ndarray:
    """Fractional top-k membership with the expected-value tie rule.

    Rows strictly above the k-th largest score get 1, rows tied with it share the
    remaining slots equally, all other rows get 0. The result sums to k.
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("scores must be a non-empty 1-D array")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores contain NaN or inf")
    n = s.size
    if not (isinstance(k, (int, np.integer)) and 1 <= k <= n):
        raise ValueError(f"k must be an integer in [1, {n}], got {k!r}")
    kth = np.partition(s, n - k)[n - k]  # k-th largest score
    above = s > kth
    tied = s == kth
    a = above.astype(float)
    a[tied] = (k - int(above.sum())) / int(tied.sum())
    return a


def excess_overlap(
    train_scores: Mapping[str, np.ndarray], budget: float, k_rule: KRule = metrics.budget_k
) -> tuple[tuple[str, ...], np.ndarray, int, int]:
    """Expected top-K overlap beyond chance for every pair of members (training rows only).

    Returns (names, C, k, n) with C[i, j] = E|T_i & T_j| - k^2/n. The diagonal is not used.
    """
    names = tuple(train_scores)
    lengths = {np.asarray(train_scores[nm]).shape[0] for nm in names}
    if len(lengths) != 1:
        raise ValueError("all members must score the same training rows")
    n = lengths.pop()
    k = k_rule(budget, n)
    A = np.vstack([topk_membership(train_scores[nm], k) for nm in names])
    C = A @ A.T - (k * k) / n
    return names, C, k, n


def rank_one_offdiag(C: np.ndarray, iters: int = 500, tol: float = 1e-12) -> tuple[np.ndarray, float]:
    """Fit C[i, j] ~ v_i * v_j for i != j by iterative diagonal completion.

    Start with a zero diagonal; repeat: take the leading eigenpair (lam, u) of the
    symmetric matrix and replace its diagonal by lam * u**2. Returns (v, lam) with
    v = sqrt(lam) * u, signed so that sum(v) > 0.
    """
    M = np.array(C, dtype=float, copy=True)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("C must be a square matrix")
    M = (M + M.T) / 2.0
    d = np.zeros(M.shape[0])
    np.fill_diagonal(M, d)
    lam, u = 0.0, np.zeros(M.shape[0])
    for _ in range(iters):
        w, U = np.linalg.eigh(M)
        lam, u = float(w[-1]), U[:, -1]
        d_new = lam * u**2
        np.fill_diagonal(M, d_new)
        if np.max(np.abs(d_new - d)) <= tol * max(1.0, float(np.max(np.abs(d_new)))):
            break
        d = d_new
    v = math.sqrt(max(lam, 0.0)) * u
    if v.sum() < 0:
        v = -v
    return v, lam


@dataclass(frozen=True)
class SpectralResult:
    names: tuple[str, ...]
    weights: np.ndarray  # non-negative, sums to 1 (equal weights on fallback)
    v: np.ndarray  # spectral estimate, proportional to the Youden index under independence
    eigenvalue: float
    fallback: bool
    reason: str  # empty unless fallback
    k: int
    n: int

    @property
    def selected(self) -> str | None:
        """Member with the largest estimate (label-free selection); None if not finite."""
        if not np.all(np.isfinite(self.v)):
            return None
        return self.names[int(np.argmax(self.v))]


def spectral_weights(
    train_scores: Mapping[str, np.ndarray], budget: float, k_rule: KRule = metrics.budget_k
) -> SpectralResult:
    """Label-free weights for at least three members, from their training scores only.

    Weights are max(v, 0) normalised to sum 1. Falls back to equal weights if the
    leading eigenvalue is not positive, any value is not finite, or fewer than two
    members get a positive weight.
    """
    if len(train_scores) < 3:
        raise ValueError("spectral weights need at least three members (two cannot be separated)")
    names, C, k, n = excess_overlap(train_scores, budget, k_rule)
    v, lam = rank_one_offdiag(C)
    m = len(names)
    pos = np.clip(v, 0.0, None)
    offdiag = C[~np.eye(m, dtype=bool)]
    lam_floor = 1e-9 * max(1.0, float(np.max(np.abs(offdiag))))  # below this, lam is rounding noise
    reason = ""
    if not (np.all(np.isfinite(v)) and math.isfinite(lam)):
        reason = "non-finite estimate"
    elif lam <= lam_floor:
        reason = "leading eigenvalue <= 0"
    elif int((pos > 0).sum()) < 2:
        reason = "fewer than two positive weights"
    if reason:
        w = np.full(m, 1.0 / m)
    else:
        w = pos / pos.sum()
    return SpectralResult(names, w, v, lam, bool(reason), reason, k, n)


def weighted_fusion(calibrated: Mapping[str, np.ndarray], result: SpectralResult) -> np.ndarray:
    """Weighted mean of already calibrated scores (e.g. min-max fitted on training scores)."""
    missing = set(result.names) - set(calibrated)
    if missing:
        raise ValueError(f"calibrated scores missing for {sorted(missing)}")
    return np.sum([w * np.asarray(calibrated[nm], dtype=float) for nm, w in zip(result.names, result.weights)], axis=0)
