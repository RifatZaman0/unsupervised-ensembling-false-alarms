"""Committee cleaning: label-free choice of training rows to drop before refitting detectors.

The ensemble acts as a committee on the training split: rows that the members jointly score
as most anomalous are removed, and every member is refitted on the remaining rows. This
module only decides which rows to keep. It never receives labels or test rows.

Removal rule (order-free): with k = metrics.budget_k(q, n) (the repository's K rule), the k
highest-scored rows are removed. If rows tied with the k-th largest score would push the count
above k, the whole tied block is kept instead, so at most k rows are removed and the row order
never matters.
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from thesis import metrics

KRule = Callable[[float, int], int]


def _check(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("scores must be a non-empty 1-D array")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores contain NaN or inf")
    return s


def removal_mask(scores: np.ndarray, q: float, k_rule: KRule = metrics.budget_k) -> np.ndarray:
    """True for the k highest-scored rows, k = k_rule(q, n); a tie crossing the cut is kept."""
    s = _check(scores)
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0, 1)")
    n = s.size
    k = k_rule(q, n)
    if not (1 <= k <= n):
        raise ValueError(f"k out of range: {k}")
    kth = np.partition(s, n - k)[n - k]
    at_or_above = s >= kth
    if int(at_or_above.sum()) == k:  # no tie crosses the cut: remove exactly k rows
        return at_or_above
    return s > kth  # a tie crosses the cut: keep the whole tied block


def minmax_train(scores: np.ndarray) -> np.ndarray:
    """Min-max scaling fitted on the same training scores; a constant member maps to 0."""
    s = _check(scores)
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def consensus_scores(train_scores: Mapping[str, np.ndarray]) -> np.ndarray:
    """Committee score on training rows: min-max per member, then the mean (primary fusion)."""
    if len(train_scores) < 2:
        raise ValueError("a committee needs at least two members")
    cols = [minmax_train(train_scores[nm]) for nm in train_scores]
    if len({c.size for c in cols}) != 1:
        raise ValueError("all members must score the same training rows")
    return np.mean(cols, axis=0)


def committee_keep_mask(
    train_scores: Mapping[str, np.ndarray], q: float, k_rule: KRule = metrics.budget_k
) -> np.ndarray:
    """Rows every member is refitted on under committee cleaning (True = keep)."""
    return ~removal_mask(consensus_scores(train_scores), q, k_rule)


def self_keep_mask(train_score: np.ndarray, q: float, k_rule: KRule = metrics.budget_k) -> np.ndarray:
    """Control: a member drops only the rows it scores highest itself (True = keep)."""
    return ~removal_mask(train_score, q, k_rule)
