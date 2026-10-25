"""Evaluation metrics, computed on the test split only (Master Plan § Evaluation protocol).

Budget mode: each method flags its top K rows, K = ceil(budget * n_test).
    FP@K = K - TP@K,  Precision@K = TP@K / K,  Recall@K = TP@K / #anomalies.
Ties at the K-th score are split in expectation (random tie-breaking): rows strictly above
the cut count fully, rows tied with the K-th score count by the share of the remaining slots.
Row order therefore never decides an alert (ADBench files store anomalies in blocks, so
breaking ties by position would leak label information into the metric). TP@K and FP@K can
be fractional for that reason.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ADBENCH_BUDGETS = (0.02, 0.05)


def budget_k(budget: float, n: int) -> int:
    """Number of alerts for a budget given as a share of the rows (at least 1)."""
    return max(1, math.ceil(round(budget * n, 9)))


def tp_at_k(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    """Expected true positives among the top k scores, ties broken at random.

    Raises ValueError for non-1-D or mismatched inputs, NaN in scores or labels, and k that is
    not an integer in 1..n.
    """
    scores, y = np.asarray(scores, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if scores.ndim != 1 or y.ndim != 1 or len(scores) != len(y):
        raise ValueError(f"scores {scores.shape} and y {y.shape} must be 1-D of equal length")
    if np.isnan(scores).any():
        raise ValueError(f"scores contain {int(np.isnan(scores).sum())} NaN value(s)")
    if np.isnan(y).any():
        raise ValueError(f"y contains {int(np.isnan(y).sum())} NaN value(s)")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError(f"k must be an integer, got {k!r}")
    if not 1 <= k <= len(scores):
        raise ValueError(f"k={k} outside 1..{len(scores)}")
    kth = np.partition(scores, len(scores) - k)[len(scores) - k]
    above, tied = scores > kth, scores == kth
    n_above = int(above.sum())
    return float(y[above].sum() + (k - n_above) * y[tied].sum() / tied.sum())


def budget_metrics(scores: np.ndarray, y: np.ndarray, budget: float) -> dict:
    k = budget_k(budget, len(y))
    tp = tp_at_k(scores, y, k)
    n_pos = int(np.sum(y))
    return {"budget": budget, "k": k, "tp_at_k": tp, "fp_at_k": k - tp,
            "precision_at_k": tp / k, "recall_at_k": tp / n_pos if n_pos else float("nan")}


def ranking_metrics(scores: np.ndarray, y: np.ndarray) -> dict:
    """PR-AUC (average precision) and ROC-AUC, for comparison with other papers."""
    return {"pr_auc": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores))}
