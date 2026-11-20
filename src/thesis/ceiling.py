"""Label-tuned weights: an upper reference for any weighting of the members.

Uses training labels, so it is a diagnostic, never a method. It answers one question:
if labels were available, would weighting the members beat picking the best one?
If not, no label-free weighting can be expected to.
"""
from __future__ import annotations

import itertools
import math
from typing import Callable

import numpy as np

from thesis import metrics

KRule = Callable[[float, int], int]


def fp_at_k(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    """False positives among the top k: k - metrics.tp_at_k (ties at the k-th score split
    in expectation, same validation)."""
    return k - metrics.tp_at_k(scores, y, k)


def simplex_grid(m: int, step: float) -> np.ndarray:
    """Every weight vector of length m whose entries are multiples of step and sum to 1."""
    units = round(1.0 / step)
    if not math.isclose(units * step, 1.0, rel_tol=0, abs_tol=1e-9) or m < 1:
        raise ValueError("1/step must be an integer and m >= 1")
    rows = []
    for bars in itertools.combinations(range(units + m - 1), m - 1):
        parts, prev = [], -1
        for b in bars:
            parts.append(b - prev - 1)
            prev = b
        parts.append(units + m - 2 - prev)
        rows.append(parts)
    return np.asarray(rows, dtype=float) / units


def tune_weights(cal_train: np.ndarray, y_train: np.ndarray, k: int, grid: np.ndarray, chunk: int = 200) -> np.ndarray:
    """Grid weight vector with the fewest training false alarms at k.

    cal_train: members x rows, calibrated training scores. Ties between weight vectors
    go to the one with the largest single weight (closest to picking one member), then
    to grid order, so weighting only wins where it is strictly better on training data.
    """
    X = np.asarray(cal_train, dtype=float)
    if X.ndim != 2 or X.shape[0] != grid.shape[1]:
        raise ValueError("cal_train must be members x rows, matching the grid width")
    fps = np.empty(len(grid))
    for start in range(0, len(grid), chunk):
        fused = grid[start : start + chunk] @ X
        fps[start : start + chunk] = [fp_at_k(f, y_train, k) for f in fused]
    best = np.flatnonzero(np.isclose(fps, fps.min(), rtol=0, atol=1e-9))
    return grid[best[np.argmax(grid[best].max(axis=1))]]


def ceiling_run(
    cal_train: np.ndarray,
    y_train: np.ndarray,
    cal_test: np.ndarray,
    y_test: np.ndarray,
    budget: float,
    step: float,
    k_rule: KRule = metrics.budget_k,
) -> dict[str, float | np.ndarray]:
    """Test false alarms of label-tuned weights against label-selected and oracle members.

    Returns test FP@K for: tuned weights (training labels), the member picked by training
    FP@K (training labels), equal weights, and the oracle-best and median member (test labels).
    """
    m = cal_train.shape[0]
    k_tr = k_rule(budget, cal_train.shape[1])
    k_te = k_rule(budget, cal_test.shape[1])
    w = tune_weights(cal_train, y_train, k_tr, simplex_grid(m, step))
    train_member_fp = np.array([fp_at_k(cal_train[i], y_train, k_tr) for i in range(m)])
    picked = int(np.argmin(train_member_fp))  # ties: first in member order
    test_member_fp = np.array([fp_at_k(cal_test[i], y_test, k_te) for i in range(m)])
    return {
        "weights": w,
        "label_selected": picked,
        "fp_tuned_weights": fp_at_k(w @ cal_test, y_test, k_te),
        "fp_label_selected_member": float(test_member_fp[picked]),
        "fp_equal_weights": fp_at_k(cal_test.mean(axis=0), y_test, k_te),
        "fp_oracle_best_member": float(test_member_fp.min()),
        "fp_median_member": float(np.median(test_member_fp)),
        "k_test": k_te,
    }
