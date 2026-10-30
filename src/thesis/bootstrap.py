"""Paired bootstrap for the fraud-data comparisons (frozen_config.json `statistics`, `hypotheses`).

Nothing is refitted. Every method's test scores (budget mode) or alerts (threshold mode) are
fixed; a resample only decides which rows are evaluated. Per resample:

  a. one set of row indices is drawn, the same for every method and every IForest seed;
  b. K is recomputed from the resampled window size (K = ceil(budget * n), metrics.budget_k);
  c. FP@K is computed from the fixed scores for every method and seed (rows drawn twice are
     exact ties, which the expected-value tie rule of metrics.tp_at_k handles);
  d. seed means are taken;
  e. the median single (median of the six singles) and the oracle (their minimum) are
     recomputed inside the resample;
  f. every difference is computed.

A resample is a list of row-index arrays, one per evaluation window: ULB has one (day 2),
IEEE-CIS has five (months 2-6). The budget statistic is the sum over windows of FP@K, each
window with its own K. RNG: numpy.random.default_rng(seed), one stream, windows drawn in order.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from thesis import metrics

RESAMPLES = 1000
SEED = 0
CI_LEVEL = 0.95
H3B_BLOCK_WEEKS = 4
MEDIAN, ORACLE, DEFAULT_IFOREST = "median_single", "oracle_best_single", "default_iforest"

Windows = list[np.ndarray]
Statistic = Callable[[Windows], dict]


# ---------------------------------------------------------------- resamples

def transaction_windows(n: int) -> Windows:
    """The full ULB test day as one window (the point estimate)."""
    return [np.arange(n)]


def draw_transactions(rng: np.random.Generator, n: int) -> Windows:
    """ULB: n test rows drawn with replacement (same size, so K does not change)."""
    return [rng.integers(0, n, n)]


def day_rows(day: np.ndarray, window: np.ndarray, windows: Sequence[int]) -> list[list[np.ndarray]]:
    """Row indices per day, grouped by window (IEEE-CIS: the test months, in order)."""
    out = []
    for w in windows:
        rows = np.flatnonzero(window == w)
        if rows.size == 0:
            raise ValueError(f"window {w} has no rows")
        d = day[rows]
        out.append([rows[d == v] for v in np.unique(d)])
    return out


def day_windows(days: list[list[np.ndarray]]) -> Windows:
    """Every window with its own days once (the point estimate)."""
    return [np.concatenate(month) for month in days]


def draw_days(rng: np.random.Generator, days: list[list[np.ndarray]]) -> Windows:
    """Month-stratified day-block resample: within each window, as many days as it has, drawn
    with replacement; a drawn day brings all its rows. Every window stays in every resample."""
    out = []
    for month in days:
        pick = rng.integers(0, len(month), len(month))
        out.append(np.concatenate([month[i] for i in pick]))
    return out


def moving_blocks(rng: np.random.Generator, n: int, block: int = H3B_BLOCK_WEEKS) -> np.ndarray:
    """Moving-block bootstrap of n ordered windows: ceil(n / block) blocks of `block`
    consecutive positions, start drawn uniformly, joined and cut to length n."""
    if not 1 <= block <= n:
        raise ValueError(f"block {block} outside 1..{n}")
    starts = rng.integers(0, n - block + 1, math.ceil(n / block))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


# ---------------------------------------------------------------- statistics

def seed_mean_fp(scores: np.ndarray, y: np.ndarray, budget: float, windows: Windows) -> tuple[float, int]:
    """Sum over windows of FP@K, mean over seeds (rows of `scores`), and the sum of the K."""
    scores = np.atleast_2d(scores)
    fp, k_sum = np.zeros(len(scores)), 0
    for idx in windows:
        k = metrics.budget_k(budget, len(idx))
        k_sum += k
        yy = y[idx]
        fp += [k - metrics.tp_at_k(s[idx], yy, k) for s in scores]
    return float(fp.mean()), k_sum


def comparator(name: str) -> str:
    return "IForest" if name == DEFAULT_IFOREST else name


def budget_statistic(scores: Mapping[str, np.ndarray], y: np.ndarray, budget: float,
                     singles: Sequence[str], comparisons: Sequence[tuple[str, str]]) -> Statistic:
    """FP@K statistic of one budget: `fp:<method>` for every method in `scores` (seed means,
    summed over windows), `fp:median_single`, `fp:oracle_best_single`, `k_sum`, and for each
    (a, b) in `comparisons` the difference `d:a-b` = fp(a) - fp(b) and `d100:a-b` = the same
    per 100 alerts (difference / k_sum * 100). b may be a method, median_single,
    oracle_best_single or default_iforest (= IForest, seed mean)."""
    if len(singles) < 2 or not set(singles) <= set(scores):
        raise ValueError(f"singles {singles} must all have scores")

    def stat(windows: Windows) -> dict:
        out, k_sum = {}, None
        for name, s in scores.items():
            out[f"fp:{name}"], k = seed_mean_fp(s, y, budget, windows)
            k_sum = k if k_sum is None else k_sum
        single_fp = np.array([out[f"fp:{n}"] for n in singles])
        out[f"fp:{MEDIAN}"] = float(np.median(single_fp))
        out[f"fp:{ORACLE}"] = float(single_fp.min())
        out["k_sum"] = k_sum
        for a, b in comparisons:
            d = out[f"fp:{a}"] - out[f"fp:{comparator(b)}"]
            out[f"d:{a}-{b}"], out[f"d100:{a}-{b}"] = d, d / k_sum * 100
        return out
    return stat


def alert_rate_statistic(alerts: Mapping[str, np.ndarray], q: float, singles: Sequence[str],
                         ensemble: str) -> Statistic:
    """H3a: per method, the mean over windows of |alert rate - (1 - q)| (alerts fixed by the
    frozen training-quantile threshold; seed mean), the median of the singles' values, and
    `d:<ensemble>-median_single`."""
    def stat(windows: Windows) -> dict:
        out = {}
        for name, a in alerts.items():
            a = np.atleast_2d(a)
            dev = [np.mean([abs(s[idx].mean() - (1 - q)) for idx in windows]) for s in a]
            out[f"dev:{name}"] = float(np.mean(dev))
        out[f"dev:{MEDIAN}"] = float(np.median([out[f"dev:{n}"] for n in singles]))
        out[f"d:{ensemble}-{MEDIAN}"] = out[f"dev:{ensemble}"] - out[f"dev:{MEDIAN}"]
        return out
    return stat


# ---------------------------------------------------------------- bootstrap and CI

def run(draw: Callable[[np.random.Generator], Windows], statistic: Statistic,
        resamples: int = RESAMPLES, seed: int = SEED) -> pd.DataFrame:
    """One RNG stream; resample i is the i-th draw, shared by every method in `statistic`."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame([statistic(draw(rng)) for _ in range(resamples)])


def percentile_ci(values, level: float = CI_LEVEL) -> tuple[float, float]:
    tail = round((1 - level) / 2 * 100, 9)    # 2.5, not 2.5000000000000004
    lo, hi = np.percentile(np.asarray(values, dtype=np.float64), [tail, 100 - tail])
    return float(lo), float(hi)


def fewer_false_alarms(ci_high: float) -> bool:
    """The frozen criterion: supported only if the whole CI of the difference is below 0."""
    return ci_high < 0


def summary(point: dict, boot: pd.DataFrame, level: float = CI_LEVEL) -> pd.DataFrame:
    """Point estimate and percentile CI for every statistic; `supported` for differences."""
    rows = []
    for key, est in point.items():
        lo, hi = percentile_ci(boot[key], level)
        rows.append({"stat": key, "estimate": est, "ci_low": lo, "ci_high": hi,
                     "supported": fewer_false_alarms(hi) if key.startswith("d:") else None})
    return pd.DataFrame(rows)


def spearman_moving_block(x: np.ndarray, y: np.ndarray, block: int = H3B_BLOCK_WEEKS,
                          resamples: int = RESAMPLES, seed: int = SEED,
                          level: float = CI_LEVEL) -> dict:
    """H3b: Spearman rho across ordered windows with a moving-block bootstrap CI (an
    association, no decision). Resamples where rho is undefined (a constant resample) are
    dropped and counted."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError("x and y need one value per window")
    rng = np.random.default_rng(seed)
    rhos = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # a constant resample gives nan, counted
        for _ in range(resamples):
            i = moving_blocks(rng, len(x), block)
            rhos.append(spearmanr(x[i], y[i]).statistic)
    rhos = np.asarray(rhos, dtype=np.float64)
    ok = rhos[~np.isnan(rhos)]
    lo, hi = percentile_ci(ok, level) if ok.size else (math.nan, math.nan)
    return {"rho": float(spearmanr(x, y).statistic), "ci_low": lo, "ci_high": hi,
            "n_windows": len(x), "n_undefined_resamples": int(rhos.size - ok.size)}
