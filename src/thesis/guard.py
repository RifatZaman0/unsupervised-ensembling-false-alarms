"""Run-time guard (RQ4, proposed for freeze-v3): leave out a member on a day where its alert
rate at the frozen training threshold is more than c times the target.

For member m on test day d, with q = 1 - budget and the threshold t_m = numpy.quantile(training
scores of m, q, method 'linear') (one per IForest seed):

    r_m,d = (alerts of m on day d) / (n_d * (1 - q)),   alert if score > t_m.

IForest: alerts summed over the seeds and n_d multiplied by the number of seeds (= the seed mean
of the alert rate). The member passes if r_m,d <= c (r = c passes). The comparison is exact: it
is made on integer alert counts with fractions.Fraction, so a ratio that is exactly c cannot be
turned into c + 1 ulp by the order of float operations. If fewer than MIN_MEMBERS members of an
ensemble pass, the static ensemble is used that day (fallback). Kept members are fused with the
frozen primary fusion (min-max + mean), calibrators from their training scores only.

Everything here is label-free: the decision for a day uses that day's own unlabelled scores.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd

from thesis import fusion

C = 3
MIN_MEMBERS = 2
CALIB, AGG = "minmax", "mean"


def target(budget) -> Fraction:
    """1 - q = the budget, exact (Fraction of its decimal string: 0.02 -> 1/50)."""
    t = Fraction(str(budget))
    if not 0 < t < 1:
        raise ValueError(f"budget {budget} outside (0, 1)")
    return t


def quantile(budget) -> float:
    """q = 1 - budget, rounded so that 1 - 0.02 is 0.98 exactly as a float literal."""
    return round(1 - float(budget), 12)


def thresholds(train_scores: np.ndarray, budget) -> np.ndarray:
    """One threshold per seed from that seed's training scores."""
    return np.quantile(np.atleast_2d(train_scores), quantile(budget), axis=1, method="linear")


def alert_count(scores: np.ndarray, thr: np.ndarray) -> int:
    """Alerts summed over seeds (rows of `scores`), strict: score > threshold."""
    return int((np.atleast_2d(scores) > thr[:, None]).sum())


def passes(alerts: int, n_rows: int, n_seeds: int, budget, c=C) -> bool:
    """r = alerts / (n_rows * n_seeds * (1 - q)) <= c, computed exactly; r = c passes."""
    if n_rows < 1 or n_seeds < 1 or alerts < 0:
        raise ValueError("need n_rows >= 1, n_seeds >= 1, alerts >= 0")
    return Fraction(int(alerts)) <= Fraction(c) * target(budget) * n_rows * n_seeds


def ratio(alerts: int, n_rows: int, n_seeds: int, budget) -> float:
    """r as a float, for reporting only (decisions use passes)."""
    return float(Fraction(int(alerts)) / (target(budget) * n_rows * n_seeds))


def kept_members(members, passed: set[str], min_members: int = MIN_MEMBERS) -> tuple[list[str], bool]:
    """Members that pass, in ensemble order; the static ensemble if fewer than min_members pass."""
    kept = [m for m in members if m in passed]
    return (kept, False) if len(kept) >= min_members else (list(members), True)


def fuse_members(train: dict, test: dict, kept, rows=None) -> tuple[np.ndarray, np.ndarray]:
    """min-max + mean of `kept`, one row per IForest seed (seeds paired as in e3_ulb.fit_stage);
    `rows` selects test columns (one day)."""
    n_seeds = max(len(np.atleast_2d(train[m])) for m in kept)
    tr, te = [], []
    for i in range(n_seeds):
        pick = [min(i, len(np.atleast_2d(train[m])) - 1) for m in kept]
        S_tr = np.column_stack([np.atleast_2d(train[m])[j] for m, j in zip(kept, pick)])
        S_te = np.column_stack([np.atleast_2d(test[m])[j] if rows is None
                                else np.atleast_2d(test[m])[j, rows] for m, j in zip(kept, pick)])
        f_tr, f_te = fusion.fuse(S_tr, S_te, CALIB, AGG, list(kept), "guard")
        tr.append(f_tr)
        te.append(f_te)
    return np.vstack(tr), np.vstack(te)


def member_days(train: dict, test: dict, day: np.ndarray, budget, c=C) -> pd.DataFrame:
    """Per member and test day: alerts at the frozen threshold, r, and the exact pass flag."""
    rows = []
    for m in train:
        thr = thresholds(train[m], budget)
        s = np.atleast_2d(test[m])
        for d in np.unique(day):
            cols = day == d
            a = alert_count(s[:, cols], thr)
            n = int(cols.sum())
            rows.append({"member": m, "day": int(d), "n_rows": n, "n_seeds": len(thr),
                         "alerts": a, "r": ratio(a, n, len(thr), budget),
                         "passes": passes(a, n, len(thr), budget, c)})
    return pd.DataFrame(rows)


def guarded(train: dict, test: dict, day: np.ndarray, ensembles: dict, budget, c=C,
            min_members: int = MIN_MEMBERS) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Daily guard for every ensemble. Returns (member-day table, guarded test scores per
    ensemble with one row per seed, decision table per ensemble x day)."""
    md = member_days({m: train[m] for e in ensembles.values() for m in e},
                     {m: test[m] for e in ensembles.values() for m in e}, day, budget, c)
    out, dec = {}, []
    for ens, members in ensembles.items():
        n_seeds = max(len(np.atleast_2d(train[m])) for m in members)
        scores = np.empty((n_seeds, len(day)))
        for d in np.unique(day):
            cols = np.flatnonzero(day == d)
            ok = set(md[(md.day == d) & md.passes].member)
            kept, fallback = kept_members(members, ok, min_members)
            _, te = fuse_members(train, test, kept, cols)
            scores[:, cols] = te if len(te) == n_seeds else np.repeat(te, n_seeds, axis=0)
            dec.append({"ensemble": ens, "day": int(d), "kept": "+".join(kept),
                        "dropped": "+".join(m for m in members if m not in kept),
                        "n_kept": len(kept), "fallback": fallback,
                        "changed": kept != list(members)})
        out[ens] = scores
    return md, out, pd.DataFrame(dec)


def daily_fp(scores: np.ndarray, y: np.ndarray, day: np.ndarray, budget) -> pd.DataFrame:
    """FP@K per test day with K_d = ceil(budget * n_d) (fixed daily budget), seed mean."""
    from thesis import metrics
    rows = []
    s = np.atleast_2d(scores)
    for d in np.unique(day):
        cols = day == d
        k = metrics.budget_k(budget, int(cols.sum()))
        fp = np.mean([k - metrics.tp_at_k(r[cols], y[cols], k) for r in s])
        rows.append({"day": int(d), "n_rows": int(cols.sum()), "k": k, "fp_at_k": float(fp)})
    return pd.DataFrame(rows)
