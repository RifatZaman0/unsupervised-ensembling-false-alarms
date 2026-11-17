"""IEEE-CIS: the single run of the freeze-v1 analyses and the freeze-v3 (RQ4) analyses.

    python -m thesis.e3_ieee                      # THE run: once, after the freeze-v3 tag
    python -m thesis.e3_ieee --dry-run [--out D]  # month 1 only, synthetic labels, never results/

Real run (refuses to start unless tags freeze-v1, freeze-v2 and freeze-v3 exist, the three
frozen files equal their tags, frozen_stress_v3.json passes the tag-time check, the working
tree is clean and no earlier output exists): month 1 fit, months 2-6 test.
- freeze-v1: E3 budget mode per month and summed over months 2-6 (H1a, H1b, H2b, descriptive
  CIs, regret) with the month-stratified day-block bootstrap; E4 threshold mode per month;
  H3a; H3b (weekly Evidently drift share against the weekly FPR of E-bench); context (ADTK on
  daily alert rates, Evidently drift of the fused E-bench score).
- freeze-v3: robust fusion (train-percentile + median: H4r over the 20 triples in threshold
  mode, H4r-cost in budget mode, E-bench / E-bench-5% / E-all as descriptive rows, the
  label-free bound check per month), pre-deployment check (days 0-20 -> 21-29), run-time
  guard per test day, daily FP@K, H4a-H4e, the drift slope, the member-day table of r. No
  month-1 label is read.
Dry run: the same code on month 1 only. Days 0-20 play month 1, days 21-29 play the test
months (21-22 -> "2", 23-24 -> "3", 25-26 -> "4", 27-28 -> "5", 29 -> "6"), each test day is
an H3b window, and the pre-deployment split is days 0-14 -> 15-20. isFraud is never read:
labels are synthetic (Bernoulli 0.035, seed 12345). It checks that the run executes end to
end; its numbers mean nothing and it writes to a scratch folder only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from thesis import (bootstrap, data, detectors, fragility, frozen, frozen_v3, guard,
                    ieee_features, ieee_stress, lof_diag, metrics, robust_fusion)
from thesis.e3_ulb import (Fitted, check_code_matches, comparisons, criterion,
                           distinct_ensembles, fit_stage, fit_thresholds, frozen_unchanged,
                           hypothesis_pairs, method_key, parse_key, threshold_table)
from thesis.export_datasets import git_state
from thesis.paths import REPO, TABLES

OUT = TABLES / "ieee_cis" / "run"
TIME, LABEL = ieee_features.TIME, ieee_features.LABEL
TAGS = {"freeze-v1": "frozen_config.json", "freeze-v2": "frozen_features_v2.json",
        "freeze-v3": "frozen_stress_v3.json"}
KNN_LOF_LIMIT_S = 1800                    # freeze-v1 detectors.knn_lof_fallback
KNN_LOF_FALLBACK_ROWS = 10_000
DRY_LABEL_RATE, DRY_LABEL_SEED = 0.035, 12345
DRY_MONTHS = {21: 2, 22: 2, 23: 3, 24: 3, 25: 4, 26: 4, 27: 5, 28: 5, 29: 6}
DRY_TRAIN_LAST, DRY_PRE_FIT_LAST = 20, 14
ADTK_ORIGIN = pd.Timestamp("2000-01-01")  # TransactionDT is relative; days get a nominal date
DESCRIPTIVE = "descriptive, no criterion applied"


class KnnLofTooSlow(RuntimeError):
    pass


# ---------------------------------------------------------------- data layout

@dataclass
class Layout:
    """Everything the run needs, in row order (training rows, then test rows)."""
    train: pd.DataFrame            # raw freeze-v2 columns, training window
    test: pd.DataFrame             # raw freeze-v2 columns, test rows
    day_train: np.ndarray
    day_test: np.ndarray
    month_test: np.ndarray         # evaluation window of each test row
    week_test: np.ndarray          # H3b window of each test row
    months: list[int]
    pre_fit: np.ndarray            # positions in train: pre-deployment fit slice
    pre_scored: np.ndarray         # positions in train: pre-deployment scored slice
    y_test: np.ndarray             # evaluation only (no month-1 label is ever read)
    labels: str                    # "isFraud" or "synthetic"


def real_layout(names) -> Layout:
    """All rows of train_transaction.csv: month 1 train, months 2-6 test. Reads isFraud."""
    df = pd.read_csv(ieee_features.RAW, usecols=[TIME, LABEL, *names])
    t = df[TIME].to_numpy()
    if not np.all(np.diff(t) >= 0):
        raise RuntimeError("TransactionDT is not sorted")
    day = data.ieee_cis_day(t)
    month = data.ieee_cis_month(day)
    tr, te = month == 1, month > 1
    assert t[tr].max() < t[te].min()                     # every training row strictly earlier
    y = df[LABEL].to_numpy(dtype=np.int8)
    fit, scored = ieee_stress.split_days(day[tr])
    return Layout(df.loc[tr, names].reset_index(drop=True), df.loc[te, names].reset_index(drop=True),
                  day[tr], day[te], month[te], data.ieee_cis_week(day)[te],
                  list(data.IEEE_TEST_MONTHS), fit, scored, y[te], LABEL)


def dry_layout(names) -> Layout:
    """Month 1 only, without isFraud: days 0-20 train, days 21-29 as five pseudo months."""
    month1, _ = ieee_features.load_month1()
    assert LABEL not in month1.columns
    day = data.ieee_cis_day(month1[TIME].to_numpy())
    tr, te = day <= DRY_TRAIN_LAST, day > DRY_TRAIN_LAST
    rng = np.random.default_rng(DRY_LABEL_SEED)
    y = (rng.random(len(day)) < DRY_LABEL_RATE).astype(np.int8)
    d_tr, d_te = day[tr], day[te]
    fit, scored = np.flatnonzero(d_tr <= DRY_PRE_FIT_LAST), np.flatnonzero(d_tr > DRY_PRE_FIT_LAST)
    month = np.array([DRY_MONTHS[int(d)] for d in d_te])
    return Layout(month1.loc[tr, names].reset_index(drop=True),
                  month1.loc[te, names].reset_index(drop=True), d_tr, d_te, month,
                  d_te - d_te.min(), sorted(set(DRY_MONTHS.values())), fit, scored,
                  y[te], "synthetic")


# ---------------------------------------------------------------- fitting (label-free)

def score_fn(limit_s: float = KNN_LOF_LIMIT_S):
    """e3_ulb.fit_stage scorer: fit on the training rows, score training and test rows; a KNN
    or LOF training-score pass slower than limit_s raises KnnLofTooSlow."""
    def fn(s: data.Split, det: str) -> dict:
        t0 = time.perf_counter()
        model, tr = detectors.fit_scores(det, s.X_train, s.seed)
        train_s = time.perf_counter() - t0
        if det in ("KNN", "LOF") and train_s > limit_s:
            raise KnnLofTooSlow(f"{det}: {train_s:.0f} s")
        te = np.asarray(model.decision_function(s.X_test), dtype=np.float64)
        if not np.isfinite(te).all():
            raise detectors.NonFiniteScores(f"{det}: non-finite test scores")
        return {"train": tr, "test": te, "runtime_s": time.perf_counter() - t0}
    return fn


def fit_all(lay: Layout, names, cfg: dict, limit_s: float = KNN_LOF_LIMIT_S) -> tuple[Fitted, dict]:
    """freeze-v2 features fitted on the training rows, then e3_ulb.fit_stage (z-score from the
    training rows, every detector, every fusion config). Labels never enter."""
    rule = ieee_features.FeatureRule.fit(lay.train, names)
    X_tr, X_te = rule.transform(lay.train), rule.transform(lay.test)
    n = len(X_tr)
    split = data.Split("ieee_cis", X_tr, X_te, np.zeros(n, np.int8), np.zeros(len(X_te), np.int8),
                       np.arange(n), np.arange(n, n + len(X_te)))
    info = {"knn_lof_fit_rows": detectors.MAX_FIT_ROWS, "knn_lof_fallback_used": False}
    try:
        fitted = fit_stage(split, cfg, score_fn(limit_s))
    except KnnLofTooSlow as e:
        old = detectors.MAX_FIT_ROWS
        detectors.MAX_FIT_ROWS = KNN_LOF_FALLBACK_ROWS
        try:
            fitted = fit_stage(split, cfg, score_fn(float("inf")))
        finally:
            detectors.MAX_FIT_ROWS = old
        info = {"knn_lof_fit_rows": KNN_LOF_FALLBACK_ROWS, "knn_lof_fallback_used": True,
                "reason": str(e)}
    return fitted, info


# ---------------------------------------------------------------- freeze-v1 analyses

def month_rows(lay: Layout) -> dict[int, np.ndarray]:
    return {m: np.flatnonzero(lay.month_test == m) for m in lay.months}


def budget_monthly(fitted: Fitted, y: np.ndarray, budgets, rows: dict) -> pd.DataFrame:
    """E3 budget mode: one row per method x seed x budget x month."""
    out = []
    for key in fitted.test:
        ens, calib, agg = parse_key(key)
        for i, s in enumerate(np.atleast_2d(fitted.test[key])):
            for m, idx in rows.items():
                rank = metrics.ranking_metrics(s[idx], y[idx])
                for b in budgets:
                    out.append({"method": key, "kind": "single" if calib is None else "ensemble",
                                "calib": calib, "agg": agg, "seed_index": i, "month": m, **rank,
                                **metrics.budget_metrics(s[idx], y[idx], b)})
    return pd.DataFrame(out)


def budget_sums(monthly: pd.DataFrame, singles) -> pd.DataFrame:
    """Sum over months of FP@K and K (seed mean), the baselines on the sums, regret."""
    per_seed = monthly.groupby(["method", "budget", "seed_index"], sort=False).agg(
        fp_sum=("fp_at_k", "sum"), k_sum=("k", "sum"), tp_sum=("tp_at_k", "sum")).reset_index()
    t = per_seed.groupby(["method", "budget"], sort=False).agg(
        fp_sum=("fp_sum", "mean"), tp_sum=("tp_sum", "mean"), k_sum=("k_sum", "first"),
        n_seeds=("seed_index", "size")).reset_index()
    extra = []
    for b, g in t.groupby("budget", sort=False):
        sg = g.set_index("method").loc[list(singles)]
        oracle = sg.fp_sum.idxmin()
        k = int(g.k_sum.iloc[0])
        extra += [{"method": bootstrap.MEDIAN, "budget": b, "fp_sum": float(np.median(sg.fp_sum)), "k_sum": k},
                  {"method": bootstrap.DEFAULT_IFOREST, "budget": b, "fp_sum": sg.loc["IForest", "fp_sum"], "k_sum": k},
                  {"method": bootstrap.ORACLE, "budget": b, "fp_sum": sg.loc[oracle, "fp_sum"], "k_sum": k,
                   "oracle_member": oracle}]
    t = pd.concat([t, pd.DataFrame(extra)], ignore_index=True)
    oracle_fp = t[t.method == bootstrap.ORACLE].set_index("budget").fp_sum
    t["regret"] = t.fp_sum - t.budget.map(oracle_fp)
    t["fp_per_100_alerts"] = t.fp_sum / t.k_sum * 100
    return t


def day_picks(rng: np.random.Generator, days: list[list[np.ndarray]]) -> list[np.ndarray]:
    """The day indices bootstrap.draw_days draws (same calls, same order, same stream)."""
    return [rng.integers(0, len(month), len(month)) for month in days]


def v1_bootstrap(fitted: Fitted, y, budgets, cfg, days, resamples, seed) -> pd.DataFrame:
    """Month-stratified day-block bootstrap of every freeze-v1 comparison (sums over months)."""
    out = []
    for b in budgets:
        stat = bootstrap.budget_statistic(fitted.test, y, b, list(fitted.singles), comparisons(cfg))
        point = stat(bootstrap.day_windows(days))
        boot = bootstrap.run(lambda rng: bootstrap.draw_days(rng, days), stat, resamples, seed)
        out.append(bootstrap.summary(point, boot).assign(budget=b))
    return pd.concat(out, ignore_index=True)


def v1_deltas(summ: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s = summ.set_index(["budget", "stat"])
    primary = cfg["budgets"]["ieee_cis"]["primary"]
    roles = hypothesis_pairs(cfg)
    rows = []
    for b in summ.budget.unique():
        for a, comp in comparisons(cfg):
            d, d100 = s.loc[(b, f"d:{a}-{comp}")], s.loc[(b, f"d100:{a}-{comp}")]
            role = roles.get((a, comp)) if b == primary else None
            rows.append({"budget": b, "method": a, "comparator": comp, "d_fp": d.estimate,
                         "d_fp_ci_low": d.ci_low, "d_fp_ci_high": d.ci_high,
                         "d_fp_per_100": d100.estimate, "d_fp_per_100_ci_low": d100.ci_low,
                         "d_fp_per_100_ci_high": d100.ci_high, "role": role or "descriptive",
                         "verdict": criterion(d.ci_high) if role else DESCRIPTIVE,
                         "ci_entirely_below_0": bool(d.ci_high < 0),
                         "ci_entirely_above_0": bool(d.ci_low > 0)})
    return pd.DataFrame(rows)


def ulb_verdicts() -> dict:
    """The ULB half of H1a and H1b (results/tables/e3_ulb/e3_ulb_hypotheses.csv)."""
    p = TABLES / "e3_ulb" / "e3_ulb_hypotheses.csv"
    if not p.exists():
        return {}
    t = pd.read_csv(p)
    return dict(zip(t["hypothesis"], t["verdict"]))


def v1_hypotheses(deltas: pd.DataFrame, cfg: dict, ulb: dict) -> pd.DataFrame:
    hyp, rows = cfg["hypotheses"], []
    d = deltas[deltas.role != "descriptive"].set_index("role")
    support = "fewer false alarms"
    for h in ("H1a", "H1b"):
        r = d.loc[h]
        ieee_ok = r.verdict == support
        ulb_ok = ulb.get(h) == support if h in ulb else None
        if hyp[h]["supported_if"] == "the criterion holds on both datasets":
            status = "supported" if ieee_ok and ulb_ok else "not supported"
        else:
            status = "supported" if ieee_ok or ulb_ok else "not supported"
        rows.append({"hypothesis": h, "dataset": "IEEE-CIS", "comparison": f"{r.method} vs {r.comparator}",
                     "d_fp": r.d_fp, "d_fp_ci_low": r.d_fp_ci_low, "d_fp_ci_high": r.d_fp_ci_high,
                     "verdict_ieee": r.verdict, "verdict_ulb": ulb.get(h), "status": status,
                     "rule": hyp[h]["supported_if"]})
    r = d.loc["H2b (secondary)"]
    rows.append({"hypothesis": "H2b (secondary)", "dataset": "IEEE-CIS",
                 "comparison": f"{r.method} vs {r.comparator}", "d_fp": r.d_fp,
                 "d_fp_ci_low": r.d_fp_ci_low, "d_fp_ci_high": r.d_fp_ci_high,
                 "verdict_ieee": r.verdict, "status": "descriptive only", "rule": hyp["H2b"]["secondary"]})
    rows.append({"hypothesis": "H2b (primary)", "dataset": "IEEE-CIS", "comparison": "E-bench vs E-lit",
                 "d_fp": 0.0, "status": hyp["H2b"]["primary"]})
    return pd.DataFrame(rows)


def threshold_monthly(fitted: Fitted, y, quantiles, rows: dict) -> pd.DataFrame:
    """E4: frozen month-1 thresholds applied unchanged to every month."""
    out = []
    for m, idx in rows.items():
        sub = Fitted(fitted.train, {k: np.atleast_2d(v)[:, idx] for k, v in fitted.test.items()},
                     fitted.singles, fitted.runtime_s)
        out.append(threshold_table(sub, y[idx], quantiles).assign(month=m))
    return pd.concat(out, ignore_index=True)


def h3a(fitted: Fitted, cfg, days, resamples, seed) -> pd.DataFrame:
    spec = cfg["hypotheses"]["H3a"]
    q = spec["quantile"]
    ens = method_key(spec["ensemble"], *cfg_primary(cfg))
    keys = [*fitted.singles, ens]
    al = {k: fitted.test[k] > fit_thresholds(fitted.train[k], q)[:, None] for k in keys}
    stat = bootstrap.alert_rate_statistic(al, q, list(fitted.singles), ens)
    point = stat(bootstrap.day_windows(days))
    boot = bootstrap.run(lambda rng: bootstrap.draw_days(rng, days), stat, resamples, seed)
    summ = bootstrap.summary(point, boot)
    r = summ.set_index("stat").loc[f"d:{ens}-{bootstrap.MEDIAN}"]
    return summ.assign(hypothesis="H3a", verdict=("supported" if r.ci_high < 0 else "not supported"))


def cfg_primary(cfg) -> tuple[str, str]:
    p = cfg["fusion"]["primary"]
    return p["calibration"], p["aggregation"]


def evidently_share(reference: pd.DataFrame, current: pd.DataFrame) -> float:
    """Share of drifted columns, Evidently DataDriftPreset with its default tests
    (freeze-v1 H3b); categorical columns as strings with missing as its own value."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset
    cat = [c for c in reference.columns if ieee_features.kind(c) == "categorical"]
    num = [c for c in reference.columns if c not in cat]

    def prep(df):
        df = df.copy()
        for c in cat:
            df[c] = df[c].astype(object).where(df[c].notna(), ieee_features.MISSING).astype(str)
        return df
    dd = DataDefinition(numerical_columns=num, categorical_columns=cat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snap = Report([DataDriftPreset()]).run(Dataset.from_pandas(prep(current), data_definition=dd),
                                               Dataset.from_pandas(prep(reference), data_definition=dd))
    for m in snap.dict()["metrics"]:
        if str(m.get("metric_name", "")).startswith("DriftedColumnsCount"):
            return float(m["value"]["share"])
    raise RuntimeError("Evidently returned no DriftedColumnsCount")


def score_drifted(reference: np.ndarray, current: np.ndarray) -> bool:
    share = evidently_share(pd.DataFrame({"score": reference}), pd.DataFrame({"score": current}))
    return share == 1.0


def h3b(fitted: Fitted, lay: Layout, cfg) -> tuple[pd.DataFrame, dict]:
    spec = cfg["hypotheses"]["H3b"]
    ens = method_key(cfg["hypotheses"]["primary_ensemble"], *cfg_primary(cfg))
    thr = fit_thresholds(fitted.train[ens], cfg["hypotheses"]["H3a"]["quantile"])
    a = np.atleast_2d(fitted.test[ens]) > thr[:, None]
    rows = []
    for w in np.unique(lay.week_test):
        idx = lay.week_test == w
        neg = (lay.y_test[idx] == 0)
        fpr = float(np.mean([(r[idx] & neg).sum() / max(neg.sum(), 1) for r in a]))
        rows.append({"week": int(w), "n_rows": int(idx.sum()), "fpr": fpr,
                     "drift_share": evidently_share(lay.train, lay.test.loc[idx])})
    t = pd.DataFrame(rows)
    ci = spec["ci"]
    res = bootstrap.spearman_moving_block(t.drift_share.to_numpy(), t.fpr.to_numpy(),
                                          ci["block_weeks"], ci["resamples"], 0)
    return t, {**res, "status": spec["status"]}


def adtk_context(fitted: Fitted, lay: Layout, cfg) -> pd.DataFrame:
    """freeze-v1 context.adtk: daily alert rates at q = 0.99, LevelShiftAD and QuantileAD."""
    from adtk.data import validate_series
    from adtk.detector import LevelShiftAD, QuantileAD
    spec = cfg["context"]["adtk"]["detectors"]
    q = cfg["hypotheses"]["H3a"]["quantile"]
    keys = [method_key(cfg["hypotheses"]["primary_ensemble"], *cfg_primary(cfg)), *fitted.singles]
    days_te, days_tr = np.unique(lay.day_test), np.unique(lay.day_train)
    rows = []
    for k in keys:
        thr = fit_thresholds(fitted.train[k], q)
        a_te = np.atleast_2d(fitted.test[k]) > thr[:, None]
        a_tr = np.atleast_2d(fitted.train[k]) > thr[:, None]
        rate_te = pd.Series([a_te[:, lay.day_test == d].mean() for d in days_te],
                            index=ADTK_ORIGIN + pd.to_timedelta(days_te, unit="D"))
        rate_tr = pd.Series([a_tr[:, lay.day_train == d].mean() for d in days_tr],
                            index=ADTK_ORIGIN + pd.to_timedelta(days_tr, unit="D"))
        ls = spec["LevelShiftAD"]
        if len(rate_te) >= 2 * ls["window"]:
            f_ls = LevelShiftAD(window=ls["window"], c=ls["c"], side=ls["side"]).fit_detect(
                validate_series(rate_te)).fillna(False)
        else:                                       # dry run only: fewer than 2 x window days
            f_ls = pd.Series(False, index=rate_te.index)
        qa = spec["QuantileAD"]
        qad = QuantileAD(low=qa["low"], high=qa["high"])
        qad.fit(validate_series(rate_tr))               # fit on the training days (month 1)
        f_q = qad.detect(validate_series(rate_te)).fillna(False)
        for d, r, a, b in zip(days_te, rate_te, f_ls, f_q):
            rows.append({"method": k, "day": int(d), "n_rows": int((lay.day_test == d).sum()),
                         "alert_rate": float(r), "level_shift": bool(a), "quantile": bool(b),
                         "flagged": bool(a or b)})
    return pd.DataFrame(rows)


def score_drift_context(fitted: Fitted, rows: dict, cfg) -> pd.DataFrame:
    ens = method_key(cfg["hypotheses"]["primary_ensemble"], *cfg_primary(cfg))
    out = []
    for m, idx in rows.items():
        for i, (tr, te) in enumerate(zip(np.atleast_2d(fitted.train[ens]), np.atleast_2d(fitted.test[ens]))):
            out.append({"month": m, "seed_index": i, "drifted": score_drifted(tr, te[idx])})
    t = pd.DataFrame(out)
    return t.assign(share_of_seeds=t.groupby("month")["drifted"].transform("mean"))


# ---------------------------------------------------------------- freeze-v3 analyses

def pre_deployment(lay: Layout, names, pool, seeds, budgets, ensembles) -> tuple:
    """Days 0-20 -> 21-29 (dry: 0-14 -> 15-20): r per member and the E-stress sets. Label-free."""
    X_fit, X_sc, _ = ieee_stress.features_for(lay.train, lay.pre_fit, lay.pre_scored, names)
    sl = lof_diag.slice_scores(X_fit, X_sc, pool, seeds)
    r_rows, sets = [], []
    for b in budgets:
        passed = set()
        for m in pool:
            thr = guard.thresholds(sl[m][0], b)
            a = guard.alert_count(sl[m][1], thr)
            n, k = sl[m][1].shape[1], len(thr)
            ok = guard.passes(a, n, k, b)
            passed |= {m} if ok else set()
            r_rows.append({"budget": b, "member": m, "alerts": a, "n_rows": n, "n_seeds": k,
                           "r": guard.ratio(a, n, k, b), "passes": ok})
        for e, members in ensembles.items():
            kept, fb = guard.kept_members(members, passed)
            sets.append({"budget": b, "ensemble": e, "kept": "+".join(kept), "fallback": fb,
                         "changed": kept != list(members)})
    return pd.DataFrame(r_rows), pd.DataFrame(sets)


def robust_scores(fitted: Fitted, ensembles) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Train-percentile + median of every ensemble (freeze-v3 robust_fusion): (train, test)."""
    return {e: robust_fusion.fuse_set(fitted.train, fitted.test, members, *robust_fusion.ROBUST)
            for e, members in ensembles.items()}


def rq4_methods(fitted: Fitted, lay: Layout, budget, ensembles, sets: pd.DataFrame, primary, robust):
    """Test scores of every RQ4 method at one budget: singles, static, guarded, E-stress, robust."""
    singles = list(fitted.singles)
    md, gscores, dec = guard.guarded(fitted.train, fitted.test, lay.day_test, ensembles, budget)
    scores = {m: fitted.test[m] for m in singles}
    for e, members in ensembles.items():
        scores[f"static:{e}"] = fitted.test[method_key(e, *primary)]
        scores[f"guarded:{e}"] = gscores[e]
        kept = sets[(sets.budget == budget) & (sets.ensemble == e)].kept.iloc[0].split("+")
        scores[f"stress:{e}"] = guard.fuse_members(fitted.train, fitted.test, kept)[1]
        scores[f"robust:{e}"] = robust[e][1]
    return scores, md, dec


def day_alerts(train: np.ndarray, test: np.ndarray, day: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Alerts (summed over seeds) and rows x seeds per test day (sorted days), frozen threshold."""
    thr = fit_thresholds(train, q)
    a = np.atleast_2d(test) > thr[:, None]
    days = np.unique(day)
    return (np.array([a[:, day == d].sum() for d in days], dtype=np.float64),
            np.array([(day == d).sum() * len(thr) for d in days], dtype=np.float64))


def month_deviation(A: np.ndarray, N: np.ndarray, M: np.ndarray, q: float, counts=None) -> float:
    """Mean over months of |log(month alert rate / (1 - q))|, a month without an alert counted as
    0.5 alert (frozen_stress_v3.json H4r.zero_alert_month): robust_fusion.deviation."""
    return robust_fusion.deviation(A, N, M, q, counts)


def h4r_threshold(methods: dict, lay: Layout, q: float, C: np.ndarray, pairs, roles) -> pd.DataFrame:
    """dD with the day-block resamples for every (method, comparator) pair; methods: name ->
    (train, test) scores."""
    days = np.unique(lay.day_test)
    month_of_day = {int(d): int(m) for d, m in zip(lay.day_test, lay.month_test)}
    M = np.array([[month_of_day[int(d)] == mo for mo in lay.months] for d in days], dtype=np.float64)
    AN = {k: day_alerts(tr, te, lay.day_test, q) for k, (tr, te) in methods.items()}

    def dev(k, w=None):
        return month_deviation(*AN[k], M, q, w)
    rows = []
    for a, c in pairs:
        boot = np.array([dev(a, w) - dev(c, w) for w in C])
        lo, hi = bootstrap.percentile_ci(boot)
        rows.append({"q": q, "method": a, "comparator": c, "D_method": dev(a), "D_comparator": dev(c),
                     "d_D": dev(a) - dev(c), "ci_low": lo, "ci_high": hi,
                     "ci_entirely_below_0": bool(hi < 0), "ci_entirely_above_0": bool(lo > 0),
                     "role": roles.get((a, c), "descriptive")})
    return pd.DataFrame(rows)


def month_matrix(lay: Layout) -> np.ndarray:
    """Test days (sorted) x months indicator."""
    days = np.unique(lay.day_test)
    month_of_day = {int(d): int(m) for d, m in zip(lay.day_test, lay.month_test)}
    return np.array([[month_of_day[int(d)] == mo for mo in lay.months] for d in days], dtype=np.float64)


def h4r_triples(fitted: Fitted, lay: Layout, q: float, C: np.ndarray, budget, y: np.ndarray):
    """H4r and H4r-cost over the 20 three-member ensembles (frozen_stress_v3.json): per triple,
    D under train-percentile + median and under min-max + mean (paired), and the daily FP@K
    totals. Returns (per-triple table, H4r result, H4r-cost result)."""
    M = month_matrix(lay)
    rows, boot, fp_r, fp_f = [], [], [], []
    for members in robust_fusion.TRIPLES:
        an, fp = {}, {}
        for name, cfg in (("robust", robust_fusion.ROBUST), ("frozen", robust_fusion.FROZEN)):
            tr, te = robust_fusion.fuse_set(fitted.train, fitted.test, members, *cfg)
            an[name] = robust_fusion.block_alerts(tr, te, lay.day_test, q)
            fp[name] = guard.daily_fp(te, y, lay.day_test, budget).fp_at_k.to_numpy()
        d_r = robust_fusion.deviation(*an["robust"], M, q)
        d_f = robust_fusion.deviation(*an["frozen"], M, q)
        rows.append({"triple": robust_fusion.triple_name(members), "q": q, "D_robust": d_r,
                     "D_frozen": d_f, "dD": d_r - d_f, "budget": budget,
                     "fp_total_robust": float(fp["robust"].sum()), "fp_total_frozen": float(fp["frozen"].sum())})
        boot.append([robust_fusion.deviation(*an["robust"], M, q, w) -
                     robust_fusion.deviation(*an["frozen"], M, q, w) for w in C])
        fp_r.append(fp["robust"])
        fp_f.append(fp["frozen"])
    tab = pd.DataFrame(rows)
    main = robust_fusion.mean_ci(tab.dD.mean(), np.array(boot).T.mean(axis=1))
    FR, FZ = np.array(fp_r).T, np.array(fp_f).T            # days x triples
    cost = robust_fusion.mean_ci(robust_fusion.cost_ratio(FR, FZ), robust_fusion.cost_ratio(FR, FZ, C))
    cost["passes"] = robust_fusion.cost_passes(cost["ci_high"])
    return tab, main, cost


def bound_monthly(fitted: Fitted, lay: Layout, ensembles, mr: pd.DataFrame, q: float) -> pd.DataFrame:
    """Label-free bound check per month (freeze-v3 hypotheses.bound_check): b = 0 and b = 1 with
    the broken member = the member with the largest month r at its own month-1 threshold."""
    out = []
    for mo in lay.months:
        cols = lay.month_test == mo
        test = {m: np.atleast_2d(fitted.test[m])[:, cols] for m in fitted.singles}
        r_mo = mr[mr.month == mo].set_index("member").r
        for e, members in ensembles.items():
            worst = r_mo.loc[list(members)].idxmax()
            for broken in ([], [worst]):
                t = robust_fusion.bound_check(fitted.train, test, list(members), broken, q)
                out.append(t.assign(ensemble=e, month=mo,
                                    broken_r=float(r_mo[worst]) if broken else np.nan))
    return pd.concat(out, ignore_index=True)


def daily_table(scores: dict, y, day, budget) -> pd.DataFrame:
    """FP@K_d per method and day (seed mean): rows = days in np.unique order."""
    cols = {k: guard.daily_fp(s, y, day, budget).set_index("day").fp_at_k for k, s in scores.items()}
    return pd.DataFrame(cols)


def day_counts(picks: list[np.ndarray], month_days: list[np.ndarray]) -> np.ndarray:
    """How often each test day (in the order of the concatenated month_days) is drawn."""
    out = []
    for p, md in zip(picks, month_days):
        out.append(np.bincount(p, minlength=len(md)))
    return np.concatenate(out)


def rq4_counts(lay: Layout, resamples: int, seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Resample x day multiplicity matrix, from the freeze-v1 day-block draws (same stream)."""
    month_days = [np.unique(lay.day_test[lay.month_test == m]) for m in lay.months]
    days = bootstrap.day_rows(lay.day_test, lay.month_test, lay.months)
    rng = np.random.default_rng(seed)
    C = np.vstack([day_counts(day_picks(rng, days), month_days) for _ in range(resamples)])
    return C, month_days


def spearman_inf(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho with average ranks; +-inf rank as the extremes; nan if a side is constant."""
    rx, ry = rankdata(x), rankdata(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def abs_log(r: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.abs(np.log(np.asarray(r, dtype=np.float64)))


def member_month_arrays(md: pd.DataFrame, lay: Layout) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Members, alerts (members x days), rows x seeds (members x days) and the day -> month
    indicator (days x months), days in sorted order (the order of rq4_counts)."""
    days = np.unique(lay.day_test)
    month_of_day = {int(d): int(m) for d, m in zip(lay.day_test, lay.month_test)}
    members = list(dict.fromkeys(md.member))
    t = md.set_index(["member", "day"])
    A = np.array([[t.loc[(m, d), "alerts"] for d in days] for m in members], dtype=np.float64)
    N = np.array([[t.loc[(m, d), "n_rows"] * t.loc[(m, d), "n_seeds"] for d in days]
                  for m in members], dtype=np.float64)
    M = np.array([[month_of_day[int(d)] == mo for mo in lay.months] for d in days], dtype=np.float64)
    return members, A, N, M


def monthly_r(md: pd.DataFrame, lay: Layout, budget, counts: np.ndarray | None = None,
              arrays=None) -> pd.DataFrame:
    """r per member and month from the member-day table; counts = day multiplicities."""
    members, A, N, M = arrays or member_month_arrays(md, lay)
    w = np.ones(A.shape[1]) if counts is None else np.asarray(counts, dtype=np.float64)
    r = (A * w) @ M / (((N * w) @ M) * float(guard.target(budget)))
    return pd.DataFrame([{"member": m, "month": mo, "r": float(r[i, j])}
                         for i, m in enumerate(members) for j, mo in enumerate(lay.months)])


def h4a(pre_r: pd.DataFrame, md: pd.DataFrame, lay: Layout, budget, C: np.ndarray, singles):
    x_m = pre_r[pre_r.budget == budget].set_index("member").r
    arrays = member_month_arrays(md, lay)

    def rho(counts):
        mr = monthly_r(md, lay, budget, counts, arrays)
        mr = mr[mr.member.isin(singles)]
        return spearman_inf(abs_log(x_m.loc[mr.member].to_numpy()), abs_log(mr.r.to_numpy()))
    point = rho(None)
    boot = np.array([rho(c) for c in C])
    ok = boot[~np.isnan(boot)]
    lo, hi = bootstrap.percentile_ci(ok) if ok.size else (np.nan, np.nan)
    pairs = monthly_r(md, lay, budget).assign(x_abs_log_r_pre=lambda d: abs_log(x_m.loc[d.member].to_numpy()),
                                              y_abs_log_r=lambda d: abs_log(d.r.to_numpy()))
    return {"hypothesis": "H4a", "estimate": point, "ci_low": lo, "ci_high": hi,
            "n_pairs": int(len(pairs[pairs.member.isin(singles)])),
            "undefined_resamples": int(boot.size - ok.size),
            "verdict": "supported" if lo > 0 else "not supported"}, pairs


def rq4_deltas(D: pd.DataFrame, C: np.ndarray, singles, pairs, budget, roles: dict) -> pd.DataFrame:
    """dFP over all test days (daily K) with the day-block resamples. D: days x methods."""
    point = D.sum(axis=0)
    boot = pd.DataFrame(C @ D.to_numpy(), columns=D.columns)
    def base(frame, name):
        sv = frame[list(singles)]
        if name == bootstrap.MEDIAN:
            return np.median(sv.to_numpy(), axis=-1)
        if name == bootstrap.ORACLE:
            return sv.to_numpy().min(axis=-1)
        return np.asarray(frame[name])
    rows = []
    for a, c in pairs:
        d_point = float(point[a] - base(point, c))
        d_boot = boot[a].to_numpy() - base(boot, c)
        lo, hi = bootstrap.percentile_ci(d_boot)
        role = roles.get((a, c))
        rows.append({"budget": budget, "method": a, "comparator": c, "fp_method": float(point[a]),
                     "fp_comparator": float(base(point, c)), "d_fp": d_point, "ci_low": lo,
                     "ci_high": hi, "ci_entirely_below_0": bool(hi < 0),
                     "ci_entirely_above_0": bool(lo > 0), "role": role or "descriptive"})
    return pd.DataFrame(rows)


def rq4_hypothesis_rows(dl: pd.DataFrame, main: dict, cost: dict, dec: pd.DataFrame, h4a_row: dict,
                        h4d: dict, v3: dict) -> pd.DataFrame:
    """H4r (main, 20 triples), H4r-cost, H4a, H4b, H4c, H4d. A member breaks if the guard
    excludes it from E-all on at least one test day at the primary budget (frozen `breaks`)."""
    hyp = v3["hypotheses"]
    broke = dec.groupby("ensemble")["changed"].any().to_dict()
    any_break = bool(broke.get("E-all", False))
    if any_break:
        ok = bool(main["ci_entirely_below_0"])
        verdict, status = ("supported" if ok else "not supported"), "decided"
    else:
        ok = not bool(main["ci_entirely_above_0"])
        verdict, status = ("no harm shown" if ok else "worse"), "no member broke: no-harm check"
    rows = [{"hypothesis": "H4r", "estimate": main["estimate"], "ci_low": main["ci_low"],
             "ci_high": main["ci_high"], "comparison": "mean over 20 triples: train-percentile + median vs min-max + mean",
             "verdict": verdict, "status": status, "rule": hyp["H4r"]["supported_if"]},
            {"hypothesis": "H4r-cost", "estimate": cost["estimate"], "ci_low": cost["ci_low"],
             "ci_high": cost["ci_high"], "comparison": "relative increase of the mean total FP@K over 20 triples",
             "verdict": "passes" if cost["passes"] else "fails", "status": "decided",
             "rule": hyp["H4r_cost"]["passes_if"]},
            {**h4a_row, "rule": hyp["H4a"]["supported_if"], "status": "decided"}]
    d = dl.set_index("role")
    for h, ens in (("H4b", "E-all"), ("H4c", "E-bench")):
        r = d.loc[h]
        if h == "H4b":
            v = "supported" if r.ci_entirely_below_0 else "not supported"
        else:
            v = "no harm shown" if not r.ci_entirely_above_0 else "worse"
        st = "decided"
        if not broke.get(ens, False):
            st = {"H4b": "not testable: no member broke (dFP = 0 by identity)",
                  "H4c": "descriptive: no member broke (dFP = 0 by identity)"}[h]
        rows.append({"hypothesis": h, "estimate": r.d_fp, "ci_low": r.ci_low, "ci_high": r.ci_high,
                     "comparison": f"{r.method} vs {r.comparator}", "verdict": v, "status": st,
                     "rule": hyp[h]["supported_if"]})
    rows.append({"hypothesis": "H4d", "estimate": h4d["rho"], "verdict": "exploratory",
                 "status": hyp["H4d"]["status"], "comparison": "F_m vs mean log r_m,month"})
    return pd.DataFrame(rows)


def h4d(fitted: Fitted, mr: pd.DataFrame, budget) -> tuple[dict, pd.DataFrame]:
    q = guard.quantile(budget)
    t = pd.DataFrame([{"member": m, "fragility": float(fragility.index(fitted.train[m], q).mean())}
                      for m in fitted.singles])
    with np.errstate(divide="ignore"):
        mean_log = mr.assign(log_r=np.log(mr.r)).groupby("member").log_r.mean()
    t["mean_log_r"] = t.member.map(mean_log)
    return {"rho": spearman_inf(t.fragility.to_numpy(), t.mean_log_r.to_numpy())}, t


def drift_slopes(lay: Layout, rows: dict, mr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    share = pd.DataFrame([{"month": m, "drift_share": evidently_share(lay.train, lay.test.loc[idx])}
                          for m, idx in rows.items()])
    t = mr.merge(share, on="month").assign(deviation=lambda d: (d.r - 1).abs())
    slopes = []
    for m, g in t.groupby("member", sort=False):
        x = g.drift_share.to_numpy()
        slope = float(np.polyfit(x, g.deviation.to_numpy(), 1)[0]) if np.ptp(x) > 0 else float("nan")
        slopes.append({"member": m, "slope": slope, "n_months": len(g)})
    return t, pd.DataFrame(slopes)


# ---------------------------------------------------------------- guards for the one real run

def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def real_run_allowed(out: Path) -> None:
    """Every condition for the single IEEE-CIS run, checked before any month 2-6 row is read."""
    for tag, path in TAGS.items():
        if git("rev-parse", "--verify", "--quiet", f"refs/tags/{tag}").returncode != 0:
            raise SystemExit(f"tag {tag} is missing: the IEEE-CIS run needs freeze-v3")
        if git("diff", "--quiet", tag, "--", path).returncode != 0:
            raise SystemExit(f"{path} differs from {tag}")
    frozen_v3.load(for_tag=True)
    if (out / "run.meta.json").exists():
        raise SystemExit(f"{out} already holds an IEEE-CIS run: it is run once")


# ---------------------------------------------------------------- main

def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="python -m thesis.e3_ieee")
    ap.add_argument("--dry-run", action="store_true", help="month 1 only, synthetic labels")
    ap.add_argument("--out", type=Path, default=None, help="dry run: output folder (not results/)")
    ap.add_argument("--resamples", type=int, default=None, help="dry run only")
    args = ap.parse_args(argv)
    state = git_state()
    t0 = time.perf_counter()
    cfg = frozen.load()
    check_code_matches(cfg)
    if args.dry_run:
        out = args.out or Path(tempfile.mkdtemp(prefix="e3_ieee_dry_"))
        if TABLES.resolve() in [out.resolve(), *out.resolve().parents]:
            raise SystemExit("a dry run never writes into results/")
        v3 = frozen_v3.load()
        resamples = args.resamples or 50
    else:
        if args.out or args.resamples:
            raise SystemExit("--out and --resamples are for the dry run only")
        if state["dirty"]:
            raise SystemExit("Working tree has uncommitted changes; commit before the run.")
        frozen_unchanged()
        out = OUT
        real_run_allowed(out)
        v3 = frozen_v3.load(for_tag=True)
        resamples = cfg["statistics"]["bootstrap"]["resamples"]
    frozen_v3.check_code_matches(v3)
    seed = cfg["statistics"]["bootstrap"]["seed"]
    names = [f["name"] for f in ieee_features.load_frozen()["features"]]
    pool, seeds = cfg["detectors"]["pool"], cfg["detectors"]["iforest_seeds"]
    budgets = cfg["budgets"]["ieee_cis"]["values"]
    primary_budget = cfg["budgets"]["ieee_cis"]["primary"]
    quantiles = cfg["thresholds"]["ieee_cis"]["training_score_quantiles"]
    ensembles = {e: m for e, m in distinct_ensembles(cfg).items() if e in v3["guard"]["ensembles"]}
    primary = cfg_primary(cfg)
    log = lambda msg: print(f"[{time.perf_counter() - t0:7.1f} s] {msg}", flush=True)

    lay = dry_layout(names) if args.dry_run else real_layout(names)
    log(f"rows: train {len(lay.train)}, test {len(lay.test)}, labels {lay.labels}")
    fitted, fit_info = fit_all(lay, names, cfg)
    log(f"fitted {len(fitted.test)} methods {fit_info}")
    y = lay.y_test                                          # labels: evaluation only from here
    rows = month_rows(lay)
    days = bootstrap.day_rows(lay.day_test, lay.month_test, lay.months)

    # freeze-v1
    monthly = budget_monthly(fitted, y, budgets, rows)
    sums = budget_sums(monthly, fitted.singles)
    summ = v1_bootstrap(fitted, y, budgets, cfg, days, resamples, seed)
    deltas = v1_deltas(summ, cfg)
    v1_hyp = v1_hypotheses(deltas, cfg, ulb_verdicts())
    log("E3 budget mode and bootstrap done")
    thr_m = threshold_monthly(fitted, y, quantiles, rows)
    h3a_t = h3a(fitted, cfg, days, resamples, seed)
    log("E4 and H3a done")
    h3b_t, h3b_res = h3b(fitted, lay, cfg)
    adtk_t = adtk_context(fitted, lay, cfg)
    sdrift = score_drift_context(fitted, rows, cfg)
    log("H3b and context done")

    # freeze-v3
    pre_r, sets = pre_deployment(lay, names, pool, seeds, budgets, ensembles)
    robust = robust_scores(fitted, ensembles)
    C, _ = rq4_counts(lay, resamples, seed)
    q_main = v3["hypotheses"]["H4r"]["q"]
    thr_methods = {f"robust:{e}": robust[e] for e in ensembles}
    thr_methods.update({f"static:{e}": (fitted.train[method_key(e, *primary)],
                                        fitted.test[method_key(e, *primary)]) for e in ensembles})
    thr_pairs = [(f"robust:{e}", f"static:{e}") for e in ensembles]
    h4r_thr = pd.concat([h4r_threshold(thr_methods, lay, q, C, thr_pairs, {})       # descriptive rows
                         for q in sorted({q_main, *quantiles}, reverse=True)], ignore_index=True)
    h4r_tab, h4r_main, h4r_cost = h4r_triples(fitted, lay, q_main, C, primary_budget, y)
    log("H4r (20 triples) and H4r-cost done")
    rq4_rows, md_all, dec_all, daily_all, h4a_row, h4a_pairs = [], [], [], [], None, None
    for b in budgets:
        scores, md, dec = rq4_methods(fitted, lay, b, ensembles, sets, primary, robust)
        D = daily_table(scores, y, lay.day_test, b)
        pairs, roles = [], {}
        for e in ensembles:
            pairs += [(f"robust:{e}", f"static:{e}"), (f"guarded:{e}", f"static:{e}"),
                      (f"guarded:{e}", f"robust:{e}"), (f"stress:{e}", f"static:{e}")]
        for m in ("guarded:E-bench", "robust:E-all", "static:E-bench"):          # H4e, descriptive
            pairs += [(m, bootstrap.MEDIAN), (m, bootstrap.ORACLE)]
        if b == primary_budget:
            roles = {("guarded:E-all", "static:E-all"): "H4b",
                     ("guarded:E-bench", "static:E-bench"): "H4c"}
            h4a_row, h4a_pairs = h4a(pre_r, md, lay, b, C, list(fitted.singles))
            mr = monthly_r(md, lay, b)
            h4d_res, h4d_t = h4d(fitted, mr[mr.member.isin(fitted.singles)], b)
            drift_t, slopes = drift_slopes(lay, rows, mr[mr.member.isin(fitted.singles)])
            bounds = bound_monthly(fitted, lay, ensembles, mr, guard.quantile(b))
            dec_primary = dec
        rq4_rows.append(rq4_deltas(D, C, fitted.singles, pairs, b, roles))
        md_all.append(md.assign(budget=b))
        dec_all.append(dec.assign(budget=b))
        daily_all.append(D.reset_index().melt(id_vars="day", var_name="method",
                                              value_name="fp_at_k").assign(budget=b))
        log(f"RQ4 budget {b} done")
    rq4 = pd.concat(rq4_rows, ignore_index=True)
    rq4_hyp = rq4_hypothesis_rows(rq4[rq4.budget == primary_budget], h4r_main, h4r_cost, dec_primary,
                                  h4a_row, h4d_res, v3)

    out.mkdir(parents=True, exist_ok=True)
    c = state["commit"]
    files = {"e3_budget_monthly": monthly, "e3_budget_sums": sums, "e3_bootstrap": summ,
             "e3_deltas": deltas, "v1_hypotheses": v1_hyp, "e4_threshold_monthly": thr_m,
             "h3a": h3a_t, "h3b_weekly": h3b_t, "context_adtk": adtk_t,
             "context_score_drift": sdrift, "rq4_pre_deployment_r": pre_r,
             "rq4_e_stress_sets": sets, "rq4_h4r_threshold": h4r_thr, "rq4_h4r_triples": h4r_tab, "rq4_bound_check": bounds,
             "rq4_member_days": pd.concat(md_all, ignore_index=True),
             "rq4_guard_decisions": pd.concat(dec_all, ignore_index=True),
             "rq4_daily_fp": pd.concat(daily_all, ignore_index=True), "rq4_deltas": rq4,
             "rq4_hypotheses": rq4_hyp, "rq4_h4a_pairs": h4a_pairs, "rq4_h4d_fragility": h4d_t,
             "rq4_drift_monthly": drift_t, "rq4_drift_slopes": slopes}
    for name, df in files.items():
        df.assign(commit=c).to_csv(out / f"{name}.csv", index=False)
    meta = {**state, "command": "python -m thesis.e3_ieee" + (" --dry-run" if args.dry_run else ""),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dry_run": args.dry_run, "labels": lay.labels,
            "rows": {"train": len(lay.train), "test": len(lay.test),
                     "pre_fit": len(lay.pre_fit), "pre_scored": len(lay.pre_scored)},
            "test_months": lay.months, "budgets": budgets, "resamples": resamples,
            "bootstrap_seed": seed, "fit": fit_info, "h3b": h3b_res,
            "tags": {t: git("rev-parse", t).stdout.strip() for t in TAGS},
            "runtime_s": round(time.perf_counter() - t0, 1)}
    (out / "run.meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n", encoding="utf-8")
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(v1_hyp[["hypothesis", "comparison", "d_fp", "d_fp_ci_low", "d_fp_ci_high", "status"]].to_string(index=False))
    print(rq4_hyp[["hypothesis", "estimate", "ci_low", "ci_high", "verdict", "status"]].to_string(index=False))
    log(f"written to {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
