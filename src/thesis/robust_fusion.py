"""Robust fusion (RQ4, freeze-v3; developed on ULB): train-percentile + median against the frozen
min-max + mean, on development data only.

    python -m thesis.robust_fusion          # writes results/tables/robust_fusion/
    python -m thesis.robust_fusion --quick  # ULB tables only, prints, writes nothing

Three tables, rebuilt with the thesis code from the score cache (the development check in
docs/notes/robust_fusion_check_2026-11-17 did the same with stand-alone scripts):
1. ULB budget mode: FP@K on day 2 at 0.1/0.2/0.5%, seed mean, singles and every ensemble under
   {min-max, train-percentile} x {mean, median};
2. ULB threshold mode: alert rate, FP and TP on day 2 at the frozen day-1 threshold
   (numpy.quantile of the fused day-1 training scores, q = 0.999 / 0.998 / 0.995);
3. ADBench static check: 15 sets x 5 seeds x 20 triples, the same four configurations, 2% and 5%.
Plus the label-free bound check of the proposition (docs/writing/notes/robust_fusion_proposition.md)
on ULB day 2.

Labels: ULB day-2 and ADBench test labels for FP@K / FP / TP only. Calibrators and thresholds come
from training scores only. The six freeze-v1 fusion configurations are not changed: median is an
extra aggregation (fusion.EXTRA_AGGREGATIONS).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd

from thesis import data, detectors, frozen, fusion, metrics, scorecache
from thesis.e3_ulb import distinct_ensembles, fit_stage, fit_thresholds, load_split
from thesis.export_datasets import git_state
from thesis.paths import TABLES

OUT = TABLES / "robust_fusion"
CONFIGS = (("minmax", "mean"), ("minmax", "median"), ("percentile", "mean"), ("percentile", "median"))
ROBUST = ("percentile", "median")
ULB_Q = (0.999, 0.998, 0.995)


def config_name(calib: str, agg: str) -> str:
    return f"{calib}+{agg}"


def fuse_set(train: dict, test: dict, members, calib: str, agg: str) -> tuple[np.ndarray, np.ndarray]:
    """Fused training and test scores, one row per IForest seed (seeds paired as in e3_ulb)."""
    n_seeds = max(len(np.atleast_2d(train[m])) for m in members)
    tr, te = [], []
    for i in range(n_seeds):
        pick = [min(i, len(np.atleast_2d(train[m])) - 1) for m in members]
        S_tr = np.column_stack([np.atleast_2d(train[m])[j] for m, j in zip(members, pick)])
        S_te = np.column_stack([np.atleast_2d(test[m])[j] for m, j in zip(members, pick)])
        a, b = fusion.fuse(S_tr, S_te, calib, agg, list(members), "robust")
        tr.append(a)
        te.append(b)
    return np.vstack(tr), np.vstack(te)


def methods(train: dict, test: dict, singles, ensembles) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {m: (np.atleast_2d(train[m]), np.atleast_2d(test[m])) for m in singles}
    for e, members in ensembles.items():
        for calib, agg in CONFIGS:
            out[f"{e}:{config_name(calib, agg)}"] = fuse_set(train, test, members, calib, agg)
    return out


def seed_mean_fp(s: np.ndarray, y: np.ndarray, budget) -> float:
    k = metrics.budget_k(budget, len(y))
    return float(np.mean([k - metrics.tp_at_k(r, y, k) for r in np.atleast_2d(s)]))


def budget_table(meth: dict, y: np.ndarray, budgets) -> pd.DataFrame:
    rows = []
    for name, (_, te) in meth.items():
        for b in budgets:
            rows.append({"method": name, "budget": b, "k": metrics.budget_k(b, len(y)),
                         "fp_at_k": seed_mean_fp(te, y, b), "n_seeds": len(te)})
    return pd.DataFrame(rows)


def threshold_table(meth: dict, y: np.ndarray, quantiles=ULB_Q) -> pd.DataFrame:
    """Frozen threshold rule on each method's own (fused) training scores; seed means."""
    rows = []
    for name, (tr, te) in meth.items():
        for q in quantiles:
            a = te > fit_thresholds(tr, q)[:, None]
            rows.append({"method": name, "q": q, "target_rate": round(1 - q, 12),
                         "alert_rate": float(a.mean(axis=1).mean()),
                         "fp": float((a & (y == 0)).sum(axis=1).mean()),
                         "tp": float((a & (y == 1)).sum(axis=1).mean()), "n_seeds": len(te)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- proposition bound

def required_above(m: int) -> int:
    """numpy median of m values > t needs at least ceil(m / 2) values > t (for even m the
    median averages the two middle values, so the larger of them must exceed t)."""
    return math.ceil(m / 2)


def bound_check(train: dict, test: dict, members, broken, q: float) -> pd.DataFrame:
    """Label-free check of the proposition on new data, per IForest seed.

    Train-percentile + median of `members`, threshold t = quantile q of the fused training
    scores. With the members in `broken` treated as arbitrary, an alert needs at least
    k = ceil(m/2) - b healthy members whose calibrated score exceeds t, so
        fused alert rate <= share of rows with >= k healthy members above t
                         <= (sum of the healthy members' rates above t) / k.
    """
    m, b = len(members), len(broken)
    if not 0 <= b <= (m - 1) // 2:
        raise ValueError(f"b = {b} broken members: the median of {m} tolerates at most {(m - 1) // 2}")
    k = required_above(m) - b
    healthy = [x for x in members if x not in broken]
    n_seeds = max(len(np.atleast_2d(train[x])) for x in members)
    rows = []
    for i in range(n_seeds):
        pick = {x: min(i, len(np.atleast_2d(train[x])) - 1) for x in members}
        cal = {x: fusion.Calibrator.fit(np.atleast_2d(train[x])[pick[x]], "percentile") for x in members}
        C_tr = np.column_stack([cal[x].transform(np.atleast_2d(train[x])[pick[x]]) for x in members])
        C_te = np.column_stack([cal[x].transform(np.atleast_2d(test[x])[pick[x]]) for x in members])
        t = float(np.quantile(np.median(C_tr, axis=1), q, method="linear"))
        alert = np.median(C_te, axis=1) > t
        above = np.column_stack([C_te[:, members.index(x)] > t for x in healthy])
        event = above.sum(axis=1) >= k
        rates = above.mean(axis=0)
        rows.append({"members": "+".join(members), "broken": "+".join(broken), "q": q, "seed_index": i,
                     "m": m, "b": b, "k_healthy_needed": k, "threshold": t,
                     "fused_alert_rate": float(alert.mean()), "event_rate": float(event.mean()),
                     "bound": float(rates.sum() / k),
                     "healthy_rates": ";".join(f"{x}={r:.6f}" for x, r in zip(healthy, rates)),
                     "alert_without_event": int((alert & ~event).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ADBench

def adbench_table(budgets=metrics.ADBENCH_BUDGETS) -> pd.DataFrame:
    """Every triple x four configurations x every split, from the E1/E2 score cache."""
    rows = []
    for name in frozen.load()["data"]["adbench"]["datasets"]:
        X, y = data.load_adbench(name)
        for seed in data.ADBENCH_SEEDS:
            split = data.preprocess(data.split_adbench(name, seed, X, y))
            if not all(scorecache.cache_path(name, seed, d).exists() for d in detectors.DETECTORS):
                raise SystemExit(f"{name} seed {seed}: score cache missing (python -m thesis.e1)")
            sc = {d: scorecache.get_scores(split, d) for d in detectors.DETECTORS}
            yt = split.y_test
            single = {(d, b): metrics.budget_metrics(sc[d]["test"], yt, b)["fp_at_k"]
                      for d in detectors.DETECTORS for b in budgets}
            for members in combinations(detectors.DETECTORS, 3):
                S_tr = np.column_stack([sc[d]["train"] for d in members])
                S_te = np.column_stack([sc[d]["test"] for d in members])
                for calib, agg in CONFIGS:
                    _, f = fusion.fuse(S_tr, S_te, calib, agg, members, name)
                    for b in budgets:
                        fps = [single[d, b] for d in detectors.DETECTORS]
                        fp = metrics.budget_metrics(f, yt, b)["fp_at_k"]
                        rows.append({"dataset": name, "seed": seed, "budget": b,
                                     "combo": "+".join(members), "calib": calib, "agg": agg,
                                     "fp_at_k": fp, "beats_best_single": fp < min(fps),
                                     "beats_median_single": fp < float(np.median(fps))})
    return pd.DataFrame(rows)


def adbench_summary(cells: pd.DataFrame) -> pd.DataFrame:
    g = cells.groupby(["budget", "calib", "agg"])
    t = g.agg(mean_fp=("fp_at_k", "mean"), share_beats_best=("beats_best_single", "mean"),
              share_beats_median=("beats_median_single", "mean"), n_cells=("fp_at_k", "size")).reset_index()
    eb = cells[cells.combo == "IForest+PCA+KNN"].groupby(["budget", "calib", "agg"]).fp_at_k.mean()
    t["mean_fp_ebench"] = [eb.loc[(r.budget, r.calib, r.agg)] for r in t.itertuples()]
    return t


# ---------------------------------------------------------------- README reference values

README_BUDGET = {   # method -> FP@K at 0.1 / 0.2 / 0.5 %
    "HBOS": (94, 214, 582), "LOF": (141, 281, 701),
    "E-bench-5%:minmax+mean": (131.4, 268.4, 657.6), "E-bench-5%:percentile+median": (122.9, 240.9, 604.6),
    "E-all:minmax+mean": (107.2, 231.8, 601.0), "E-all:percentile+median": (103.6, 233.2, 606.0),
    "E-bench:minmax+mean": (111.0, 238.0, 623.4), "E-bench:percentile+median": (112.6, 241.4, 620.2)}
README_THRESHOLD = {   # method -> (alert rate %, FP, TP) at q = 0.998
    "LOF": (8.22, 11480, 30), "HBOS": (0.164, 173, 57),
    "E-bench-5%:minmax+mean": (0.473, 622, 40), "E-bench-5%:minmax+median": (1.678, 2309, 41),
    "E-bench-5%:percentile+mean": (0.313, 403, 35), "E-bench-5%:percentile+median": (0.219, 260, 47),
    "E-all:minmax+mean": (0.180, 206, 47), "E-all:percentile+mean": (0.274, 355, 29),
    "E-all:percentile+median": (0.179, 206, 45)}
README_ADBENCH = {   # (budget, calib, agg) -> mean FP over all cells
    (0.02, "minmax", "mean"): 48.9, (0.02, "percentile", "mean"): 53.8, (0.02, "percentile", "median"): 51.0,
    (0.05, "minmax", "mean"): 131.5, (0.05, "percentile", "mean"): 155.1, (0.05, "percentile", "median"): 139.2}
README_ADBENCH_EBENCH = {(0.02, "minmax", "mean"): 47.7, (0.02, "percentile", "median"): 48.4}


def _ndigits(v) -> int:
    s = repr(float(v)).rstrip("0").rstrip(".")
    return len(s.split(".")[1]) if "." in s else 0


def readme_differences(budget: pd.DataFrame, thr: pd.DataFrame, adb: pd.DataFrame | None) -> list[str]:
    """Every README value against the rebuilt one, after rounding to the README's precision."""
    bad = []
    bt = budget.set_index(["method", "budget"]).fp_at_k
    for m, vals in README_BUDGET.items():
        for b, v in zip((0.001, 0.002, 0.005), vals):
            got = round(bt[(m, b)], max(_ndigits(v), 1 if isinstance(v, float) else 0))
            if abs(got - v) > 1e-9 and round(bt[(m, b)]) != v:
                bad.append(f"budget {m} {b}: README {v}, rebuilt {bt[(m, b)]:.3f}")
    tt = thr[thr.q == 0.998].set_index("method")
    for m, (rate, fp, tp) in README_THRESHOLD.items():
        r = tt.loc[m]
        if round(r.alert_rate * 100, _ndigits(rate)) != rate or round(r.fp) != fp or round(r.tp) != tp:
            bad.append(f"threshold {m}: README {rate}% {fp} {tp}, rebuilt {r.alert_rate * 100:.4f}% {r.fp:.1f} {r.tp:.1f}")
    if adb is not None:
        a = adb.set_index(["budget", "calib", "agg"])
        for key, v in README_ADBENCH.items():
            if round(a.loc[key, "mean_fp"], 1) != v:
                bad.append(f"ADBench {key}: README {v}, rebuilt {a.loc[key, 'mean_fp']:.3f}")
        for key, v in README_ADBENCH_EBENCH.items():
            if round(a.loc[key, "mean_fp_ebench"], 1) != v:
                bad.append(f"ADBench E-bench {key}: README {v}, rebuilt {a.loc[key, 'mean_fp_ebench']:.3f}")
    return bad


# ---------------------------------------------------------------- main

def ulb_inputs(cfg: dict):
    split = load_split()
    base = fit_stage(split, cfg)                 # blinded split, cached E3 scores
    return base, split.y_test


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="python -m thesis.robust_fusion")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    state = git_state()
    if state["dirty"] and not args.quick:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    t0 = time.perf_counter()
    cfg = frozen.load()
    ensembles = distinct_ensembles(cfg)
    base, y = ulb_inputs(cfg)
    pool = list(base.singles)
    meth = methods(base.train, base.test, pool, ensembles)
    budget = budget_table(meth, y, cfg["budgets"]["ulb"]["values"])
    thr = threshold_table(meth, y)
    bounds = pd.concat([bound_check(base.train, base.test, ensembles[e], ["LOF"], q)
                        for e in ("E-bench-5%", "E-all") for q in ULB_Q], ignore_index=True)
    adb_cells = adb = None
    if not args.quick:
        adb_cells = adbench_table()
        adb = adbench_summary(adb_cells)
    diff = readme_differences(budget, thr, adb)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(budget.pivot(index="method", columns="budget", values="fp_at_k").round(1).to_string())
    print(thr[thr.q == 0.998].round(5).to_string(index=False))
    print(bounds.groupby(["members", "q"])[["fused_alert_rate", "event_rate", "bound",
                                            "alert_without_event"]].mean().to_string())
    if adb is not None:
        print(adb.round(3).to_string(index=False))
    print("README differences:", diff or "none")
    if args.quick:
        return
    if diff:
        raise SystemExit("rebuilt tables differ from the README; nothing written")
    OUT.mkdir(parents=True, exist_ok=True)
    c = state["commit"]
    budget.assign(commit=c).to_csv(OUT / "ulb_budget.csv", index=False)
    thr.assign(commit=c).to_csv(OUT / "ulb_threshold.csv", index=False)
    bounds.assign(commit=c).to_csv(OUT / "ulb_bound_check.csv", index=False)
    adb.assign(commit=c).to_csv(OUT / "adbench_static.csv", index=False)
    adb_cells.assign(commit=c).to_csv(OUT / "adbench_static_cells.csv", index=False)
    meta = {**state, "command": "python -m thesis.robust_fusion",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "configs": [config_name(*x) for x in CONFIGS], "robust": config_name(*ROBUST),
            "ulb_quantiles": list(ULB_Q), "ulb_budgets": cfg["budgets"]["ulb"]["values"],
            "adbench_budgets": list(metrics.ADBENCH_BUDGETS), "score_cache": scorecache.cache_key(),
            "labels_read": "ULB day-2 and ADBench test labels for FP@K / FP / TP only; calibrators "
                           "and thresholds from training scores only; the bound check is label-free",
            "readme": "docs/notes/robust_fusion_check_2026-11-17/README.md: every value matches "
                      "after rounding to the README's precision",
            "note": "Development data (ULB, ADBench). Developed on ULB. freeze-v1 fusion configs unchanged.",
            "runtime_s": round(time.perf_counter() - t0, 1)}
    (OUT / "robust_fusion.meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
