"""Guarded ensembles against the best and the median single member on ULB (development data).

    python -m thesis.guard_ulb            # 1000 resamples per scheme, writes results/tables/e3_ulb/stress/
    python -m thesis.guard_ulb --quick    # 20 resamples, prints, writes nothing

Links RQ4 to RQ1: is a guarded ensemble (thesis.guard, the rule proposed for freeze-v3:
one-sided, absolute r at q = 1 - budget, c = 3, static fallback below two passing members)
still better than the singles? Comparators at every ULB budget:
- median_single and oracle_best_single (day-2 labels, an upper bound), as in E3;
- chosen_single: the single with the lowest FP@K on the last 12 h of day 1, with every member
  fitted on the first 12 h (thesis.lof_diag L2b). This reads day-1 labels; it mirrors the
  proposed H4e comparator (best single chosen on month 1). The chosen member is then scored
  as in E3 (fitted on day 1, scored on day 2).
Two bootstrap schemes, the same 1000 resamples for every method: the frozen ULB scheme
(transaction-level) and an hour-block scheme (the 24 hours of day 2 drawn with replacement,
each with all its rows). A day-block CI is impossible on ULB: day 2 is a single day.
ULB is development data for RQ4; nothing here is a pre-registered result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from thesis import bootstrap, data, frozen, guard, lof_diag, metrics
from thesis.e3_ulb import distinct_ensembles, fit_stage, load_split, method_key
from thesis.export_datasets import git_state
from thesis.paths import TABLES

OUT = TABLES / "e3_ulb" / "stress"
QUICK_RESAMPLES = 20
HOUR = 3600
CHOSEN = "chosen_single"


def hours(t_test: np.ndarray) -> np.ndarray:
    """Hour of day 2 (0-23) from Time; the block of the hour-block scheme."""
    h = np.floor((np.asarray(t_test) - data.ULB_DAY_SECONDS) / HOUR).astype(int)
    if h.min() < 0 or h.max() > 23:
        raise ValueError("rows outside day 2")
    return h


def seed_mean_fp(s: np.ndarray, y: np.ndarray, budget) -> float:
    k = metrics.budget_k(budget, len(y))
    return float(np.mean([k - metrics.tp_at_k(r, y, k) for r in np.atleast_2d(s)]))


def chosen_single(slice_scores: dict, y_late: np.ndarray, budget, pool) -> tuple[str, pd.DataFrame]:
    """The member with the lowest FP@K on the scored half of day 1 (pool order breaks ties)."""
    tab = pd.DataFrame([{"budget": budget, "member": m,
                         "fp_at_k_day1_late": seed_mean_fp(slice_scores[m][1], y_late, budget)}
                        for m in pool])
    best = tab.loc[tab["fp_at_k_day1_late"].idxmin(), "member"]
    return best, tab.assign(chosen=tab["member"] == best)


def comparisons(ensembles, chosen: str) -> list[tuple[str, str]]:
    """Static and guarded version of every ensemble against the median, the oracle and the
    chosen single, and guarded against static."""
    pairs = []
    for e in ensembles:
        for kind in ("static", "guarded"):
            for comp in (bootstrap.MEDIAN, bootstrap.ORACLE, chosen):
                pairs.append((f"{kind}:{e}", comp))
        pairs.append((f"guarded:{e}", f"static:{e}"))
    return pairs


def deltas(scores: dict, y: np.ndarray, budget, singles, pairs, draws: dict, resamples: int,
           seed: int, chosen: str) -> pd.DataFrame:
    """Paired bootstrap of dFP@K for every pair, once per scheme (same seed for each scheme)."""
    stat = bootstrap.budget_statistic(scores, y, budget, singles, pairs)
    point = stat(bootstrap.transaction_windows(len(y)))
    rows = []
    for scheme, draw in draws.items():
        boot = bootstrap.run(draw, stat, resamples=resamples, seed=seed)
        summ = bootstrap.summary(point, boot, bootstrap.CI_LEVEL).set_index("stat")
        for a, c in pairs:
            d, d100 = summ.loc[f"d:{a}-{c}"], summ.loc[f"d100:{a}-{c}"]
            rows.append({"budget": budget, "scheme": scheme, "method": a,
                         "comparator": CHOSEN if c == chosen else c, "comparator_member": c,
                         "fp_method": point[f"fp:{a}"],
                         "fp_comparator": point[f"fp:{bootstrap.comparator(c)}"],
                         "d_fp": d.estimate, "d_fp_ci_low": d.ci_low, "d_fp_ci_high": d.ci_high,
                         "d_fp_per_100": d100.estimate, "d_fp_per_100_ci_low": d100.ci_low,
                         "d_fp_per_100_ci_high": d100.ci_high,
                         "ci_entirely_below_0": bool(d.ci_high < 0),
                         "ci_entirely_above_0": bool(d.ci_low > 0)})
    return pd.DataFrame(rows)


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="python -m thesis.guard_ulb")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    state = git_state()
    if state["dirty"] and not args.quick:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    t0 = time.perf_counter()
    cfg = frozen.load()
    pool, seeds = cfg["detectors"]["pool"], cfg["detectors"]["iforest_seeds"]
    budgets = cfg["budgets"]["ulb"]["values"]
    boot_cfg = cfg["statistics"]["bootstrap"]
    ensembles = distinct_ensembles(cfg)

    split = load_split()
    base = fit_stage(split, cfg)                            # blinded split, cached E3 scores
    one_day = np.zeros(len(split.y_test), dtype=int)
    X, y_all, t = data.load_ulb()
    assert np.array_equal(X[split.idx_train], split.X_train)
    fit, late = lof_diag.control_slices(t, "L2b")
    assert t[late].max() < data.ULB_DAY_SECONDS             # day 1 only
    sl = lof_diag.slice_scores(X[fit], X[late], pool, seeds)
    y = split.y_test                                        # day-2 labels: evaluation only
    days = bootstrap.day_rows(hours(split.t_test), one_day, [0])
    resamples = QUICK_RESAMPLES if args.quick else boot_cfg["resamples"]
    draws = {"transaction": lambda rng: bootstrap.draw_transactions(rng, len(y)),
             "hour-block": lambda rng: bootstrap.draw_days(rng, days)}

    dl, dec_all, chosen_tabs, md_all = [], [], [], []
    for b in budgets:
        md, gscores, dec = guard.guarded(base.train, base.test, one_day, ensembles, b)
        md_all.append(md.assign(budget=b))
        dec_all.append(dec.assign(budget=b))
        chosen, ctab = chosen_single(sl, y_all[late], b, pool)
        chosen_tabs.append(ctab)
        scores = {m: base.test[m] for m in pool}
        for e in ensembles:
            scores[f"static:{e}"] = base.test[method_key(e, guard.CALIB, guard.AGG)]
            scores[f"guarded:{e}"] = gscores[e]
        dl.append(deltas(scores, y, b, pool, comparisons(ensembles, chosen), draws, resamples,
                         boot_cfg["seed"], chosen))
        print(f"budget {b}: done ({time.perf_counter() - t0:.0f} s)", flush=True)
    dl, dec_all = pd.concat(dl, ignore_index=True), pd.concat(dec_all, ignore_index=True)
    chosen_tab = pd.concat(chosen_tabs, ignore_index=True)
    md_all = pd.concat(md_all, ignore_index=True)

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 300)
    print(md_all[["budget", "member", "alerts", "r", "passes"]].round(3).to_string(index=False))
    print(dec_all[["budget", "ensemble", "kept", "dropped", "fallback"]].to_string(index=False))
    print(chosen_tab.round(2).to_string(index=False))
    print(dl[["budget", "scheme", "method", "comparator", "comparator_member", "fp_method",
              "fp_comparator", "d_fp", "d_fp_ci_low", "d_fp_ci_high"]].round(1).to_string(index=False))
    print(f"runtime {time.perf_counter() - t0:.0f} s")
    if args.quick:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    c = state["commit"]
    dl.assign(commit=c).to_csv(OUT / "guard_vs_singles.csv", index=False)
    dec_all.assign(commit=c).to_csv(OUT / "guard_vs_singles_decisions.csv", index=False)
    md_all.assign(commit=c).to_csv(OUT / "guard_vs_singles_member_r.csv", index=False)
    chosen_tab.assign(commit=c).to_csv(OUT / "guard_vs_singles_chosen.csv", index=False)
    meta = {**state, "command": "python -m thesis.guard_ulb",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "guard": {"side": "one-sided", "basis": "absolute", "q": "1 - budget", "c": guard.C,
                      "ties": "r = c passes; exact on integer alert counts (fractions.Fraction)",
                      "fallback": f"static ensemble if fewer than {guard.MIN_MEMBERS} members pass",
                      "fusion": f"{guard.CALIB}+{guard.AGG}"},
            "labels_read": "day-2 labels for FP@K; day-1 labels (12-24 h) only to choose "
                           "chosen_single; member decisions are label-free",
            "chosen_single": "lowest FP@K on day 1, 12-24 h, members fitted on day 1, 0-12 h "
                             "(lof_diag L2b); pool order breaks ties",
            "schemes": {"transaction": boot_cfg["ulb"],
                        "hour-block": "the 24 hours of day 2 drawn with replacement, each with "
                                      "all its rows; K recomputed from the resample size"},
            "budgets": budgets, "resamples": resamples, "bootstrap_seed": boot_cfg["seed"],
            "note": "ULB is development data for RQ4. A day-block CI is impossible on ULB (one "
                    "test day); the hour-block scheme is the block analogue.",
            "runtime_s": round(time.perf_counter() - t0, 1)}
    (OUT / "guard_vs_singles.meta.json").write_text(json.dumps(meta, indent=1) + "\n",
                                                    encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
