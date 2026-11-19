"""E-stress and the run-time guard on ULB (development data for RQ4).

    python -m thesis.e_stress            # full run (1000 resamples), writes results/tables/e3_ulb/stress/
    python -m thesis.e_stress --quick    # 20 resamples, prints the tables, writes nothing

Two mechanisms leave members out of a frozen ensemble (fusion stays min-max + mean):
- pre-deployment check: r_m from the half-day split inside day 1 (fit on the first 12 h, score
  the last 12 h; thesis.lof_diag), computed before any day-2 row is seen;
- run-time guard: r_m on day 2 at the member's frozen day-1 threshold, from unlabelled day-2
  scores (the decision for a day uses that day's own scores, as daily top-K review does).
r_m = alert rate / (1 - q) at q = RULE_Q. Candidate rules: one-sided (r <= c) or two-sided
(1/c <= r <= c), on the absolute r or on r relative to the median member, for c in CS. If no
member of an ensemble passes, the static ensemble is used (fallback).

No rule is chosen here. Every candidate is evaluated, and the table shows what each would do.
Member decisions never see a label: they use day-1 features (pre-deployment) or unlabelled
day-2 scores (guard). Day-2 labels are used only for FP@K and the threshold-mode metrics, as in
E3. ULB is development data for the stress test, and this output is reported as such.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from thesis import bootstrap, data, detectors, frozen, fusion, lof_diag
from thesis.e3_ulb import (Fitted, budget_runs, distinct_ensembles, fit_stage, load_split,
                           threshold_table)
from thesis.export_datasets import git_state
from thesis.paths import TABLES

OUT = TABLES / "e3_ulb" / "stress"
RULE_Q = 0.998                     # the frozen ULB threshold quantile of the 0.2% primary budget
CS = (2, 3, 5)
SIDES = ("one-sided", "two-sided")
BASES = ("absolute", "relative")
MECHANISMS = ("pre-deployment", "guard")
QUICK_RESAMPLES = 20
CALIB, AGG = "minmax", "mean"      # frozen primary fusion


# ---------------------------------------------------------------- rules (label-free)

def rules() -> list[dict]:
    return [{"rule": f"{side} {basis} c={c}", "side": side, "basis": basis, "c": c}
            for side in SIDES for basis in BASES for c in CS]


def passes(value: float, side: str, c: float) -> bool:
    """One-sided: value <= c. Two-sided: 1/c <= value <= c."""
    if side == "one-sided":
        return bool(value <= c)
    if side == "two-sided":
        return bool(1 / c <= value <= c)
    raise ValueError(f"unknown side {side!r}")


def passing(r_tab: pd.DataFrame, rule: dict) -> set[str]:
    """Members that pass `rule`, from a lof_diag.stress_table at RULE_Q (one row per member)."""
    col = "r" if rule["basis"] == "absolute" else "r_over_median"
    return {row.member for row in r_tab.itertuples()
            if passes(getattr(row, col), rule["side"], rule["c"])}


def kept_members(members: list[str], passed: set[str]) -> tuple[list[str], bool]:
    """Members of the ensemble that pass, in ensemble order; all members if none pass."""
    kept = [m for m in members if m in passed]
    return (kept, False) if kept else (list(members), True)


def decisions(r_tabs: dict[str, pd.DataFrame], ensembles: dict[str, list[str]]) -> pd.DataFrame:
    """One row per mechanism x rule x ensemble: kept members, dropped members, fallback."""
    rows = []
    for mech, tab in r_tabs.items():
        for rule in rules():
            passed = passing(tab, rule)
            for ens, members in ensembles.items():
                kept, fallback = kept_members(members, passed)
                rows.append({"mechanism": mech, **rule, "ensemble": ens,
                             "kept": "+".join(kept), "dropped": "+".join(m for m in members
                                                                          if m not in kept),
                             "n_kept": len(kept), "fallback": fallback,
                             "changed": kept != list(members)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- fused scores

def set_name(kept: list[str]) -> str:
    """Method key of a member set (no ':' so it reads as a single method in e3_ulb tables)."""
    return "{" + "+".join(kept) + "}"


def fuse_members(train: dict[str, np.ndarray], test: dict[str, np.ndarray],
                 kept: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """min-max + mean of `kept`, one row per IForest seed, seeds paired as in e3_ulb.fit_stage."""
    n_seeds = max(len(train[m]) for m in kept)
    tr, te = [], []
    for i in range(n_seeds):
        pick = [min(i, len(train[m]) - 1) for m in kept]
        S_tr = np.column_stack([train[m][j] for m, j in zip(kept, pick)])
        S_te = np.column_stack([test[m][j] for m, j in zip(kept, pick)])
        f_tr, f_te = fusion.fuse(S_tr, S_te, CALIB, AGG, kept, "ulb")
        tr.append(f_tr)
        te.append(f_te)
    return np.vstack(tr), np.vstack(te)


def variant_fitted(base: Fitted, dec: pd.DataFrame, ensembles: dict[str, list[str]]) -> Fitted:
    """The six singles (plain names, as references), every static ensemble and every member set
    that some rule produces, as one Fitted."""
    train = {m: base.train[m] for m in base.singles}
    test = {m: base.test[m] for m in base.singles}
    sets = {tuple(ensembles[e]) for e in ensembles} | {tuple(k.split("+")) for k in dec.kept}
    for kept in sorted(sets, key=lambda s: (len(s), s)):
        key = set_name(list(kept))
        train[key], test[key] = fuse_members(base.train, base.test, list(kept))
    return Fitted(train, test, base.singles, base.runtime_s)


# ---------------------------------------------------------------- evaluation (labels)

def comparisons(dec: pd.DataFrame, ensembles: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Each changed member set against the static ensemble it came from."""
    pairs = []
    for r in dec[dec.changed].itertuples():
        pair = (set_name(r.kept.split("+")), set_name(ensembles[r.ensemble]))
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def deltas(fitted: Fitted, y: np.ndarray, budgets, pairs, resamples: int, seed: int,
           ci: float) -> pd.DataFrame:
    """Paired transaction-level bootstrap (frozen ULB settings) of dFP@K per pair and budget."""
    scores = {**{m: fitted.test[m] for p in pairs for m in p},
              **{s: fitted.test[s] for s in API_SINGLES}}
    rows = []
    for b in budgets:
        stat = bootstrap.budget_statistic(scores, y, b, list(API_SINGLES), pairs)
        point = stat(bootstrap.transaction_windows(len(y)))
        boot = bootstrap.run(lambda rng: bootstrap.draw_transactions(rng, len(y)), stat,
                             resamples=resamples, seed=seed)
        summ = bootstrap.summary(point, boot, ci).set_index("stat")
        for a, c in pairs:
            d, d100 = summ.loc[f"d:{a}-{c}"], summ.loc[f"d100:{a}-{c}"]
            rows.append({"budget": b, "variant": a, "static": c, "d_fp": d.estimate,
                         "d_fp_ci_low": d.ci_low, "d_fp_ci_high": d.ci_high,
                         "d_fp_per_100": d100.estimate, "d_fp_per_100_ci_low": d100.ci_low,
                         "d_fp_per_100_ci_high": d100.ci_high,
                         "ci_entirely_below_0": bool(d.ci_high < 0),
                         "ci_entirely_above_0": bool(d.ci_low > 0)})
    return pd.DataFrame(rows)


# bootstrap.budget_statistic needs two single detectors in its input; their median and oracle
# rows are not used here. Two one-seed detectors keep the resampling cheap.
API_SINGLES = ("PCA", "ECOD")


def budget_summary(fitted: Fitted, y: np.ndarray, budgets) -> pd.DataFrame:
    runs = budget_runs(fitted, y, budgets)
    cols = ["fp_at_k", "tp_at_k", "precision_at_k", "recall_at_k", "pr_auc", "roc_auc"]
    t = runs.groupby(["method", "budget", "k"], sort=False)[cols].mean().reset_index()
    t["n_seeds"] = runs.groupby(["method", "budget", "k"], sort=False).size().to_numpy()
    t["fp_per_100_alerts"] = t.fp_at_k / t.k * 100
    return t


# ---------------------------------------------------------------- main

def r_tables(t: np.ndarray, X: np.ndarray, base: Fitted, pool, seeds) -> dict[str, pd.DataFrame]:
    """Label-free r at RULE_Q: pre-deployment from the half-day split, guard from day 2."""
    fit, late = lof_diag.control_slices(t, "L2b")
    pre = lof_diag.stress_table(lof_diag.slice_scores(X[fit], X[late], pool, seeds), [RULE_Q],
                                "pre-deployment (day 1: 0-12 h -> 12-24 h)")
    guard = lof_diag.stress_table({m: (base.train[m], base.test[m]) for m in pool}, [RULE_Q],
                                  "guard (day 1 threshold -> day 2)")
    return {"pre-deployment": pre, "guard": guard}


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="python -m thesis.e_stress")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    state = git_state()
    if state["dirty"] and not args.quick:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    t0 = time.perf_counter()
    cfg = frozen.load()
    pool, seeds = cfg["detectors"]["pool"], cfg["detectors"]["iforest_seeds"]
    budgets, quantiles = cfg["budgets"]["ulb"]["values"], cfg["thresholds"]["ulb"]["training_score_quantiles"]
    boot_cfg = cfg["statistics"]["bootstrap"]
    ensembles = distinct_ensembles(cfg)

    split = load_split()
    base = fit_stage(split, cfg)                              # blinded split, cached scores
    X, t = data.load_ulb_unlabelled()
    assert np.array_equal(X[split.idx_train], split.X_train)  # same rows, no label column
    r_tabs = r_tables(t, X, base, pool, seeds)
    dec = decisions(r_tabs, ensembles)
    fitted = variant_fitted(base, dec, ensembles)
    y = split.y_test                                          # labels: evaluation only
    budget = budget_summary(fitted, y, budgets)
    thr = threshold_table(fitted, y, quantiles)
    pairs = comparisons(dec, ensembles)
    resamples = QUICK_RESAMPLES if args.quick else boot_cfg["resamples"]
    dl = deltas(fitted, y, budgets, pairs, resamples, boot_cfg["seed"], 0.95)

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
    r_all = pd.concat(r_tabs.values(), ignore_index=True)
    print(r_all[["run", "member", "r", "r_over_median"]].round(3).to_string(index=False))
    print(dec[dec.changed][["mechanism", "rule", "ensemble", "kept", "dropped", "fallback"]]
          .to_string(index=False))
    print(budget[budget.budget == cfg["budgets"]["ulb"]["primary"]]
          [["method", "fp_at_k", "fp_per_100_alerts", "n_seeds"]].round(2).to_string(index=False))
    print(dl.round(2).to_string(index=False))
    if args.quick:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    c = state["commit"]
    r_all.assign(commit=c).to_csv(OUT / "stress_r.csv", index=False)
    dec.assign(commit=c).to_csv(OUT / "stress_decisions.csv", index=False)
    budget.assign(commit=c).to_csv(OUT / "stress_budget.csv", index=False)
    thr.assign(commit=c).to_csv(OUT / "stress_threshold.csv", index=False)
    dl.assign(commit=c).to_csv(OUT / "stress_deltas.csv", index=False)
    meta = {**state, "command": "python -m thesis.e_stress",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "labels_read": "day-2 labels for FP@K and threshold-mode metrics only; member "
                           "decisions are label-free",
            "rule_q": RULE_Q, "c_values": list(CS), "sides": list(SIDES), "bases": list(BASES),
            "fallback": "static ensemble if no member passes", "fusion": f"{CALIB}+{AGG}",
            "budgets": budgets, "quantiles": quantiles, "resamples": resamples,
            "bootstrap_seed": boot_cfg["seed"], "bootstrap": boot_cfg["ulb"],
            "member_sets": list(fitted.test), "comparisons": [list(p) for p in pairs],
            "note": "ULB is development data for the stress test. No rule is chosen here.",
            "runtime_s": round(time.perf_counter() - t0, 1)}
    (OUT / "stress.meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
