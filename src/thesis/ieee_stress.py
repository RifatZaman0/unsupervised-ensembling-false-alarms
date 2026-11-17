"""Temporal stress test on IEEE-CIS month 1 (training window only, label-free).

    python -m thesis.ieee_stress    # writes results/tables/ieee_cis/stress_month1{,_rules,
                                    # _alert_days,_feature_tails}.csv, .meta.json and .md

Fit window: month 1, days 0-20 (weeks 1-3). Scored slice: days 21-29 (the rest of month 1,
"week 4"); days 21-27 (exactly seven days) are reported as a sensitivity check. Only month-1
rows are loaded (ieee_features.load_month1: the isFraud column and the rows of months 2-6 are
never read). Everything that is fitted (feature medians and frequency maps, the z-score,
every detector, the thresholds) is fitted on the fit window only, as freeze-v2 states.

For member m and frozen IEEE-CIS quantile q:
    r_m = (share of scored rows above m's fit-window threshold) / (1 - q)
reported as an absolute value and relative to the median member, with pass/fail under a
one-sided rule (r <= c) and a two-sided rule (1/c <= r <= c) for several c. No c is chosen
here; that belongs to freeze-v3. Development output, not part of a frozen evaluation.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from thesis import config, data, detectors, frozen, ieee_features
from thesis.export_datasets import git_state
from thesis.lof_diag import slice_scores, stress_table
from thesis.paths import TABLES

OUT = TABLES / "ieee_cis"
FIT_LAST_DAY = 20                         # weeks 1-3 = days 0-20
SCORED_DAYS = (21, 29)                    # main split: the rest of month 1
SENSITIVITY_DAYS = (21, 27)               # exactly seven days
CS = (2, 3, 5)                            # candidate c values, reported, not chosen


def split_days(day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of the fit window (days 0-20) and the scored slice (days 21-29).
    Refuses any row outside month 1."""
    day = np.asarray(day)
    if day.min() < 0 or day.max() >= data.IEEE_MONTH_DAYS:
        raise ValueError("rows outside month 1 (days 0-29)")
    fit = np.flatnonzero(day <= FIT_LAST_DAY)
    scored = np.flatnonzero((day >= SCORED_DAYS[0]) & (day <= SCORED_DAYS[1]))
    assert day[fit].max() < day[scored].min()
    return fit, scored


def features_for(month1: pd.DataFrame, fit: np.ndarray, scored: np.ndarray,
                 names) -> tuple[np.ndarray, np.ndarray, ieee_features.FeatureRule]:
    """freeze-v2 features: the rule's parameters come from the fit-window rows only."""
    rule = ieee_features.FeatureRule.fit(month1.iloc[fit], names)
    return rule.transform(month1.iloc[fit]), rule.transform(month1.iloc[scored]), rule


def rule_table(tab: pd.DataFrame, cs=CS) -> pd.DataFrame:
    """Pass/fail per member, q and c: one-sided r <= c, two-sided 1/c <= r <= c, applied to
    the absolute r and to r relative to the median member."""
    rows = []
    for r in tab.itertuples():
        for c in cs:
            for basis, value in (("absolute", r.r), ("relative to median", r.r_over_median)):
                rows.append({"split": r.run, "member": r.member, "q": r.q, "c": c,
                             "basis": basis, "value": value,
                             "one_sided_pass": bool(value <= c),
                             "two_sided_pass": bool(1 / c <= value <= c)})
    return pd.DataFrame(rows)


def alert_days(scores, day_fit: np.ndarray, day_scored: np.ndarray, quantiles) -> pd.DataFrame:
    """Alerts per day at each member's fit-window threshold, for fit-window and scored rows
    (IForest: mean over seeds). Shows whether a threshold is set by a burst on a few days."""
    rows = []
    for m, (s_fit, s_sc) in scores.items():
        for q in quantiles:
            thr = np.quantile(s_fit, q, axis=1, method="linear")
            for part, s, d in (("fit", s_fit, day_fit), ("scored", s_sc, day_scored)):
                hits = s > thr[:, None]
                for dd in np.unique(d):
                    rows.append({"member": m, "q": q, "part": part, "day": int(dd),
                                 "rows": int((d == dd).sum()),
                                 "alerts": float(hits[:, d == dd].sum(axis=1).mean())})
    return pd.DataFrame(rows)


def feature_tails(X_fit: np.ndarray, X_scored: np.ndarray, names) -> pd.DataFrame:
    """Per feature (after the freeze-v2 transform): share of rows outside the fit-window 0.5%
    and 99.5% quantiles, the scored-slice mean shift and the SD ratio, in fit-window SD."""
    sd = np.where(X_fit.std(axis=0) > 0, X_fit.std(axis=0), 1.0)
    rows = []
    for j, n in enumerate(names):
        lo, hi = np.quantile(X_fit[:, j], [0.005, 0.995])
        rows.append({"feature": n,
                     "fit_outside_share": float(((X_fit[:, j] < lo) | (X_fit[:, j] > hi)).mean()),
                     "scored_outside_share": float(((X_scored[:, j] < lo) | (X_scored[:, j] > hi)).mean()),
                     "mean_shift_sd": float((X_scored[:, j].mean() - X_fit[:, j].mean()) / sd[j]),
                     "sd_ratio": float(X_scored[:, j].std() / sd[j])})
    return pd.DataFrame(rows)


def summary_md(tab: pd.DataFrame, rules: pd.DataFrame, meta: dict) -> str:
    lines = ["# IEEE-CIS month 1: temporal stress test (label-free, training window only)", "",
             f"Commit `{meta['commit']}` · `{meta['command']}` · created {meta['created_utc']}", "",
             f"Fit window days 0-{FIT_LAST_DAY} ({meta['n_fit']:,} rows); scored slice days "
             f"{SCORED_DAYS[0]}-{SCORED_DAYS[1]} ({meta['n_scored']:,} rows); sensitivity: days "
             f"{SENSITIVITY_DAYS[0]}-{SENSITIVITY_DAYS[1]} ({meta['n_scored_sensitivity']:,} rows). "
             "No label read; no row of months 2-6 loaded. Feature parameters, z-score, detectors "
             "and thresholds fitted on the fit window only. Development output; c is not chosen.", ""]
    for split in tab.run.unique():
        lines += [f"## {split}", "", "| member | q | alert rate | r | r / median r |",
                  "|---|---|---|---|---|"]
        for r in tab[tab.run == split].itertuples():
            rng = f" ({r.r_min:.2f}-{r.r_max:.2f} over seeds)" if r.n_seeds > 1 else ""
            lines.append(f"| {r.member} | {r.q:g} | {100 * r.alert_rate:.3f}% | {r.r:.2f}{rng} "
                         f"| {r.r_over_median:.2f} |")
        lines.append("")
    main_split = tab.run.iloc[0]
    lines += [f"## Rules on {main_split} (pass = P, fail = F)", "",
              "| member | q | basis | " + " | ".join(f"one-sided c={c} | two-sided c={c}" for c in CS) + " |",
              "|---|---|---|" + "---|" * (2 * len(CS))]
    sub = rules[rules.split == main_split]
    for (m, q, basis), g in sub.groupby(["member", "q", "basis"], sort=False):
        cells = []
        for c in CS:
            row = g[g.c == c].iloc[0]
            cells += ["P" if row.one_sided_pass else "F", "P" if row.two_sided_pass else "F"]
        lines.append(f"| {m} | {q:g} | {basis} | " + " | ".join(cells) + " |")
    lines += ["", "Fit-window and scored alerts per day: `stress_month1_alert_days.csv`. Per-feature "
              "tails, mean shift and SD ratio: `stress_month1_feature_tails.csv`."]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> None:
    if argv:
        raise SystemExit("usage: python -m thesis.ieee_stress")
    state = git_state()
    if state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    t0 = time.perf_counter()
    cfg = frozen.load()
    pool, seeds = cfg["detectors"]["pool"], cfg["detectors"]["iforest_seeds"]
    quantiles = cfg["thresholds"]["ieee_cis"]["training_score_quantiles"]
    names = [f["name"] for f in ieee_features.load_frozen()["features"]]
    month1, info = ieee_features.load_month1()
    assert "isFraud" not in month1.columns
    day = data.ieee_cis_day(month1["TransactionDT"].to_numpy())
    fit, scored = split_days(day)
    X_fit, X_scored, rule = features_for(month1, fit, scored, names)
    t_score = time.perf_counter()
    scores = slice_scores(X_fit, X_scored, pool, seeds)
    score_s = time.perf_counter() - t_score
    seven = (day[scored] >= SENSITIVITY_DAYS[0]) & (day[scored] <= SENSITIVITY_DAYS[1])
    main_name = f"days 0-{FIT_LAST_DAY} -> {SCORED_DAYS[0]}-{SCORED_DAYS[1]}"
    sens_name = f"days 0-{FIT_LAST_DAY} -> {SENSITIVITY_DAYS[0]}-{SENSITIVITY_DAYS[1]}"
    tab = pd.concat([stress_table(scores, quantiles, main_name),
                     stress_table({m: (a, b[:, seven]) for m, (a, b) in scores.items()},
                                  quantiles, sens_name)], ignore_index=True)
    rules = rule_table(tab)
    days = alert_days(scores, day[fit], day[scored], quantiles)
    tails = feature_tails(X_fit, X_scored, names)
    meta = {**state, "command": "python -m thesis.ieee_stress",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "labels_read": False, "rows_read": "month 1 only (ieee_features.load_month1)",
            "features": "freeze-v2 list (frozen_features_v2.json); medians, frequency maps and "
                        "z-score fitted on the fit window", "n_features": len(names),
            "fit_days": [0, FIT_LAST_DAY], "scored_days": list(SCORED_DAYS),
            "sensitivity_days": list(SENSITIVITY_DAYS), "n_fit": int(len(fit)),
            "n_scored": int(len(scored)), "n_scored_sensitivity": int(seven.sum()),
            "quantiles": quantiles, "iforest_seeds": seeds, "c_values_reported": list(CS),
            "knn_lof_fit_sample": {"max_rows": detectors.MAX_FIT_ROWS,
                                   "seed": config.KNN_LOF_SAMPLE_SEED},
            "threshold_rule": "numpy.quantile(fit-window scores, q), method 'linear'; alert if "
                              "score > threshold",
            "r": "alert rate on the scored slice / (1 - q); IForest: mean over seeds",
            "constant_fit_columns": [n for j, n in enumerate(names) if np.ptp(X_fit[:, j]) == 0],
            "scoring_runtime_s": round(score_s, 1), "month1": info,
            "note": "Development output before freeze-v3: label-free, month 1 only. No c is "
                    "chosen here.", "runtime_s": round(time.perf_counter() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    tab.assign(commit=state["commit"]).to_csv(OUT / "stress_month1.csv", index=False)
    rules.assign(commit=state["commit"]).to_csv(OUT / "stress_month1_rules.csv", index=False)
    days.assign(commit=state["commit"]).to_csv(OUT / "stress_month1_alert_days.csv", index=False)
    tails.assign(commit=state["commit"]).to_csv(OUT / "stress_month1_feature_tails.csv", index=False)
    (OUT / "stress_month1.meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    (OUT / "stress_month1.md").write_text(summary_md(tab, rules, meta), encoding="utf-8")
    print((OUT / "stress_month1.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main(sys.argv[1:])
