"""E3 on ULB: the frozen ensembles against the single detectors, day 1 fit, day 2 test, plus the
day-1 -> day-2 threshold check (E4 on ULB).

    python -m thesis.e3_ulb             # full run, writes results/{tables,figures}/e3_ulb/
    python -m thesis.e3_ulb --quick     # 20 resamples, prints the hypotheses, writes nothing

Every choice comes from frozen_config.json (thesis.frozen.load); the run refuses to start if
the file differs from tag freeze-v1 or if the code no longer matches it (check_code_matches).

Fitting stage (no labels, no day-2 row): z-score with day-1 statistics, the six detectors fitted
on day 1 (IForest per frozen seed, KNN/LOF on the frozen fit sample), per-member calibrators on
day-1 training scores, fused training scores, and the quantile thresholds on each method's own
training scores. The scoring split handed to this stage has its labels replaced by zeros
(`blind`), so no label can reach it. Day-2 labels are used only by the metrics.

Budget mode (every frozen ULB budget): FP@K, P@K, R@K, PR-AUC, ROC-AUC for the six singles and
the three distinct ensembles (E-bench = E-lit, E-bench-5%, E-all) x six fusion configs, seed
means; median single, default IForest, oracle-best single (chosen with test labels: an upper
bound) and regret against the oracle. Paired bootstrap CIs (thesis.bootstrap) for every ensemble
x fusion config against median single, default IForest and oracle, and for the H2b secondary
check; H1a, H1b and H2b answered with the frozen criterion.

Threshold mode (every frozen ULB quantile): threshold = training-score quantile, applied
unchanged to day 2; alert count, alert rate vs 1 - q, FPR, recall, precision.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyod

from thesis import bootstrap, config, data, detectors, frozen, fusion, metrics, scorecache
from thesis.export_datasets import git_state
from thesis.export_setup import booktabs
from thesis.paths import FIGURES, REPO, TABLES

OUT = TABLES / "e3_ulb"
FIG_OUT = FIGURES / "e3_ulb"
DATASET = "ulb"
FREEZE_TAG = "freeze-v1"
SEP = ":"                     # method key = "<ensemble>:<calibration>+<aggregation>"
QUICK_RESAMPLES = 20
NULL, SUPPORT = "no difference shown", "fewer false alarms"

ScoreFn = Callable[[data.Split, str], dict]


# ---------------------------------------------------------------- frozen file and code

def frozen_unchanged(tag: str = FREEZE_TAG) -> None:
    """The tag exists and `git diff <tag> -- frozen_config.json` is empty (working tree too)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    if git("rev-parse", "--verify", "--quiet", f"refs/tags/{tag}").returncode != 0:
        raise RuntimeError(f"tag {tag} not found: no fraud-data result before the freeze")
    diff = git("diff", "--quiet", tag, "--", "frozen_config.json")
    if diff.returncode != 0:
        raise RuntimeError(f"frozen_config.json differs from {tag}")


def check_code_matches(cfg: dict) -> None:
    """The settings the code uses are the frozen ones (the code is not configured from prose)."""
    det = cfg["detectors"]
    checks = {
        "detector params": det["params"] == detectors.params(),
        "PyOD version": det["library"] == f"PyOD {pyod.__version__}",
        "KNN/LOF fit sample": det["knn_lof_fit_sample"]["max_rows"] == detectors.MAX_FIT_ROWS,
        "KNN/LOF sample seed": det["knn_lof_fit_sample"]["seed"].endswith(
            f"{config.KNN_LOF_SAMPLE_SEED} when the split has none (ULB, IEEE-CIS)"),
        "default protocol": config.PROTOCOL == config.Protocol(),
        "score cache spec": cfg["score_cache"]["key"] == scorecache.cache_key(),
        "threshold rule": cfg["thresholds"]["common"]["rule"] == THRESHOLD_RULE,
        "k rule": cfg["rules"]["k_rule"] == config.PROTOCOL.k_rule,
        "ULB split": f"Time < {data.ULB_DAY_SECONDS} s" in cfg["data"]["ulb"]["split"],
        "ULB bootstrap": cfg["statistics"]["bootstrap"]["ulb"].startswith("transaction-level"),
        "bootstrap code": cfg["statistics"]["bootstrap"]["code"] == "thesis.bootstrap",
        "metrics": (cfg["metrics"]["budget"] == ["fp_at_k", "precision_at_k", "recall_at_k"]
                    and cfg["metrics"]["ranking"] == ["pr_auc", "roc_auc"]),
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise RuntimeError(f"code does not match frozen_config.json: {bad}")


def ci_level(cfg: dict) -> float:
    m = re.match(r"(\d+(?:\.\d+)?)% percentile interval", cfg["statistics"]["bootstrap"]["ci"])
    if not m:
        raise ValueError("cannot read the CI level from statistics.bootstrap.ci")
    return float(m.group(1)) / 100


def fusion_configs(cfg: dict) -> list[tuple[str, str]]:
    """Primary first, then the sensitivity configs in file order."""
    return [(c["calibration"], c["aggregation"])
            for c in [cfg["fusion"]["primary"], *cfg["fusion"]["sensitivity"]]]


def distinct_ensembles(cfg: dict) -> dict[str, list[str]]:
    """Ensembles evaluated once each (an alias is reported under its target's results)."""
    return {n: e["members"] for n, e in cfg["ensembles"].items() if e["role"] != "alias"}


def aliases(cfg: dict) -> dict[str, str]:
    """target -> alias name (E-bench -> E-lit)."""
    return {e["alias_of"]: n for n, e in cfg["ensembles"].items() if e["role"] == "alias"}


def method_key(ensemble: str, calib: str, agg: str) -> str:
    return f"{ensemble}{SEP}{calib}+{agg}"


def parse_key(key: str) -> tuple[str, str | None, str | None]:
    if SEP not in key:
        return key, None, None
    ens, cfg = key.split(SEP)
    calib, agg = cfg.split("+")
    return ens, calib, agg


# ---------------------------------------------------------------- split and fitting stage

def load_split() -> data.Split:
    split = data.split_ulb()
    data.assert_temporal(split)            # every training Time < every test Time
    return split


def blind(split: data.Split) -> data.Split:
    """The split the fitting stage sees: labels replaced by zeros."""
    return replace(split, y_train=np.zeros_like(split.y_train), y_test=np.zeros_like(split.y_test))


def cached_scores(split: data.Split, det: str) -> dict:
    return scorecache.get_scores(split, det)


@dataclass
class Fitted:
    """Per method: training and day-2 scores, one row per IForest seed (one row if unseeded)."""
    train: dict[str, np.ndarray]
    test: dict[str, np.ndarray]
    singles: tuple[str, ...]
    runtime_s: dict[str, float]


def fit_stage(split: data.Split, cfg: dict, score_fn: ScoreFn = cached_scores) -> Fitted:
    """Everything that is fitted: input scaling, detectors, calibrators and fused training
    scores. Gets a blinded split, so labels cannot reach it."""
    s = data.standardize(blind(split))
    pool, seeds = cfg["detectors"]["pool"], cfg["detectors"]["iforest_seeds"]
    train, test, runtime = {}, {}, {}
    for det in pool:
        runs = [score_fn(replace(s, seed=sd), det)
                for sd in (seeds if det in detectors.SEEDED else [None])]
        train[det] = np.vstack([r["train"] for r in runs])
        test[det] = np.vstack([r["test"] for r in runs])
        runtime[det] = float(np.mean([r["runtime_s"] for r in runs]))
    for ens, members in distinct_ensembles(cfg).items():
        n_seeds = max(len(train[m]) for m in members)
        for calib, agg in fusion_configs(cfg):
            tr, te = [], []
            for i in range(n_seeds):
                pick = [min(i, len(train[m]) - 1) for m in members]
                S_tr = np.column_stack([train[m][j] for m, j in zip(members, pick)])
                S_te = np.column_stack([test[m][j] for m, j in zip(members, pick)])
                f_tr, f_te = fusion.fuse(S_tr, S_te, calib, agg, members, DATASET)
                tr.append(f_tr)
                te.append(f_te)
            key = method_key(ens, calib, agg)
            train[key], test[key] = np.vstack(tr), np.vstack(te)
    return Fitted(train, test, tuple(pool), runtime)


THRESHOLD_RULE = ("threshold = numpy.quantile(training scores, q) (method 'linear'); "
                  "alert if score > threshold")


def fit_thresholds(train_scores: np.ndarray, q: float) -> np.ndarray:
    """One threshold per seed, from that method's own training scores (THRESHOLD_RULE)."""
    return np.quantile(np.atleast_2d(train_scores), q, axis=1, method="linear")


def alerts(test_scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.atleast_2d(test_scores) > thresholds[:, None]


# ---------------------------------------------------------------- budget mode

def method_rows(fitted: Fitted) -> list[dict]:
    rows = []
    for key in fitted.test:
        ens, calib, agg = parse_key(key)
        rows.append({"method": key, "kind": "single" if calib is None else "ensemble",
                     "name": ens, "calib": calib, "agg": agg})
    return rows


def budget_runs(fitted: Fitted, y: np.ndarray, budgets) -> pd.DataFrame:
    """One row per method x seed x budget."""
    rows = []
    for m in method_rows(fitted):
        for i, s in enumerate(fitted.test[m["method"]]):
            rank = metrics.ranking_metrics(s, y)
            for b in budgets:
                rows.append({**m, "seed_index": i, **rank, **metrics.budget_metrics(s, y, b)})
    return pd.DataFrame(rows)


METRIC_COLS = ["fp_at_k", "tp_at_k", "precision_at_k", "recall_at_k", "pr_auc", "roc_auc"]


def budget_table(runs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Seed means per method x budget, the baselines, and regret against the oracle."""
    keys = ["method", "kind", "name", "calib", "agg", "budget", "k"]
    t = runs.groupby(keys, dropna=False, sort=False)[METRIC_COLS].mean().reset_index()
    t["n_seeds"] = runs.groupby(keys, dropna=False, sort=False).size().to_numpy()
    singles = cfg["baselines"]["single_detectors"]
    extra = []
    for b, g in t.groupby("budget", sort=False):
        sg = g[g.method.isin(singles)].set_index("method").loc[singles]
        oracle = sg.fp_at_k.idxmin()                    # first in pool order on a tie
        base = {"kind": "baseline", "calib": None, "agg": None, "budget": b,
                "k": int(g.k.iloc[0]), "n_seeds": None}
        extra.append({**base, "method": bootstrap.MEDIAN, "name": "median of the six singles",
                      "fp_at_k": float(np.median(sg.fp_at_k))})
        extra.append({**base, "method": bootstrap.DEFAULT_IFOREST, "name": "IForest",
                      **sg.loc["IForest", METRIC_COLS].to_dict()})
        extra.append({**base, "method": bootstrap.ORACLE, "name": oracle,
                      "oracle_ties": int((sg.fp_at_k == sg.fp_at_k.min()).sum()),
                      **sg.loc[oracle, METRIC_COLS].to_dict()})
    t = pd.concat([t, pd.DataFrame(extra)], ignore_index=True)
    oracle_fp = t[t.method == bootstrap.ORACLE].set_index("budget").fp_at_k
    t["regret"] = t.fp_at_k - t.budget.map(oracle_fp)
    t["fp_per_100_alerts"] = t.fp_at_k / t.k * 100
    return t


def comparisons(cfg: dict) -> list[tuple[str, str]]:
    """Every distinct ensemble x fusion config against the three comparators (descriptive CIs,
    H1a, H1b), plus the H2b secondary check (E-bench-5% vs E-lit, primary fusion)."""
    out = []
    for ens in distinct_ensembles(cfg):
        for calib, agg in fusion_configs(cfg):
            for comp in (bootstrap.MEDIAN, bootstrap.DEFAULT_IFOREST, bootstrap.ORACLE):
                out.append((method_key(ens, calib, agg), comp))
    out.append(h2b_secondary_pair(cfg))
    return out


def h2b_secondary_pair(cfg: dict) -> tuple[str, str]:
    """E-bench-5% vs E-lit at the primary fusion; E-lit is evaluated as its alias target."""
    text = cfg["hypotheses"]["H2b"]["secondary"]
    m = re.match(r"(\S+) vs (\S+) at the primary budget", text)
    if not m:
        raise ValueError(f"cannot read the H2b secondary pair from {text!r}")
    a, b = m.groups()
    ens = cfg["ensembles"]
    b = ens[b]["alias_of"] if ens[b]["role"] == "alias" else b
    calib, agg = fusion_configs(cfg)[0]
    return method_key(a, calib, agg), method_key(b, calib, agg)


def bootstrap_budget(fitted: Fitted, y: np.ndarray, budget: float, cfg: dict,
                     resamples: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point estimate + paired transaction-level bootstrap of one budget (statistics.bootstrap):
    the same seed and one stream for every budget, so every budget sees the same resamples."""
    boot_cfg = cfg["statistics"]["bootstrap"]
    stat = bootstrap.budget_statistic(fitted.test, y, budget, list(fitted.singles),
                                      comparisons(cfg))
    n = len(y)
    point = stat(bootstrap.transaction_windows(n))
    boot = bootstrap.run(lambda rng: bootstrap.draw_transactions(rng, n), stat,
                         resamples=resamples or boot_cfg["resamples"], seed=boot_cfg["seed"])
    summ = bootstrap.summary(point, boot, ci_level(cfg))
    summ.insert(0, "budget", budget)
    return summ, boot


def delta_table(summ: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per comparison x budget: dFP@K and dFP per 100 alerts with CIs and the verdict."""
    s = summ.set_index(["budget", "stat"])
    rows = []
    for b in summ.budget.unique():
        for a, comp in comparisons(cfg):
            d, d100 = s.loc[(b, f"d:{a}-{comp}")], s.loc[(b, f"d100:{a}-{comp}")]
            ens, calib, agg = parse_key(a)
            rows.append({"budget": b, "method": a, "ensemble": ens, "calib": calib, "agg": agg,
                         "comparator": comp, "d_fp": d.estimate, "d_fp_ci_low": d.ci_low,
                         "d_fp_ci_high": d.ci_high, "d_fp_per_100": d100.estimate,
                         "d_fp_per_100_ci_low": d100.ci_low, "d_fp_per_100_ci_high": d100.ci_high,
                         "verdict": SUPPORT if bool(d.supported) else NULL,
                         "ci_entirely_above_0": bool(d.ci_low > 0)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- hypotheses

def hypotheses(deltas: pd.DataFrame, cfg: dict, n_resamples: int) -> pd.DataFrame:
    """H1a, H1b (ULB part and what it means for the two-dataset decision), H2b."""
    hyp = cfg["hypotheses"]
    calib, agg = fusion_configs(cfg)[0]
    primary_budget = cfg["budgets"]["ulb"]["primary"]
    d = deltas[deltas.budget == primary_budget].set_index(["method", "comparator"])
    alias = aliases(cfg)
    rows = []
    for h in ("H1a", "H1b"):
        spec = hyp[h]
        if f"ULB {primary_budget}" not in spec["budget"]:
            raise ValueError(f"{h}: ULB budget in the file is not {primary_budget}")
        r = d.loc[(method_key(spec["ensemble"], calib, agg), spec["comparator"])]
        holds = r.verdict == SUPPORT
        rule = spec["supported_if"]
        if rule == "the criterion holds on both datasets":
            status = "pending IEEE-CIS" if holds else "not supported (fails on ULB; needs both)"
        elif rule == "the criterion holds on at least one dataset":
            status = "supported (holds on ULB; needs at least one)" if holds else "pending IEEE-CIS"
        else:
            raise ValueError(f"{h}: unknown rule {rule!r}")
        rows.append(_hyp_row(h, spec["ensemble"], alias, spec["comparator"], r, primary_budget,
                             calib, agg, status, n_resamples))
    rows.append({"hypothesis": "H2b (primary)", "dataset": "ULB", "budget": primary_budget,
                 "fusion": f"{calib}+{agg}", "comparison": f"E-bench vs {alias.get('E-bench')}",
                 "d_fp": 0.0, "verdict": "degenerate", "status": hyp["H2b"]["primary"],
                 "resamples": None})
    a, b = h2b_secondary_pair(cfg)
    r = d.loc[(a, b)]
    rows.append(_hyp_row("H2b (secondary)", parse_key(a)[0], alias, b, r, primary_budget, calib,
                         agg, "descriptive only (a superiority check; non-inferiority is not tested)",
                         n_resamples))
    return pd.DataFrame(rows)


def _hyp_row(h, ens, alias, comp, r, budget, calib, agg, status, n_resamples) -> dict:
    name = f"{ens} (= {alias[ens]})" if ens in alias else ens
    comp_ens = parse_key(comp)[0]
    comp_name = f"{alias[comp_ens]} (= {comp_ens})" if comp_ens in alias else comp
    return {"hypothesis": h, "dataset": "ULB", "budget": budget, "fusion": f"{calib}+{agg}",
            "comparison": f"{name} vs {comp_name}", "d_fp": r.d_fp, "d_fp_ci_low": r.d_fp_ci_low,
            "d_fp_ci_high": r.d_fp_ci_high, "d_fp_per_100": r.d_fp_per_100,
            "d_fp_per_100_ci_low": r.d_fp_per_100_ci_low,
            "d_fp_per_100_ci_high": r.d_fp_per_100_ci_high, "verdict": r.verdict,
            "ci_entirely_above_0": r.ci_entirely_above_0, "status": status,
            "resamples": n_resamples}


# ---------------------------------------------------------------- threshold mode

def threshold_table(fitted: Fitted, y: np.ndarray, quantiles) -> pd.DataFrame:
    """Per method x q: day-1 threshold applied unchanged to day 2; seed means."""
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    rows = []
    for m in method_rows(fitted):
        tr, te = fitted.train[m["method"]], fitted.test[m["method"]]
        for q in quantiles:
            thr = fit_thresholds(tr, q)
            a = alerts(te, thr)
            train_rate = (np.atleast_2d(tr) > thr[:, None]).mean(axis=1)
            tp = (a & (y == 1)).sum(axis=1)
            count = a.sum(axis=1)
            fp = count - tp
            prec = np.where(count > 0, tp / np.maximum(count, 1), np.nan)
            rows.append({**m, "q": q, "target_rate": 1 - q, "n_seeds": len(te),
                         "train_alert_rate": train_rate.mean(), "alert_count": count.mean(),
                         "alert_rate": (count / len(y)).mean(),
                         "rate_minus_target": (count / len(y)).mean() - (1 - q),
                         "fpr": (fp / n_neg).mean(), "recall": (tp / n_pos).mean(),
                         "precision": float(np.mean(prec)), "seeds_without_alerts": int((count == 0).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- outputs

def display(key: str, cfg: dict) -> str:
    ens, calib, agg = parse_key(key)
    alias = aliases(cfg)
    name = f"{ens} (= {alias[ens]})" if ens in alias else ens
    if calib is None:
        return {bootstrap.MEDIAN: "median single", bootstrap.DEFAULT_IFOREST: "default IForest",
                bootstrap.ORACLE: "oracle-best single"}.get(key, name)
    return f"{name}, {fusion.LABELS[calib]} + {fusion.LABELS[agg]}"


def fmt(x, nd=1) -> str:
    return "" if pd.isna(x) else f"{x:.{nd}f}"


def markdown(meta: dict, hyp: pd.DataFrame, budget: pd.DataFrame, deltas: pd.DataFrame,
             thr: pd.DataFrame, cfg: dict) -> str:
    pb = cfg["budgets"]["ulb"]["primary"]
    calib, agg = fusion_configs(cfg)[0]
    L = [f"# E3 on ULB (day 1 fit, day 2 test)", "",
         f"Commit `{meta['commit']}` · frozen config = tag `{FREEZE_TAG}` · created "
         f"{meta['created_utc']} · `python -m thesis.e3_ulb`", "",
         f"Day 1: {meta['n_train']:,} rows (Time {meta['time_train_max']:.0f} s at most); day 2: "
         f"{meta['n_test']:,} rows (Time {meta['time_test_min']:.0f} s at least), "
         f"**{meta['day2_frauds']} frauds** ({100 * meta['day2_fraud_rate']:.3f}%). "
         f"K at {', '.join(f'{b:.1%}' for b in meta['budgets'])}: "
         f"{', '.join(str(k) for k in meta['k'])}. Bootstrap: {meta['resamples']} paired "
         f"transaction-level resamples, seed {meta['bootstrap_seed']}, 95% percentile CI.", "",
         "## Hypotheses (criterion: 'fewer false alarms' only if the 95% CI of dFP@K lies "
         "entirely below 0; otherwise 'no difference shown')", "",
         "| Hypothesis | Comparison | dFP@K [95% CI] | per 100 alerts [95% CI] | ULB result | Status |",
         "|---|---|---|---|---|---|"]
    for r in hyp.itertuples():
        ci = (f"{fmt(r.d_fp)} [{fmt(r.d_fp_ci_low)}, {fmt(r.d_fp_ci_high)}]"
              if not pd.isna(getattr(r, "d_fp_ci_low", np.nan)) else fmt(r.d_fp))
        ci100 = (f"{fmt(r.d_fp_per_100)} [{fmt(r.d_fp_per_100_ci_low)}, "
                 f"{fmt(r.d_fp_per_100_ci_high)}]"
                 if not pd.isna(getattr(r, "d_fp_per_100_ci_low", np.nan)) else "")
        above = " (CI entirely above 0: more false alarms; descriptive)" if r.ci_entirely_above_0 in (True, 1) and not pd.isna(r.ci_entirely_above_0) else ""
        L.append(f"| {r.hypothesis} | {r.comparison} | {ci} | {ci100} | {r.verdict}{above} | "
                 f"{r.status} |")
    L += ["", "The frozen criterion has two outcomes. 'CI entirely above 0' is a descriptive "
          "note, not a hypothesis outcome.", "",
          f"## Budget mode at the primary budget {pb:.1%} (seed means; regret = FP@K − "
          "FP@K of the oracle-best single)", "",
          "| Method | FP@K | P@K | R@K | PR-AUC | ROC-AUC | Regret |", "|---|---|---|---|---|---|---|"]
    g = budget[budget.budget == pb]
    g = g[(g.kind != "ensemble") | ((g.calib == calib) & (g["agg"] == agg))]
    for r in g.itertuples():
        name = display(r.method, cfg) + (f" ({r.name})" if r.method == bootstrap.ORACLE else "")
        L.append(f"| {name} | {fmt(r.fp_at_k)} | {fmt(r.precision_at_k, 3)} | "
                 f"{fmt(r.recall_at_k, 3)} | {fmt(r.pr_auc, 3)} | {fmt(r.roc_auc, 3)} | "
                 f"{fmt(r.regret)} |")
    L += ["", "All budgets, fusion configs and CIs: `e3_ulb_budget.csv`, `e3_ulb_deltas.csv`.", "",
          f"## Threshold mode (day-1 quantile threshold applied unchanged to day 2; "
          f"{fusion.LABELS[calib]} + {fusion.LABELS[agg]} for ensembles; seed means)", "",
          "| Method | q | Alerts | Alert rate | 1 − q | FPR | Recall | Precision |",
          "|---|---|---|---|---|---|---|---|"]
    t = thr[(thr.kind != "ensemble") | ((thr.calib == calib) & (thr["agg"] == agg))]
    for r in t.itertuples():
        L.append(f"| {display(r.method, cfg)} | {r.q} | {fmt(r.alert_count)} | "
                 f"{r.alert_rate:.4%} | {r.target_rate:.1%} | {r.fpr:.4%} | {fmt(r.recall, 3)} | "
                 f"{fmt(r.precision, 3)} |")
    L += ["", "The oracle-best single is chosen with day-2 labels: an upper bound, not a "
          "deployable method. E-lit is the same triple as E-bench and is evaluated once.", ""]
    return "\n".join(L)


def tex_tables(budget: pd.DataFrame, hyp: pd.DataFrame, thr: pd.DataFrame, cfg: dict,
               commit: str) -> dict[str, str]:
    pb = cfg["budgets"]["ulb"]["primary"]
    calib, agg = fusion_configs(cfg)[0]
    g = budget[budget.budget == pb]
    g = g[(g.kind != "ensemble") | ((g.calib == calib) & (g["agg"] == agg))]
    bt = pd.DataFrame({
        "Method": [display(m, cfg) + (f" ({n})" if m == bootstrap.ORACLE else "")
                   for m, n in zip(g.method, g.name)],
        "FP@K": g.fp_at_k.map(fmt), "P@K": g.precision_at_k.map(lambda v: fmt(v, 3)),
        "R@K": g.recall_at_k.map(lambda v: fmt(v, 3)), "PR-AUC": g.pr_auc.map(lambda v: fmt(v, 3)),
        "ROC-AUC": g.roc_auc.map(lambda v: fmt(v, 3)), "Regret": g.regret.map(fmt)})
    k = int(g.k.iloc[0])
    h = hyp
    ht = pd.DataFrame({
        "Hypothesis": h.hypothesis, "Comparison": h.comparison,
        "dFP@K [95% CI]": [f"{fmt(a)} [{fmt(b)}, {fmt(c)}]" if not pd.isna(b) else
                           f"{fmt(a)} (identity)"
                           for a, b, c in zip(h.d_fp, h.d_fp_ci_low, h.d_fp_ci_high)],
        "ULB result": h.verdict})
    t = thr[(thr.kind != "ensemble") | ((thr.calib == calib) & (thr["agg"] == agg))]
    tt = pd.DataFrame({
        "Method": t.method.map(lambda m: display(m, cfg)), "q": t.q,
        "Alert rate (%)": (100 * t.alert_rate).map(lambda v: f"{v:.3f}"),
        "1 - q (%)": (100 * t.target_rate).map(lambda v: f"{v:.1f}"),
        "FPR (%)": (100 * t.fpr).map(lambda v: f"{v:.3f}"),
        "Recall": t.recall.map(lambda v: fmt(v, 3)), "Precision": t.precision.map(lambda v: fmt(v, 3))})
    src = "results/tables/e3_ulb"
    return {
        "e3_ulb_budget.tex": booktabs(
            bt, f"ULB day 2, budget mode at {pb:.1%} (K = {k}), seed means; ensembles with "
                f"min-max + mean. The oracle-best single is chosen with test labels (upper "
                f"bound).".replace("%", "\\%"),
            "tab:e3-ulb-budget", commit, f"{src}/e3_ulb_budget.csv", "lrrrrrr"),
        "e3_ulb_hypotheses.tex": booktabs(
            ht, "ULB part of the frozen hypotheses: paired transaction-level bootstrap, 95\\% "
                "percentile CI; 'fewer false alarms' only if the CI lies entirely below 0.",
            "tab:e3-ulb-hypotheses", commit, f"{src}/e3_ulb_hypotheses.csv", "llll"),
        "e3_ulb_threshold.tex": booktabs(
            tt, "ULB threshold check: day-1 training-score quantile thresholds applied unchanged "
                "to day 2 (seed means; ensembles with min-max + mean).",
            "tab:e3-ulb-threshold", commit, f"{src}/e3_ulb_threshold.csv", "lrrrrrr"),
    }


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="python -m thesis.e3_ulb")
    parser.add_argument("--quick", action="store_true",
                        help=f"{QUICK_RESAMPLES} resamples, print the hypotheses, write nothing")
    args = parser.parse_args(argv)
    frozen_unchanged()
    cfg = frozen.load()
    check_code_matches(cfg)
    state = git_state()
    if not args.quick and state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    t0 = time.perf_counter()
    split = load_split()
    fitted = fit_stage(split, cfg)
    print(f"fitting stage done ({time.perf_counter() - t0:.0f} s)", flush=True)
    y = split.y_test
    budgets = cfg["budgets"]["ulb"]["values"]
    runs = budget_runs(fitted, y, budgets)
    budget = budget_table(runs, cfg)
    resamples = QUICK_RESAMPLES if args.quick else cfg["statistics"]["bootstrap"]["resamples"]
    summ = []
    for b in budgets:
        t1 = time.perf_counter()
        summ.append(bootstrap_budget(fitted, y, b, cfg, resamples)[0])
        print(f"bootstrap {b:.1%}: {resamples} resamples, {time.perf_counter() - t1:.0f} s",
              flush=True)
    summ = pd.concat(summ, ignore_index=True)
    deltas = delta_table(summ, cfg)
    hyp = hypotheses(deltas, cfg, resamples)
    thr = threshold_table(fitted, y, cfg["thresholds"]["ulb"]["training_score_quantiles"])
    print(hyp[["hypothesis", "comparison", "d_fp", "d_fp_ci_low", "d_fp_ci_high", "verdict",
               "status"]].to_string(index=False))
    if args.quick:
        return
    from thesis import e3_ulb_figures
    meta = {**state, "command": "python -m thesis.e3_ulb",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "frozen_config": f"identical to tag {FREEZE_TAG}",
            "cache_key": scorecache.cache_key(), "n_train": len(split.y_train),
            "n_test": len(y), "time_train_max": float(split.t_train.max()),
            "time_test_min": float(split.t_test.min()), "day2_frauds": int(y.sum()),
            "day2_fraud_rate": float(y.mean()), "budgets": budgets,
            "k": [metrics.budget_k(b, len(y)) for b in budgets],
            "quantiles": cfg["thresholds"]["ulb"]["training_score_quantiles"],
            "resamples": resamples, "bootstrap_seed": cfg["statistics"]["bootstrap"]["seed"],
            "detector_runtime_s": fitted.runtime_s,
            "runtime_s": round(time.perf_counter() - t0, 1),
            "note": "Fitting stage (scaling, detectors, calibrators, thresholds) on day 1 with "
                    "labels replaced by zeros; day-2 labels used only by the metrics. The "
                    "oracle-best single is chosen with test labels (upper bound)."}
    OUT.mkdir(parents=True, exist_ok=True)
    c = meta["commit"]
    runs.assign(commit=c).to_csv(OUT / "e3_ulb_runs.csv", index=False)
    budget.assign(commit=c).to_csv(OUT / "e3_ulb_budget.csv", index=False)
    summ.assign(commit=c).to_csv(OUT / "e3_ulb_bootstrap.csv", index=False)
    deltas.assign(commit=c).to_csv(OUT / "e3_ulb_deltas.csv", index=False)
    hyp.assign(commit=c).to_csv(OUT / "e3_ulb_hypotheses.csv", index=False)
    thr.assign(commit=c).to_csv(OUT / "e3_ulb_threshold.csv", index=False)
    (OUT / "e3_ulb.meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    (OUT / "e3_ulb_summary.md").write_text(markdown(meta, hyp, budget, deltas, thr, cfg),
                                           encoding="utf-8")
    for name, text in tex_tables(budget, hyp, thr, cfg, c).items():
        (OUT / name).write_text(text, encoding="utf-8")
    e3_ulb_figures.write(deltas, thr, cfg, FIG_OUT, c)
    print(f"wrote {OUT} and {FIG_OUT} ({meta['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main(sys.argv[1:])
