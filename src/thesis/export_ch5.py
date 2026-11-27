"""Chapter 5 tables and figures, rebuilt from the committed result files only.

    python -m thesis.export_ch5

Reads (never writes) results/tables/ieee_cis/run/ (the single IEEE-CIS run),
results/tables/robust_fusion/ and results/tables/e3_ulb/lof/. Nothing is recomputed from data or
scores: values are only selected, combined and rounded. Every plain-unit sentence is built from
the CSVs, and every sentence that states a direction or a comparison is checked against the
values first (ValueError if it would be false).

Writes:
- results/tables/ieee_cis/ch5/*.tex           hypothesis summary table, H4r summary table, guard-event
                                              table, appendix tables, and ch5.meta.json
- results/figures/ieee_cis/h4r_tradeoff.pdf   trade-off figure (alert-rate deviation vs FP@K change)
- results/figures/ieee_cis/h4a_scatter.pdf    pre-deployment figure (pre-deployment vs monthly |log r|)
- results/figures/e3_ulb/lof/l2_controls_r.pdf  ULB LOF break (r per control split)
- results/tables/robust_fusion/adbench_static.tex  ADBench static cost of robust fusion
Figures: PDF (+ PNG preview), 16 cm wide, no title inside, black-and-white readable (two inks
plus marker shape and fill); captions live in LaTeX.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thesis import metrics  # noqa: E402
from thesis.e2_figures import INK, MUTED, _save  # noqa: E402
from thesis.export_datasets import git_state  # noqa: E402
from thesis.export_setup import booktabs  # noqa: E402
from thesis.paths import FIGURES, TABLES  # noqa: E402

RUN = TABLES / "ieee_cis" / "run"
OUT = TABLES / "ieee_cis" / "ch5"
FIG = FIGURES / "ieee_cis"
RF = TABLES / "robust_fusion"
LOF = TABLES / "e3_ulb" / "lof"
LOF_FIG = FIGURES / "e3_ulb" / "lof"
PRIMARY = 0.02
MEMBERS = ("IForest", "PCA", "ECOD", "HBOS", "KNN", "LOF")
NAMED = {"IForest+PCA+KNN": "E-bench", "IForest+KNN+LOF": "E-bench-5%"}
RUN_FILES = ("rq4_hypotheses", "v1_hypotheses", "h3a", "e3_deltas", "rq4_h4r_triples",
             "rq4_h4r_threshold", "rq4_member_days", "rq4_guard_decisions", "rq4_daily_fp",
             "rq4_pre_deployment_r", "rq4_e_stress_sets", "rq4_h4a_pairs", "rq4_bound_check",
             "rq4_drift_monthly", "rq4_drift_slopes", "rq4_h4d_fragility", "context_adtk",
             "context_score_drift")
WIDTH_CM, PT = 16.0, 8
MINUS = "−"


# ---------------------------------------------------------------- loading and formatting

def load_run(run=None) -> tuple[dict[str, pd.DataFrame], dict]:
    run = RUN if run is None else run
    t = {f: pd.read_csv(run / f"{f}.csv") for f in RUN_FILES}
    meta = json.loads((run / "run.meta.json").read_text(encoding="utf-8"))
    if meta.get("dry_run") or meta.get("dirty"):
        raise ValueError("the Chapter 5 exports need the real, clean IEEE-CIS run")
    commits = {c for df in t.values() for c in df["commit"].unique()}
    if commits != {meta["commit"]}:
        raise ValueError(f"run files from several commits: {commits}")
    return t, meta


def num(v: float, nd: int, plus: bool = False) -> str:
    s = f"{v:+.{nd}f}" if plus and v != 0 else f"{v:.{nd}f}"
    return s.replace("-", MINUS)


def est_ci(e, lo, hi, nd, plus=False, unit="") -> str:
    return f"{num(e, nd, plus)}{unit} [{num(lo, nd, plus)}{unit}, {num(hi, nd, plus)}{unit}]"


def short(commit: str) -> str:
    return str(commit)[:7]


# ---------------------------------------------------------------- facts used in several outputs

def guard_facts(t: dict) -> dict:
    md = t["rq4_member_days"]
    md = md[md["budget"] == PRIMARY]
    dec = t["rq4_guard_decisions"]
    dec = dec[dec["budget"] == PRIMARY]
    one = md[md["member"] == MEMBERS[0]].sort_values("day")
    days = one["day"].to_numpy()
    k_sum = sum(metrics.budget_k(PRIMARY, int(n)) for n in one["n_rows"])
    excluded = {m: sorted(md[(md["member"] == m) & ~md["passes"]]["day"].astype(int)) for m in MEMBERS}
    changed = {e: sorted(g[g["changed"]]["day"].astype(int)) for e, g in dec.groupby("ensemble")}
    return {"n_days": len(days), "k_per_day": k_sum / len(days), "excluded": excluded,
            "changed": changed, "event_days": changed.get("E-all", [])}


def pre_r(t: dict) -> pd.Series:
    p = t["rq4_pre_deployment_r"]
    return p[p["budget"] == PRIMARY].set_index("member")["r"]


def h4a_note(t: dict, g: dict) -> str:
    """Which members broke and where they stood in the pre-deployment check; raises if the
    sentence would not be true."""
    r = pre_r(t)
    dev = np.abs(np.log(r.loc[list(MEMBERS)]))
    n_ev = len(g["event_days"])
    every = [m for m in MEMBERS if len(g["excluded"][m]) == n_ev and n_ev > 0]
    some = [m for m in MEMBERS if 0 < len(g["excluded"][m]) < n_ev]
    smallest = list(dev.sort_values().index[:len(every)])
    if not every or sorted(every) != sorted(smallest):
        raise ValueError("H4a note: the members excluded on every event day are not the ones "
                         "with the smallest pre-deployment deviation")
    below = [m for m in dev.sort_values(ascending=False).index if r[m] < 1][:2]
    txt = (f"Carried by members below target in the check ({', '.join(f'{m} r = {r[m]:.2f}' for m in below)}). "
           f"{' and '.join(every)}, excluded by the guard on all {n_ev} event days, had the "
           f"smallest check deviation (r = {', '.join(f'{r[m]:.2f}' for m in every)})")
    if some:
        txt += (f"; {' and '.join(some)}, excluded on fewer days, were below target in the check "
                f"(r = {', '.join(f'{r[m]:.2f}' for m in some)})")
        if any(r[m] >= 1 for m in some):
            raise ValueError("H4a note: a partly excluded member was not below target")
    return txt


# ---------------------------------------------------------------- hypothesis summary table

def hypothesis_table(t: dict, meta: dict) -> pd.DataFrame:
    h = t["rq4_hypotheses"].set_index("hypothesis")
    v1 = t["v1_hypotheses"].set_index("hypothesis")
    dl = t["e3_deltas"]
    dl = dl[(dl["budget"] == PRIMARY)].set_index("role")
    tri = t["rq4_h4r_triples"]
    g = guard_facts(t)
    rows = []

    def add(hyp, comp, est, crit, verdict, plain):
        rows.append({"Hypothesis": hyp, "Comparison": comp, "Estimate [95% CI]": est,
                     "Criterion": crit, "Verdict": verdict, "In plain units": plain})

    r = h.loc["H4r"]
    better = int((tri["dD"] < 0).sum())
    add("H4r", "train-percentile + median vs min-max + mean, 20 triples, threshold mode, q = 0.98",
        est_ci(r["estimate"], r["ci_low"], r["ci_high"], 4), "CI entirely below 0", r["verdict"],
        f"Typical monthly miss of the alert target ×{math.exp(tri['D_frozen'].mean()):.2f} → "
        f"×{math.exp(tri['D_robust'].mean()):.2f}; better in {better} of {len(tri)} ensembles")
    r = h.loc["H4r-cost"]
    d_tot = tri["fp_total_robust"].mean() - tri["fp_total_frozen"].mean()
    add("H4r-cost", "relative change of mean total FP@K, 20 triples, daily K",
        est_ci(100 * r["estimate"], 100 * r["ci_low"], 100 * r["ci_high"], 2, True, "%"),
        "upper CI bound ≤ +5%", r["verdict"],
        f"{num(d_tot, 0, True)} FP over {g['n_days']} days ({num(d_tot / g['n_days'], 1, True)} per "
        f"day at about {g['k_per_day']:.0f} alerts per day)")
    r = h.loc["H4a"]
    add("H4a", f"Spearman ρ, pre-deployment vs monthly |log r|, {int(r['n_pairs'])} pairs",
        est_ci(r["estimate"], r["ci_low"], r["ci_high"], 3), "CI entirely above 0", r["verdict"],
        h4a_note(t, g))
    for hyp, ens in (("H4b", "E-all"), ("H4c", "E-bench")):
        r = h.loc[hyp]
        ch = g["changed"].get(ens, [])
        if hyp == "H4b":
            plain = (f"The guard changed E-all on {len(ch)} of {g['n_days']} days; "
                     f"{num(r['estimate'] / g['n_days'], 2, True)} FP per day")
            crit = "CI entirely below 0"
        else:
            daily = t["rq4_daily_fp"]
            daily = daily[daily["budget"] == PRIMARY].pivot(index="day", columns="method", values="fp_at_k")
            same = all(daily.loc[d, "guarded:E-bench"] == daily.loc[d, "static:E-bench"] for d in ch)
            dropped = sorted(set("+".join(t["rq4_guard_decisions"].query(
                "budget == @PRIMARY and ensemble == 'E-bench' and changed")["dropped"]).split("+")))
            if not (ch and same and r["estimate"] == 0):
                raise ValueError("H4c sentence assumes changed days with unchanged FP@K")
            plain = (f"{', '.join(dropped)} dropped on {len(ch)} days ({', '.join(map(str, ch))}); FP@K "
                     f"unchanged on each, so dFP = 0 (a result, not 0 by identity)")
            crit = "CI not entirely above 0"
        add(hyp, f"E-guard({ens}) vs static {ens}, daily K",
            est_ci(r["estimate"], r["ci_low"], r["ci_high"], 1, True), crit, r["verdict"], plain)
    for hyp in ("H1a", "H1b"):
        r, d = v1.loc[hyp], dl.loc[hyp]
        per100 = num(d["d_fp_per_100"], 2 if hyp == "H1a" else 1, True)
        if hyp == "H1a":
            plain = f"{per100} FP per 100 alerts; ULB: {r['verdict_ulb']}"
            crit = "CI entirely below 0 on both ULB and IEEE-CIS"
        else:
            if not d["d_fp_ci_low"] > 0:
                raise ValueError("H1b sentence assumes a CI entirely above 0")
            plain = (f"The ensemble is worse than default IForest: {per100} FP per 100 alerts, CI "
                     f"entirely above 0; ULB: {r['verdict_ulb']}")
            crit = "CI entirely below 0 on at least one dataset"
        comp = "E-bench vs " + ("median single" if hyp == "H1a" else "default IForest")
        add(hyp, comp + ", monthly K", est_ci(r["d_fp"], r["d_fp_ci_low"], r["d_fp_ci_high"], 1, True),
            crit, r["status"], plain)
    d = dl.loc["H2b (secondary)"]
    add("H2b (secondary)", "E-bench-5% vs E-bench, monthly K",
        est_ci(d["d_fp"], d["d_fp_ci_low"], d["d_fp_ci_high"], 1, True), "none (descriptive)",
        "descriptive", f"{num(d['d_fp_per_100'], 1, True)} FP per 100 alerts")
    a = t["h3a"].set_index("stat")
    key = next(s for s in a.index if s.startswith("d:"))
    ens = key[2:key.index("-median_single")]
    add("H3a", "E-bench vs median single, |monthly alert rate − 1%|, q = 0.99",
        est_ci(a.loc[key, "estimate"], a.loc[key, "ci_low"], a.loc[key, "ci_high"], 5, True),
        "CI entirely below 0", a.loc[key, "verdict"],
        f"Monthly alert rate misses the 1% target by {100 * a.loc['dev:' + ens, 'estimate']:.2f} "
        f"percentage points (E-bench) vs {100 * a.loc['dev:median_single', 'estimate']:.2f} (median single)")
    b = meta["h3b"]
    add("H3b", f"Spearman ρ, weekly drift share vs FPR, {b['n_windows']} weeks",
        est_ci(b["rho"], b["ci_low"], b["ci_high"], 3), "none (exploratory)", "exploratory",
        "Association only, no decision")
    add("H4d", "Spearman ρ, fragility index vs mean log r, 6 members",
        f"{num(h.loc['H4d', 'estimate'], 3)} (no CI)", "none (exploratory)", "exploratory",
        "Association only, no decision; appendix")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- H4r summary table

def h4r_main_table(t: dict) -> pd.DataFrame:
    h = t["rq4_hypotheses"].set_index("hypothesis").loc["H4r"]
    tri = t["rq4_h4r_triples"]
    rows = [{"Ensemble": f"Mean of the {len(tri)} triples (H4r)",
             "D robust": f"{tri['D_robust'].mean():.3f}", "D frozen": f"{tri['D_frozen'].mean():.3f}",
             "dD [95% CI]": est_ci(h["estimate"], h["ci_low"], h["ci_high"], 3),
             "Role": h["verdict"]}]
    thr = t["rq4_h4r_threshold"]
    for r in thr[thr["q"] == 0.98].itertuples():
        rows.append({"Ensemble": r.method.split(":", 1)[1], "D robust": f"{r.D_method:.3f}",
                     "D frozen": f"{r.D_comparator:.3f}",
                     "dD [95% CI]": est_ci(r.d_D, r.ci_low, r.ci_high, 3), "Role": "descriptive"})
    return pd.DataFrame(rows)


def h4r_triples_table(t: dict) -> pd.DataFrame:
    tri = t["rq4_h4r_triples"]
    return pd.DataFrame({
        "Ensemble": [f"{x.replace('+', ', ')} ({NAMED[x]})" if x in NAMED else x.replace("+", ", ")
                     for x in tri["triple"]],
        "D robust": tri["D_robust"].map(lambda v: f"{v:.3f}"),
        "D frozen": tri["D_frozen"].map(lambda v: f"{v:.3f}"),
        "dD": tri["dD"].map(lambda v: num(v, 3, True)),
        "FP robust": tri["fp_total_robust"].map(lambda v: f"{v:.1f}"),
        "FP frozen": tri["fp_total_frozen"].map(lambda v: f"{v:.1f}"),
        "FP change (%)": (100 * (tri["fp_total_robust"] / tri["fp_total_frozen"] - 1)).map(
            lambda v: num(v, 1, True))})


# ---------------------------------------------------------------- guard-event table

def guard_table(t: dict) -> pd.DataFrame:
    g = guard_facts(t)
    days = g["event_days"]
    md = t["rq4_member_days"]
    md = md[md["budget"] == PRIMARY].set_index(["member", "day"])
    dec = t["rq4_guard_decisions"]
    dec = dec[dec["budget"] == PRIMARY].set_index(["ensemble", "day"])
    daily = t["rq4_daily_fp"]
    daily = daily[daily["budget"] == PRIMARY].pivot(index="day", columns="method", values="fp_at_k")
    rows = [["Rows that day"] + [f"{int(md.loc[(MEMBERS[0], d), 'n_rows']):,}" for d in days],
            ["Daily K"] + [str(metrics.budget_k(PRIMARY, int(md.loc[(MEMBERS[0], d), 'n_rows']))) for d in days]]
    for m in MEMBERS:
        rows.append([f"r, {m}"] + [f"{md.loc[(m, d), 'r']:.2f}" + ("" if md.loc[(m, d), "passes"] else " †")
                                   for d in days])
    for e in ("E-all", "E-bench", "E-bench-5%"):
        cells = []
        for d in days:
            r = dec.loc[(e, d)]
            cells.append("static (fallback)" if r["fallback"] else
                         (r["kept"].replace("+", ", ") if r["changed"] else "unchanged"))
        rows.append([f"Kept, {e}"] + cells)
    for e in ("E-all", "E-bench"):
        rows.append([f"FP@K, {e}: static / guarded"] +
                    [f"{daily.loc[d, 'static:' + e]:.1f} / {daily.loc[d, 'guarded:' + e]:.1f}" for d in days])
    return pd.DataFrame(rows, columns=["", *[f"Day {d}" for d in days]])


# ---------------------------------------------------------------- appendix tables

def bound_table(t: dict) -> pd.DataFrame:
    b = t["rq4_bound_check"].copy()
    b["broken"] = b["broken"].fillna("")
    if not ((b["fused_alert_rate"] <= b["event_rate"]) & (b["event_rate"] <= b["bound"] + 1e-12)).all():
        raise ValueError("bound check: fused <= event <= bound fails on a seed row")
    g = b.groupby(["ensemble", "month", "b", "broken"], sort=False).agg(
        fused=("fused_alert_rate", "mean"), event=("event_rate", "mean"), bound=("bound", "mean"),
        broken_r=("broken_r", "first")).reset_index()
    if not ((g["fused"] <= g["event"] + 1e-12) & (g["event"] <= g["bound"] + 1e-12)).all():
        raise ValueError("bound check: fused <= event <= bound fails")
    return pd.DataFrame({
        "Ensemble": g["ensemble"], "Month": g["month"], "b": g["b"],
        "Broken member (r)": [f"{m} ({r:.2f})" if m else "none" for m, r in zip(g["broken"], g["broken_r"])],
        "Fused alert rate (%)": (100 * g["fused"]).map(lambda v: f"{v:.3f}"),
        "Event rate (%)": (100 * g["event"]).map(lambda v: f"{v:.3f}"),
        "Bound (%)": (100 * g["bound"]).map(lambda v: f"{v:.3f}")})


def drift_table(t: dict) -> pd.DataFrame:
    d = t["rq4_drift_monthly"]
    months = sorted(d["month"].unique())
    sl = t["rq4_drift_slopes"].set_index("member")["slope"]
    rows = []
    for m in MEMBERS:
        g = d[d["member"] == m].set_index("month")["r"]
        rows.append([m] + [f"{g.loc[x]:.3f}" for x in months] + [num(sl.loc[m], 2, True)])
    share = d.groupby("month")["drift_share"].first()
    rows.append(["Drift share"] + [f"{share.loc[x]:.3f}" for x in months] + [""])
    return pd.DataFrame(rows, columns=["Member", *[f"r, month {x}" for x in months], "Slope"])


def h4d_table(t: dict) -> pd.DataFrame:
    f = t["rq4_h4d_fragility"].set_index("member").loc[list(MEMBERS)].reset_index()
    return pd.DataFrame({"Member": f["member"], "Fragility index F": f["fragility"].map(lambda v: f"{v:.3f}"),
                         "Mean log r, months 2–6": f["mean_log_r"].map(lambda v: num(v, 3, True))})


def pre_deployment_table(t: dict) -> pd.DataFrame:
    p = t["rq4_pre_deployment_r"]
    budgets = sorted(p["budget"].unique())
    rows = []
    for m in MEMBERS:
        g = p[p["member"] == m].set_index("budget")
        rows.append([m] + [f"{g.loc[b, 'r']:.3f}" + ("" if g.loc[b, "passes"] else " †") for b in budgets])
    return pd.DataFrame(rows, columns=["Member", *[f"r, budget {100 * b:g}%" for b in budgets]])


def context_table(t: dict) -> pd.DataFrame:
    c = t["context_adtk"]
    g = c.groupby("method", sort=False).agg(days=("day", "size"), q=("quantile", "sum"),
                                           ls=("level_shift", "sum"), any=("flagged", "sum")).reset_index()
    return pd.DataFrame({"Series (q = 0.99)": g["method"].str.replace(":minmax+mean", " (min-max + mean)", regex=False),
                         "Days": g["days"], "QuantileAD": g["q"], "LevelShiftAD": g["ls"], "Either": g["any"]})


# ---------------------------------------------------------------- ADBench static cost

def adbench_table(a: pd.DataFrame) -> pd.DataFrame:
    name = {("minmax", "mean"): "min-max + mean (freeze-v1)", ("minmax", "median"): "min-max + median",
            ("percentile", "mean"): "train-percentile + mean",
            ("percentile", "median"): "train-percentile + median"}
    return pd.DataFrame({
        "Budget": a["budget"].map(lambda b: f"{100 * b:g}%"),
        "Fusion": [name[(c, g)] for c, g in zip(a["calib"], a["agg"])],
        "FP@K, triples": a["mean_fp"].map(lambda v: f"{v:.1f}"),
        "FP@K, E-bench": a["mean_fp_ebench"].map(lambda v: f"{v:.1f}"),
        "Beats best (%)": (100 * a["share_beats_best"]).map(lambda v: f"{v:.1f}"),
        "Beats median (%)": (100 * a["share_beats_median"]).map(lambda v: f"{v:.1f}")})


# ---------------------------------------------------------------- figures

def _axes(height_cm: float):
    plt.rcParams.update({"font.size": PT, "pdf.fonttype": 42, "axes.titlesize": PT})
    fig, ax = plt.subplots(figsize=(WIDTH_CM / 2.54, height_cm / 2.54), layout="constrained")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=PT)
    return fig, ax


def tradeoff_figure(t: dict):
    """Trade-off figure: per triple, dD (x) against the relative change of total FP@K (y, %); the mean with
    the H4r and H4r-cost CIs; the +5% margin."""
    tri = t["rq4_h4r_triples"]
    h = t["rq4_hypotheses"].set_index("hypothesis")
    x = tri["dD"].to_numpy()
    y = 100 * (tri["fp_total_robust"] / tri["fp_total_frozen"] - 1).to_numpy()
    named = tri["triple"].isin(list(NAMED)).to_numpy()
    fig, ax = _axes(9.0)
    ax.axhline(0, color=MUTED, lw=0.6, zorder=0)
    ax.axvline(0, color=MUTED, lw=0.6, zorder=0)
    ax.axhline(100 * 0.05, color=INK, lw=0.9, ls="--", zorder=1)
    ax.scatter(x[~named], y[~named], s=26, facecolors="white", edgecolors=MUTED, linewidths=1.0,
               label="three-member ensemble", zorder=2)
    ax.scatter(x[named], y[named], s=30, c=INK, marker="s", edgecolors="white", linewidths=0.5,
               label="E-bench / E-bench-5%", zorder=3)
    for xi, yi, name in zip(x[named], y[named], tri["triple"][named]):
        ax.annotate(NAMED[name], (xi, yi), xytext=(5, 3), textcoords="offset points", fontsize=PT, color=INK)
    mx, my = h.loc["H4r", "estimate"], 100 * h.loc["H4r-cost", "estimate"]
    ax.errorbar(mx, my, xerr=[[mx - h.loc["H4r", "ci_low"]], [h.loc["H4r", "ci_high"] - mx]],
                yerr=[[my - 100 * h.loc["H4r-cost", "ci_low"]], [100 * h.loc["H4r-cost", "ci_high"] - my]],
                fmt="D", ms=6, color=INK, mfc=INK, mec="white", ecolor=INK, elinewidth=1.0, capsize=2.5,
                label="mean over the 20 (H4r, H4r-cost), 95% CI", zorder=4)
    ax.text(0.005, 5, "+5% margin (H4r-cost)", transform=ax.get_yaxis_transform(), fontsize=PT,
            color=INK, va="bottom")
    ax.set_xlabel("change in monthly alert-rate deviation dD (robust − min-max + mean; < 0 = closer to target)")
    ax.set_ylabel("change in total FP@K, daily K (%)")
    fig.legend(frameon=False, fontsize=PT, loc="outside lower center", ncols=3)
    return fig


H4A_GROUPS = (("excluded on every event day", dict(marker="^", s=34, c=INK, edgecolors="white", linewidths=0.5)),
              ("excluded on fewer days", dict(marker="s", s=28, c=MUTED, edgecolors="white", linewidths=0.5)),
              ("never excluded", dict(marker="o", s=26, facecolors="white", edgecolors=MUTED, linewidths=1.0)))


def h4a_groups(t: dict) -> dict[str, str]:
    g = guard_facts(t)
    n = len(g["event_days"])
    return {m: H4A_GROUPS[0][0] if n and len(g["excluded"][m]) == n else
            H4A_GROUPS[1][0] if g["excluded"][m] else H4A_GROUPS[2][0] for m in MEMBERS}


def h4a_figure(t: dict):
    """Pre-deployment figure: x = pre-deployment |log r|, y = monthly |log r| (six members x five months)."""
    p = t["rq4_h4a_pairs"]
    grp = h4a_groups(t)
    fig, ax = _axes(8.5)
    for label, style in H4A_GROUPS:
        sel = p[p["member"].map(grp) == label]
        if len(sel):
            ax.scatter(sel["x_abs_log_r_pre"], sel["y_abs_log_r"], label=label, zorder=3, **style)
    tops = p.groupby("member").agg(x=("x_abs_log_r_pre", "first"), y=("y_abs_log_r", "max")).sort_values("x")
    prev_x, flip = -1.0, False
    for m, r in tops.iterrows():
        flip = (r.x - prev_x < 0.12) and not flip
        ax.annotate(m, (r.x, r.y), xytext=(0, 12 if flip else 4), textcoords="offset points",
                    ha="center", fontsize=PT, color=INK)
        prev_x = r.x
    ax.set_xlabel("pre-deployment deviation |log r| (fit days 0–20, scored days 21–29)")
    ax.set_ylabel("monthly deviation |log r| (months 2–6)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=float(p["y_abs_log_r"].max()) * 1.18)
    ax.legend(frameon=False, fontsize=PT, loc="upper left")
    return fig


L2_RUNS = (("L2a", "L2a: random halves\nof day 1"), ("L2b", "L2b: first → last\n12 h of day 1"),
           ("L2c", "L2c: day 2 → day 1"), ("L2d", "L2d: day 1 → day 2\n(as in E3)"))
L2_MARKERS = {"IForest": "o", "PCA": "s", "ECOD": "v", "HBOS": "P", "KNN": "X", "LOF": "D"}


def lof_figure(l2: pd.DataFrame, q: float = 0.998):
    """ULB: r per member and control split at q = 0.998; LOF in ink, the others muted."""
    d = l2[l2["q"] == q]
    fig, ax = _axes(8.0)
    ax.axhline(1, color=MUTED, lw=0.8, ls=":", zorder=0)
    ax.axhline(3, color=INK, lw=0.9, ls="--", zorder=0)
    offs = np.linspace(-0.25, 0.25, len(MEMBERS))
    for off, m in zip(offs, MEMBERS):
        g = d[d["member"] == m].set_index("run")["r"]
        xs = np.arange(len(L2_RUNS)) + off
        ys = [g.loc[r] for r, _ in L2_RUNS]
        lof = m == "LOF"
        ax.scatter(xs, ys, marker=L2_MARKERS[m], s=34 if lof else 24, label=m, zorder=3 if lof else 2,
                   c=INK if lof else "white", edgecolors="white" if lof else MUTED, linewidths=0.5 if lof else 0.9)
        if lof:
            for xi, yi in zip(xs, ys):
                if yi > 3:
                    ax.annotate(f"{yi:.1f}", (xi, yi), xytext=(6, -2), textcoords="offset points",
                                fontsize=PT, color=INK)
    ax.set_yscale("log")
    ticks = [0.1, 0.3, 1, 3, 10, 30]
    ax.set_yticks(ticks, [f"{v:g}" for v in ticks])
    ax.minorticks_off()
    ax.set_xticks(range(len(L2_RUNS)), [lab for _, lab in L2_RUNS])
    ax.set_ylabel(f"r = alert rate / target (q = {q:g}, log scale)")
    ax.text(0.995, 3, "guard limit c = 3", transform=ax.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=PT, color=INK)
    ax.text(1.5, 0.96, "on target (r = 1)", ha="center", va="top", fontsize=PT, color=MUTED)
    ax.legend(frameon=False, fontsize=PT, loc="upper left", ncols=3)
    return fig


# ---------------------------------------------------------------- captions and writing

def tex_outputs(t: dict, meta: dict, me: str) -> dict[str, str]:
    """name -> LaTeX source of the hypothesis summary, H4r summary, guard-event and appendix tables."""
    dc = short(meta["commit"])
    src = "results/tables/ieee_cis/run/"
    cm = f"{me} (data {dc})"
    g = guard_facts(t)
    h = t["rq4_hypotheses"].set_index("hypothesis")
    p = t["rq4_pre_deployment_r"]
    stress = t["rq4_e_stress_sets"]
    sd = t["context_score_drift"]
    out = {}
    out["hypotheses.tex"] = booktabs(
        hypothesis_table(t, meta),
        "IEEE-CIS, the single run (months 2--6, budget 2\\%): every hypothesis with its frozen "
        "criterion (freeze-v1, freeze-v3). 95\\% CIs from 1000 month-stratified day-block "
        "resamples. FP = false positives; daily K = $\\lceil 0.02\\, n_d \\rceil$ per test day, monthly "
        "K per month. H1a and H1b combine both fraud datasets; the ULB verdict is in the last column.",
        "tab:ieee-hypotheses", cm, src + "rq4_hypotheses.csv, v1_hypotheses.csv, h3a.csv, e3_deltas.csv, "
        "rq4_h4r_triples.csv, rq4_member_days.csv, rq4_guard_decisions.csv, rq4_daily_fp.csv, "
        "rq4_pre_deployment_r.csv, run.meta.json",
        align=">{\\raggedright\\arraybackslash}p{1.25cm}>{\\raggedright\\arraybackslash}p{2.75cm}"
              ">{\\raggedright\\arraybackslash}p{2.3cm}>{\\raggedright\\arraybackslash}p{2.0cm}"
              ">{\\raggedright\\arraybackslash}p{1.45cm}>{\\raggedright\\arraybackslash}p{4.6cm}",
        colsep="2.5pt", size="\\scriptsize")
    out["h4r_summary.tex"] = booktabs(
        h4r_main_table(t),
        "H4r in threshold mode (q = 0.98): D = mean over months 2--6 of $|\\log r|$, r = monthly "
        "alert rate at the frozen month-1 threshold divided by its 2\\% target (a month without an "
        "alert counts as 0.5 alert); dD = D(train-percentile + median) $-$ D(min-max + mean), paired. "
        "Robust = train-percentile + median; frozen = min-max + mean (freeze-v1 primary). "
        "The first row decides H4r; the named ensembles are descriptive.",
        "tab:h4r-summary", cm, src + "rq4_hypotheses.csv, rq4_h4r_triples.csv, rq4_h4r_threshold.csv",
        align="lrrrl", colsep="4pt", size="\\small")
    out["h4r_triples.tex"] = booktabs(
        h4r_triples_table(t),
        "H4r and H4r-cost per three-member ensemble (q = 0.98; budget 2\\%, daily K). FP = total "
        "FP@K over the test days, IForest-containing ensembles averaged over five seeds. The "
        "pre-registered cost statistic is the ratio of the mean totals (H4r-cost); the per-ensemble "
        "change is descriptive. Robust = train-percentile + median; frozen = min-max + mean.",
        "tab:h4r-triples", cm, src + "rq4_h4r_triples.csv", align="lrrrrrr", colsep="3pt",
        size="\\footnotesize")
    out["guard_events.tex"] = booktabs(
        guard_table(t),
        f"The run-time guard on IEEE-CIS (budget 2\\%, c = 3): the {len(g['event_days'])} test days on "
        f"which it changed E-all, out of {g['n_days']}. r = alert rate of the member that day at its "
        "frozen month-1 threshold divided by the target; $\\dagger$: r $>$ 3, member excluded that day. "
        "Fewer than two passing members: the static ensemble is used (fallback). IForest: mean of five seeds.",
        "tab:guard-events", cm, src + "rq4_member_days.csv, rq4_guard_decisions.csv, rq4_daily_fp.csv",
        align="l" + "r" * len(g["event_days"]), colsep="4pt", size="\\small")
    out["bound_check.tex"] = booktabs(
        bound_table(t),
        "Label-free bound check of train-percentile + median per test month (q = 0.98, mean of the "
        "IForest seeds): fused alert rate $\\le$ share of rows with at least $\\lceil m/2\\rceil - b$ healthy "
        "members above the threshold (event rate) $\\le$ the bound of the proposition. b = 1 treats the "
        "member with the largest monthly r as broken.",
        "tab:bound-check", cm, src + "rq4_bound_check.csv", align="lrrlrrr", colsep="4pt",
        size="\\scriptsize")
    out["drift_slopes.tex"] = booktabs(
        drift_table(t),
        "Monthly r per member (q = 0.98) and the Evidently drift share of each month against month 1. "
        "Slope: least-squares slope of $|r - 1|$ on the drift share over the five months "
        "(exploratory, no CI).",
        "tab:drift-slopes", cm, src + "rq4_drift_monthly.csv, rq4_drift_slopes.csv",
        align="lrrrrrr", colsep="4pt", size="\\small")
    out["h4d_fragility.tex"] = booktabs(
        h4d_table(t),
        f"H4d (exploratory, developed after seeing ULB): fragility index from the month-1 training "
        f"scores (q = 0.98) and the mean monthly log r. Spearman $\\rho$ = {num(h.loc['H4d', 'estimate'], 3)} "
        "over the six members; no decision.",
        "tab:h4d", cm, src + "rq4_h4d_fragility.csv, rq4_hypotheses.csv", align="lrr", size="\\small")
    failed = p[~p["passes"]]
    changed = int(stress["changed"].sum())
    out["pre_deployment.tex"] = booktabs(
        pre_deployment_table(t),
        "Pre-deployment check on IEEE-CIS month 1 (fit days 0--20, scored days 21--29, label-free): "
        "r per member at each budget; $\\dagger$: r $>$ 3. "
        + (f"No member failed at any budget, so E-stress equals the static ensemble everywhere "
           f"({changed} of {len(stress)} ensemble sets changed)." if failed.empty and changed == 0 else
           f"{len(failed)} member-budget pairs failed; {changed} of {len(stress)} ensemble sets changed."),
        "tab:pre-deployment", cm, src + "rq4_pre_deployment_r.csv, rq4_e_stress_sets.csv",
        align="lrrr", size="\\small")
    out["context.tex"] = booktabs(
        context_table(t),
        "Context (exploratory, E5): days flagged by ADTK on the daily alert rate at the frozen "
        "q = 0.99 threshold, months 2--6. Score drift of E-bench (Evidently, per month and seed): "
        f"{int(sd['drifted'].sum())} of {len(sd)} tests flagged.",
        "tab:context", cm, src + "context_adtk.csv, context_score_drift.csv", align="lrrrr",
        size="\\small")
    return out


def adbench_tex(a: pd.DataFrame, me: str) -> str:
    return booktabs(
        adbench_table(a),
        "Static cost of robust fusion on ADBench (no drift). FP@K, triples: mean over 15 datasets, 5 "
        "split seeds and the 20 three-member ensembles (1500 cells per row); FP@K, E-bench: the same "
        "for E-bench only; beats best / median: share of cells in which the ensemble has fewer FP@K "
        "than its best or its median member. Development data; the +5\\% "
        "margin of H4r-cost was set from the 2\\% rows before IEEE-CIS.",
        "tab:adbench-static", f"{me} (data {short(a['commit'].iloc[0])})",
        "results/tables/robust_fusion/adbench_static.csv",
        align="l>{\\raggedright\\arraybackslash}p{3.3cm}rrrr", colsep="3pt",
        size="\\footnotesize")


def main(argv: list[str]) -> None:
    if argv:
        raise SystemExit("usage: python -m thesis.export_ch5")
    state = git_state()
    if state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    if RUN.resolve() in [OUT.resolve(), *OUT.resolve().parents]:
        raise SystemExit("the Chapter 5 exports never write into the run folder")
    me = short(state["commit"])
    t, meta = load_run()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    LOF_FIG.mkdir(parents=True, exist_ok=True)
    written = []
    for name, tex in tex_outputs(t, meta, me).items():
        (OUT / name).write_text(tex, encoding="utf-8")
        written.append(f"results/tables/ieee_cis/ch5/{name}")
    a = pd.read_csv(RF / "adbench_static.csv")
    (RF / "adbench_static.tex").write_text(adbench_tex(a, me), encoding="utf-8")
    written.append("results/tables/robust_fusion/adbench_static.tex")
    l2 = pd.read_csv(LOF / "l2_controls.csv")
    for fig, stem in ((tradeoff_figure(t), FIG / "h4r_tradeoff"), (h4a_figure(t), FIG / "h4a_scatter"),
                      (lof_figure(l2), LOF_FIG / "l2_controls_r")):
        _save(fig, stem, me)
        written += [str(stem.relative_to(FIGURES.parent.parent)).replace("\\", "/") + e for e in (".pdf", ".png")]
    m = {**state, "command": "python -m thesis.export_ch5",
         "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "reads": {"results/tables/ieee_cis/run/": short(meta["commit"]),
                   "results/tables/robust_fusion/adbench_static.csv": short(a["commit"].iloc[0]),
                   "results/tables/e3_ulb/lof/l2_controls.csv": short(l2["commit"].iloc[0])},
         "recomputed": "nothing: values are selected, combined and rounded only",
         "figures": {"width_cm": WIDTH_CM, "font_pt": PT, "inks": [INK, MUTED]},
         "written": written}
    (OUT / "ch5.meta.json").write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
    print("\n".join(written))


if __name__ == "__main__":
    main(sys.argv[1:])
