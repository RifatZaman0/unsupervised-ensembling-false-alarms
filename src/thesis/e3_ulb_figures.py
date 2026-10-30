"""E3 ULB figures, from the E3 ULB tables (called by thesis.e3_ulb).

Outputs in figures/e3_ulb/ (pdf + png, commit in the file metadata):
- delta_primary_budget: every ensemble x fusion config at the primary ULB budget, dFP per 100
  alerts against the median single, the default IForest and the oracle-best single, with the
  95% paired-bootstrap CI; the primary fusion config highlighted. Left of 0 = fewer false alarms.
- threshold_check: day-2 alert rate of every single and every ensemble (primary fusion) under
  each frozen day-1 quantile threshold, against the target 1 - q.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thesis import bootstrap, fusion  # noqa: E402
from thesis.e2_figures import E_LIT_COLOR, INK, MUTED, OTHER_COLOR, _save  # noqa: E402

COMPARATORS = {bootstrap.MEDIAN: "vs median single", bootstrap.DEFAULT_IFOREST: "vs default IForest",
               bootstrap.ORACLE: "vs oracle-best single"}


def _label(ens: str, calib: str, agg: str, cfg: dict) -> str:
    from thesis.e3_ulb import aliases
    alias = aliases(cfg)
    name = f"{ens} (= {alias[ens]})" if ens in alias else ens
    return f"{name}, {fusion.LABELS[calib]} + {fusion.LABELS[agg]}"


def delta_figure(deltas: pd.DataFrame, cfg: dict, out, commit: str) -> None:
    from thesis.e3_ulb import fusion_configs
    pb = cfg["budgets"]["ulb"]["primary"]
    primary = fusion_configs(cfg)[0]
    d = deltas[(deltas.budget == pb) & deltas.comparator.isin(list(COMPARATORS))]
    order = d.drop_duplicates("method")[["method", "ensemble", "calib", "agg"]].reset_index(drop=True)
    ypos = {m: len(order) - 1 - i for i, m in enumerate(order.method)}
    fig, axes = plt.subplots(1, len(COMPARATORS), figsize=(11, 0.32 * len(order) + 1.4),
                             sharey=True)
    for ax, (comp, title) in zip(axes, COMPARATORS.items()):
        g = d[d.comparator == comp]
        for r in g.itertuples():
            is_primary = (r.calib, r.agg) == primary
            color = E_LIT_COLOR if is_primary else OTHER_COLOR
            y = ypos[r.method]
            ax.plot([r.d_fp_per_100_ci_low, r.d_fp_per_100_ci_high], [y, y], color=color, lw=2,
                    solid_capstyle="round")
            ax.plot(r.d_fp_per_100, y, "o", ms=6, color=color, mec="white", mew=1)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(title, fontsize=10, color=INK)
        ax.grid(axis="x", color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("dFP@K per 100 alerts (95% CI)", fontsize=9, color=MUTED)
    axes[0].set_yticks(list(ypos.values()))
    axes[0].set_yticklabels([_label(r.ensemble, r.calib, r.agg, cfg) for r in order.itertuples()],
                            fontsize=8, color=INK)
    fig.suptitle(f"ULB day 2, budget {pb:.1%}: ensemble minus comparator (left of 0 = fewer "
                 f"false alarms); highlighted: primary fusion", fontsize=10, color=INK)
    fig.tight_layout()
    _save(fig, out / "delta_primary_budget", commit)


def threshold_figure(thr: pd.DataFrame, cfg: dict, out, commit: str) -> None:
    from thesis.e3_ulb import display, fusion_configs
    calib, agg = fusion_configs(cfg)[0]
    t = thr[(thr.kind != "ensemble") | ((thr.calib == calib) & (thr["agg"] == agg))]
    qs = list(dict.fromkeys(t.q))
    methods = list(dict.fromkeys(t.method))
    ypos = {m: len(methods) - 1 - i for i, m in enumerate(methods)}
    fig, axes = plt.subplots(1, len(qs), figsize=(9, 0.32 * len(methods) + 1.4), sharey=True)
    for ax, q in zip(np.atleast_1d(axes), qs):
        g = t[t.q == q]
        for r in g.itertuples():
            color = E_LIT_COLOR if r.kind == "ensemble" else OTHER_COLOR
            ax.plot(100 * r.alert_rate, ypos[r.method], "o", ms=6, color=color, mec="white", mew=1)
        ax.axvline(100 * (1 - q), color=INK, lw=0.8, ls="--")
        ax.set_title(f"q = {q} (target {100 * (1 - q):.1f}% of transactions)", fontsize=10,
                     color=INK)
        ax.set_xlabel("day-2 alert rate (%)", fontsize=9, color=MUTED)
        ax.grid(axis="x", color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(list(ypos.values()))
    ax0.set_yticklabels([display(m, cfg) for m in methods], fontsize=8, color=INK)
    fig.suptitle("ULB: day-1 quantile threshold applied unchanged to day 2 (ensembles "
                 "highlighted; dashed = 1 - q)", fontsize=10, color=INK)
    fig.tight_layout()
    _save(fig, out / "threshold_check", commit)


def write(deltas: pd.DataFrame, thr: pd.DataFrame, cfg: dict, out, commit: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    delta_figure(deltas, cfg, out, commit)
    threshold_figure(thr, cfg, out, commit)
