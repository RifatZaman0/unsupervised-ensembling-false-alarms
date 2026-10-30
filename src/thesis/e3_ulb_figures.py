"""E3 ULB figures, from the E3 ULB tables (called by thesis.e3_ulb, or rebuilt alone).

    python -m thesis.e3_ulb_figures     # redraw both figures and rewrite the three .tex tables
                                        # from results/tables/e3_ulb/*.csv (no refit, no bootstrap)

Outputs in figures/e3_ulb/ (pdf + png, commit in the file metadata):
- delta_primary_budget: every ensemble x fusion config at the primary ULB budget, dFP per 100
  alerts against the median single, the default IForest and the oracle-best single, with the
  95% paired-bootstrap CI; the primary fusion config highlighted. Left of 0 = fewer false alarms.
- threshold_check: day-2 alert rate of every single and every ensemble (primary fusion) under
  each frozen day-1 quantile threshold, against the target 1 - q.

Both are drawn at the TDK text width (16 cm) with no text below 8 pt, so they are printed at
scale 1 and every label stays readable. The standalone rebuild reads the tables only: it plots
the same numbers and computes nothing new.
"""
from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from thesis import bootstrap, fusion  # noqa: E402
from thesis.e2_figures import E_LIT_COLOR, INK, MUTED, OTHER_COLOR, _save  # noqa: E402

COMPARATORS = {bootstrap.MEDIAN: "vs median single", bootstrap.DEFAULT_IFOREST: "vs default IForest",
               bootstrap.ORACLE: "vs oracle-best single"}
WIDTH_CM = 16.0                  # TDK text block: 21 cm A4 minus 2 x 2.5 cm margins
MIN_PT = 8                       # smallest text at print size
TICK_PT, LABEL_PT, TITLE_PT = 8, 8, 9


def _label(ens: str, calib: str, agg: str, cfg: dict) -> str:
    from thesis.e3_ulb import aliases
    alias = aliases(cfg)
    name = f"{ens} (= {alias[ens]})" if ens in alias else ens
    return f"{name}, {fusion.LABELS[calib]} + {fusion.LABELS[agg]}"


def _style(ax) -> None:
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=TICK_PT)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))


def delta_fig(deltas: pd.DataFrame, cfg: dict):
    from thesis.e3_ulb import fusion_configs
    pb = cfg["budgets"]["ulb"]["primary"]
    primary = fusion_configs(cfg)[0]
    d = deltas[np.isclose(deltas.budget, pb) & deltas.comparator.isin(list(COMPARATORS))]
    order = d.drop_duplicates("method")[["method", "ensemble", "calib", "agg"]].reset_index(drop=True)
    ypos = {m: len(order) - 1 - i for i, m in enumerate(order.method)}
    fig, axes = plt.subplots(1, len(COMPARATORS), sharey=True, layout="constrained",
                             figsize=(WIDTH_CM / 2.54, (0.24 * len(order) + 1.5)))
    for ax, (comp, title) in zip(axes, COMPARATORS.items()):
        g = d[d.comparator == comp]
        for r in g.itertuples():
            color = E_LIT_COLOR if (r.calib, r.agg) == primary else OTHER_COLOR
            y = ypos[r.method]
            ax.plot([r.d_fp_per_100_ci_low, r.d_fp_per_100_ci_high], [y, y], color=color, lw=1.6,
                    solid_capstyle="round")
            ax.plot(r.d_fp_per_100, y, "o", ms=4.5, color=color, mec="white", mew=0.8)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(title, fontsize=TITLE_PT, color=INK)
        _style(ax)
        ax.set_xlabel("dFP@K per 100 alerts", fontsize=LABEL_PT, color=MUTED)
    for ax in axes[1:]:          # shared y: no labels there, and no spine next to the 0 line
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(list(ypos.values()))
    axes[0].set_yticklabels([_label(r.ensemble, r.calib, r.agg, cfg) for r in order.itertuples()],
                            fontsize=TICK_PT, color=INK)
    fig.suptitle(f"ULB day 2, budget {pb:.1%}: ensemble minus comparator with 95% CI\n"
                 f"(left of 0 = fewer false alarms; blue = primary fusion)",
                 fontsize=TITLE_PT, color=INK)
    return fig


def threshold_fig(thr: pd.DataFrame, cfg: dict):
    from thesis.e3_ulb import display, fusion_configs
    calib, agg = fusion_configs(cfg)[0]
    t = thr[(thr.kind != "ensemble") | ((thr.calib == calib) & (thr["agg"] == agg))]
    qs = list(dict.fromkeys(t.q))
    methods = list(dict.fromkeys(t.method))
    ypos = {m: len(methods) - 1 - i for i, m in enumerate(methods)}
    fig, axes = plt.subplots(1, len(qs), sharey=True, layout="constrained",
                             figsize=(WIDTH_CM / 2.54, 0.26 * len(methods) + 1.5))
    for ax, q in zip(np.atleast_1d(axes), qs):
        g = t[t.q == q]
        for r in g.itertuples():
            color = E_LIT_COLOR if r.kind == "ensemble" else OTHER_COLOR
            ax.plot(100 * r.alert_rate, ypos[r.method], "o", ms=5, color=color, mec="white",
                    mew=0.8)
        ax.axvline(100 * (1 - q), color=INK, lw=0.8, ls="--")
        ax.set_title(f"q = {q} (target {100 * (1 - q):.1f}%)", fontsize=TITLE_PT, color=INK)
        ax.set_xlabel("day-2 alert rate (%)", fontsize=LABEL_PT, color=MUTED)
        _style(ax)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(list(ypos.values()))
    ax0.set_yticklabels([display(m, cfg) for m in methods], fontsize=TICK_PT, color=INK)
    fig.suptitle("ULB: day-1 quantile threshold applied unchanged to day 2\n"
                 "(blue = ensembles; dashed = target 1 - q)", fontsize=TITLE_PT, color=INK)
    return fig


def delta_figure(deltas: pd.DataFrame, cfg: dict, out, commit: str) -> None:
    _save(delta_fig(deltas, cfg), out / "delta_primary_budget", commit)


def threshold_figure(thr: pd.DataFrame, cfg: dict, out, commit: str) -> None:
    _save(threshold_fig(thr, cfg), out / "threshold_check", commit)


def write(deltas: pd.DataFrame, thr: pd.DataFrame, cfg: dict, out, commit: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    delta_figure(deltas, cfg, out, commit)
    threshold_figure(thr, cfg, out, commit)


def main(argv: list[str]) -> None:
    """Redraw both figures from the committed E3 tables (no fitting, no bootstrap)."""
    from thesis import frozen
    from thesis.e3_ulb import FIG_OUT, OUT
    from thesis.export_datasets import git_state
    if argv:
        raise SystemExit("usage: python -m thesis.e3_ulb_figures")
    state = git_state()
    if state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    cfg = frozen.load()
    deltas = pd.read_csv(OUT / "e3_ulb_deltas.csv")
    thr = pd.read_csv(OUT / "e3_ulb_threshold.csv")
    tables_commit = str(deltas["commit"].iloc[0])
    write(deltas, thr, cfg, FIG_OUT, state["commit"])
    # the .tex tables, re-formatted from the same CSVs (layout only; numbers as in the CSVs)
    from thesis.e3_ulb import tex_tables
    stamp = f"{state['commit']} (tables {tables_commit[:7]})"
    for name, text in tex_tables(pd.read_csv(OUT / "e3_ulb_budget.csv"),
                                 pd.read_csv(OUT / "e3_ulb_hypotheses.csv"), thr, cfg,
                                 stamp).items():
        (OUT / name).write_text(text, encoding="utf-8")
    (FIG_OUT / "e3_ulb_figures.meta.json").write_text(json.dumps({
        **state, "command": "python -m thesis.e3_ulb_figures",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tables_commit": tables_commit,
        "source": "results/tables/e3_ulb/e3_ulb_deltas.csv, e3_ulb_threshold.csv",
        "width_cm": WIDTH_CM, "min_font_pt": MIN_PT,
        "figures": ["delta_primary_budget", "threshold_check"],
        "note": "Redrawn from the tables at the TDK text width; same numbers as the tables "
                "(nothing recomputed)."}, indent=1) + "\n", encoding="utf-8")
    print(textwrap.dedent(f"""\
        wrote {FIG_OUT} (tables from {tables_commit[:7]}, figures from {state['commit'][:7]})"""))


if __name__ == "__main__":
    main(sys.argv[1:])
