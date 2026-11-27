"""E2 figures and the triple LaTeX table, from the E2 tables (no scores needed).

    python -m thesis.e2_figures            # rebuild from results/tables/e2/
    python -m thesis.e2_figures --dedup    # same protocol switches as thesis.e2

thesis.e2 calls write() in the same run, so tables and figures normally name one commit; a
separate rebuild records both its own commit and the commit of the tables it read.

Primary fusion config only (min-max + mean), seed means per dataset (e2_per_dataset).
Gain = -(FP@K ensemble - FP@K member) / K x 100 = false alarms saved per 100 alerts; positive =
the ensemble raises fewer false alarms than that member (best member chosen with test labels,
an oracle reference).

Outputs in figures/e2/:
- overlap_vs_gain.{pdf,png}: one point per combo x dataset (35 x 15 = 525 per budget), x =
  training top-K Jaccard of the members, y = gain vs the best member. H2a predicts Spearman
  rho(overlap, gain) < 0. overlap_vs_gain.csv: pooled rho (points share datasets, so its p-value
  is descriptive) and the within-dataset rho over the 35 combos, for both overlap measures and
  both reference members.
- pair_gain_heatmap.{pdf,png}: 6 x 6, mean gain of each pair vs its best member over the 15
  sets, and in how many sets it beats that member. pair_gain.csv: the same, plus vs median.
- triple_ranking.{pdf,png}: the 20 triples by average FP@K rank, Friedman p and Nemenyi CD.
- e2_figures.meta.json (commits, sources, H2a numbers).
In tables/e2/: e2_triples.tex (booktabs; triple ranking at both budgets with the Friedman test).
The commit is also stored in the PDF/PNG metadata.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from thesis import config, detectors, fusion  # noqa: E402
from thesis.e1 import out_dir  # noqa: E402
from thesis.e1_corr import CMAP  # noqa: E402
from thesis.e2 import OUT, SEP, combo_name, spearman  # noqa: E402
from thesis.export_datasets import git_state  # noqa: E402
from thesis.paths import FIGURES  # noqa: E402

FIG_OUT = FIGURES / "e2"
E_LIT = combo_name(("IForest", "PCA", "KNN"))          # the published default
PAIR_COLOR, TRIPLE_COLOR = "#2a78d6", "#eb6834"         # categorical slots 1, 2
E_LIT_COLOR, OTHER_COLOR = "#2a78d6", "#a3a29b"    # highlight one, the rest muted
INK, MUTED = "#0b0b0b", "#6b6a64"
# Overlap and gain are rounded before ranking: exact ties (common, see the tie shares) must not
# be broken by 1-ulp float noise, e.g. from a CSV round trip (pandas read_csv is not bit-exact).
TIE_DECIMALS = 9


def primary(per_ds: pd.DataFrame) -> pd.DataFrame:
    """Primary-config rows with gain columns (false alarms saved per 100 alerts)."""
    calib, agg = fusion.PRIMARY
    df = per_ds[(per_ds.calib == calib) & (per_ds["agg"] == agg)].copy()
    df["gain_best"] = (-100 * df["delta_fp_best_rel"]).round(TIE_DECIMALS)
    df["gain_median"] = (-100 * df["delta_fp_median_rel"]).round(TIE_DECIMALS)
    return df


def overlap_gain_stats(prim: pd.DataFrame) -> pd.DataFrame:
    """Per budget x overlap measure x reference member: pooled Spearman rho over all
    combo x dataset points, and rho within each dataset over its combos (median, #negative)."""
    rows = []
    for b, g in prim.groupby("budget", sort=True):
        for ov in ("train_jaccard", "train_spearman"):
            for ref in ("best", "median"):
                y = f"gain_{ref}"
                d = g[["dataset", ov, y]].dropna().round({ov: TIE_DECIMALS, y: TIE_DECIMALS})
                rho, p = stats.spearmanr(d[ov], d[y])
                within = pd.Series({ds: spearman(h[ov].to_numpy(), h[y].to_numpy())
                                    for ds, h in d.groupby("dataset")}).dropna()
                rows.append({"budget": b, "overlap": ov, "reference": ref, "n_points": len(d),
                             "n_dropped_nan": len(g) - len(d), "rho_pooled": float(rho),
                             "p_pooled": float(p), "rho_within_median": float(within.median()),
                             "n_datasets_rho_negative": int((within < 0).sum()),
                             "n_datasets": len(within)})
    return pd.DataFrame(rows)


def pair_gain(prim: pd.DataFrame) -> pd.DataFrame:
    """Per budget and pair: mean gain over the datasets and #datasets beating the member."""
    g = prim[prim["size"] == 2].groupby(["budget", "combo"], sort=True)
    df = pd.DataFrame({"mean_gain_best": g["gain_best"].mean(),
                       "mean_gain_median": g["gain_median"].mean(),
                       "n_beats_best": g["beats_best"].sum().astype(int),
                       "n_ties_best": g["ties_best"].sum().astype(int),
                       "n_beats_median": g["beats_median"].sum().astype(int),
                       "n_datasets": g["dataset"].nunique()}).reset_index()
    df[["det_a", "det_b"]] = df["combo"].str.split(SEP, expand=True)
    return df


def pair_matrix(pg: pd.DataFrame, value: str, dets=detectors.DETECTORS) -> pd.DataFrame:
    """Symmetric detector x detector matrix of one pair_gain column; NaN on the diagonal."""
    m = pd.DataFrame(np.nan, index=list(dets), columns=list(dets))
    for r in pg.itertuples():
        m.loc[r.det_a, r.det_b] = m.loc[r.det_b, r.det_a] = getattr(r, value)
    return m


def _save(fig, path_stem, commit: str) -> None:
    for ext in ("pdf", "png"):
        meta = {"Subject": f"commit {commit}"} if ext == "pdf" else {"Comment": f"commit {commit}"}
        fig.savefig(f"{path_stem}.{ext}", dpi=300 if ext == "png" else None, metadata=meta)
    plt.close(fig)


def _clean(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def scatter(prim: pd.DataFrame, st: pd.DataFrame, path_stem, commit: str) -> None:
    budgets = sorted(prim.budget.unique())
    fig, axes = plt.subplots(1, len(budgets), figsize=(3.4 * len(budgets), 3.2), sharey=True)
    for ax, b in zip(np.atleast_1d(axes), budgets):
        g = prim[prim.budget == b]
        ax.axhline(0, color=MUTED, lw=0.8, ls="--", zorder=0)
        for size, color, marker, label in ((2, PAIR_COLOR, "o", "pairs"),
                                           (3, TRIPLE_COLOR, "^", "triples")):
            h = g[g["size"] == size]
            ax.scatter(h.train_jaccard, h.gain_best, s=14, c=color, marker=marker, alpha=0.7,
                       linewidths=0.4, edgecolors="white", label=f"{label} ({len(h)})")
        s = st[(st.budget == b) & (st.overlap == "train_jaccard")
               & (st.reference == "best")].iloc[0]
        ax.text(0.98, 0.03, f"Spearman ρ = {s.rho_pooled:+.3f}\n(n = {s.n_points}; within-set "
                            f"median {s.rho_within_median:+.2f})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK)
        ax.set_title(f"Budget {b:.0%}", fontsize=9)
        ax.set_xlabel("Training top-K overlap of the members (Jaccard)", fontsize=8.5)
        _clean(ax)
    axes = np.atleast_1d(axes)
    axes[0].set_ylabel("False alarms saved vs best member\n(per 100 alerts)", fontsize=8.5)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7.5, frameon=False, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, path_stem, commit)


def heatmap(pg: pd.DataFrame, path_stem, commit: str) -> None:
    budgets = sorted(pg.budget.unique())
    mats = {b: pair_matrix(pg[pg.budget == b], "mean_gain_best") for b in budgets}
    beats = {b: pair_matrix(pg[pg.budget == b], "n_beats_best") for b in budgets}
    n = int(pg.n_datasets.max())
    vmax = max(np.nanmax(np.abs(m.to_numpy())) for m in mats.values())
    cmap = CMAP.with_extremes(bad="white")
    fig, axes = plt.subplots(1, len(budgets), figsize=(3.6 * len(budgets) + 0.6, 3.5))
    for ax, b in zip(np.atleast_1d(axes), budgets):
        m, k = mats[b], len(mats[b])
        im = ax.imshow(np.ma.masked_invalid(m.to_numpy()), cmap=cmap, vmin=-vmax, vmax=vmax)
        for i in range(k):
            for j in range(k):
                v = m.iat[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:+.1f}\n{int(beats[b].iat[i, j])}/{n}", ha="center",
                        va="center", fontsize=6.5, linespacing=1.1,
                        color="white" if abs(v) > 0.6 * vmax else INK)
        ax.set_xticks(range(k), m.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(k), m.index, fontsize=8)
        ax.set_xticks(np.arange(k + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(k + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", lw=2)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"Budget {b:.0%}", fontsize=9)
    cb = fig.colorbar(im, ax=list(np.atleast_1d(axes)), fraction=0.025, pad=0.03)
    cb.set_label("False alarms saved vs best member\n(per 100 alerts, mean over sets)",
                 fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_visible(False)
    calib, agg = (fusion.LABELS[x] for x in fusion.PRIMARY)
    fig.suptitle(f"Pairs, {calib} + {agg}: mean gain; "
                 f"below it, #sets (of {n}) where the pair beats its best member", fontsize=8.5)
    _save(fig, path_stem, commit)


def triple_ranking(summ: pd.DataFrame, fried: pd.DataFrame, path_stem, commit: str) -> None:
    calib, agg = fusion.PRIMARY
    tri = summ[(summ["size"] == 3) & (summ.calib == calib) & (summ["agg"] == agg)]
    budgets = sorted(tri.budget.unique())
    n_combos = tri.combo.nunique()
    fig, axes = plt.subplots(1, len(budgets), figsize=(3.6 * len(budgets), 4.3))
    for ax, b in zip(np.atleast_1d(axes), budgets):
        g = tri[tri.budget == b].sort_values(["avg_rank_fp_at_k", "combo"])
        f = fried[(fried.budget == b) & (fried["size"] == 3)].iloc[0]
        y = np.arange(len(g))
        colors = [E_LIT_COLOR if c == E_LIT else OTHER_COLOR for c in g.combo]
        ax.hlines(y, 1, g.avg_rank_fp_at_k, color="#e4e3df", lw=0.8, zorder=0)
        ax.scatter(g.avg_rank_fp_at_k, y, s=22, c=colors, edgecolors="white", linewidths=0.5,
                   zorder=2)
        top = g.avg_rank_fp_at_k.iloc[0]
        ax.plot([top, top + f.nemenyi_cd], [-1.3, -1.3], color=INK, lw=1.2)
        ax.text(top + f.nemenyi_cd / 2, -1.6, f"Nemenyi CD = {f.nemenyi_cd:.2f}", ha="center",
                va="bottom", fontsize=7.5, color=INK)
        ax.set_yticks(y, g.combo, fontsize=7)
        ax.set_ylim(len(g) - 0.4, -2.6)
        ax.set_xlim(1, n_combos)
        ax.set_xticks([1, 5, 10, 15, 20] if n_combos == 20 else ax.get_xticks())
        ax.set_title(f"Budget {b:.0%}: Friedman p = {f.friedman_p:.3f}", fontsize=9)
        ax.set_xlabel(f"Average FP@K rank among the {n_combos} triples", fontsize=8.5)
        _clean(ax)
    axes = np.atleast_1d(axes)
    axes[0].scatter([], [], s=22, c=E_LIT_COLOR, label=f"E-lit = E-bench ({E_LIT})")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7.5, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, path_stem, commit)


def latex_triples(summ: pd.DataFrame, fried: pd.DataFrame, meta: dict) -> str:
    """booktabs table: triples sorted by average rank at the first (smallest) budget."""
    calib, agg = fusion.PRIMARY
    tri = summ[(summ["size"] == 3) & (summ.calib == calib) & (summ["agg"] == agg)]
    budgets = sorted(tri.budget.unique())
    by = {b: tri[tri.budget == b].sort_values(["avg_rank_fp_at_k", "combo"]).reset_index(drop=True)
          for b in budgets}
    # competition ranking: equal average ranks share the best position (1, 2, 2, 4)
    pos = {b: dict(zip(by[b].combo, by[b].avg_rank_fp_at_k.round(TIE_DECIMALS)
                       .rank(method="min").astype(int))) for b in budgets}
    cols = {b: by[b].set_index("combo") for b in budgets}
    n = int(tri.n_datasets.max())
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{Budget {round(b * 100)}\\%}}" for b in budgets)
    rules = " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(budgets)))
    sub = " & ".join(["Avg.\\ rank (pos.)", "$<$ best", "$<$ median"] * len(budgets))
    lines = [f"% commit {meta['commit']}; source e2_summary.csv, e2_friedman.csv "
             f"({calib} + {agg})",
             "\\begin{table}[t]", "\\centering", "\\small",
             "\\setlength{\\tabcolsep}{5pt}",          # fits the 16 cm text block
             f"\\caption{{The {len(by[budgets[0]])} detector triples "
             f"({fusion.LABELS[calib]} calibration, {fusion.LABELS[agg]} fusion) by average FP@K rank over {n} ADBench sets (1 = fewest false "
             f"alarms), sorted by the {round(budgets[0] * 100)}\\% budget. $<$ best / $<$ median: "
             f"number of sets where the triple raises fewer false alarms than its best / "
             f"median member. Equal average ranks share a position. E-lit and E-bench are both "
             f"IForest+PCA+KNN.}}",
             "\\label{tab:e2-triples}",
             "\\begin{tabular}{l" + "rrr" * len(budgets) + "}", "\\toprule",
             f" & {head} \\\\", rules, f"Triple & {sub} \\\\", "\\midrule"]
    for c in by[budgets[0]].combo:
        cells = []
        for b in budgets:
            r = cols[b].loc[c]
            cells += [f"{r.avg_rank_fp_at_k:.2f} ({pos[b][c]})", f"{int(r.n_beats_best)}/{n}",
                      f"{int(r.n_beats_median)}/{n}"]
        lines.append(f"{c} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    ftxt = []
    for b in budgets:
        f = fried[(fried.budget == b) & (fried["size"] == 3)].iloc[0]
        ftxt.append(f"\\multicolumn{{3}}{{c}}{{$\\chi^2 = {f.friedman_chi2:.2f}$, "
                    f"$p = {f.friedman_p:.3f}$, CD $= {f.nemenyi_cd:.2f}$}}")
    lines += ["Friedman & " + " & ".join(ftxt) + " \\\\", "\\bottomrule", "\\end{tabular}",
              "\\end{table}"]
    return "\n".join(lines) + "\n"


def write(per_ds: pd.DataFrame, summ: pd.DataFrame, fried: pd.DataFrame, fig_dir, tab_dir,
          meta: dict, tables_commit: str | None = None) -> dict:
    fig_dir.mkdir(parents=True, exist_ok=True)
    commit = meta["commit"]
    prim = primary(per_ds)
    st, pg = overlap_gain_stats(prim), pair_gain(prim)
    scatter(prim, st, fig_dir / "overlap_vs_gain", commit)
    heatmap(pg, fig_dir / "pair_gain_heatmap", commit)
    triple_ranking(summ, fried, fig_dir / "triple_ranking", commit)
    st.assign(commit=commit).to_csv(fig_dir / "overlap_vs_gain.csv", index=False)
    pg.assign(commit=commit).to_csv(fig_dir / "pair_gain.csv", index=False)
    (tab_dir / "e2_triples.tex").write_text(latex_triples(summ, fried, meta), encoding="utf-8")
    h2a = st[(st.overlap == "train_jaccard") & (st.reference == "best")]
    info = {**{k: meta[k] for k in ("commit", "dirty", "cache_key", "protocol", "created_utc")},
            "tables_commit": tables_commit or commit,
            "source": "e2_per_dataset.csv, e2_summary.csv, e2_friedman.csv (seed means per "
                      "dataset)",
            "primary_config": fusion.PRIMARY, "e_lit": E_LIT,
            "gain": "-(FP@K ensemble - FP@K member) / K x 100 (false alarms saved per 100 "
                    "alerts); best member chosen per run with test labels (oracle reference)",
            "h2a_rho_jaccard_vs_gain_best": {f"{r.budget:.0%}": {
                "rho_pooled": r.rho_pooled, "p_pooled": r.p_pooled, "n_points": r.n_points,
                "rho_within_median": r.rho_within_median,
                "n_datasets_rho_negative": r.n_datasets_rho_negative} for r in h2a.itertuples()},
            "figures": ["overlap_vs_gain", "pair_gain_heatmap", "triple_ranking"],
            "latex": "tables/e2/e2_triples.tex"}
    (fig_dir / "e2_figures.meta.json").write_text(json.dumps(info, indent=1) + "\n",
                                                  encoding="utf-8")
    return info


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="python -m thesis.e2_figures")
    config.add_arguments(parser)
    protocol = config.from_args(parser.parse_args(argv))
    state = git_state()
    if state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    tab = out_dir(protocol, OUT)
    per_ds = pd.read_csv(tab / "e2_per_dataset.csv")
    summ = pd.read_csv(tab / "e2_summary.csv")
    fried = pd.read_csv(tab / "e2_friedman.csv")
    tables_commit = str(per_ds["commit"].iloc[0])
    runs_meta = json.loads((tab / "e2_runs.meta.json").read_text(encoding="utf-8"))
    meta = {**state, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cache_key": runs_meta["cache_key"], "protocol": protocol.as_dict()}
    if tables_commit != state["commit"]:
        print(f"note: tables are from {tables_commit[:7]}, figures from {state['commit'][:7]}")
    info = write(per_ds, summ, fried, out_dir(protocol, FIG_OUT), tab, meta, tables_commit)
    print(json.dumps(info["h2a_rho_jaccard_vs_gain_best"], indent=1))
    print("wrote", out_dir(protocol, FIG_OUT))


if __name__ == "__main__":
    main(sys.argv[1:])
