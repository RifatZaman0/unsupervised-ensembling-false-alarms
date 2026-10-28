"""E1 duplicate robustness check: main E1 (ADBench duplicates kept) vs E1 with exact
(features, label) duplicates removed (`python -m thesis.e1 --dedup`).

Built from the two e1_per_dataset.csv files only (no scoring). Compared per budget:
average FP@K and PR-AUC ranks of the six detectors, the Kendall tau between the two rank
orders, the Friedman p-values, and per dataset whether the detector ranking (Spearman) and the
best detector stay the same. FP@K values themselves are not compared one to one: removing rows
changes the split and usually n_test and K. A dataset counts as changed when dedup removes at
least one row (not when n_test changes: letter loses 2 rows and keeps n_test = 640). Datasets
with nothing removed must come out identical, which is checked.

    python -m thesis.e1 --dedup      # first, writes tables/e1/variants/dedup/
    python -m thesis.e1_dedup

Outputs in tables/e1/variants/dedup/: dedup_vs_main.csv (detector x budget),
dedup_vs_main_per_dataset.csv, dedup_vs_main.md and dedup_vs_main.meta.json (both source
commits and this commit).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pandas as pd
from scipy import stats

from thesis import config, data, e1_figures, e1_summary
from thesis.e1 import OUT, out_dir
from thesis.export_datasets import git_state

DEDUP_DIR = out_dir(config.Protocol(adbench_dedup=True))


def rows_removed_per_dataset(datasets) -> dict[str, int]:
    """Rows that --dedup drops per dataset, from the raw files (duplicates kept on load)."""
    config.use(config.Protocol())
    return {n: data.duplicate_stats(*data.load_adbench(n))["rows_removed"] for n in datasets}


def compare(main_pd: pd.DataFrame, dedup_pd: pd.DataFrame,
            rows_removed: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """(detector x budget table, dataset x budget table, per-budget summaries).

    rows_removed: rows dropped by --dedup per dataset; 0 means the dataset is unchanged."""
    sm, sd = e1_summary.summary(main_pd), e1_summary.summary(dedup_pd)
    cols = ["avg_rank_fp_at_k", "avg_rank_pr_auc", "n_best_fp_at_k", "mean_fp_at_k", "mean_pr_auc"]
    det = sm[["budget", "detector", *cols]].merge(sd[["budget", "detector", *cols]],
                                                 on=["budget", "detector"],
                                                 suffixes=("_main", "_dedup"))
    det["delta_avg_rank_fp_at_k"] = det.avg_rank_fp_at_k_dedup - det.avg_rank_fp_at_k_main
    det = det.sort_values(["budget", "avg_rank_fp_at_k_main", "detector"], ignore_index=True)

    key = ["dataset", "budget", "detector"]
    both = main_pd[[*key, "rank_fp_at_k", "fp_at_k", "n_test"]].merge(
        dedup_pd[[*key, "rank_fp_at_k", "fp_at_k", "n_test"]], on=key, suffixes=("_main", "_dedup"))
    ds_rows = []
    for (ds, b), g in both.groupby(["dataset", "budget"], sort=True):
        best = lambda col: sorted(g.detector[g[col] == g[col].min()])
        rho = stats.spearmanr(g.rank_fp_at_k_main, g.rank_fp_at_k_dedup)[0]
        ds_rows.append({"dataset": ds, "budget": b, "rows_removed": int(rows_removed.get(ds, 0)),
                        "n_test_main": int(g.n_test_main.iloc[0]),
                        "n_test_dedup": int(g.n_test_dedup.iloc[0]),
                        "spearman_ranks": float(rho),  # NaN if one side has all ranks tied
                        "best_main": "/".join(best("rank_fp_at_k_main")),
                        "best_dedup": "/".join(best("rank_fp_at_k_dedup")),
                        "identical": bool((g.fp_at_k_main == g.fp_at_k_dedup).all()
                                          and (g.n_test_main == g.n_test_dedup).all())})
    per_ds = pd.DataFrame(ds_rows)
    per_ds["best_changed"] = per_ds.best_main != per_ds.best_dedup

    summaries = []
    for b, g in det.groupby("budget", sort=True):
        fm = e1_figures.friedman_nemenyi(main_pd, "fp_at_k", b)
        fd = e1_figures.friedman_nemenyi(dedup_pd, "fp_at_k", b)
        tau = stats.kendalltau(g.avg_rank_fp_at_k_main, g.avg_rank_fp_at_k_dedup)[0]
        p = per_ds[per_ds.budget == b]
        changed = p[p.rows_removed > 0]
        summaries.append({
            "budget": b, "kendall_tau_avg_rank_fp_at_k": float(tau),
            "top_main": min(fm["avg_ranks"], key=fm["avg_ranks"].get),
            "top_dedup": min(fd["avg_ranks"], key=fd["avg_ranks"].get),
            "friedman_p_main": fm["friedman_p"], "friedman_p_dedup": fd["friedman_p"],
            "groups_dedup": fd["groups"],
            "max_abs_delta_avg_rank": float(g.delta_avg_rank_fp_at_k.abs().max()),
            "datasets_changed_by_dedup": sorted(changed.dataset),
            "datasets_best_changed": sorted(p.dataset[p.best_changed]),
            "unchanged_datasets_identical": bool(p[p.rows_removed == 0].identical.all())})
    return det, per_ds, summaries


def markdown(det: pd.DataFrame, summaries: list, meta: dict) -> str:
    lines = ["# E1 duplicate robustness check: duplicates kept (main) vs removed (--dedup)", "",
             f"Main E1 commit `{meta['commit_main'][:7]}` · dedup E1 commit "
             f"`{meta['commit_dedup'][:7]}` · comparison commit `{meta['commit'][:7]}` · "
             f"created {meta['created_utc']}", "",
             "Ranks per dataset (seed means), 1 = fewest false alarms; averaged over the 15 sets. "
             "FP@K values are not compared one to one, because removing rows changes n_test and K."]
    for s in summaries:
        g = det[det.budget == s["budget"]]
        lines += ["", f"## Budget {s['budget']:.0%}", "",
                  f"Kendall τ between the two average-rank orders: {s['kendall_tau_avg_rank_fp_at_k']:.3f}. "
                  f"Top detector: {s['top_main']} (main), {s['top_dedup']} (dedup). "
                  f"Friedman p: {s['friedman_p_main']:.3g} (main), {s['friedman_p_dedup']:.3g} (dedup). "
                  f"Datasets changed by dedup: {len(s['datasets_changed_by_dedup'])}; best detector "
                  f"changed on: {', '.join(s['datasets_best_changed']) or 'none'}.", "",
                  "| Detector | avg rank FP@K main | dedup | Δ | avg rank PR-AUC main | dedup "
                  "| #best FP@K main | dedup |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in g.itertuples():
            lines.append(f"| {r.detector} | {r.avg_rank_fp_at_k_main:.2f} | "
                         f"{r.avg_rank_fp_at_k_dedup:.2f} | {r.delta_avg_rank_fp_at_k:+.2f} | "
                         f"{r.avg_rank_pr_auc_main:.2f} | {r.avg_rank_pr_auc_dedup:.2f} | "
                         f"{r.n_best_fp_at_k_main} | {r.n_best_fp_at_k_dedup} |")
    return "\n".join(lines) + "\n"


def write(main_dir, dedup_dir, meta: dict,
          rows_removed: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    src = {}
    for tag, d in (("main", main_dir), ("dedup", dedup_dir)):
        m = json.loads((d / "e1_runs.meta.json").read_text(encoding="utf-8"))
        if m["protocol"]["adbench_dedup"] != (tag == "dedup") or m["dirty"]:
            raise ValueError(f"{d}: expected a clean {tag} E1 run, got {m['protocol']}")
        src[tag] = m
    det, per_ds, summaries = compare(pd.read_csv(main_dir / "e1_per_dataset.csv"),
                                     pd.read_csv(dedup_dir / "e1_per_dataset.csv"), rows_removed)
    meta = {**meta, "commit_main": src["main"]["commit"], "commit_dedup": src["dedup"]["commit"],
            "cache_key_main": src["main"]["cache_key"], "cache_key_dedup": src["dedup"]["cache_key"]}
    det.assign(commit=meta["commit"]).to_csv(dedup_dir / "dedup_vs_main.csv", index=False)
    per_ds.assign(commit=meta["commit"]).to_csv(dedup_dir / "dedup_vs_main_per_dataset.csv",
                                                index=False)
    (dedup_dir / "dedup_vs_main.md").write_text(markdown(det, summaries, meta), encoding="utf-8")
    (dedup_dir / "dedup_vs_main.meta.json").write_text(
        json.dumps({**meta, "summaries": summaries}, indent=1) + "\n", encoding="utf-8")
    return det, per_ds, summaries


def main(argv: list[str]) -> None:
    state = git_state()
    if state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    meta = {**state, "command": " ".join(["python -m thesis.e1_dedup", *argv]),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write(OUT, DEDUP_DIR, meta, rows_removed_per_dataset(data.ADBENCH_15))
    print((DEDUP_DIR / "dedup_vs_main.md").read_text(encoding="utf-8"))
    print("wrote", DEDUP_DIR)


if __name__ == "__main__":
    main(sys.argv[1:])
