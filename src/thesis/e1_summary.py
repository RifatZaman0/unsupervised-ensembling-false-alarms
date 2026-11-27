"""E1 summary table (first written for an earlier version of this work): detectors compared over the 15 ADBench sets, per budget.

Built from e1_runs.csv only (no scoring). Seeds are averaged within a dataset first, so every
dataset counts once. Per dataset, detectors are ranked by FP@K (1 = fewest false alarms) and by
PR-AUC (1 = highest); ties get the average rank. These per-dataset ranks are the input of the
Friedman/Nemenyi CD diagram.

Outputs, next to e1_runs.csv:
    e1_per_dataset.csv  dataset x detector x budget: seed mean and std, ranks
    e1_summary.csv      detector x budget: mean over datasets, average ranks, #datasets best
    e1_summary.md       the summary as readable tables
Each carries the commit (column `commit` in the CSVs, header line in the .md).
"""
from __future__ import annotations

import pandas as pd

METRICS = ["fp_at_k", "precision_at_k", "recall_at_k", "pr_auc", "roc_auc"]


def per_dataset(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["dataset", "budget", "detector"], sort=True)
    df = g[METRICS].mean()
    df = df.join(g[METRICS].std(ddof=1).add_suffix("_sd"))
    df = df.join(g[["k", "n_test", "anomalies_test"]].first())
    df = df.join(g["seed"].nunique().rename("n_seeds")).reset_index()
    by = df.groupby(["dataset", "budget"])
    df["rank_fp_at_k"] = by["fp_at_k"].rank(method="average", ascending=True)
    df["rank_pr_auc"] = by["pr_auc"].rank(method="average", ascending=False)
    df["best_fp_at_k"] = df["fp_at_k"] == by["fp_at_k"].transform("min")
    return df


def summary(per_ds: pd.DataFrame) -> pd.DataFrame:
    g = per_ds.groupby(["budget", "detector"])
    df = g[METRICS].mean().add_prefix("mean_")
    df["avg_rank_fp_at_k"] = g["rank_fp_at_k"].mean()
    df["avg_rank_pr_auc"] = g["rank_pr_auc"].mean()
    df["n_best_fp_at_k"] = g["best_fp_at_k"].sum().astype(int)
    df["n_datasets"] = g["dataset"].nunique()
    return df.reset_index().sort_values(["budget", "avg_rank_fp_at_k", "detector"],
                                        ignore_index=True)


def markdown(summ: pd.DataFrame, meta: dict) -> str:
    p = meta["protocol"]
    lines = [
        "# E1 summary: single detectors on the 15 ADBench sets",
        "",
        f"Commit `{meta['commit']}` · cache key `{meta['cache_key']}` · source `e1_runs.csv` · "
        f"created {meta['created_utc']}",
        "",
        f"Protocol: z-score inputs {'on' if p['zscore_inputs'] else 'off'}, "
        f"K = {p['k_rule']}(budget × n_test), "
        f"PCA zero-variance drop {'on' if p['pca_drop_zero_variance'] else 'off'}, "
        f"ADBench duplicates {'dropped' if p['adbench_dedup'] else 'kept'}.",
        "Means are over datasets of seed means (5 seeds). Ranks per dataset, 1 = best; "
        "#best = datasets with the lowest FP@K (ties count for each tied detector).",
    ]
    for budget, part in summ.groupby("budget", sort=True):
        lines += ["", f"## Budget {budget:.0%} of test rows", "",
                  "| Detector | P@K | R@K | FP@K | PR-AUC | ROC-AUC | avg rank FP@K "
                  "| avg rank PR-AUC | #best FP@K |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in part.itertuples():
            lines.append(
                f"| {r.detector} | {r.mean_precision_at_k:.3f} | {r.mean_recall_at_k:.3f} "
                f"| {r.mean_fp_at_k:.1f} | {r.mean_pr_auc:.3f} | {r.mean_roc_auc:.3f} "
                f"| {r.avg_rank_fp_at_k:.2f} | {r.avg_rank_pr_auc:.2f} "
                f"| {r.n_best_fp_at_k}/{r.n_datasets} |")
    return "\n".join(lines) + "\n"


def write(out_dir, meta: dict) -> pd.DataFrame:
    """Read out_dir/e1_runs.csv, write the three summary files; returns the summary."""
    runs = pd.read_csv(out_dir / "e1_runs.csv")
    per_ds = per_dataset(runs)
    summ = summary(per_ds)
    per_ds.assign(commit=meta["commit"]).to_csv(out_dir / "e1_per_dataset.csv", index=False)
    summ.assign(commit=meta["commit"]).to_csv(out_dir / "e1_summary.csv", index=False)
    (out_dir / "e1_summary.md").write_text(markdown(summ, meta), encoding="utf-8")
    return summ
