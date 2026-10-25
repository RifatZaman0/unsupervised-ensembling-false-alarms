"""E1, single detectors on the 15 ADBench sets: fill the score cache and write per-run metrics.

    python -m thesis.e1            # all 15 datasets x 5 seeds x 6 detectors
    python -m thesis.e1 cardio     # a subset, for a quick check (no results written)

Writes results/tables/e1/e1_runs.csv (one row per dataset x seed x detector x budget) and
e1_runs.meta.json with the commit. Summary table, CD diagram and correlation heatmap are
built from this file and the cache in the next step. Fraud data is not touched here.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from thesis import data, detectors, metrics, scorecache
from thesis.export_datasets import git_state
from thesis.paths import TABLES

OUT = TABLES / "e1"


def run(datasets) -> pd.DataFrame:
    rows = []
    for name in datasets:
        X, y = data.load_adbench(name)
        for seed in data.ADBENCH_SEEDS:
            split = data.standardize(data.split_adbench(name, seed, X, y))
            for det in detectors.DETECTORS:
                sc = scorecache.get_scores(split, det)
                base = {"dataset": name, "seed": seed, "detector": det,
                        "n_test": len(split.y_test), "anomalies_test": int(split.y_test.sum()),
                        "runtime_s": sc["runtime_s"], **metrics.ranking_metrics(sc["test"], split.y_test)}
                for b in metrics.ADBENCH_BUDGETS:
                    rows.append({**base, **metrics.budget_metrics(sc["test"], split.y_test, b)})
            print(f"{name} seed {seed} done", flush=True)
    return pd.DataFrame(rows)


def main(argv: list[str]) -> None:
    subset = argv[1:]
    state = git_state()
    if not subset and state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    datasets = subset or sorted(data.ADBENCH_15, key=str.lower)
    df = run(datasets)
    if subset:
        print(df.groupby(["dataset", "detector", "budget"])[["fp_at_k", "pr_auc"]].mean())
        return
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "e1_runs.csv", index=False)
    (OUT / "e1_runs.meta.json").write_text(json.dumps({
        **state, "command": "python -m thesis.e1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cache_key": scorecache.cache_key(), "preprocessing": scorecache.PREPROCESSING,
        "detector_params": detectors.PARAMS, "budgets": list(metrics.ADBENCH_BUDGETS),
        "k_rule": "K = ceil(budget * n_test); ties at the K-th score split in expectation",
        "note": "runtime_s = fit + score on one core, from the run that filled the cache",
    }, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUT, len(df), "rows")


if __name__ == "__main__":
    main(sys.argv)
