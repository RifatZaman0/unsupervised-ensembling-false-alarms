"""E1, single detectors on the 15 ADBench sets: per-run metrics and every E1 output, one command.

    python -m thesis.e1                    # all 15 datasets x 5 seeds x 6 detectors
    python -m thesis.e1 cardio             # a subset, for a quick check (no results written)
    python -m thesis.e1 --no-zscore --k-rule round --no-pca-drop --dedup   # protocol switches

Scores come from the score cache; only scores missing for the current cache key are computed
(the count is printed and recorded). Writes results/tables/e1/e1_runs.csv (one row per
dataset x seed x detector x budget), e1_runs.meta.json with the commit, and the summary
tables (thesis.e1_summary). With non-default switches (thesis.config) everything goes to
results/tables/e1/variants/<switches>/ instead, so the main outputs are never overwritten.
Fraud data is not touched here.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone

import pandas as pd

from thesis import config, data, detectors, e1_summary, metrics, scorecache
from thesis.export_datasets import git_state
from thesis.paths import TABLES

OUT = TABLES / "e1"
NAN = float("nan")


def run(datasets) -> tuple[pd.DataFrame, dict]:
    """Per-run metrics, plus counts of scores loaded from / added to the cache and the runs
    with non-finite scores (only possible with the PCA drop switched off; their metrics are NaN)."""
    rows, stats = [], {"scores_from_cache": 0, "scores_computed": 0, "non_finite": []}
    for name in datasets:
        X, y = data.load_adbench(name)
        for seed in data.ADBENCH_SEEDS:
            split = data.preprocess(data.split_adbench(name, seed, X, y))
            for det in detectors.DETECTORS:
                cached = scorecache.cache_path(name, seed, det).exists()
                base = {"dataset": name, "seed": seed, "detector": det,
                        "n_test": len(split.y_test), "anomalies_test": int(split.y_test.sum())}
                try:
                    sc = scorecache.get_scores(split, det)
                except detectors.NonFiniteScores:
                    warnings.warn(f"{det} on {name} seed {seed}: non-finite scores, metrics NaN")
                    stats["non_finite"].append(f"{name}/seed{seed}/{det}")
                    for b in metrics.ADBENCH_BUDGETS:
                        rows.append({**base, "runtime_s": NAN, "pr_auc": NAN, "roc_auc": NAN,
                                     "budget": b, "k": metrics.budget_k(b, len(split.y_test)),
                                     "tp_at_k": NAN, "fp_at_k": NAN, "precision_at_k": NAN,
                                     "recall_at_k": NAN})
                    continue
                stats["scores_from_cache" if cached else "scores_computed"] += 1
                base.update(runtime_s=sc["runtime_s"],
                            **metrics.ranking_metrics(sc["test"], split.y_test))
                for b in metrics.ADBENCH_BUDGETS:
                    rows.append({**base, **metrics.budget_metrics(sc["test"], split.y_test, b)})
            print(f"{name} seed {seed} done", flush=True)
    return pd.DataFrame(rows), stats


def out_dir(protocol: config.Protocol):
    tag = protocol.tag()
    return OUT / "variants" / tag if tag else OUT


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="python -m thesis.e1")
    parser.add_argument("datasets", nargs="*", help="subset for a quick check (writes nothing)")
    config.add_arguments(parser)
    args = parser.parse_args(argv)
    protocol = config.from_args(args)
    config.use(protocol)
    state = git_state()
    if not args.datasets and state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    datasets = args.datasets or sorted(data.ADBENCH_15, key=str.lower)
    df, stats = run(datasets)
    print(f"scores: {stats['scores_from_cache']} from cache {scorecache.cache_key()}, "
          f"{stats['scores_computed']} computed, {len(stats['non_finite'])} non-finite")
    if args.datasets:
        print(df.groupby(["dataset", "detector", "budget"])[["fp_at_k", "pr_auc"]].mean())
        return
    out = out_dir(protocol)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "e1_runs.csv", index=False)
    meta = {
        **state, "command": " ".join(["python -m thesis.e1", *argv]),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cache_key": scorecache.cache_key(), "protocol": protocol.as_dict(),
        "preprocessing": scorecache.preprocessing(),
        "detector_params": detectors.params(), "budgets": list(metrics.ADBENCH_BUDGETS),
        "k_rule": f"K = {protocol.k_rule}(budget * n_test), at least 1; ties at the K-th "
                  "score split in expectation",
        **stats,
        "note": "runtime_s = fit + score on one core, from the run that filled the cache",
    }
    (out / "e1_runs.meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    e1_summary.write(out, meta)
    print("wrote", out, len(df), "rows + summary")


if __name__ == "__main__":
    main(sys.argv[1:])
