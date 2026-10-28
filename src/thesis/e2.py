"""E2, complementarity on the 15 ADBench sets: every pair and triple of the six detectors under
the six fusion configs, from the cached E1 scores (Master Plan § Experiments, E2).

    python -m thesis.e2                    # 15 datasets x 5 seeds x 35 combos x 6 configs x 2 budgets
    python -m thesis.e2 cardio             # a subset, for a quick check (no results written)
    python -m thesis.e2 --dedup            # same protocol switches as thesis.e1

Per run (dataset x seed) and combo (15 pairs + 20 triples):
- Overlap between the members, from TRAINING scores only (no labels, no test rows): Spearman rho
  and top-K Jaccard, K = K-rule(budget * n_train). For a triple, the mean of its three pairs.
  Jaccard follows the tie rule of thesis.metrics: rows tied with the K-th score are in the top K
  with probability (free slots / tied rows), and Jaccard = E|A & B| / (2K - E|A & B|).
- Each fusion config (thesis.fusion.fuse, calibrators fitted on training scores): FP@K, P@K,
  R@K, PR-AUC, ROC-AUC of the fused test score.
- delta_fp_best = FP@K(ensemble) - FP@K(best member), delta_fp_median = FP@K(ensemble) -
  FP@K(median member); negative = fewer false alarms than that member. The best member is
  picked per run with test labels (an oracle reference, ADBench only); the median member of a
  pair is the mean of its two members. *_rel = delta / K, comparable across datasets.

Outputs in results/tables/e2/: e2_runs.csv (one row per dataset x seed x combo x config x
budget), e2_runs.meta.json, e2_per_dataset.csv (seed means, rank of the combo among combos of
the same size per dataset / config / budget), e2_summary.csv (combo x config x budget: average
rank, #datasets beating the best / median member), e2_headline.csv (size x config x budget:
share of combo-dataset cells beating / tying the best / median member), e2_friedman.csv
(primary config, per budget and size: Friedman test over the combos, Nemenyi CD) and
e2_summary.md (headline, and the triple ranking under the primary config with its Friedman
test, from which E-bench is chosen). Each names the commit. The figures (thesis.e2_figures) are
written in the same run. Fraud data is not touched here.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from thesis import config, data, detectors, e1_figures, fusion, metrics, scorecache
from thesis.e1 import OUT as E1_OUT
from thesis.e1 import out_dir
from thesis.export_datasets import git_state
from thesis.paths import TABLES

OUT = TABLES / "e2"
COMBOS = tuple(c for r in (2, 3) for c in combinations(detectors.DETECTORS, r))
SEP = "+"


def combo_name(members) -> str:
    return SEP.join(members)


def topk_weights(scores: np.ndarray, k: int) -> np.ndarray:
    """Probability that each row is among the top k, ties at the K-th score broken at random
    (the rule of metrics.tp_at_k: tp_at_k(s, y, k) == y @ topk_weights(s, k))."""
    scores = np.asarray(scores, dtype=np.float64)
    kth = np.partition(scores, len(scores) - k)[len(scores) - k]
    above, tied = scores > kth, scores == kth
    w = above.astype(np.float64)
    w[tied] = (k - above.sum()) / tied.sum()
    return w


def jaccard_topk(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Top-k Jaccard of two score vectors on the same rows, expected-value tie rule."""
    inter = float(topk_weights(a, k) @ topk_weights(b, k))
    return inter / (2 * k - inter)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """NaN if either score vector is constant."""
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b)[0])


def overlap(train: dict[str, np.ndarray], members, budget: float) -> dict:
    """Training-score agreement of a combo: mean over its member pairs. Uses training scores
    only, so neither labels nor test rows can enter."""
    k = metrics.budget_k(budget, len(next(iter(train.values()))))
    pairs = list(combinations(members, 2))
    rhos = [spearman(train[a], train[b]) for a, b in pairs]
    return {"k_train": k,
            "train_spearman": float(np.nanmean(rhos)) if not np.isnan(rhos).all() else np.nan,
            "train_jaccard": float(np.mean([jaccard_topk(train[a], train[b], k)
                                            for a, b in pairs]))}


def run_split(train: dict[str, np.ndarray], test: dict[str, np.ndarray], y_test: np.ndarray,
              base: dict, combos=COMBOS) -> list[dict]:
    """All rows for one dataset x seed. train/test: detector -> cached scores."""
    member = {(d, b): metrics.budget_metrics(test[d], y_test, b)["fp_at_k"]
              for d in test for b in metrics.ADBENCH_BUDGETS}
    rows = []
    for members in combos:
        ov = {b: overlap(train, members, b) for b in metrics.ADBENCH_BUDGETS}
        S_train = np.column_stack([train[d] for d in members])
        S_test = np.column_stack([test[d] for d in members])
        for calib, agg in fusion.FUSION_CONFIGS:
            _, fused = fusion.fuse(S_train, S_test, calib, agg, members, base["dataset"])
            rank = metrics.ranking_metrics(fused, y_test)
            for b in metrics.ADBENCH_BUDGETS:
                bm = metrics.budget_metrics(fused, y_test, b)
                fps = {d: member[d, b] for d in members}
                best = min(fps.values())
                med = float(np.median(list(fps.values())))
                rows.append({**base, "combo": combo_name(members), "size": len(members),
                             "calib": calib, "agg": agg, **rank, **bm, **ov[b],
                             "best_member": "/".join(d for d in members if fps[d] == best),
                             "fp_best_member": best, "fp_median_member": med,
                             "delta_fp_best": bm["fp_at_k"] - best,
                             "delta_fp_median": bm["fp_at_k"] - med,
                             "delta_fp_best_rel": (bm["fp_at_k"] - best) / bm["k"],
                             "delta_fp_median_rel": (bm["fp_at_k"] - med) / bm["k"]})
    return rows


def run(datasets) -> tuple[pd.DataFrame, dict]:
    rows, stats_ = [], {"scores_from_cache": 0, "scores_computed": 0}
    for name in datasets:
        X, y = data.load_adbench(name)
        for seed in data.ADBENCH_SEEDS:
            split = data.preprocess(data.split_adbench(name, seed, X, y))
            train, test = {}, {}
            for det in detectors.DETECTORS:
                cached = scorecache.cache_path(name, seed, det).exists()
                sc = scorecache.get_scores(split, det)
                stats_["scores_from_cache" if cached else "scores_computed"] += 1
                train[det], test[det] = sc["train"], sc["test"]
            base = {"dataset": name, "seed": seed, "n_test": len(split.y_test),
                    "anomalies_test": int(split.y_test.sum())}
            rows += run_split(train, test, split.y_test, base)
            print(f"{name} seed {seed} done", flush=True)
    return pd.DataFrame(rows), stats_


MEANS = ["fp_at_k", "precision_at_k", "recall_at_k", "pr_auc", "roc_auc", "delta_fp_best",
         "delta_fp_median", "delta_fp_best_rel", "delta_fp_median_rel", "train_spearman",
         "train_jaccard"]
CELL = ["dataset", "budget", "size", "calib", "agg"]


def per_dataset(runs: pd.DataFrame) -> pd.DataFrame:
    """Seed means per dataset x combo x config x budget; FP@K rank among the combos of the same
    size (1 = fewest false alarms, ties averaged); beats_* on the seed means."""
    g = runs.groupby([*CELL, "combo"], sort=True)
    df = g[MEANS].mean()
    df = df.join(g[["k", "n_test"]].first())
    df = df.join(g["seed"].nunique().rename("n_seeds"))
    df = df.join((g["delta_fp_best"].apply(lambda s: (s < 0).mean())
                  .rename("share_runs_beats_best"))).reset_index()
    df["rank_fp_at_k"] = df.groupby(CELL)["fp_at_k"].rank(method="average")
    df["beats_best"] = df["delta_fp_best"] < 0
    df["beats_median"] = df["delta_fp_median"] < 0
    df["ties_best"] = df["delta_fp_best"] == 0
    df["ties_median"] = df["delta_fp_median"] == 0
    return df


def summary(per_ds: pd.DataFrame) -> pd.DataFrame:
    """Combo x config x budget over the datasets (each dataset counts once)."""
    g = per_ds.groupby(["budget", "size", "calib", "agg", "combo"])
    df = g[["fp_at_k", "pr_auc", "delta_fp_best", "delta_fp_median", "delta_fp_best_rel",
            "delta_fp_median_rel", "train_spearman", "train_jaccard"]].mean().add_prefix("mean_")
    df["avg_rank_fp_at_k"] = g["rank_fp_at_k"].mean()
    df["n_beats_best"] = g["beats_best"].sum().astype(int)
    df["n_beats_median"] = g["beats_median"].sum().astype(int)
    df["n_datasets"] = g["dataset"].nunique()
    return df.reset_index().sort_values(["budget", "size", "calib", "agg", "avg_rank_fp_at_k",
                                         "combo"], ignore_index=True)


def headline(per_ds: pd.DataFrame) -> pd.DataFrame:
    """Size x config x budget: share of (combo, dataset) cells where the ensemble has fewer /
    exactly as many false alarms as its best / median member (the rest: more), and the median
    relative delta."""
    g = per_ds.groupby(["budget", "size", "calib", "agg"])
    df = pd.DataFrame({"n_cells": g.size(),
                       "share_beats_best": g["beats_best"].mean(),
                       "share_ties_best": g["ties_best"].mean(),
                       "share_beats_median": g["beats_median"].mean(),
                       "share_ties_median": g["ties_median"].mean(),
                       "median_delta_fp_best_rel": g["delta_fp_best_rel"].median(),
                       "median_delta_fp_median_rel": g["delta_fp_median_rel"].median()})
    return df.reset_index()


def friedman(per_ds: pd.DataFrame) -> pd.DataFrame:
    """Primary config, per budget and size: Friedman test over the combos of that size (FP@K
    seed means, 15 datasets as blocks) and the Nemenyi CD (thesis.e1_figures, alpha 0.05).
    n_within_cd = combos whose average rank is less than CD above the lowest one. Sizes with
    fewer than 3 combos (a subset run) are skipped: the test needs at least 3."""
    calib, agg = fusion.PRIMARY
    prim = per_ds[(per_ds.calib == calib) & (per_ds["agg"] == agg)]
    rows = []
    for (b, size), g in prim.groupby(["budget", "size"], sort=True):
        if g.combo.nunique() < 3:
            continue
        r = e1_figures.friedman_nemenyi(g, "fp_at_k", b, item="combo")
        ranks = r["avg_ranks"]                       # ascending
        top = next(iter(ranks))
        rows.append({"budget": b, "size": size, "calib": calib, "agg": agg,
                     "n_datasets": r["n_datasets"], "n_combos": r["k_detectors"],
                     "friedman_chi2": r["friedman_chi2"], "friedman_p": r["friedman_p"],
                     "iman_davenport_F": r["iman_davenport_F"],
                     "iman_davenport_p": r["iman_davenport_p"], "alpha": r["alpha"],
                     "nemenyi_cd": r["cd"], "lowest_avg_rank_combo": top,
                     "lowest_avg_rank": ranks[top], "highest_avg_rank": max(ranks.values()),
                     "n_within_cd": sum(v - ranks[top] < r["cd"] for v in ranks.values())})
    return pd.DataFrame(rows)


def markdown(head: pd.DataFrame, summ: pd.DataFrame, meta: dict,
             fried: pd.DataFrame | None = None) -> str:
    p = meta["protocol"]
    lines = ["# E2 summary: pairs and triples of detectors on the 15 ADBench sets", "",
             f"Commit `{meta['commit']}` · cache key `{meta['cache_key']}` · source `e2_runs.csv` · "
             f"created {meta['created_utc']}", "",
             f"Protocol: z-score inputs {'on' if p['zscore_inputs'] else 'off'}, "
             f"K = {p['k_rule']}(budget × n_test), "
             f"PCA zero-variance drop {'on' if p['pca_drop_zero_variance'] else 'off'}, "
             f"ADBench duplicates {'dropped' if p['adbench_dedup'] else 'kept'}. "
             f"Primary fusion config: {fusion.PRIMARY[0]} + {fusion.PRIMARY[1]}.",
             "Seed means per dataset. Beats best / median = fewer false alarms at K than the "
             "best / median member (best chosen with test labels, an oracle reference). "
             "Ties = exactly as many false alarms; loses = 1 − beats − ties. "
             "Δ/K = (FP@K ensemble − FP@K member) / K; negative = fewer false alarms.", "",
             "## How often an ensemble beats its members (share of combo × dataset cells)", "",
             "| Budget | Size | Calibration | Aggregation | beats best | ties best "
             "| beats median | ties median | median Δ/K vs best | median Δ/K vs median |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in head.itertuples():
        lines.append(f"| {r.budget:.0%} | {r.size} | {r.calib} | {r.agg} | "
                     f"{r.share_beats_best:.3f} | {r.share_ties_best:.3f} | "
                     f"{r.share_beats_median:.3f} | {r.share_ties_median:.3f} | "
                     f"{r.median_delta_fp_best_rel:+.4f} | {r.median_delta_fp_median_rel:+.4f} |")
    calib, agg = fusion.PRIMARY
    for b, g in summ[(summ["size"] == 3) & (summ.calib == calib) & (summ["agg"] == agg)].groupby(
            "budget", sort=True):
        lines += ["", f"## Triple ranking, {calib} + {agg}, budget {b:.0%}", "",
                  "| # | Triple | avg rank FP@K (of 20) | mean FP@K | mean PR-AUC "
                  "| #beats best | #beats median | train ρ | train Jaccard |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(g.itertuples(), 1):
            lines.append(f"| {i} | {r.combo} | {r.avg_rank_fp_at_k:.2f} | {r.mean_fp_at_k:.1f} | "
                         f"{r.mean_pr_auc:.3f} | {r.n_beats_best}/{r.n_datasets} | "
                         f"{r.n_beats_median}/{r.n_datasets} | {r.mean_train_spearman:.3f} | "
                         f"{r.mean_train_jaccard:.3f} |")
        sel = fried[(fried.budget == b) & (fried["size"] == 3)] if fried is not None and len(fried) else None
        if sel is not None and len(sel):
            f = sel.iloc[0]
            lines += ["", f"Friedman test over the {f.n_combos} triples ({f.n_datasets} datasets): "
                          f"χ² = {f.friedman_chi2:.2f}, p = {f.friedman_p:.3f} "
                          f"(Iman-Davenport p = {f.iman_davenport_p:.3f}); Nemenyi CD = "
                          f"{f.nemenyi_cd:.2f} at α = {f.alpha}; {f.n_within_cd} of "
                          f"{f.n_combos} triples lie within CD of the lowest average rank. "
                          "Not significant means the triples cannot be told apart on these "
                          "datasets: the top row has the lowest average rank, it is not shown "
                          "to be better."]
    if fried is not None and len(fried):
        lines += ["", f"## Friedman tests over the combos of one size, {calib} + {agg}", "",
                  "| Budget | Size | combos | χ² | p | Iman-Davenport p | Nemenyi CD "
                  "| lowest avg rank | within CD |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for f in fried.itertuples():
            lines.append(f"| {f.budget:.0%} | {f.size} | {f.n_combos} | {f.friedman_chi2:.2f} | "
                         f"{f.friedman_p:.3f} | {f.iman_davenport_p:.3f} | {f.nemenyi_cd:.2f} | "
                         f"{f.lowest_avg_rank_combo} ({f.lowest_avg_rank:.2f}) | "
                         f"{f.n_within_cd}/{f.n_combos} |")
    return "\n".join(lines) + "\n"


def members_match_e1(runs: pd.DataFrame, e1_runs: pd.DataFrame) -> float:
    """Largest |difference| between the member FP@K used here and E1's single-detector FP@K,
    via the best member and the median (= mean of the two) of each pair, which together pin
    both members; 0 when both come from the same scores."""
    key = ["dataset", "seed", "budget"]
    e1 = e1_runs.set_index([*key, "detector"])["fp_at_k"]
    pairs = runs[(runs["size"] == 2) & (runs.calib == fusion.PRIMARY[0])
                 & (runs["agg"] == fusion.PRIMARY[1])]
    worst = 0.0
    for r in pairs.itertuples():
        e1_fps = [e1[(r.dataset, r.seed, r.budget, d)] for d in r.combo.split(SEP)]
        worst = max(worst, abs(min(e1_fps) - r.fp_best_member),
                    abs(float(np.mean(e1_fps)) - r.fp_median_member))
    return worst


def write(runs: pd.DataFrame, out, meta: dict) -> tuple[pd.DataFrame, ...]:
    """Tables; returns (per_dataset, summary, headline, friedman)."""
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "e2_runs.csv", index=False)
    (out / "e2_runs.meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    per_ds = per_dataset(runs)
    summ, head, fried = summary(per_ds), headline(per_ds), friedman(per_ds)
    per_ds.assign(commit=meta["commit"]).to_csv(out / "e2_per_dataset.csv", index=False)
    summ.assign(commit=meta["commit"]).to_csv(out / "e2_summary.csv", index=False)
    head.assign(commit=meta["commit"]).to_csv(out / "e2_headline.csv", index=False)
    fried.assign(commit=meta["commit"]).to_csv(out / "e2_friedman.csv", index=False)
    (out / "e2_summary.md").write_text(markdown(head, summ, meta, fried), encoding="utf-8")
    return per_ds, summ, head, fried


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="python -m thesis.e2")
    parser.add_argument("datasets", nargs="*", help="subset for a quick check (writes nothing)")
    config.add_arguments(parser)
    args = parser.parse_args(argv)
    protocol = config.from_args(args)
    config.use(protocol)
    state = git_state()
    if not args.datasets and state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    datasets = args.datasets or sorted(data.ADBENCH_15, key=str.lower)
    runs, stats_ = run(datasets)
    print(f"scores: {stats_['scores_from_cache']} from cache {scorecache.cache_key()}, "
          f"{stats_['scores_computed']} computed")
    if args.datasets:
        print(headline(per_dataset(runs)).to_string(index=False))
        return
    e1_runs = out_dir(protocol, E1_OUT) / "e1_runs.csv"
    meta = {
        **state, "command": " ".join(["python -m thesis.e2", *argv]),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cache_key": scorecache.cache_key(), "protocol": protocol.as_dict(),
        "preprocessing": scorecache.preprocessing(), "detector_params": detectors.params(),
        "budgets": list(metrics.ADBENCH_BUDGETS), "fusion_configs": fusion.FUSION_CONFIGS,
        "primary_config": fusion.PRIMARY, "n_combos": len(COMBOS),
        "k_rule": f"K = {protocol.k_rule}(budget * n_test), at least 1; ties at the K-th "
                  "score split in expectation; training overlap uses K over n_train",
        **stats_,
        "max_abs_diff_member_fp_vs_e1": (members_match_e1(runs, pd.read_csv(e1_runs))
                                         if e1_runs.exists() else None),
        "note": "best member chosen per run with test labels (oracle reference, ADBench only); "
                "overlap from training scores only",
    }
    from thesis import e2_figures              # imports this module
    out = out_dir(protocol, OUT)
    per_ds, summ, head, fried = write(runs, out, meta)
    fig_dir = out_dir(protocol, e2_figures.FIG_OUT)
    e2_figures.write(per_ds, summ, fried, fig_dir, out, meta)
    print(head.to_string(index=False))
    print(fried.to_string(index=False))
    print("figures:", fig_dir)
    print("member FP@K vs E1, max |diff|:", meta["max_abs_diff_member_fp_vs_e1"])
    print("wrote", out, len(runs), "rows + summary")


if __name__ == "__main__":
    main(sys.argv[1:])
