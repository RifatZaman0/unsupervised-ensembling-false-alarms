"""E1 detector agreement: Spearman correlation of the six detectors' test scores.

For each dataset and seed, the 6 x 6 Spearman matrix of the test scores (cached E1 scores, no
labels). Seeds are averaged within a dataset first, then datasets are averaged, so every
dataset counts once (as in e1_summary). A detector with constant test scores on a run has no
defined rank correlation; that pair is NaN for the run and left out of the means (counted in
n_runs).

    python -m thesis.e1_corr

Outputs in figures/e1/: spearman_test_scores.{pdf,png} (the averaged matrix),
spearman_per_dataset.csv (seed-averaged matrix per dataset, long format) and spearman.meta.json
(commit, averaged matrix, least / most correlated pairs). The commit is also stored in the
PDF/PNG metadata.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from thesis import config, data, detectors, scorecache  # noqa: E402
from thesis.export_datasets import git_state  # noqa: E402
from thesis.paths import FIGURES  # noqa: E402

FIG_OUT = FIGURES / "e1"
# Diverging: red (negative) - neutral gray (0) - blue (positive)
CMAP = LinearSegmentedColormap.from_list("rho", ["#b8322f", "#f0efec", "#184f95"])


def spearman(S: np.ndarray, names) -> pd.DataFrame:
    """Spearman matrix of the columns of S (rows = records); constant columns give NaN."""
    df = pd.DataFrame(S, columns=list(names))
    with np.errstate(invalid="ignore", divide="ignore"):
        return df.corr(method="spearman")


def per_run(datasets, dets=detectors.DETECTORS) -> pd.DataFrame:
    """Long frame: dataset, seed, det_a, det_b, rho (each unordered pair once, a < b in dets order)."""
    rows = []
    for name in datasets:
        X, y = data.load_adbench(name)
        for seed in data.ADBENCH_SEEDS:
            split = data.preprocess(data.split_adbench(name, seed, X, y))
            S = np.column_stack([scorecache.get_scores(split, d)["test"] for d in dets])
            m = spearman(S, dets)
            rows += [{"dataset": name, "seed": seed, "det_a": a, "det_b": b, "rho": m.loc[a, b]}
                     for a, b in combinations(dets, 2)]
    return pd.DataFrame(rows)


def per_dataset(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["dataset", "det_a", "det_b"], sort=False)["rho"]
    return g.agg(rho="mean", rho_sd="std", n_runs="count").reset_index()


def average(per_ds: pd.DataFrame, dets=detectors.DETECTORS) -> pd.DataFrame:
    """Symmetric matrix of the mean over datasets, 1 on the diagonal."""
    mean = per_ds.groupby(["det_a", "det_b"])["rho"].mean()
    m = pd.DataFrame(np.eye(len(dets)), index=list(dets), columns=list(dets))
    for (a, b), v in mean.items():
        m.loc[a, b] = m.loc[b, a] = v
    return m


def extreme_pairs(avg: pd.DataFrame) -> tuple[tuple[str, str, float], tuple[str, str, float]]:
    """(least correlated pair, most correlated pair) off the diagonal."""
    pairs = [(a, b, float(avg.loc[a, b])) for a, b in combinations(avg.index, 2)]
    return min(pairs, key=lambda p: p[2]), max(pairs, key=lambda p: p[2])


def heatmap(avg: pd.DataFrame, path_stem, commit: str, n_datasets: int) -> None:
    k = len(avg)
    fig, ax = plt.subplots(figsize=(4.8, 4.1))
    im = ax.imshow(avg.to_numpy(), cmap=CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(k), avg.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(k), avg.index, fontsize=9)
    for i in range(k):
        for j in range(k):
            v = avg.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else "#0b0b0b")
    ax.set_xticks(np.arange(k + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(k + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman ρ", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    ax.set_title(f"Test-score rank correlation, mean over {n_datasets} sets × 5 seeds",
                 fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        meta = {"Subject": f"commit {commit}"} if ext == "pdf" else {"Comment": f"commit {commit}"}
        fig.savefig(f"{path_stem}.{ext}", dpi=300 if ext == "png" else None, metadata=meta)
    plt.close(fig)


def write(runs: pd.DataFrame, fig_dir, meta: dict,
          dets=detectors.DETECTORS) -> tuple[pd.DataFrame, dict]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    per_ds = per_dataset(runs)
    avg = average(per_ds, dets)
    lo, hi = extreme_pairs(avg)
    n = per_ds["dataset"].nunique()
    heatmap(avg, fig_dir / "spearman_test_scores", meta["commit"], n)
    per_ds.assign(commit=meta["commit"]).to_csv(fig_dir / "spearman_per_dataset.csv", index=False)
    info = {**meta, "n_datasets": n, "n_runs": int(runs.groupby(["dataset", "seed"]).ngroups),
            "nan_pairs": int(runs["rho"].isna().sum()),
            "source": "cached E1 test scores (no labels); Spearman per dataset x seed, "
                      "mean over seeds, then mean over datasets",
            "average": {a: {b: float(avg.loc[a, b]) for b in avg.columns} for a in avg.index},
            "least_correlated": {"pair": lo[:2], "rho": lo[2]},
            "most_correlated": {"pair": hi[:2], "rho": hi[2]}}
    (fig_dir / "spearman.meta.json").write_text(json.dumps(info, indent=1) + "\n",
                                                encoding="utf-8")
    return avg, info


def main(argv: list[str]) -> None:
    state = git_state()
    if state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    config.use(config.Protocol())
    runs = per_run(sorted(data.ADBENCH_15, key=str.lower))
    meta = {**state, "command": " ".join(["python -m thesis.e1_corr", *argv]),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cache_key": scorecache.cache_key(), "protocol": config.PROTOCOL.as_dict()}
    avg, info = write(runs, FIG_OUT, meta)
    print("Mean Spearman rho of test scores (15 sets x 5 seeds):")
    print(avg.round(3).to_string())
    lo, hi = info["least_correlated"], info["most_correlated"]
    print(f"least correlated: {lo['pair'][0]}-{lo['pair'][1]} rho = {lo['rho']:.3f}")
    print(f"most correlated:  {hi['pair'][0]}-{hi['pair'][1]} rho = {hi['rho']:.3f}")
    print(f"NaN pairs (constant scores): {info['nan_pairs']}; wrote {FIG_OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
