"""E1 critical-difference diagrams (Demšar 2006): Friedman test + Nemenyi post-hoc over the
15 ADBench sets, from the same per-dataset seed means as the summary table (e1_summary).

One diagram per budget on FP@K, the primary metric (rank 1 = fewest false alarms). Two
detectors whose average ranks differ by less than the critical difference
    CD = q_alpha * sqrt(k (k + 1) / (6 N)),  q_alpha = studentized range(1 - alpha, k, inf) / sqrt(2)
are not significantly different; such groups are joined by a bar. alpha = 0.05.

Outputs: figures/e1/cd_fp_at_k_<budget>.{pdf,png} and figures/e1/cd.meta.json (commit, test
statistics, average ranks, groups). The commit is also stored in the PDF/PNG metadata.
"""
from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ALPHA = 0.05


def nemenyi_cd(k: int, n: int, alpha: float = ALPHA) -> float:
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2)
    return float(q * math.sqrt(k * (k + 1) / (6 * n)))


def friedman_nemenyi(per_ds: pd.DataFrame, metric: str = "fp_at_k", budget: float = 0.05,
                     lower_is_better: bool = True, alpha: float = ALPHA) -> dict:
    """Friedman test (chi-square and Iman-Davenport F) and Nemenyi CD for one metric and budget.

    per_ds: e1_summary.per_dataset output (one row per dataset x budget x detector, seed means).
    """
    wide = per_ds[per_ds.budget == budget].pivot(index="dataset", columns="detector",
                                                 values=metric)
    if wide.isna().any().any():
        raise ValueError(f"{metric} at budget {budget}: missing values, cannot rank")
    n, k = wide.shape
    ranks = wide.rank(axis=1, method="average", ascending=lower_is_better)
    avg = ranks.mean().sort_values()
    chi2, p = stats.friedmanchisquare(*(wide[c].to_numpy() for c in wide.columns))
    f_id = (n - 1) * chi2 / (n * (k - 1) - chi2) if n * (k - 1) != chi2 else math.inf
    p_id = float(stats.f.sf(f_id, k - 1, (k - 1) * (n - 1)))
    cd = nemenyi_cd(k, n, alpha)
    return {"metric": metric, "budget": budget, "n_datasets": n, "k_detectors": k,
            "alpha": alpha, "friedman_chi2": float(chi2), "friedman_p": float(p),
            "iman_davenport_F": float(f_id), "iman_davenport_p": p_id, "cd": cd,
            "avg_ranks": {d: float(r) for d, r in avg.items()},
            "groups": groups(avg, cd)}


def groups(avg: pd.Series, cd: float) -> list[list[str]]:
    """Maximal sets of detectors (consecutive in rank order) whose rank span is < cd."""
    names, r = list(avg.index), avg.to_numpy()
    spans = []
    for i in range(len(r)):
        j = i
        while j + 1 < len(r) and r[j + 1] - r[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in spans):
            spans.append((i, j))
    return [names[a:b + 1] for a, b in spans]


def cd_diagram(res: dict, path_stem, commit: str) -> None:
    """Demšar-style diagram: rank axis (1 = best on the left), detectors on both sides,
    a bar per group of detectors that are not significantly different, the CD on top."""
    avg, k, cd = res["avg_ranks"], res["k_detectors"], res["cd"]
    names, ranks = list(avg), list(avg.values())
    half = math.ceil(k / 2)
    fig, ax = plt.subplots(figsize=(6.4, 1.2 + 0.32 * half))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-(half + 1.2), 1.6)
    ax.axis("off")
    ax.hlines(0, 1, k, color="black", lw=1)
    for t in range(1, k + 1):
        ax.vlines(t, 0, 0.15, color="black", lw=1)
        ax.text(t, 0.3, str(t), ha="center", va="bottom", fontsize=9)
    ax.hlines(1.2, 1, 1 + cd, color="black", lw=1.5)
    ax.vlines([1, 1 + cd], 1.1, 1.3, color="black", lw=1.5)
    ax.text(1 + cd / 2, 1.35, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9)
    for i, (name, r) in enumerate(zip(names, ranks)):
        left = i < half
        y = -(i + 1) if left else -(k - i)
        x_text = 0.6 if left else k + 0.4
        ax.plot([r, r, x_text], [0, y * 0.5 - 0.3, y * 0.5 - 0.3], color="black", lw=0.8)
        ax.text(x_text + (-0.05 if left else 0.05), y * 0.5 - 0.3, f"{name} ({r:.2f})",
                ha="right" if left else "left", va="center", fontsize=9)
    for g, members in enumerate(res["groups"]):
        lo, hi = avg[members[0]], avg[members[-1]]
        y = -0.15 - 0.12 * (g + 1)
        ax.hlines(y, lo - 0.03, hi + 0.03, color="black", lw=3)
    title = (f"FP@K, budget {res['budget']:.0%}: Friedman p = {res['friedman_p']:.3g}, "
             f"N = {res['n_datasets']}, alpha = {res['alpha']}")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        meta = {"Subject": f"commit {commit}"} if ext == "pdf" else {"Comment": f"commit {commit}"}
        fig.savefig(f"{path_stem}.{ext}", dpi=300 if ext == "png" else None, metadata=meta)
    plt.close(fig)


def write(per_ds: pd.DataFrame, fig_dir, meta: dict, budgets) -> list[dict]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for b in budgets:
        res = friedman_nemenyi(per_ds, "fp_at_k", b)
        stem = fig_dir / f"cd_fp_at_k_{round(b * 100)}pct"
        cd_diagram(res, stem, meta["commit"])
        results.append({**res, "figure": stem.name})
    (fig_dir / "cd.meta.json").write_text(json.dumps({
        "commit": meta["commit"], "dirty": meta["dirty"], "cache_key": meta["cache_key"],
        "protocol": meta["protocol"], "created_utc": meta["created_utc"],
        "source": "e1_per_dataset.csv (seed means per dataset; same data as e1_summary)",
        "test": "Friedman (scipy.stats.friedmanchisquare) + Nemenyi CD, Demsar 2006",
        "results": results}, indent=1) + "\n", encoding="utf-8")
    return results
