"""Label-free LOF diagnosis on ULB: how far day-2 rows lie from the day-1 data LOF is fitted on.

    python -m thesis.e3_ulb_lof        # writes results/tables/e3_ulb/lof_diagnosis.{csv,meta.json,md}

LOF ranked day 2 worse than at random in E3 (ROC-AUC 0.379 in e3_ulb_budget.csv). This module
describes, without any label, the two quantities LOF depends on:
- the k-nearest-neighbour distances LOF uses (k = PyOD's default n_neighbors, the setting used
  in E3), measured against the same fit sample E3 uses (z-scored with day-1 statistics; at most
  config.KNN_LOF_MAX_FIT_ROWS day-1 rows drawn with config.KNN_LOF_SAMPLE_SEED; sampled rows
  leave-one-out, as in detectors.fit_scores), for every day-1 row and every day-2 row;
- the per-feature mean shift of day 2 against day-1 training statistics, in day-1 standard
  deviations, and the ratio of the standard deviations.
The Class column is never read (data.load_ulb_unlabelled). The result is descriptive: it was
added after freeze-v1, is not part of the frozen evaluation, and shows how day 2 differs from
day 1, not why LOF fails.
"""
from __future__ import annotations

import inspect
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pyod.models.lof import LOF
from sklearn.neighbors import NearestNeighbors

from thesis import config, data, detectors
from thesis.e3_ulb import OUT
from thesis.export_datasets import git_state

QUANTILES = (0.5, 0.9, 0.99)
TAIL_QUANTILES = (0.99, 0.998)            # 0.998: the frozen q for the 0.2% threshold check


def lof_k() -> int:
    """The k LOF uses in E3; it must be PyOD's default."""
    k = detectors.params()["LOF"]["n_neighbors"]
    default = inspect.signature(LOF.__init__).parameters["n_neighbors"].default
    if k != default:
        raise SystemExit(f"LOF n_neighbors {k} is not the PyOD default {default}")
    return k


def fit_sample(n: int, max_rows: int | None = None, seed: int | None = None) -> np.ndarray:
    """Row indices of the KNN / LOF fit sample, drawn exactly as detectors.fit_scores draws it."""
    max_rows = detectors.MAX_FIT_ROWS if max_rows is None else max_rows
    if n <= max_rows:
        return np.arange(n)
    rng = np.random.default_rng(config.KNN_LOF_SAMPLE_SEED if seed is None else seed)
    return np.sort(rng.choice(n, max_rows, replace=False))


def knn_distances(X_train: np.ndarray, X_test: np.ndarray, k: int,
                  max_rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Distances to the k nearest fit-sample rows, for every training row (sampled rows
    leave-one-out) and every test row. Inputs are already scaled. No labels."""
    sample = fit_sample(len(X_train), max_rows)
    nn = NearestNeighbors(n_neighbors=k).fit(X_train[sample])   # LOF: minkowski, p = 2
    d_train = np.empty((len(X_train), k))
    d_train[sample] = nn.kneighbors()[0]                        # excludes the row itself
    rest = np.setdiff1d(np.arange(len(X_train)), sample)
    if len(rest):
        d_train[rest] = nn.kneighbors(X_train[rest])[0]
    return d_train, nn.kneighbors(X_test)[0]


def feature_shift(X_train: np.ndarray, X_test: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Day-2 mean minus day-1 mean, in day-1 standard deviations (the input scaler of E3)."""
    sc = data.Standardizer.fit(X_train)
    z_test = sc.transform(X_test)
    return pd.DataFrame({"feature": names, "day1_mean": sc.mean, "day1_sd": sc.scale,
                         "day2_mean": X_test.mean(axis=0),
                         "shift_sd": z_test.mean(axis=0),
                         "sd_ratio": X_test.std(axis=0) / sc.scale})


def diagnose(X_train: np.ndarray, X_test: np.ndarray, names: list[str], k: int,
             max_rows: int | None = None) -> pd.DataFrame:
    """One table, long format: section, item, day1, day2, ratio (day2 / day1), shift_sd."""
    sc = data.Standardizer.fit(X_train)                         # day-1 statistics only
    d1, d2 = knn_distances(sc.transform(X_train), sc.transform(X_test), k, max_rows)
    rows = []
    for section, a, b in (("k_distance", d1[:, -1], d2[:, -1]),
                          ("mean_knn_distance", d1.mean(axis=1), d2.mean(axis=1))):
        for q in QUANTILES:
            v1, v2 = float(np.quantile(a, q)), float(np.quantile(b, q))
            rows.append((section, f"q{q:g}", v1, v2, v2 / v1, np.nan))
        rows.append((section, "mean", float(a.mean()), float(b.mean()),
                     float(b.mean() / a.mean()), np.nan))
    for q in TAIL_QUANTILES:
        cut = float(np.quantile(d1[:, -1], q))
        s1, s2 = float((d1[:, -1] > cut).mean()), float((d2[:, -1] > cut).mean())
        rows.append(("share_above_day1_k_distance_quantile", f"q{q:g}", s1, s2, s2 / s1, np.nan))
    fs = feature_shift(X_train, X_test, names)
    rows += [("feature_mean", r.feature, r.day1_mean, r.day2_mean, r.sd_ratio, r.shift_sd)
             for r in fs.itertuples()]
    return pd.DataFrame(rows, columns=["section", "item", "day1", "day2", "ratio", "shift_sd"])


def summary_md(tab: pd.DataFrame, meta: dict) -> str:
    def get(section, item):
        return tab[(tab.section == section) & (tab.item == item)].iloc[0]
    kd, kd99 = get("k_distance", "q0.5"), get("k_distance", "q0.99")
    sh = get("share_above_day1_k_distance_quantile", "q0.99")
    sh998 = get("share_above_day1_k_distance_quantile", "q0.998")
    f = tab[tab.section == "feature_mean"].assign(abs_shift=lambda d: d.shift_sd.abs())
    top = f.sort_values("abs_shift", ascending=False).head(5)
    big = int((f.abs_shift > 0.5).sum())
    lines = [
        "# ULB: label-free LOF diagnosis, day 1 against day 2", "",
        f"Commit `{meta['commit']}` · `{meta['command']}` · created {meta['created_utc']}", "",
        f"No label is read (`Class` column never loaded). Inputs z-scored with day-1 statistics; "
        f"k = {meta['k']} (PyOD default, as in E3); fit sample {meta['fit_rows']:,} of "
        f"{meta['n_day1']:,} day-1 rows (seed {meta['sample_seed']}), sampled rows "
        f"leave-one-out; {meta['n_day2']:,} day-2 rows. Descriptive, added after `freeze-v1`, "
        "not part of the frozen evaluation.", "",
        "## k-nearest-neighbour distances (z-scored units)", "",
        "| statistic | day 1 | day 2 | day 2 / day 1 |", "|---|---|---|---|"]
    for section, label in (("k_distance", f"distance to the {meta['k']}th neighbour"),
                           ("mean_knn_distance", f"mean distance to {meta['k']} neighbours")):
        for item in [f"q{q:g}" for q in QUANTILES] + ["mean"]:
            r = get(section, item)
            lines.append(f"| {label}, {item.replace('q', 'quantile ') if item != 'mean' else 'mean'} "
                         f"| {r.day1:.3f} | {r.day2:.3f} | {r.ratio:.2f} |")
    lines += ["",
              f"Share of rows whose {meta['k']}th-neighbour distance exceeds the day-1 0.99 "
              f"quantile: day 1 {sh.day1:.2%}, day 2 {sh.day2:.2%}; above the day-1 0.998 "
              f"quantile: day 1 {sh998.day1:.2%}, day 2 {sh998.day2:.2%}.", "",
              "## Feature means of day 2 against day-1 statistics", "",
              f"{big} of {len(f)} features shift by more than 0.5 day-1 standard deviations. "
              "Largest shifts:", "",
              "| feature | shift (day-1 SD) | SD ratio day 2 / day 1 |", "|---|---|---|"]
    lines += [f"| {r.item} | {r.shift_sd:+.3f} | {r.ratio:.3f} |" for r in top.itertuples()]
    lines += ["",
              f"Reading: the median {meta['k']}th-neighbour distance is {kd.day1:.3f} on day 1 "
              f"and {kd.day2:.3f} on day 2 (0.99 quantile {kd99.day1:.3f} and {kd99.day2:.3f}). "
              "This describes how far day-2 rows lie from the day-1 fit sample; it does not by "
              "itself show why LOF ranks frauds low.", ""]
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    if argv:
        raise SystemExit("usage: python -m thesis.e3_ulb_lof")
    state = git_state()
    if state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    t0 = time.perf_counter()
    X, t = data.load_ulb_unlabelled()
    train = t < data.ULB_DAY_SECONDS
    X1, X2 = X[train], X[~train]
    assert t[train].max() < t[~train].min()                    # day 1 strictly before day 2
    k = lof_k()
    tab = diagnose(X1, X2, list(data.ULB_FEATURES), k)
    meta = {**state, "command": "python -m thesis.e3_ulb_lof",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "labels_read": False, "columns_read": list(data.ULB_FEATURES) + ["Time"],
            "k": k, "fit_rows": int(min(len(X1), detectors.MAX_FIT_ROWS)),
            "sample_seed": config.KNN_LOF_SAMPLE_SEED, "n_day1": int(len(X1)),
            "n_day2": int(len(X2)), "scaling": "z-score with day-1 statistics",
            "metric": "Euclidean (sklearn NearestNeighbors, as PyOD LOF: minkowski, p = 2)",
            "runtime_s": round(time.perf_counter() - t0, 1),
            "note": "Descriptive and label-free; added after freeze-v1, not part of the frozen "
                    "evaluation. Day-1 rows in the fit sample are leave-one-out; other rows are "
                    "measured like new rows, as in detectors.fit_scores."}
    OUT.mkdir(parents=True, exist_ok=True)
    tab.assign(commit=state["commit"]).to_csv(OUT / "lof_diagnosis.csv", index=False)
    (OUT / "lof_diagnosis.meta.json").write_text(json.dumps(meta, indent=1) + "\n",
                                                 encoding="utf-8")
    (OUT / "lof_diagnosis.md").write_text(summary_md(tab, meta), encoding="utf-8")
    print((OUT / "lof_diagnosis.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main(sys.argv[1:])
