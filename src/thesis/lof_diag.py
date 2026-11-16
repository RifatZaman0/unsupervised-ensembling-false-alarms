"""LOF diagnosis and the temporal stress test on ULB. Label-free, descriptive, after freeze-v1.

    python -m thesis.lof_diag l0     # LOF recomputed with scikit-learn against the cached scores
    python -m thesis.lof_diag l2b    # stress test: fit on the first 12 h of day 1, score the last 12 h
    python -m thesis.lof_diag l2     # controls L2a-L2d (random halves, half days, swapped days, E3)
    python -m thesis.lof_diag l1     # score quantiles and histograms, day 1 against day 2

Stress statistic of member m (one fit slice, one later slice):

    r_m = (share of later-slice rows above the member's threshold) / (1 - q)

The threshold is the q quantile of the member's own scores on the fit slice (the E3 threshold
rule), and the input scaler is fitted on the fit slice only, so nothing from the later slice
reaches a fitted parameter. r_m = 1 means the member keeps its target alert rate.

No label is read anywhere in this module (data.load_ulb_unlabelled; the score cache holds
scores and row positions only). ULB is development data for the stress test. Nothing here
changes frozen_config.json or a frozen result.
Outputs: results/tables/e3_ulb/lof/ and results/figures/e3_ulb/lof/.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import LocalOutlierFactor

from thesis import config, data, detectors, frozen, scorecache
from thesis.e3_ulb import alerts, fit_thresholds
from thesis.e3_ulb_lof import fit_sample, knn_distances, lof_k
from thesis.export_datasets import git_state
from thesis.paths import FIGURES, TABLES

OUT = TABLES / "e3_ulb" / "lof"
FIG_OUT = FIGURES / "e3_ulb" / "lof"
HALF_DAY_SECONDS = data.ULB_DAY_SECONDS // 2        # 43,200 s: the L2b cut inside day 1
L2A_SEED = 0                                        # random halves of day 1
RUNS = {"L2a": "random 50% of day 1 -> other 50% of day 1",
        "L2b": "day 1, first 12 h -> day 1, last 12 h",
        "L2c": "day 2 -> day 1",
        "L2d": "day 1 -> day 2 (E3, cached scores)"}


# ---------------------------------------------------------------- slices and scores

def control_slices(t: np.ndarray, run: str) -> tuple[np.ndarray, np.ndarray]:
    """Row positions (fit slice, scored slice) of one control run. Uses Time only."""
    day1 = t < data.ULB_DAY_SECONDS
    if run == "L2a":
        idx = np.random.default_rng(L2A_SEED).permutation(np.flatnonzero(day1))
        half = len(idx) // 2
        return np.sort(idx[:half]), np.sort(idx[half:])
    if run == "L2b":
        fit, late = np.flatnonzero(t < HALF_DAY_SECONDS), np.flatnonzero(day1 & (t >= HALF_DAY_SECONDS))
        assert t[fit].max() < t[late].min()             # fit slice strictly earlier
        return fit, late
    if run == "L2c":
        return np.flatnonzero(~day1), np.flatnonzero(day1)
    if run == "L2d":
        return np.flatnonzero(day1), np.flatnonzero(~day1)
    raise ValueError(f"unknown run {run!r}")


def slice_scores(X_fit: np.ndarray, X_score: np.ndarray, pool, seeds,
                 score_fn=detectors.score) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per member: (fit-slice scores, scored-slice scores), one row per IForest seed.

    The scaler and every detector are fitted on X_fit only; X_score is only transformed and
    scored."""
    sc = data.Standardizer.fit(X_fit)
    Z_fit, Z_score = sc.transform(X_fit), sc.transform(X_score)
    out = {}
    for det in pool:
        runs = [score_fn(det, Z_fit, Z_score, sd)
                for sd in (seeds if det in detectors.SEEDED else [None])]
        out[det] = (np.vstack([r[0] for r in runs]), np.vstack([r[1] for r in runs]))
    return out


def cached_day_scores(t: np.ndarray, pool, seeds) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The E3 scores (day 1 fit, day 2 scored) from the score cache; no label is stored there."""
    fit, late = control_slices(t, "L2d")
    out = {}
    for det in pool:
        tr, te = [], []
        for sd in (seeds if det in detectors.SEEDED else [None]):
            path = scorecache.cache_path("ulb", sd, det)
            if not path.exists():
                raise SystemExit(f"{path} is missing; run python -m thesis.e3_ulb first")
            with np.load(path) as z:
                assert np.array_equal(z["idx_train"], fit) and np.array_equal(z["idx_test"], late)
                tr.append(z["train"])
                te.append(z["test"])
        out[det] = (np.vstack(tr), np.vstack(te))
    return out


def stress_table(scores: dict[str, tuple[np.ndarray, np.ndarray]], quantiles,
                 run: str) -> pd.DataFrame:
    """r_m per member and q. Thresholds come from the fit-slice scores only."""
    rows = []
    for det, (s_fit, s_late) in scores.items():
        for q in quantiles:
            thr = fit_thresholds(s_fit, q)                      # one per seed, fit slice only
            rate = alerts(s_late, thr).mean(axis=1)             # later slice, per seed
            target = round(1 - q, 12)
            rows.append({"run": run, "member": det, "q": q, "target_rate": target,
                         "n_seeds": len(thr), "n_fit": s_fit.shape[1], "n_scored": s_late.shape[1],
                         "fit_alert_rate": float(alerts(s_fit, thr).mean()),
                         "alert_rate": float(rate.mean()), "r": float(rate.mean() / target),
                         "r_min": float(rate.min() / target), "r_max": float(rate.max() / target)})
    tab = pd.DataFrame(rows)
    tab["r_over_median"] = tab["r"] / tab.groupby(["run", "q"])["r"].transform("median")
    return tab


def k_distance_ratio(X_fit: np.ndarray, X_score: np.ndarray, k: int) -> dict:
    """Median distance to the k-th neighbour in the fit sample: fit slice, scored slice, ratio."""
    sc = data.Standardizer.fit(X_fit)
    d_fit, d_score = knn_distances(sc.transform(X_fit), sc.transform(X_score), k)
    a, b = float(np.median(d_fit[:, -1])), float(np.median(d_score[:, -1]))
    return {"k_distance_median_fit": a, "k_distance_median_scored": b, "k_distance_ratio": b / a}


# ---------------------------------------------------------------- L0: recomputation

def recompute_lof(X1: np.ndarray, X2: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """LOF with scikit-learn only: day-1 scaler, the E3 fit sample, sampled rows leave-one-out,
    all other rows scored as new rows. Higher = more anomalous."""
    sc = data.Standardizer.fit(X1)
    Z1, Z2 = sc.transform(X1), sc.transform(X2)
    sample = fit_sample(len(Z1))
    lof = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Z1[sample])
    s1 = np.empty(len(Z1))
    s1[sample] = -lof.negative_outlier_factor_
    rest = np.setdiff1d(np.arange(len(Z1)), sample)
    if len(rest):
        s1[rest] = -lof.score_samples(Z1[rest])
    return s1, -lof.score_samples(Z2)


def agreement(own: np.ndarray, ref: np.ndarray, top_k: int) -> dict:
    top_own = set(np.argsort(-own, kind="stable")[:top_k].tolist())
    top_ref = set(np.argsort(-ref, kind="stable")[:top_k].tolist())
    return {"n": len(ref), "spearman": float(spearmanr(own, ref)[0]),
            "max_abs_diff": float(np.abs(own - ref).max()),
            "max_rel_diff": float((np.abs(own - ref) / np.abs(ref)).max()),
            "top_k": top_k, "top_k_shared": len(top_own & top_ref)}


def l0_table(X: np.ndarray, t: np.ndarray, top_k: int) -> tuple[pd.DataFrame, dict]:
    day1 = t < data.ULB_DAY_SECONDS
    k = lof_k()
    s1, s2 = recompute_lof(X[day1], X[~day1], k)
    c1, c2 = (a[0] for a in cached_day_scores(t, ["LOF"], [])["LOF"])
    rows = [{"part": "day 1 (fit, leave-one-out for sampled rows)", **agreement(s1, c1, top_k)},
            {"part": "day 2 (scored)", **agreement(s2, c2, top_k)}]
    # direction: rows moved 50 day-1 standard deviations away must score above every day-1 row
    sc = data.Standardizer.fit(X[day1])
    far = recompute_far(sc.transform(X[day1]), sc.transform(X[~day1][:200]) + 50.0, k)
    checks = {"time_not_a_feature": "Time" not in data.ULB_FEATURES,
              "scaler_fitted_on": "day 1 only (the recomputation uses a day-1 scaler and "
                                  "matches the cached scores)",
              "higher_is_more_anomalous": bool(far.min() > c1.max()),
              "pass": bool(all(r["spearman"] >= 0.9999 and r["top_k_shared"] == top_k
                               for r in rows))}
    return pd.DataFrame(rows), checks


def recompute_far(Z1: np.ndarray, Z_far: np.ndarray, k: int) -> np.ndarray:
    sample = fit_sample(len(Z1))
    return -LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Z1[sample]).score_samples(Z_far)


# ---------------------------------------------------------------- L1: distributions

L1_QUANTILES = (0.5, 0.9, 0.99, 0.995, 0.998, 0.999, 1.0)
L1_MEMBERS = ("LOF", "KNN")
L1_X_FLOOR_QUANTILE = 0.001                         # histogram x-axis: lower scores go in the first bin


def l1_table(scores: dict[str, tuple[np.ndarray, np.ndarray]], n_day1: int,
             quantiles) -> pd.DataFrame:
    """Score quantiles and alert rates at the day-1 thresholds for (a) day-1 rows in the fit
    sample, (b) day-1 rows outside it, (c) day 2."""
    sample = fit_sample(n_day1)
    in_sample = np.zeros(n_day1, dtype=bool)
    in_sample[sample] = True
    rows = []
    for det in L1_MEMBERS:
        s1, s2 = scores[det][0][0], scores[det][1][0]
        thr = {q: float(fit_thresholds(s1, q)[0]) for q in quantiles}
        groups = {"day 1, fit sample (leave-one-out)": s1[in_sample],
                  "day 1, outside the fit sample": s1[~in_sample], "day 1, all": s1, "day 2": s2}
        for g, s in groups.items():
            row = {"member": det, "group": g, "n": len(s)}
            row |= {f"score_q{q:g}": float(np.quantile(s, q)) for q in L1_QUANTILES}
            for q in quantiles:
                row |= {f"threshold_q{q:g}": thr[q], f"alert_rate_q{q:g}": float((s > thr[q]).mean())}
            rows.append(row)
    return pd.DataFrame(rows)


def l1_figure(scores: dict[str, tuple[np.ndarray, np.ndarray]], quantiles, path_stem) -> dict:
    """Histograms of LOF and KNN scores (log x-axis), day 1 against day 2, with the day-1
    thresholds. 16 cm wide, drawn at print size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    width_cm, height_cm, pt = 16.0, 7.8, 8
    plt.rcParams.update({"font.size": pt, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(width_cm / 2.54, height_cm / 2.54),
                             layout="constrained")
    for ax, det in zip(axes, L1_MEMBERS):
        s1, s2 = scores[det][0][0], scores[det][1][0]
        # the x-axis starts at the pooled 0.1% quantile; lower scores are counted in the first bin
        floor = float(np.quantile(np.concatenate([s1, s2]), L1_X_FLOOR_QUANTILE))
        lo, hi = np.log10(floor), np.log10(max(s1.max(), s2.max()))
        bins = np.logspace(lo, hi, 121)
        for s, label, colour in ((s1, "day 1 (fit)", "#1f4e79"), (s2, "day 2", "#c0504d")):
            ax.hist(np.maximum(s, floor), bins=bins, weights=np.full(len(s), 100.0 / len(s)),
                    histtype="step", linewidth=1.0, color=colour, label=label)
        for q, style in zip(quantiles, ("--", ":")):
            thr = float(fit_thresholds(s1, q)[0])
            ax.axvline(thr, color="black", linestyle=style, linewidth=0.8,
                       label=f"day-1 threshold, q = {q:g}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{det} score (log scale)", fontsize=pt)
        ax.set_title(det, fontsize=pt + 1)
        ax.tick_params(labelsize=pt)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("share of rows per bin (%, log scale)", fontsize=pt)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=pt, frameon=False, loc="outside lower center", ncols=4)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300)
    plt.close(fig)
    return {"width_cm": width_cm, "height_cm": height_cm, "min_font_pt": pt, "bins": 120,
            "x_axis_starts_at_pooled_quantile": L1_X_FLOOR_QUANTILE}


# ---------------------------------------------------------------- output

def write(name: str, tab: pd.DataFrame, state: dict, extra: dict, t0: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tab.assign(commit=state["commit"]).to_csv(OUT / f"{name}.csv", index=False)
    meta = {**state, "command": f"python -m thesis.lof_diag {extra.pop('step')}",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "labels_read": False, "columns_read": list(data.ULB_FEATURES) + ["Time"],
            "scaling": "z-score, fitted on the fit slice only",
            "threshold_rule": "numpy.quantile(fit-slice scores, q), method 'linear'; alert if "
                              "score > threshold",
            "knn_lof_fit_sample": {"max_rows": detectors.MAX_FIT_ROWS,
                                   "seed": config.KNN_LOF_SAMPLE_SEED},
            "note": "Descriptive and label-free; after freeze-v1, not part of the frozen "
                    "evaluation. ULB is development data for the stress test.",
            **extra, "runtime_s": round(time.perf_counter() - t0, 1)}
    (OUT / f"{name}.meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")


def main(argv: list[str]) -> None:
    if len(argv) != 1 or argv[0] not in ("l0", "l1", "l2", "l2b"):
        raise SystemExit("usage: python -m thesis.lof_diag l0|l1|l2|l2b")
    step = argv[0]
    state = git_state()
    if state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    t0 = time.perf_counter()
    cfg = frozen.load()
    pool, seeds = cfg["detectors"]["pool"], cfg["detectors"]["iforest_seeds"]
    quantiles = cfg["thresholds"]["ulb"]["training_score_quantiles"]
    X, t = data.load_ulb_unlabelled()
    n_day1, n_day2 = int((t < data.ULB_DAY_SECONDS).sum()), int((t >= data.ULB_DAY_SECONDS).sum())
    pd.set_option("display.width", 200, "display.max_columns", 30)

    if step == "l0":
        top_k = int(np.ceil(0.002 * n_day2))            # K of the 0.2% budget on day 2 (281)
        tab, checks = l0_table(X, t, top_k)
        write("l0_recompute", tab, state, {"step": step, "k": lof_k(), "checks": checks,
              "reference": "cached E3 scores (PyOD LOF)", "recomputed_with":
              "sklearn.neighbors.LocalOutlierFactor(novelty=True)"}, t0)
        print(tab.to_string(index=False), "\n", json.dumps(checks, indent=1))
    elif step == "l1":
        scores = cached_day_scores(t, L1_MEMBERS, seeds)
        tab = l1_table(scores, n_day1, quantiles)
        fig = l1_figure(scores, quantiles, FIG_OUT / "l1_hist")
        write("l1_quantiles", tab, state, {"step": step, "figure": {**fig, "files": [
              "results/figures/e3_ulb/lof/l1_hist.pdf", "results/figures/e3_ulb/lof/l1_hist.png"]},
              "scores": "cached E3 scores, day 1 fit and day 2"}, t0)
        print(tab.to_string(index=False))
    else:
        runs = ["L2b"] if step == "l2b" else list(RUNS)
        parts, kd = [], {}
        for run in runs:
            fit, late = control_slices(t, run)
            scores = (cached_day_scores(t, pool, seeds) if run == "L2d"
                      else slice_scores(X[fit], X[late], pool, seeds))
            parts.append(stress_table(scores, quantiles, run))
            kd[run] = k_distance_ratio(X[fit], X[late], lof_k())
            print(run, "done", round(time.perf_counter() - t0, 1), "s", flush=True)
        tab = pd.concat(parts, ignore_index=True)
        for col in ("k_distance_median_fit", "k_distance_median_scored", "k_distance_ratio"):
            tab[col] = tab["run"].map(lambda r: kd[r][col])
        name = "l2b_stress" if step == "l2b" else "l2_controls"
        write(name, tab, state, {"step": step, "runs": {r: RUNS[r] for r in runs},
              "half_day_seconds": HALF_DAY_SECONDS, "l2a_seed": L2A_SEED, "quantiles": quantiles,
              "iforest_seeds": seeds, "r": "alert rate on the scored slice / (1 - q); mean over "
              "IForest seeds, r_min and r_max over seeds", "k": lof_k()}, t0)
        print(tab.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
