"""Threshold fragility index per member (exploratory, proposed for freeze-v3 as H4d).

    python -m thesis.fragility            # ULB day 1 and IEEE-CIS month 1, writes results/tables/
    python -m thesis.fragility --ulb      # ULB day 1 only (cached E3 scores, seconds)

Idea (developed after seeing ULB, from the calibration-fragility remark of Heydari & Nyarko
2026): a member whose threshold sits close to the bulk of its training scores flips many rows
into alerts after a small shift of the bulk. Definition, from training scores only:

    t   = numpy.quantile(training scores, q, method 'linear')     (the frozen threshold)
    med = median of the training scores
    F_m = share of training scores > t - DELTA * (t - med), divided by (1 - q)

i.e. the alert ratio r the member would show on its own training data if every score moved up
by DELTA times the threshold-to-median gap. F >= 1; a large F means a fragile threshold.
F does not change under any increasing affine map of the scores (a * s + b, a > 0), so it can
be compared across detectors whose scores have offsets or negative values (PyOD IForest
scores are centred near 0; HBOS sits on a large offset). DELTA = 0.1 is fixed once, before any
value was computed, and is not tuned. IForest: F per seed, mean over the five seeds.

Label-free: only training scores (ULB day 1, IEEE-CIS month 1) are used; no test row is
scored. No exclusion rule is based on F.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from thesis import data, detectors, frozen, ieee_features, lof_diag
from thesis.export_datasets import git_state
from thesis.paths import TABLES

DELTA = 0.1
OUT_ULB = TABLES / "e3_ulb" / "stress"
OUT_IEEE = TABLES / "ieee_cis"


def index(train_scores: np.ndarray, q: float, delta: float = DELTA) -> np.ndarray:
    """F per seed (rows of train_scores)."""
    s = np.atleast_2d(np.asarray(train_scores, dtype=np.float64))
    t = np.quantile(s, q, axis=1, method="linear")
    med = np.median(s, axis=1)
    cut = t - delta * (t - med)
    return (s > cut[:, None]).mean(axis=1) / round(1 - q, 12)


def table(train: dict, quantiles, run: str, delta: float = DELTA) -> pd.DataFrame:
    rows = []
    for m, s in train.items():
        for q in quantiles:
            f = index(s, q, delta)
            rows.append({"run": run, "member": m, "q": q, "delta": delta, "n_seeds": len(f),
                         "n_train": np.atleast_2d(s).shape[1], "fragility": float(f.mean()),
                         "fragility_min": float(f.min()), "fragility_max": float(f.max())})
    tab = pd.DataFrame(rows)
    tab["rank"] = tab.groupby(["run", "q"])["fragility"].rank(ascending=False, method="average")
    return tab


def quantiles_for(cfg: dict, dataset: str) -> list[float]:
    """q = 1 - budget for every budget, plus the frozen threshold quantiles."""
    qs = {round(1 - b, 12) for b in cfg["budgets"][dataset]["values"]}
    qs |= set(cfg["thresholds"][dataset]["training_score_quantiles"])
    return sorted(qs, reverse=True)


def ulb_training_scores(cfg: dict) -> dict:
    """Day-1 training scores from the E3 score cache (no label is stored there)."""
    _, t = data.load_ulb_unlabelled()
    sc = lof_diag.cached_day_scores(t, cfg["detectors"]["pool"], cfg["detectors"]["iforest_seeds"])
    return {m: a for m, (a, _) in sc.items()}


def ulb_overshoot(cfg: dict, quantiles) -> pd.DataFrame:
    """Development check on ULB: day-2 r of each member at its day-1 threshold (label-free),
    from the same cached scores; for the Spearman of F against log r."""
    _, t = data.load_ulb_unlabelled()
    sc = lof_diag.cached_day_scores(t, cfg["detectors"]["pool"], cfg["detectors"]["iforest_seeds"])
    return lof_diag.stress_table(sc, quantiles, "ULB day 1 -> day 2")


def ieee_month1_training_scores(cfg: dict) -> tuple[dict, dict]:
    """Every member fitted on all of IEEE-CIS month 1 (freeze-v2 features fitted on month 1,
    z-score from month 1); training scores only. Month-1 rows only, isFraud never read."""
    names = [f["name"] for f in ieee_features.load_frozen()["features"]]
    month1, info = ieee_features.load_month1()
    assert "isFraud" not in month1.columns
    rule = ieee_features.FeatureRule.fit(month1, names)
    Z = data.Standardizer.fit(X := rule.transform(month1)).transform(X)
    out = {}
    for det in cfg["detectors"]["pool"]:
        seeds = cfg["detectors"]["iforest_seeds"] if det in detectors.SEEDED else [None]
        out[det] = np.vstack([detectors.fit_scores(det, Z, sd)[1] for sd in seeds])
    return out, info


def spearman_against(frag: pd.DataFrame, over: pd.DataFrame) -> pd.DataFrame:
    """Per q: Spearman over members of F against log r (descriptive)."""
    rows = []
    for q, g in frag.groupby("q"):
        o = over[over.q == q].set_index("member")
        if o.empty:
            continue
        g = g.set_index("member").loc[o.index]
        lr = np.log(np.maximum(o["r"].to_numpy(), 1e-12))
        rho = spearmanr(g["fragility"].to_numpy(), lr)[0]
        rows.append({"q": q, "n_members": len(g), "spearman_F_vs_log_r": float(rho)})
    return pd.DataFrame(rows)


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="python -m thesis.fragility")
    ap.add_argument("--ulb", action="store_true", help="ULB day 1 only")
    args = ap.parse_args(argv)
    state = git_state()
    if state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    t0 = time.perf_counter()
    cfg = frozen.load()
    meta = {**state, "command": "python -m thesis.fragility" + (" --ulb" if args.ulb else ""),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "definition": "F = share of training scores > t - DELTA (t - median), / (1 - q); "
                          "t = numpy.quantile(training scores, q, 'linear'); IForest: mean of "
                          "the five seeds", "delta": DELTA,
            "delta_note": "fixed once before any value was computed; not tuned",
            "labels_read": False,
            "note": "Exploratory (developed after seeing ULB). No exclusion rule uses F."}
    pd.set_option("display.width", 200)
    q_ulb = quantiles_for(cfg, "ulb")
    ulb = table(ulb_training_scores(cfg), q_ulb, "ULB day 1")
    over = ulb_overshoot(cfg, q_ulb)
    rho = spearman_against(ulb, over)
    print(ulb.round(3).to_string(index=False))
    print(rho.round(3).to_string(index=False))
    OUT_ULB.mkdir(parents=True, exist_ok=True)
    c = state["commit"]
    ulb.assign(commit=c).to_csv(OUT_ULB / "fragility_ulb.csv", index=False)
    rho.assign(commit=c).to_csv(OUT_ULB / "fragility_ulb_spearman.csv", index=False)
    (OUT_ULB / "fragility_ulb.meta.json").write_text(json.dumps(
        {**meta, "quantiles": q_ulb, "rows": "day 1 (Time < 86,400 s), E3 score cache",
         "spearman": "development check: F (day 1) against log r (day 2 at the day-1 "
                     "threshold), over the six members; label-free",
         "runtime_s": round(time.perf_counter() - t0, 1)}, indent=1) + "\n", encoding="utf-8")
    if args.ulb:
        return
    q_ieee = quantiles_for(cfg, "ieee_cis")
    train, info = ieee_month1_training_scores(cfg)
    ieee = table(train, q_ieee, "IEEE-CIS month 1")
    print(ieee.round(3).to_string(index=False))
    OUT_IEEE.mkdir(parents=True, exist_ok=True)
    ieee.assign(commit=c).to_csv(OUT_IEEE / "fragility_month1.csv", index=False)
    (OUT_IEEE / "fragility_month1.meta.json").write_text(json.dumps(
        {**meta, "quantiles": q_ieee, "rows": "month 1 only (ieee_features.load_month1)",
         "features": "freeze-v2 list; medians, frequency maps and z-score fitted on month 1",
         "month1": info, "runtime_s": round(time.perf_counter() - t0, 1)}, indent=1) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
