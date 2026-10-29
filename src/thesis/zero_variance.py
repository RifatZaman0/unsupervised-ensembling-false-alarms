"""Zero-variance check of detector training scores, before the freeze (B06b).

    python -m thesis.zero_variance

Under z-score calibration a member whose training scores are constant maps to 0 (fusion.py);
the supervisor has not confirmed that rule yet. This counts how often it would apply:
- ADBench: every cached training-score vector under the main cache key (15 sets x 5 seeds x 6);
- ULB: the 6 detectors fitted on day 1 with the planned preprocessing (z-score with day-1
  statistics, KNN / LOF on the 30,000-row sample, IForest with seeds 0-4).
Training scores only: the ULB labels are never read and no day-2 row is scored.
Writes results/tables/setup/zero_variance.csv (one row per member) and .meta.json; the ULB fit
times are reused by the runtime table (thesis.export_setup).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from thesis import data, detectors, scorecache
from thesis.export_datasets import git_state
from thesis.paths import CACHE, TABLES

OUT = TABLES / "setup"
MAIN_CACHE_KEY = "21ba8c24b9"
ULB_IFOREST_SEEDS = data.ADBENCH_SEEDS


def is_zero_variance(s: np.ndarray) -> bool:
    """The case fusion.Calibrator maps to 0 under z-score: constant training scores (range
    exactly 0; the float std of a constant vector need not be exactly 0)."""
    return bool(np.ptp(np.asarray(s, dtype=np.float64)) == 0)


def spread(s: np.ndarray) -> dict:
    s = np.asarray(s, dtype=np.float64)
    return {"n_rows": len(s), "std": float(s.std()), "ptp": float(np.ptp(s)),
            "n_unique": int(np.unique(s).size), "zero_variance": is_zero_variance(s)}


def adbench_rows(key: str = MAIN_CACHE_KEY) -> list[dict]:
    root = CACHE / key
    rows = []
    for path in sorted(root.glob("*/seed*/*.npz")):
        with np.load(path) as z:
            rows.append({"data": "ADBench", "dataset": path.parts[-3],
                         "seed": int(path.parts[-2].removeprefix("seed")),
                         "detector": path.stem, **spread(z["train"]),
                         "runtime_s": float(z["runtime_s"])})
    return rows


def ulb_rows() -> list[dict]:
    X = data.load_ulb_day1_features()
    X = data.Standardizer.fit(X).transform(X)
    rows = []
    for det in detectors.DETECTORS:
        for seed in (ULB_IFOREST_SEEDS if det in detectors.SEEDED else (None,)):
            t0 = time.perf_counter()
            fitted, s = detectors.fit_scores(det, X, seed)
            rows.append({"data": "ULB day 1", "dataset": "ULB", "seed": seed, "detector": det,
                         **spread(s), "runtime_s": time.perf_counter() - t0,
                         "pca_dropped": getattr(fitted, "n_dropped_", None)})
            print(f"ULB {det} seed {seed}: std {rows[-1]['std']:.4g}, "
                  f"{rows[-1]['runtime_s']:.1f} s", flush=True)
    return rows


def main() -> None:
    state = git_state()
    if state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before exporting results.")
    if scorecache.cache_key() != MAIN_CACHE_KEY:
        raise SystemExit(f"cache key is {scorecache.cache_key()}, expected {MAIN_CACHE_KEY}")
    ad, ulb = adbench_rows(), ulb_rows()
    if len(ad) != 15 * 5 * 6:
        raise SystemExit(f"expected 450 cached ADBench score files, found {len(ad)}")
    df = pd.DataFrame(ad + ulb)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "zero_variance.csv", index=False)
    counts = {d: {"members": int(len(g)), "zero_variance": int(g.zero_variance.sum()),
                  "min_std": float(g["std"].min()),
                  "min_std_member": " ".join(str(v) for v in
                                             g.loc[g["std"].idxmin(), ["dataset", "seed", "detector"]])}
              for d, g in df.groupby("data")}
    (OUT / "zero_variance.meta.json").write_text(json.dumps({
        **state, "command": "python -m thesis.zero_variance",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cache_key": MAIN_CACHE_KEY, "counts": counts,
        "note": "Training scores only. ULB: day-1 features (Class column never read), z-scored "
                "with day-1 statistics; no day-2 row scored. zero_variance = training scores "
                "constant (range exactly 0), the case fusion maps to 0 under z-score "
                "calibration",
    }, indent=1) + "\n", encoding="utf-8")
    for d, c in counts.items():
        print(f"{d}: {c['zero_variance']} of {c['members']} members with zero-variance training "
              f"scores (smallest std {c['min_std']:.4g}, {c['min_std_member']})")


if __name__ == "__main__":
    main()
