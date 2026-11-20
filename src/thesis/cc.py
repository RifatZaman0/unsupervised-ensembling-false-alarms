"""Committee cleaning test (prereg_cc.md): runner for the refit check and for Step 1, the oracle ceiling.

    python -m thesis.cc check-refit            # all training rows kept reproduces the cached scores
    python -m thesis.cc check-refit cardio     # the same for a subset
    python -m thesis.cc step1                  # oracle ceiling and gate; writes results once

Refit (prereg_cc.md Section 3, step 3): for one ADBench split and a boolean mask over its training
rows, each of the six detectors is rebuilt on the kept training rows only: z-score fitted on the
kept rows (thesis.data.Standardizer), the detector fitted on the kept rows with the E1 parameters
and seed = split seed (thesis.detectors.score, which gives fit-time training scores, leave-one-out
for KNN and LOF), and every test row scored. Test rows are never removed or reordered. Refits are
kept in memory only; nothing is written to the score cache (cache/scores/<key>/ stays as it is).

Reproduction check (Section 10, item 3): the refit with every training row kept must give the
cached training and test scores of each member bit for bit (np.array_equal). `check-refit` checks
it for all 450 members; Step 1 repeats it on every split it uses and writes nothing if one differs.

Step 1 (Section 5; training labels, diagnostic only): oracle cleaning (OC) drops every training
anomaly, refits the six members on the remaining rows and fuses them (min-max fitted on the kept
rows' training scores, then the mean; thesis.fusion.fuse). Per dataset x seed x budget (2%, 5%):
FP@K of OC E-all, plain E-all (cached scores, min-max fitted on all training scores + mean), plain
IForest (the cached E1 member), the oracle-best and median plain member and the oracle-best member
after oracle cleaning (test labels, references only). Seed means are rounded to TIE_DECIMALS before
anything is compared.

Gate (E-all, 2%), all three must hold: (1) OC E-all has fewer false alarms than plain E-all on more
datasets than the reverse (ties not counted); (2) the mean FP@K of OC E-all over the 15 datasets is
at least 5% below that of plain E-all; (3) it is below the mean FP@K of plain IForest. Conditions
(2) and (3) are evaluated exactly (fractions.Fraction of the rounded seed means).

Outputs in results/tables/cc/: cc_step1_runs.csv, cc_step1_per_dataset.csv, cc_step1_gate.csv,
cc_step1.meta.json. Step 1 refuses a dirty tree, a missing tag prereg-cc, a HEAD that does not
descend from it, a prereg_cc.md that differs from the tag, existing Step 1 outputs, a refit that
does not reproduce the cache, plain member FP@K that do not reproduce e1_runs.csv and any change
to the score cache folder during the run. Fraud data is not touched here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction

import numpy as np
import pandas as pd

from thesis import config, data, detectors, fusion, metrics, scorecache
from thesis.export_datasets import git_state
from thesis.paths import CACHE, REPO, TABLES

OUT = TABLES / "cc"
E1_RUNS = TABLES / "e1" / "e1_runs.csv"
PREREG_TAG = "prereg-cc"
PREREG_FILE = "prereg_cc.md"
CACHE_KEY = "21ba8c24b9"
BUDGETS = metrics.ADBENCH_BUDGETS
E_ALL = tuple(detectors.DETECTORS)
TIE_DECIMALS = 9
GATE_BUDGET = 0.02
GATE_MIN_REDUCTION = Fraction(5, 100)  # OC mean at least 5% below plain E-all
STEP1_FILES = ("cc_step1_runs.csv", "cc_step1_per_dataset.csv", "cc_step1_gate.csv",
               "cc_step1.meta.json")


# ------------------------------------------------------------------ refit pipeline

def refit_member(split: data.Split, keep: np.ndarray, det: str) -> tuple[np.ndarray, np.ndarray]:
    """E1 pipeline for one detector on the kept training rows of a raw (not preprocessed) split.

    Returns (training scores of the kept rows, scores of every test row). Takes no labels; the
    z-score and the detector see the kept training rows only."""
    keep = np.asarray(keep)
    if keep.dtype != bool or keep.shape != (len(split.X_train),):
        raise ValueError("keep must be a boolean mask over the training rows")
    if not keep.any():
        raise ValueError("keep removes every training row")
    X_fit, X_test = split.X_train[keep], split.X_test
    if config.PROTOCOL.zscore_inputs:
        sc = data.Standardizer.fit(X_fit)
        X_fit, X_test = sc.transform(X_fit), sc.transform(X_test)
    return detectors.score(det, X_fit, X_test, split.seed)


def refit_all(split: data.Split, keep: np.ndarray, members=E_ALL) -> tuple[dict, dict]:
    train, test = {}, {}
    for det in members:
        train[det], test[det] = refit_member(split, keep, det)
    return train, test


def oracle_keep_mask(y_train: np.ndarray) -> np.ndarray:
    """Oracle cleaning (Step 1 only, diagnostic): keep exactly the normal training rows."""
    y = np.asarray(y_train)
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("labels must be 0/1")
    return y == 0


def fused_test(train: dict, test: dict, members=E_ALL) -> np.ndarray:
    """Min-max fitted on each member's training scores, then the mean (primary fusion)."""
    S_tr = np.column_stack([train[d] for d in members])
    S_te = np.column_stack([test[d] for d in members])
    return fusion.fuse(S_tr, S_te, *fusion.PRIMARY)[1]


def fp_at_k(scores: np.ndarray, y: np.ndarray, budget: float) -> float:
    return metrics.budget_metrics(scores, y, budget)["fp_at_k"]


def refit_matches_cache(split: data.Split, cached_train: dict, cached_test: dict,
                        members=E_ALL) -> list[dict]:
    """One row per member: does the refit with all training rows kept give the cached scores?"""
    keep = np.ones(len(split.X_train), dtype=bool)
    rows = []
    for det in members:
        tr, te = refit_member(split, keep, det)
        rows.append({"dataset": split.dataset, "seed": split.seed, "detector": det,
                     "train_identical": bool(np.array_equal(tr, cached_train[det])),
                     "test_identical": bool(np.array_equal(te, cached_test[det])),
                     "max_abs_diff": float(max(np.abs(tr - cached_train[det]).max(),
                                               np.abs(te - cached_test[det]).max()))})
    return rows


def cached_scores(raw: data.Split) -> tuple[dict, dict]:
    """Cached E1 scores of the six members; never computes (a missing file stops the run)."""
    pre = data.preprocess(raw)
    train, test = {}, {}
    for det in E_ALL:
        if not scorecache.cache_path(raw.dataset, raw.seed, det).exists():
            raise SystemExit(f"score cache missing for {raw.dataset} seed {raw.seed} {det} "
                             f"(key {scorecache.cache_key()}); run python -m thesis.e1 first")
        sc = scorecache.get_scores(pre, det)
        train[det], test[det] = sc["train"], sc["test"]
    return train, test


def raw_splits(datasets):
    for name in datasets:
        X, y = data.load_adbench(name)
        for seed in data.ADBENCH_SEEDS:
            yield data.split_adbench(name, seed, X, y)


def check_refit(datasets) -> pd.DataFrame:
    rows = []
    for raw in raw_splits(datasets):
        rows += refit_matches_cache(raw, *cached_scores(raw))
        print(f"{raw.dataset} seed {raw.seed} checked", flush=True)
    return pd.DataFrame(rows)


def refit_summary(chk: pd.DataFrame) -> dict:
    return {"n_members": int(len(chk)), "n_train_identical": int(chk["train_identical"].sum()),
            "n_test_identical": int(chk["test_identical"].sum()),
            "max_abs_diff": float(chk["max_abs_diff"].max()) if len(chk) else float("nan")}


def refit_ok(summary: dict, expected: int) -> bool:
    return (summary["n_members"] == expected and summary["n_train_identical"] == expected
            and summary["n_test_identical"] == expected)


# ------------------------------------------------------------------ Step 1 (prereg Section 5)

FP_COLS = ["fp_oc_eall", "fp_plain_eall", "fp_plain_iforest", "fp_best_plain_member",
           "fp_median_plain_member", "fp_best_oc_member"]


def step1_rows(raw: data.Split, plain_train: dict, plain_test: dict, oc_train: dict,
               oc_test: dict, keep: np.ndarray) -> list[dict]:
    """One row per budget for one split. Training labels entered only through `keep` (oracle
    cleaning); test labels enter only through FP@K."""
    y = raw.y_test
    plain_fused = fused_test(plain_train, plain_test)
    oc_fused = fused_test(oc_train, oc_test)
    rows = []
    for b in BUDGETS:
        plain_fp = np.array([fp_at_k(plain_test[d], y, b) for d in E_ALL])
        oc_fp = np.array([fp_at_k(oc_test[d], y, b) for d in E_ALL])
        row = {"dataset": raw.dataset, "seed": raw.seed, "budget": b,
               "n_train": len(raw.y_train), "n_train_anomalies": int(raw.y_train.sum()),
               "n_kept_oc": int(keep.sum()), "n_test": len(y),
               "k_test": metrics.budget_k(b, len(y)),
               "fp_oc_eall": fp_at_k(oc_fused, y, b),
               "fp_plain_eall": fp_at_k(plain_fused, y, b),
               "fp_plain_iforest": float(plain_fp[E_ALL.index("IForest")]),
               "fp_best_plain_member": float(plain_fp.min()),
               "best_plain_member": E_ALL[int(plain_fp.argmin())],
               "fp_median_plain_member": float(np.median(plain_fp)),
               "fp_best_oc_member": float(oc_fp.min()),
               "best_oc_member": E_ALL[int(oc_fp.argmin())]}
        for i, d in enumerate(E_ALL):
            row[f"fp_plain_{d}"] = float(plain_fp[i])
        for i, d in enumerate(E_ALL):
            row[f"fp_oc_{d}"] = float(oc_fp[i])
        rows.append(row)
    return rows


def run_step1(datasets) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 1 rows and the reproduction check of every split used."""
    rows, checks = [], []
    for raw in raw_splits(datasets):
        t0 = time.perf_counter()
        plain_train, plain_test = cached_scores(raw)
        checks += refit_matches_cache(raw, plain_train, plain_test)
        keep = oracle_keep_mask(raw.y_train)
        oc_train, oc_test = refit_all(raw, keep)
        rows += step1_rows(raw, plain_train, plain_test, oc_train, oc_test, keep)
        print(f"{raw.dataset} seed {raw.seed} done ({time.perf_counter() - t0:.1f} s)", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(checks)


def per_dataset(runs: pd.DataFrame) -> pd.DataFrame:
    """Seed means per dataset x budget, rounded to TIE_DECIMALS; OC - plain on the rounded means."""
    g = runs.groupby(["dataset", "budget"], sort=True)
    df = g[FP_COLS].mean().round(TIE_DECIMALS)
    df = df.join(g["seed"].nunique().rename("n_seeds")).join(g["k_test"].first()).reset_index()
    d = df["fp_oc_eall"] - df["fp_plain_eall"]
    df["delta_oc_vs_plain"] = d
    df["outcome"] = np.select([d < 0, d > 0], ["oc fewer", "plain fewer"], "tie")
    return df


def _exact_sum(s: pd.Series) -> Fraction:
    return sum((Fraction(str(round(float(v), TIE_DECIMALS))) for v in s), Fraction(0))


def gate_table(per_ds: pd.DataFrame) -> pd.DataFrame:
    """Per budget: the three gate conditions (the decision uses the 2% row only)."""
    rows = []
    for b, g in per_ds.groupby("budget", sort=True):
        n = len(g)
        oc, plain, iforest = (_exact_sum(g[c]) for c in
                              ("fp_oc_eall", "fp_plain_eall", "fp_plain_iforest"))
        n_oc, n_plain = int((g["outcome"] == "oc fewer").sum()), int((g["outcome"] == "plain fewer").sum())
        c1 = n_oc > n_plain
        c2 = plain > 0 and oc <= (1 - GATE_MIN_REDUCTION) * plain
        c3 = oc < iforest
        rows.append({"budget": b, "n_datasets": n, "n_oc_fewer": n_oc, "n_plain_fewer": n_plain,
                     "n_ties": int((g["outcome"] == "tie").sum()),
                     "mean_fp_oc_eall": float(oc / n), "mean_fp_plain_eall": float(plain / n),
                     "mean_fp_plain_iforest": float(iforest / n),
                     "mean_fp_best_plain_member": float(g["fp_best_plain_member"].mean()),
                     "mean_fp_median_plain_member": float(g["fp_median_plain_member"].mean()),
                     "mean_fp_best_oc_member": float(g["fp_best_oc_member"].mean()),
                     "rel_reduction_oc_vs_plain": float(1 - oc / plain) if plain else float("nan"),
                     "cond1_oc_fewer_on_more_datasets": bool(c1),
                     "cond2_mean_at_least_5pct_below_plain": bool(c2),
                     "cond3_mean_below_iforest": bool(c3),
                     "all_three": bool(c1 and c2 and c3),
                     "is_gate": bool(np.isclose(b, GATE_BUDGET))})
    return pd.DataFrame(rows)


def gate_passes(gate: pd.DataFrame) -> bool:
    row = gate[gate["is_gate"]]
    if len(row) != 1:
        raise ValueError(f"expected one gate row, found {len(row)}")
    return bool(row["all_three"].iloc[0])


def members_match_e1(runs: pd.DataFrame, e1_runs=E1_RUNS) -> dict:
    """Plain member FP@K used here against E1's single-detector FP@K (round-trip parser)."""
    e1 = pd.read_csv(e1_runs, float_precision="round_trip",
                     usecols=["dataset", "seed", "detector", "budget", "fp_at_k"])
    cols = [f"fp_plain_{d}" for d in E_ALL]
    m = runs[["dataset", "seed", "budget", *cols]].melt(["dataset", "seed", "budget"], cols,
                                                         "detector", "fp_here")
    m["detector"] = m["detector"].str.removeprefix("fp_plain_")
    j = m.merge(e1, on=["dataset", "seed", "budget", "detector"], how="left")
    diff = (j["fp_here"] - j["fp_at_k"]).abs()
    return {"n_compared": int(j["fp_at_k"].notna().sum()),
            "n_unmatched": int(j["fp_at_k"].isna().sum()),
            "n_not_identical": int((diff != 0).sum()),
            "max_abs_diff": float(diff.max()) if len(j) else float("nan")}


# ------------------------------------------------------------------ guards

def tag_commit(tag: str = PREREG_TAG) -> str | None:
    r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}"],
                       cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip() or None


def head_contains(commit: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                          cwd=REPO).returncode == 0


def prereg_unchanged(path: str = PREREG_FILE, tag: str = PREREG_TAG) -> bool:
    return subprocess.run(["git", "diff", "--quiet", tag, "--", path], cwd=REPO).returncode == 0


def check_can_write(state: dict, prereg: str | None, contains, unchanged, files, out=OUT) -> None:
    """A step writes once, from a clean tree that descends from prereg-cc with prereg_cc.md
    unchanged since the tag."""
    if state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    if prereg is None:
        raise SystemExit(f"tag {PREREG_TAG} does not exist; tag the pre-registration first.")
    if not contains(prereg):
        raise SystemExit(f"HEAD does not descend from {PREREG_TAG} ({prereg}).")
    if not unchanged():
        raise SystemExit(f"{PREREG_FILE} differs from {PREREG_TAG}; the pre-registration is fixed.")
    existing = [f for f in files if (out / f).exists()]
    if existing:
        raise SystemExit(f"results already exist ({existing[0]}); each step runs once.")


def cache_listing(root=CACHE) -> dict[str, tuple[int, int]]:
    """Every file under the score cache with its size and mtime (ns), to prove it is untouched."""
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


def step1_main(argv: list[str]) -> None:
    datasets = sorted(data.ADBENCH_15, key=str.lower)
    state, prereg = git_state(), tag_commit()
    check_can_write(state, prereg, head_contains, prereg_unchanged, STEP1_FILES, OUT)
    before = cache_listing()
    t0 = time.perf_counter()
    runs, chk = run_step1(datasets)
    refit = refit_summary(chk)
    e1_check = members_match_e1(runs)
    cache_unchanged = cache_listing() == before
    print("refit with all rows vs cache:", refit)
    print("plain member FP@K vs e1_runs.csv:", e1_check)
    print("score cache unchanged:", cache_unchanged)
    if not refit_ok(refit, len(datasets) * len(data.ADBENCH_SEEDS) * len(E_ALL)):
        raise SystemExit("refit with all training rows does not reproduce the cache; nothing written")
    if e1_check["n_unmatched"] or e1_check["n_not_identical"]:
        raise SystemExit("plain member FP@K do not reproduce e1_runs.csv; nothing written")
    if not cache_unchanged:
        raise SystemExit("the score cache changed during the run; nothing written")
    per_ds = per_dataset(runs)
    gate = gate_table(per_ds)
    passed = gate_passes(gate)
    OUT.mkdir(parents=True, exist_ok=True)
    c = state["commit"]
    runs.assign(commit=c).to_csv(OUT / "cc_step1_runs.csv", index=False)
    per_ds.assign(commit=c).to_csv(OUT / "cc_step1_per_dataset.csv", index=False)
    gate.assign(commit=c).to_csv(OUT / "cc_step1_gate.csv", index=False)
    back = pd.read_csv(OUT / "cc_step1_per_dataset.csv", float_precision="round_trip")
    rebuilt = gate_table(back.drop(columns="commit"))
    rebuild_identical = bool(rebuilt.equals(gate))
    meta = {**state, "command": " ".join(["python -m thesis.cc", *argv]),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "prereg": PREREG_FILE, "prereg_tag": PREREG_TAG, "prereg_tag_commit": prereg,
            "step": "Step 1: oracle ceiling and gate (training labels; diagnostic only)",
            "cache_key": scorecache.cache_key(), "protocol": config.PROTOCOL.as_dict(),
            "preprocessing": scorecache.preprocessing() + " (refits: fitted on the kept rows)",
            "detector_params": detectors.params(), "budgets": list(BUDGETS),
            "seeds": list(data.ADBENCH_SEEDS), "datasets": datasets,
            "oracle_cleaning": "every training anomaly removed (training labels); the six "
                               "members refitted on the normal training rows; test rows unchanged",
            "fusion": "E-all: min-max fitted on each member's training scores (kept rows for OC), "
                      "then the mean (thesis.fusion.fuse, primary config)",
            "k_rule": "K = ceil(budget * n_test), at least 1 (metrics.budget_k); ties at the K-th "
                      "score split in expectation",
            "tie_rules": f"dataset outcome on seed means rounded to {TIE_DECIMALS} decimals; "
                         "best member ties: first in detector order (name only)",
            "gate": {"member_set": "+".join(E_ALL), "budget": GATE_BUDGET,
                     "rule": "pass if all three hold: (1) OC E-all fewer false alarms than plain "
                             "E-all on more datasets than the reverse (ties not counted); (2) mean "
                             "FP@K of OC E-all at least 5% below plain E-all; (3) mean FP@K of OC "
                             "E-all below plain IForest; (2) and (3) exact on the rounded means",
                     "passed": passed},
            "refit_all_rows_vs_cache": refit, "plain_members_vs_e1_runs": e1_check,
            "score_cache_unchanged": cache_unchanged,
            "gate_rebuilt_from_csv_identical": rebuild_identical,
            "runtime_s": round(time.perf_counter() - t0, 1),
            "note": "training labels used only for oracle cleaning (diagnostic); test labels only "
                    "for FP@K and the oracle references; refits kept in memory, not cached; "
                    "ADBench only"}
    (OUT / "cc_step1.meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(gate.to_string(index=False))
    if not rebuild_identical:
        raise SystemExit("gate differs when rebuilt from cc_step1_per_dataset.csv")
    print("gate", "PASSED" if passed else "FAILED", "; wrote", OUT)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="python -m thesis.cc")
    parser.add_argument("command", choices=["check-refit", "step1"])
    parser.add_argument("datasets", nargs="*",
                        help="check-refit only: subset to check (Step 1 always runs on all 15)")
    args = parser.parse_args(argv)
    if config.PROTOCOL != config.Protocol() or scorecache.cache_key() != CACHE_KEY:
        raise SystemExit(f"default protocol and cache key {CACHE_KEY} required, "
                         f"got {scorecache.cache_key()}")
    if args.command == "check-refit":
        datasets = args.datasets or sorted(data.ADBENCH_15, key=str.lower)
        before = cache_listing()
        chk = check_refit(datasets)
        s = refit_summary(chk)
        s["score_cache_unchanged"] = cache_listing() == before
        print(json.dumps(s, indent=1))
        bad = chk[~(chk["train_identical"] & chk["test_identical"])]
        if len(bad):
            print(bad.to_string(index=False))
        if not refit_ok(s, len(datasets) * len(data.ADBENCH_SEEDS) * len(E_ALL)) \
                or not s["score_cache_unchanged"]:
            raise SystemExit("refit with all training rows does not reproduce the cache")
        return
    if args.datasets:
        raise SystemExit("step1 runs on all 15 datasets only (no real-data subset runs)")
    step1_main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
