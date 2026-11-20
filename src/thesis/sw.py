"""Spectral weighting test (prereg_sw.md): runner for the checks and for Step 1, the ceiling gate.

    python -m thesis.sw check-eq   # equal weights reproduce E2's FP@K for the 20 triples
    python -m thesis.sw step1      # ceiling gate (training labels, diagnostic); writes results

Step 1 (prereg_sw.md Section 5): for E-all (grid step 0.1, 3,003 weight vectors) and each of the
20 triples (step 0.05, 231 vectors), at 2% and 5%, the weights with the fewest TRAINING false
alarms (training labels) are evaluated on the test split and compared with the member that has
the fewest training false alarms (training labels too), evaluated on test. Calibration: min-max
fitted on each member's training scores (thesis.fusion.Calibrator, not clipped).

Gate: continue to Step 2 only if, for E-all at 2%, the tuned weights raise fewer test false
alarms than the label-selected member on more datasets than the reverse (seed means; exact ties
not counted). Seed means are rounded to TIE_DECIMALS before they are compared.

Outputs in results/tables/sw/: sw_step1_runs.csv (dataset x seed x member set x budget),
sw_step1_per_dataset.csv (seed means), sw_step1_gate.csv (member set x budget: datasets where
the tuned weights / the selected member have fewer test false alarms, ties), sw_step1.meta.json.
Step 1 refuses a dirty tree and a HEAD that does not contain the tag prereg-sw.
Fraud data is not touched here. Step 2 is not part of this module yet (prereg_sw.md Section 6).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd

from thesis import ceiling, config, data, detectors, fusion, metrics, scorecache
from thesis.export_datasets import git_state
from thesis.paths import REPO, TABLES

OUT = TABLES / "sw"
E2_RUNS = TABLES / "e2" / "e2_runs.csv"
PREREG_TAG = "prereg-sw"
BUDGETS = metrics.ADBENCH_BUDGETS
E_ALL = tuple(detectors.DETECTORS)
TRIPLES = tuple(combinations(detectors.DETECTORS, 3))
STEP = {6: 0.1, 3: 0.05}  # grid step by member-set size (prereg_sw.md Section 5)
GATE = {"member_set": "+".join(E_ALL), "budget": 0.02}
TIE_DECIMALS = 9


def member_sets() -> tuple[tuple[str, ...], ...]:
    return (E_ALL, *TRIPLES)


def load_split(name: str, seed: int, X=None, y=None):
    """Cached training/test scores of all six detectors for one dataset x seed."""
    split = data.preprocess(data.split_adbench(name, seed, X, y))
    train, test = {}, {}
    for det in detectors.DETECTORS:
        if not scorecache.cache_path(name, seed, det).exists():
            raise SystemExit(f"score cache missing for {name} seed {seed} {det} "
                             f"(key {scorecache.cache_key()}); run python -m thesis.e1 first")
        sc = scorecache.get_scores(split, det)
        train[det], test[det] = sc["train"], sc["test"]
    return split, train, test


def calibrated(train: dict, test: dict, members) -> tuple[np.ndarray, np.ndarray]:
    """Members x rows: min-max fitted on each member's training scores, applied to both splits."""
    cals = [fusion.Calibrator.fit(train[d], fusion.PRIMARY[0]) for d in members]
    return (np.vstack([c.transform(train[d]) for c, d in zip(cals, members)]),
            np.vstack([c.transform(test[d]) for c, d in zip(cals, members)]))


def splits(datasets):
    for name in datasets:
        X, y = data.load_adbench(name)
        for seed in data.ADBENCH_SEEDS:
            split, train, test = load_split(name, seed, X, y)
            yield name, seed, split, train, test


def eq_rows(datasets) -> pd.DataFrame:
    """FP@K of the equal-weight triples (min-max + mean) as Step 1 computes it."""
    rows = []
    for name, seed, split, train, test in splits(datasets):
        for members in TRIPLES:
            _, cal_te = calibrated(train, test, members)
            for b in BUDGETS:
                k = metrics.budget_k(b, cal_te.shape[1])
                rows.append({"dataset": name, "seed": seed, "combo": "+".join(members),
                             "budget": b,
                             "fp_equal_weights": ceiling.fp_at_k(cal_te.mean(axis=0),
                                                                 split.y_test, k)})
    return pd.DataFrame(rows)


def compare_with_e2(eq: pd.DataFrame, e2_runs=E2_RUNS) -> dict:
    """Join the equal-weight FP@K with e2_runs.csv (primary config, triples). Read with the
    round-trip float parser, so equal means bit-identical."""
    e2 = pd.read_csv(e2_runs, float_precision="round_trip",
                     usecols=["dataset", "seed", "combo", "size", "calib", "agg", "budget",
                              "fp_at_k"])
    e2 = e2[(e2["size"] == 3) & (e2["calib"] == fusion.PRIMARY[0])
            & (e2["agg"] == fusion.PRIMARY[1])]
    j = eq.merge(e2, on=["dataset", "seed", "combo", "budget"], how="outer", indicator=True)
    both = j[j["_merge"] == "both"]
    diff = (both["fp_equal_weights"] - both["fp_at_k"]).abs()
    return {"n_compared": int(len(both)), "n_unmatched": int((j["_merge"] != "both").sum()),
            "n_not_identical": int((diff != 0).sum()),
            "max_abs_diff": float(diff.max()) if len(both) else float("nan")}


def run_step1(datasets, sets=None) -> pd.DataFrame:
    sets = sets or member_sets()
    grids = {m: ceiling.simplex_grid(m, STEP[m]) for m in {len(s) for s in sets}}
    rows = []
    for name, seed, split, train, test in splits(datasets):
        t0 = time.perf_counter()
        for members in sets:
            cal_tr, cal_te = calibrated(train, test, members)
            for b in BUDGETS:
                out = ceiling.ceiling_run(cal_tr, split.y_train, cal_te, split.y_test, b,
                                          STEP[len(members)])
                rows.append({
                    "dataset": name, "seed": seed, "member_set": "+".join(members),
                    "size": len(members), "budget": b, "grid_step": STEP[len(members)],
                    "n_grid": len(grids[len(members)]),
                    "n_train": cal_tr.shape[1], "n_test": cal_te.shape[1],
                    "k_train": metrics.budget_k(b, cal_tr.shape[1]), "k_test": out["k_test"],
                    "weights": ";".join(f"{d}:{w:.2f}" for d, w in zip(members, out["weights"])),
                    "max_weight": float(out["weights"].max()),
                    "label_selected_member": members[out["label_selected"]],
                    "fp_tuned_weights": out["fp_tuned_weights"],
                    "fp_label_selected_member": out["fp_label_selected_member"],
                    "fp_equal_weights": out["fp_equal_weights"],
                    "fp_oracle_best_member": out["fp_oracle_best_member"],
                    "fp_median_member": out["fp_median_member"],
                })
        print(f"{name} seed {seed} done ({time.perf_counter() - t0:.1f} s)", flush=True)
    return pd.DataFrame(rows)


FP_COLS = ["fp_tuned_weights", "fp_label_selected_member", "fp_equal_weights",
           "fp_oracle_best_member", "fp_median_member"]


def per_dataset(runs: pd.DataFrame) -> pd.DataFrame:
    """Seed means per dataset x member set x budget; tuned - selected on the rounded means."""
    g = runs.groupby(["dataset", "member_set", "size", "budget"], sort=True)
    df = g[FP_COLS + ["max_weight"]].mean().join(g["k_test"].first())
    df = df.join(g["seed"].nunique().rename("n_seeds")).reset_index()
    d = (df["fp_tuned_weights"].round(TIE_DECIMALS)
         - df["fp_label_selected_member"].round(TIE_DECIMALS))
    df["delta_tuned_vs_selected"] = d
    df["outcome"] = np.select([d < 0, d > 0], ["tuned fewer", "selected fewer"], "tie")
    return df


def gate_table(per_ds: pd.DataFrame) -> pd.DataFrame:
    g = per_ds.groupby(["member_set", "size", "budget"], sort=False)
    df = pd.DataFrame({
        "n_datasets": g.size(),
        "n_tuned_fewer": g["outcome"].apply(lambda s: int((s == "tuned fewer").sum())),
        "n_selected_fewer": g["outcome"].apply(lambda s: int((s == "selected fewer").sum())),
        "n_ties": g["outcome"].apply(lambda s: int((s == "tie").sum())),
        "mean_fp_tuned_weights": g["fp_tuned_weights"].mean(),
        "mean_fp_label_selected_member": g["fp_label_selected_member"].mean(),
        "mean_fp_equal_weights": g["fp_equal_weights"].mean(),
        "mean_fp_oracle_best_member": g["fp_oracle_best_member"].mean(),
    }).reset_index()
    df["tuned_wins_more"] = df["n_tuned_fewer"] > df["n_selected_fewer"]
    df["is_gate"] = (df["member_set"] == GATE["member_set"]) & np.isclose(df["budget"],
                                                                          GATE["budget"])
    return df.sort_values(["budget", "size", "member_set"], ascending=[True, False, True],
                          ignore_index=True)


def gate_passes(gate: pd.DataFrame) -> bool:
    row = gate[gate["is_gate"]]
    if len(row) != 1:
        raise ValueError(f"expected one gate row, found {len(row)}")
    return bool(row["tuned_wins_more"].iloc[0])


def tag_commit(tag: str = PREREG_TAG) -> str | None:
    r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}"],
                       cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip() or None


def head_contains(commit: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                          cwd=REPO).returncode == 0


def check_can_write(state: dict, prereg: str | None, contains) -> None:
    """Step 1 writes results only from a clean tree that contains the pre-registration tag."""
    if state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    if prereg is None:
        raise SystemExit(f"tag {PREREG_TAG} does not exist; tag the pre-registration first.")
    if not contains(prereg):
        raise SystemExit(f"HEAD does not contain {PREREG_TAG} ({prereg}).")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="python -m thesis.sw")
    parser.add_argument("command", choices=["check-eq", "step1"])
    parser.add_argument("datasets", nargs="*", help="subset for a quick check (writes nothing)")
    args = parser.parse_args(argv)
    if config.PROTOCOL != config.Protocol() or scorecache.cache_key() != "21ba8c24b9":
        raise SystemExit(f"default protocol and cache key 21ba8c24b9 required, "
                         f"got {scorecache.cache_key()}")
    datasets = args.datasets or sorted(data.ADBENCH_15, key=str.lower)
    if args.command == "check-eq":
        res = compare_with_e2(eq_rows(datasets))
        print(json.dumps(res, indent=1))
        if args.datasets:
            return
        if res["n_unmatched"] or res["n_not_identical"]:
            raise SystemExit("equal weights do not reproduce e2_runs.csv")
        return
    state = git_state()
    prereg = tag_commit()
    if not args.datasets:
        check_can_write(state, prereg, head_contains)
    t0 = time.perf_counter()
    runs = run_step1(datasets)
    per_ds = per_dataset(runs)
    gate = gate_table(per_ds)
    eq = runs[runs["size"] == 3].rename(columns={"member_set": "combo"})
    eq_check = compare_with_e2(eq[["dataset", "seed", "combo", "budget", "fp_equal_weights"]])
    print(gate.to_string(index=False))
    print("equal weights vs e2_runs.csv:", eq_check)
    if args.datasets:
        return
    passed = gate_passes(gate)
    meta = {**state, "command": " ".join(["python -m thesis.sw", *argv]),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "prereg": "prereg_sw.md", "prereg_tag": PREREG_TAG, "prereg_tag_commit": prereg,
            "step": "Step 1: ceiling gate (training labels; diagnostic only)",
            "cache_key": scorecache.cache_key(), "protocol": config.PROTOCOL.as_dict(),
            "preprocessing": scorecache.preprocessing(), "detector_params": detectors.params(),
            "budgets": list(BUDGETS), "seeds": list(data.ADBENCH_SEEDS),
            "datasets": datasets, "calibration": "minmax fitted on training scores, not clipped",
            "grid_step": {"6": STEP[6], "3": STEP[3]},
            "n_grid": {"6": len(ceiling.simplex_grid(6, STEP[6])),
                       "3": len(ceiling.simplex_grid(3, STEP[3]))},
            "k_rule": "K = ceil(budget * n), at least 1 (metrics.budget_k); K_train over n_train "
                      "for tuning and selection, K_test over n_test for evaluation; ties at the "
                      "K-th score split in expectation",
            "tie_rules": "weight vectors tied on training FP@K: largest single weight, then grid "
                         "order; label-selected member tied on training FP@K: first in member "
                         "order; dataset outcome on seed means rounded to 9 decimals",
            "gate": {"member_set": GATE["member_set"], "budget": GATE["budget"],
                     "rule": "pass if tuned weights have fewer test false alarms than the "
                             "label-selected member on more datasets than the reverse (seed "
                             "means; ties not counted)", "passed": passed},
            "equal_weights_vs_e2_runs": eq_check,
            "runtime_s": round(time.perf_counter() - t0, 1),
            "note": "training labels used for tuning and member selection (diagnostic); test "
                    "labels only for evaluation and the oracle references; ADBench only"}
    if eq_check["n_unmatched"] or eq_check["n_not_identical"]:
        raise SystemExit("equal weights do not reproduce e2_runs.csv; nothing written")
    OUT.mkdir(parents=True, exist_ok=True)
    runs.assign(commit=state["commit"]).to_csv(OUT / "sw_step1_runs.csv", index=False)
    per_ds.assign(commit=state["commit"]).to_csv(OUT / "sw_step1_per_dataset.csv", index=False)
    gate.assign(commit=state["commit"]).to_csv(OUT / "sw_step1_gate.csv", index=False)
    (OUT / "sw_step1.meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print("gate", "PASSED" if passed else "FAILED", "; wrote", OUT)


if __name__ == "__main__":
    main(sys.argv[1:])
