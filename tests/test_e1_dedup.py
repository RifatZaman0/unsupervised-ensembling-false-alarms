"""Duplicate table and dedup robustness check: duplicate counts are exact, the comparison
spots rank changes and best-detector changes, unchanged datasets must be identical, and a
run with the wrong protocol is refused."""
import json

import numpy as np
import pandas as pd
import pytest

from thesis import data, e1_dedup, e1_summary

DETS = ("A", "B", "C")


def test_duplicate_stats_counts():
    X = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0, 0, 1, 1, 1, 0])
    st = data.duplicate_stats(X, y)
    # (1,2,0) x2, (1,2,1) x1, (3,4,1) x2, (5,6,0) x1
    assert st["rows_in_dup_groups"] == 4 and st["anomalies_in_dup_groups"] == 2
    assert st["rows_removed"] == 2 and st["anomalies_removed"] == 1
    assert st["label_conflict_rows"] == 3          # the three [1, 2] rows
    assert st["n_rows_dedup"] == len(data.dedup_rows(X, y)[1]) == 4
    assert st["anomaly_rate_dedup"] == pytest.approx(2 / 4)


def test_duplicate_stats_without_duplicates():
    X = np.arange(12.0).reshape(6, 2)
    st = data.duplicate_stats(X, np.array([0, 0, 0, 0, 0, 1]))
    assert st["rows_in_dup_groups"] == st["rows_removed"] == st["label_conflict_rows"] == 0


@pytest.mark.skipif(not any(data.ADBENCH_DIR.glob("*.npz")), reason="ADBench not downloaded")
def test_duplicate_stats_on_real_mammography():
    # one feature vector occurs 3329 times (29.8% of rows), 7 of them labelled anomalies
    st = data.duplicate_stats(*data.load_adbench("mammography"))
    assert (st["n_rows"], st["rows_removed"], st["anomalies_removed"]) == (11183, 3334, 6)
    assert st["label_conflict_rows"] == 3329
    assert st["rows_in_dup_groups"] / st["n_rows"] == pytest.approx(0.298, abs=5e-4)


def per_ds(fp: dict, n_test: dict, budget=0.05) -> pd.DataFrame:
    """fp: {dataset: {detector: fp_at_k}} -> e1_summary.per_dataset-like frame."""
    runs = pd.DataFrame([{"dataset": d, "seed": 0, "detector": det, "budget": budget,
                          "fp_at_k": v, "precision_at_k": 0.0, "recall_at_k": 0.0,
                          "pr_auc": 1.0 / (1 + v), "roc_auc": 0.5, "k": 10,
                          "n_test": n_test[d], "anomalies_test": 5}
                         for d, row in fp.items() for det, v in row.items()])
    return e1_summary.per_dataset(runs)


MAIN = {"d1": {"A": 1, "B": 2, "C": 3}, "d2": {"A": 1, "B": 3, "C": 2},
        "d3": {"A": 2, "B": 1, "C": 3}}
NONE_REMOVED = {"d1": 0, "d2": 0, "d3": 0}


def test_compare_finds_changes_and_identical_unchanged_sets():
    dedup = {**MAIN, "d1": {"A": 3, "B": 2, "C": 1}}   # dedup flips d1 only
    det, ds, (s,) = e1_dedup.compare(per_ds(MAIN, {"d1": 100, "d2": 50, "d3": 50}),
                                     per_ds(dedup, {"d1": 80, "d2": 50, "d3": 50}),
                                     {"d1": 50, "d2": 0, "d3": 0})
    assert s["datasets_changed_by_dedup"] == ["d1"]
    assert s["datasets_best_changed"] == ["d1"]
    assert s["unchanged_datasets_identical"]
    d1 = ds[ds.dataset == "d1"].iloc[0]
    assert d1.spearman_ranks == pytest.approx(-1.0) and (d1.best_main, d1.best_dedup) == ("A", "C")
    a = det[det.detector == "A"].iloc[0]
    assert a.avg_rank_fp_at_k_main == pytest.approx(4 / 3)
    assert a.delta_avg_rank_fp_at_k == pytest.approx(2 / 3)


def test_same_input_gives_tau_one():
    n = {"d1": 100, "d2": 50, "d3": 50}
    _, ds, (s,) = e1_dedup.compare(per_ds(MAIN, n), per_ds(MAIN, n), NONE_REMOVED)
    assert s["kendall_tau_avg_rank_fp_at_k"] == pytest.approx(1.0)
    assert s["friedman_p_main"] == s["friedman_p_dedup"] and not ds.best_changed.any()


def test_unchanged_dataset_that_differs_is_flagged():
    n = {"d1": 100, "d2": 50, "d3": 50}
    dedup = {**MAIN, "d2": {"A": 9, "B": 3, "C": 2}}   # nothing removed but different results
    _, _, (s,) = e1_dedup.compare(per_ds(MAIN, n), per_ds(dedup, n), NONE_REMOVED)
    assert not s["unchanged_datasets_identical"]


def test_rows_removed_with_same_n_test_counts_as_changed():
    # like letter: 2 rows removed, n_test still rounds to the same value
    n = {"d1": 100, "d2": 50, "d3": 50}
    dedup = {**MAIN, "d2": {"A": 9, "B": 3, "C": 2}}
    _, _, (s,) = e1_dedup.compare(per_ds(MAIN, n), per_ds(dedup, n), {"d1": 0, "d2": 2, "d3": 0})
    assert s["datasets_changed_by_dedup"] == ["d2"]
    assert s["unchanged_datasets_identical"]


def write_run(d, dedup: bool, fp, n):
    d.mkdir(parents=True)
    per_ds(fp, n).to_csv(d / "e1_per_dataset.csv", index=False)
    (d / "e1_runs.meta.json").write_text(json.dumps(
        {"commit": "dedc0de" if dedup else "a1a1a1a", "dirty": False, "cache_key": "k",
         "protocol": {"adbench_dedup": dedup}}))


def test_write_records_both_source_commits_and_refuses_wrong_protocol(tmp_path):
    n = {"d1": 100, "d2": 50, "d3": 50}
    write_run(tmp_path / "main", False, MAIN, n)
    write_run(tmp_path / "dedup", True, MAIN, n)
    e1_dedup.write(tmp_path / "main", tmp_path / "dedup", {"commit": "c0ffee1", "created_utc": "t"},
                   NONE_REMOVED)
    meta = json.loads((tmp_path / "dedup" / "dedup_vs_main.meta.json").read_text())
    assert (meta["commit_main"], meta["commit_dedup"], meta["commit"]) == ("a1a1a1a", "dedc0de", "c0ffee1")
    assert set(pd.read_csv(tmp_path / "dedup" / "dedup_vs_main.csv")["commit"]) == {"c0ffee1"}
    with pytest.raises(ValueError):
        e1_dedup.write(tmp_path / "dedup", tmp_path / "main", {"commit": "x", "created_utc": "t"},
                       NONE_REMOVED)
