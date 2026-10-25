"""E1 summary table: seeds averaged within a dataset, each dataset counts once, ranks and ties
as documented, commit recorded in every output."""
import pandas as pd
import pytest

from thesis import e1_summary

META = {"commit": "abc1234def", "cache_key": "21ba8c24b9", "created_utc": "2026-10-25T00:00:00",
        "protocol": {"zscore_inputs": True, "k_rule": "ceil", "pca_drop_zero_variance": True,
                     "adbench_dedup": False}}


def runs():
    """2 datasets x 2 seeds x 3 detectors, one budget. On d1, A and B tie on FP@K."""
    fp = {("d1", "A"): (10, 12), ("d1", "B"): (11, 11), ("d1", "C"): (20, 22),
          ("d2", "A"): (30, 30), ("d2", "B"): (5, 7), ("d2", "C"): (40, 44)}
    rows = []
    for (ds, det), fps in fp.items():
        for seed, f in enumerate(fps):
            k = 50
            rows.append({"dataset": ds, "seed": seed, "detector": det, "n_test": 1000,
                         "anomalies_test": 60, "runtime_s": 0.1, "budget": 0.05, "k": k,
                         "tp_at_k": k - f, "fp_at_k": f, "precision_at_k": (k - f) / k,
                         "recall_at_k": (k - f) / 60, "pr_auc": 1 - f / 100,
                         "roc_auc": 1 - f / 200})
    return pd.DataFrame(rows)


def test_per_dataset_means_and_tied_ranks():
    pd_ = e1_summary.per_dataset(runs()).set_index(["dataset", "detector"])
    assert pd_.loc[("d1", "A"), "fp_at_k"] == 11 and pd_.loc[("d1", "A"), "n_seeds"] == 2
    assert pd_.loc[("d1", "A"), "rank_fp_at_k"] == pd_.loc[("d1", "B"), "rank_fp_at_k"] == 1.5
    assert pd_.loc[("d1", "A"), "best_fp_at_k"] and pd_.loc[("d1", "B"), "best_fp_at_k"]
    assert pd_.loc[("d2", "C"), "rank_pr_auc"] == 3  # lowest PR-AUC ranks last


def test_summary_averages_over_datasets():
    s = e1_summary.summary(e1_summary.per_dataset(runs())).set_index("detector")
    assert s.loc["B", "mean_fp_at_k"] == pytest.approx((11 + 6) / 2)
    assert s.loc["B", "avg_rank_fp_at_k"] == pytest.approx((1.5 + 1) / 2)
    assert s.loc["A", "avg_rank_fp_at_k"] == pytest.approx((1.5 + 2) / 2)
    assert s.loc["B", "n_best_fp_at_k"] == 2 and s.loc["C", "n_best_fp_at_k"] == 0
    assert list(s.index) == ["B", "A", "C"]  # sorted by average FP@K rank
    assert (s["n_datasets"] == 2).all()


def test_write_records_the_commit_everywhere(tmp_path):
    runs().to_csv(tmp_path / "e1_runs.csv", index=False)
    e1_summary.write(tmp_path, META)
    for f in ("e1_summary.csv", "e1_per_dataset.csv"):
        assert (pd.read_csv(tmp_path / f)["commit"] == META["commit"]).all()
    md = (tmp_path / "e1_summary.md").read_text(encoding="utf-8")
    assert META["commit"] in md and "Budget 5% of test rows" in md and "| B |" in md
