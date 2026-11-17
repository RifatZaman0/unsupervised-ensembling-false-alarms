"""LaTeX exports of the stress tables (thesis.export_stress): values come from the CSVs."""
import pandas as pd
import pytest

from thesis import export_stress as x


def test_l2_cells_are_the_csv_values_rounded():
    rows = []
    for run in x.RUN_LABELS:
        for m, r998, r995 in (("KNN", 1.006315, 1.117215), ("LOF", 41.100978, 34.070604)):
            rows += [{"run": run, "member": m, "q": 0.998, "r": r998, "k_distance_ratio": 1.444},
                     {"run": run, "member": m, "q": 0.995, "r": r995, "k_distance_ratio": 1.444}]
    t = x.l2_table(pd.DataFrame(rows))
    assert list(t.Member) == ["KNN", "LOF", "20th-NN distance ratio"]
    assert t.iloc[1, 4] == "41.10 (34.07)" and t.iloc[0, 1] == "1.01 (1.12)"
    assert t.iloc[2, 1] == "1.44" and t.shape[1] == 5


def test_feature_table_keeps_only_kept_columns():
    df = pd.DataFrame({"column": ["a", "b"], "type": ["numerical", "categorical"],
                       "transform": ["median imputation", "frequency encoding"],
                       "missing_share_month1": [0.146875, 0.7], "kept": [True, False]})
    t = x.feature_table(df)
    assert list(t.Feature) == ["a"] and t.iloc[0, 3] == "14.69"


def test_compact_rules():
    g = pd.DataFrame({"side": ["one-sided", "one-sided", "two-sided"],
                      "basis": ["relative", "relative", "relative"], "c": [3, 2, 3]})
    assert x.compact_rules(g) == "1-sided rel. c = 2, 3; 2-sided rel. c = 3"
    assert x.compact_rules(pd.DataFrame({"side": ["one-sided"] * 12, "basis": ["absolute"] * 12,
                                         "c": list(range(12))})) == "all 12"


def test_ulb_summary_takes_the_primary_budget_delta_of_the_right_pair():
    dec = pd.DataFrame({"mechanism": ["guard", "guard"], "rule": ["r1", "r1"],
                        "side": ["one-sided"] * 2, "basis": ["absolute"] * 2, "c": [2, 2],
                        "ensemble": ["E-bench-5%", "E-all"], "kept": ["IForest+KNN", "IForest+PCA+ECOD+HBOS+KNN"],
                        "changed": [True, True]})
    dl = pd.DataFrame({"budget": [0.002, 0.005, 0.002],
                       "variant": ["{IForest+KNN}", "{IForest+KNN}", "{IForest+PCA+ECOD+HBOS+KNN}"],
                       "static": ["{IForest+KNN+LOF}", "{IForest+KNN+LOF}", "{IForest+PCA+ECOD+HBOS+KNN+LOF}"],
                       "d_fp": [-32.0, -99.0, -1.2], "d_fp_ci_low": [-41.6, -99.0, -5.4],
                       "d_fp_ci_high": [-23.0, -99.0, 2.0]})
    t = x.ulb_summary_table(dec, dl).set_index("Ensemble")
    assert t.loc["E-bench-5%", "dFP@K at 0.2% [95% CI]"] == "-32.0 [-41.6, -23.0]"
    assert t.loc["E-all", "dFP@K at 0.2% [95% CI]"] == "-1.2 [-5.4, +2.0]"
    assert t.loc["E-bench-5%", "Kept"] == "IForest, KNN"


def test_one_commit_refuses_mixed_tables():
    with pytest.raises(SystemExit):
        x.one_commit(pd.DataFrame({"commit": ["a", "b"]}))
