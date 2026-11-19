"""Threshold fragility index (thesis.fragility)."""
import numpy as np
import pandas as pd
import pytest

from thesis import fragility as f


def test_index_counts_scores_above_the_moved_threshold():
    s = np.arange(101.0)[None, :]                 # q 0.9 -> t 90, median 50, cut 90 - 4 = 86
    assert f.index(s, 0.9).tolist() == pytest.approx([14 / 101 / 0.1])
    # a score exactly at the cut is not counted (strict, as the alert rule)
    s2 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0]])  # q 0.8 -> t 8, median 5, cut 7.7
    assert f.index(s2, 0.8, delta=0.1)[0] == pytest.approx(3 / 11 / 0.2)
    assert f.index(s2, 0.8, delta=1.0)[0] == pytest.approx(5 / 11 / 0.2)  # cut = median 5
    # skewed: median 5 differs from the mean 13.2; q 0.85 is 8.5 (linear), 8 (lower)
    s3 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100.0]])
    assert f.index(s3, 0.8)[0] == pytest.approx(3 / 11 / 0.2)      # cut 7.7 (mean: 8.52 -> 2)
    assert f.index(s3, 0.85)[0] == pytest.approx(2 / 11 / 0.15)    # cut 8.15 (lower: 7.7 -> 3)


def test_index_is_affine_invariant_and_per_seed():
    rng = np.random.default_rng(0)
    s = rng.gamma(2.0, size=(3, 2000))
    base = f.index(s, 0.98)
    assert np.allclose(f.index(3.5 * s - 40.0, 0.98), base)
    assert len(base) == 3 and not np.allclose(base, base[0])
    assert np.all(base >= 1 - 1e-9)


def test_compressed_threshold_is_more_fragile():
    rng = np.random.default_rng(1)
    heavy = rng.pareto(1.5, size=(1, 20000))       # threshold far in a long tail
    narrow = rng.uniform(1, 2, size=(1, 20000))    # threshold close to the bulk
    assert f.index(narrow, 0.99)[0] > f.index(heavy, 0.99)[0]


def test_table_ranks_within_run_and_q():
    tr = {"A": np.arange(101.0)[None, :], "B": np.linspace(0, 1, 101)[None, :] ** 4}
    t = f.table(tr, [0.9, 0.95], "x")
    assert set(t.q) == {0.9, 0.95} and len(t) == 4
    for q, g in t.groupby("q"):
        g = g.set_index("member")
        top = g["fragility"].idxmax()
        assert g.loc[top, "rank"] == 1.0 and sorted(g["rank"]) == [1.0, 2.0]
    assert (t.delta == f.DELTA).all() and f.DELTA == 0.1


def test_quantiles_are_one_minus_budget_plus_the_frozen_ones():
    cfg = {"budgets": {"ieee_cis": {"values": [0.005, 0.01, 0.02]}},
           "thresholds": {"ieee_cis": {"training_score_quantiles": [0.99, 0.97]}}}
    assert f.quantiles_for(cfg, "ieee_cis") == [0.995, 0.99, 0.98, 0.97]


def test_spearman_against_log_r():
    frag = pd.DataFrame({"member": ["A", "B", "C"], "q": 0.9, "fragility": [1.0, 2.0, 3.0]})
    over = pd.DataFrame({"member": ["C", "A", "B"], "q": 0.9, "r": [9.0, 0.5, 1.0]})
    assert f.spearman_against(frag, over).spearman_F_vs_log_r.iloc[0] == pytest.approx(1.0)
