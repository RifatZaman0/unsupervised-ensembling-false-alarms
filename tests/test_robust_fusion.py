"""Robust fusion (thesis.robust_fusion, fusion median) and the proposition bound."""
import math

import numpy as np
import pandas as pd
import pytest

from thesis import frozen, fusion, lof_diag, data, robust_fusion as rf
from thesis.paths import TABLES


def test_median_is_an_extra_option_and_the_frozen_configs_are_unchanged():
    assert fusion.FUSION_CONFIGS == (("minmax", "mean"), ("minmax", "max"), ("zscore", "mean"),
                                     ("zscore", "max"), ("percentile", "mean"), ("percentile", "max"))
    assert "median" not in fusion.AGGREGATIONS and fusion.PRIMARY == ("minmax", "mean")
    frozen.load()                                   # freeze-v1 validation still passes
    rng = np.random.default_rng(0)
    tr, te = rng.normal(size=(50, 3)), rng.normal(size=(20, 3))
    f_tr, f_te = fusion.fuse(tr, te, "percentile", "median")
    cal = [fusion.Calibrator.fit(tr[:, j], "percentile") for j in range(3)]
    expect = np.median(np.column_stack([c.transform(te[:, j]) for j, c in enumerate(cal)]), axis=1)
    assert np.array_equal(f_te, expect)
    with pytest.raises(ValueError):
        fusion.fuse(tr, te, "minmax", "trimmed")


@pytest.fixture(scope="module")
def ulb():
    cfg = frozen.load()
    return rf.ulb_inputs(cfg), cfg


def test_frozen_primary_outputs_are_unchanged(ulb):
    """The frozen min-max + mean rows of E3 (budget and threshold mode) rebuild exactly."""
    (base, y), cfg = ulb
    budget = pd.read_csv(TABLES / "e3_ulb" / "e3_ulb_budget.csv")
    thr = pd.read_csv(TABLES / "e3_ulb" / "e3_ulb_threshold.csv")
    meth = rf.methods(base.train, base.test, list(base.singles), {
        e: m for e, m in rf.distinct_ensembles(cfg).items()})
    bt = rf.budget_table(meth, y, cfg["budgets"]["ulb"]["values"]).set_index(["method", "budget"])
    tt = rf.threshold_table(meth, y, [0.998, 0.995]).set_index(["method", "q"])
    for e in ("E-bench", "E-bench-5%", "E-all"):
        key = f"{e}:minmax+mean"
        for r in budget[budget.method == key].itertuples():
            assert bt.loc[(key, r.budget), "fp_at_k"] == pytest.approx(r.fp_at_k, abs=1e-9)
        for r in thr[thr.method == key].itertuples():
            assert tt.loc[(key, r.q), "alert_rate"] == pytest.approx(r.alert_rate, abs=1e-12)
    for s in base.singles:
        for r in budget[budget.method == s].itertuples():
            assert bt.loc[(s, r.budget), "fp_at_k"] == pytest.approx(r.fp_at_k, abs=1e-9)


def test_rebuilt_tables_match_the_readme(ulb):
    (base, y), cfg = ulb
    meth = rf.methods(base.train, base.test, list(base.singles), rf.distinct_ensembles(cfg))
    budget = rf.budget_table(meth, y, cfg["budgets"]["ulb"]["values"])
    thr = rf.threshold_table(meth, y)
    assert rf.readme_differences(budget, thr, None) == []
    worse = budget.copy()
    worse.loc[worse.method == "E-all:percentile+median", "fp_at_k"] += 1.0
    assert rf.readme_differences(worse, thr, None)       # a changed value is reported


def test_required_above_and_the_even_m_counterexample():
    assert [rf.required_above(m) for m in (3, 4, 5, 6)] == [2, 2, 3, 3]
    row = np.array([1, 1, 1, 0, 0, 0.0])                   # 3 of 6 above t = 0.4
    assert np.median(row) > 0.4 and (row > 0.4).sum() == 3


def synthetic(broken_scale=50.0, seed=0):
    rng = np.random.default_rng(seed)
    tr = {m: rng.normal(size=(1, 4000)) for m in ("A", "B", "C")}
    te = {m: rng.normal(size=(1, 3000)) for m in ("A", "B", "C")}
    te["C"] = te["C"] * broken_scale + broken_scale     # C breaks: almost every row far above
    return tr, te


def test_bound_holds_for_an_arbitrary_broken_member():
    tr, te = synthetic()
    b = rf.bound_check(tr, te, ["A", "B", "C"], ["C"], 0.99).iloc[0]
    assert b.alert_without_event == 0 and b.k_healthy_needed == 1
    assert b.fused_alert_rate <= b.event_rate <= b.bound + 1e-12
    cal = {m: fusion.Calibrator.fit(tr[m][0], "percentile") for m in "ABC"}
    med_tr = np.median(np.column_stack([cal[m].transform(tr[m][0]) for m in "ABC"]), axis=1)
    assert b.threshold == np.quantile(med_tr, 0.99)       # t from the fused training scores


def test_mean_has_no_such_bound():
    """Counterexample (proposition): every row alerts under the mean, no healthy member is above t."""
    t = 0.9
    C = np.tile([0.86, 0.86, 1.0], (100, 1))           # healthy, healthy, broken (at its maximum)
    healthy_rate = (C[:, :2] > t).mean(axis=0)
    assert healthy_rate.sum() == 0.0
    assert (C.mean(axis=1) > t).mean() == 1.0           # mean: alert rate 1, bound would be 0
    assert (np.median(C, axis=1) > t).mean() == 0.0     # median: no alert


def test_bound_check_refuses_too_many_broken_members():
    tr, te = synthetic()
    with pytest.raises(ValueError):
        rf.bound_check(tr, te, ["A", "B", "C"], ["B", "C"], 0.99)
    tr["D"], te["D"] = tr["A"], te["A"]
    with pytest.raises(ValueError):                      # m = 4 tolerates floor(3/2) = 1
        rf.bound_check(tr, te, ["A", "B", "C", "D"], ["B", "C"], 0.99)


def test_bound_on_ulb_day_2_label_free():
    """The proposition on real development data, without any label: LOF treated as broken."""
    X, t = data.load_ulb_unlabelled()
    sc = lof_diag.cached_day_scores(t, list(rf.detectors.DETECTORS), [0, 1, 2, 3, 4])
    train = {m: a for m, (a, _) in sc.items()}
    test = {m: b for m, (_, b) in sc.items()}
    cases = [(["IForest", "KNN", "LOF"], ["LOF"]), (list(rf.detectors.DETECTORS), ["LOF"]),
             (list(rf.detectors.DETECTORS), ["LOF", "KNN"])]
    for members, broken in cases:
        k = math.ceil(len(members) / 2) - len(broken)       # accepted even-m count
        for q in rf.ULB_Q:
            for r in rf.bound_check(train, test, members, broken, q).itertuples():
                assert r.alert_without_event == 0
                assert r.fused_alert_rate <= r.event_rate <= r.bound + 1e-12
                assert r.k_healthy_needed == k
    assert [math.ceil(m / 2) - b for m, b in ((3, 1), (6, 1), (6, 2))] == [1, 2, 1]


def test_threshold_table_uses_the_training_quantile():
    tr = np.arange(11.0)[None, :]                       # q 0.8 -> 8.0
    te = np.array([[8.0, 8.5, 9.0, 1.0]])
    y = np.array([0, 1, 0, 0])
    t = rf.threshold_table({"A": (tr, te)}, y, [0.8]).iloc[0]
    assert t.alert_rate == 0.5 and t.tp == 1 and t.fp == 1


def test_triples_are_all_20_three_member_ensembles():
    assert len(rf.TRIPLES) == 20 and len(set(rf.TRIPLES)) == 20
    assert ("IForest", "PCA", "KNN") in rf.TRIPLES and ("IForest", "KNN", "LOF") in rf.TRIPLES
    assert rf.ROBUST == ("percentile", "median") and rf.FROZEN == fusion.PRIMARY


def test_fp_rows_and_sorted_scores_equal_tp_at_k():
    rng = np.random.default_rng(1)
    n = 3000
    S = np.round(rng.normal(size=(6, n)), 1)            # many ties
    y = (rng.random(n) < 0.05).astype(int)
    blk = rng.integers(0, 24, n)
    k = 60
    assert np.allclose(rf.fp_rows(S, y, k), [k - rf.metrics.tp_at_k(r, y, k) for r in S], atol=0)
    ss = rf.SortedScores(S, y, blk, prefix=200)          # short prefix: the fallback is exercised
    for _ in range(5):
        p = rng.integers(0, 24, 24)
        w = np.bincount(p, minlength=24)
        idx = np.concatenate([np.flatnonzero(blk == b) for b in p])
        kk = rf.metrics.budget_k(0.02, len(idx))
        expect = [kk - rf.metrics.tp_at_k(r[idx], y[idx], kk) for r in S]
        assert np.array_equal(ss.fp(w, kk), np.array(expect))


def test_deviation_zero_alert_rule_and_weights():
    A, N = np.array([2.0, 0.0, 0.0]), np.array([100.0, 100.0, 100.0])
    M = np.array([[1, 0], [0, 1], [0, 1]], dtype=float)
    assert rf.deviation(A, N, M, 0.98) == pytest.approx(np.mean([0.0, abs(np.log(0.5 / 4))]))
    assert rf.deviation(A, N, M, 0.98, [2, 1, 0]) == pytest.approx(np.mean([0.0, abs(np.log(0.5 / 2))]))
    assert rf.ZERO_ALERTS == 0.5


def test_cost_ratio_and_margin():
    fr = np.array([[10.0, 20.0], [10.0, 20.0]])          # blocks x triples: totals 20, 40
    fz = np.array([[10.0, 30.0], [10.0, 30.0]])          # totals 20, 60
    assert rf.cost_ratio(fr, fz) == pytest.approx(60 / 80 - 1)   # ratio of the mean totals (not of ratios)
    C = np.array([[2, 0], [1, 1]])
    assert np.allclose(rf.cost_ratio(fr, fz, C), [60 / 80 - 1, 60 / 80 - 1])
    assert rf.COST_MARGIN == 0.05 and rf.cost_passes(0.05) and not rf.cost_passes(0.0500001)


def test_triple_deviations_pair_the_two_fusions():
    rng = np.random.default_rng(0)
    names = ["A", "B", "C"]
    tr = {m: rng.normal(size=(1, 2000)) for m in names}
    te = {m: rng.normal(size=(1, 1200)) for m in names}
    te["C"] = te["C"] + 6.0                             # C breaks on the new data
    block = np.repeat(np.arange(4), 300)
    M = np.ones((4, 1))
    C = np.array([[1, 1, 1, 1], [4, 0, 0, 0]])
    tab, boot = rf.triple_deviations(tr, te, block, M, 0.99, C, triples=[tuple(names)])
    r = tab.iloc[0]
    assert r.dD == pytest.approx(r.D_robust - r.D_frozen) and r.dD < 0   # median tolerates C
    assert boot.shape == (2, 1) and boot[0, 0] == pytest.approx(r.dD)


def test_sorted_scores_fall_back_when_a_tie_group_runs_past_the_prefix():
    S = np.concatenate([np.full(300, 5.0), np.linspace(4.0, 0.0, 700)])[None, :]
    y = np.zeros(1000, dtype=int)
    y[::7] = 1
    blk = np.arange(1000) % 10
    ss = rf.SortedScores(S, y, blk, prefix=200)          # the 300 tied top scores run past 200
    w = np.ones(10, dtype=int)
    expect = 150 - rf.metrics.tp_at_k(S[0], y, 150)        # K inside the prefix, ties beyond it
    assert ss.fp(w, 150)[0] == expect
