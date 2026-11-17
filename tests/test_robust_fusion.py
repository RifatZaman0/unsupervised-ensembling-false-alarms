"""Robust fusion (thesis.robust_fusion, fusion median) and the proposition bound."""
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
    for members in (["IForest", "KNN", "LOF"], list(rf.detectors.DETECTORS)):
        for q in rf.ULB_Q:
            for r in rf.bound_check(train, test, members, ["LOF"], q).itertuples():
                assert r.alert_without_event == 0
                assert r.fused_alert_rate <= r.event_rate <= r.bound + 1e-12
                assert r.k_healthy_needed == (1 if len(members) == 3 else 2)


def test_threshold_table_uses_the_training_quantile():
    tr = np.arange(11.0)[None, :]                       # q 0.8 -> 8.0
    te = np.array([[8.0, 8.5, 9.0, 1.0]])
    y = np.array([0, 1, 0, 0])
    t = rf.threshold_table({"A": (tr, te)}, y, [0.8]).iloc[0]
    assert t.alert_rate == 0.5 and t.tp == 1 and t.fp == 1
