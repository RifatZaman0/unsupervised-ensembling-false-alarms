"""Run-time guard (thesis.guard) and the ULB comparison with the singles (thesis.guard_ulb)."""
import numpy as np
import pandas as pd
import pytest

from thesis import bootstrap, fusion, guard as g, guard_ulb as gu


def test_target_and_quantile_are_exact():
    assert g.target(0.02) == g.target("0.02") and g.target(0.02).denominator == 50
    assert g.quantile(0.02) == 0.98 and g.quantile(0.002) == 0.998
    with pytest.raises(ValueError):
        g.target(1.5)


def test_r_equal_to_c_passes_and_is_exact():
    assert g.passes(6, 100, 1, 0.02, 3) and not g.passes(7, 100, 1, 0.02, 3)
    # 9 / (60 * 0.03) = 5 exactly; in floats 9 / (60 * 0.03) = 5.000000000000001
    assert 9 / (60 * 0.03) > 5
    assert g.passes(9, 60, 1, 0.03, 5) and not g.passes(10, 60, 1, 0.03, 5)
    assert g.passes(0, 10, 1, 0.02, 3)
    with pytest.raises(ValueError):
        g.passes(1, 0, 1, 0.02, 3)


def test_seeds_are_summed_as_the_seed_mean():
    # 5 seeds x 100 rows: 30 alerts in total = seed-mean rate 0.06 = 3 x 0.02 -> passes
    assert g.passes(30, 100, 5, 0.02, 3) and not g.passes(31, 100, 5, 0.02, 3)
    assert g.ratio(30, 100, 5, 0.02) == pytest.approx(3.0)


def test_threshold_is_the_linear_quantile_and_alerts_are_strict():
    s_tr = np.arange(11.0)[None, :]               # q 0.8 -> 8.0 exactly; q 0.85 -> 8.5
    thr = g.thresholds(s_tr, 0.2)
    assert thr.tolist() == [8.0]
    assert g.alert_count(np.array([[8.0, 8.2, 9.0]]), thr) == 2      # 8.0 is no alert
    assert g.thresholds(s_tr, 0.15).tolist() == [8.5]                # "lower" would give 8


def test_kept_members_falls_back_below_two():
    assert g.kept_members(["IForest", "KNN", "LOF"], {"IForest", "KNN"}) == (["IForest", "KNN"], False)
    assert g.kept_members(["IForest", "KNN", "LOF"], {"KNN"}) == (["IForest", "KNN", "LOF"], True)
    assert g.kept_members(["IForest", "KNN", "LOF"], set()) == (["IForest", "KNN", "LOF"], True)
    assert g.kept_members(["A", "B"], {"A"}, min_members=1) == (["A"], False)


def two_day_case():
    """A, B, C stable; C overshoots on day 1 only. A has two seeds."""
    rng = np.random.default_rng(0)
    tr = {"A": rng.normal(size=(2, 500)), "B": rng.normal(size=(1, 500)), "C": rng.normal(size=(1, 500))}
    te = {m: rng.normal(size=(len(s), 400)) for m, s in tr.items()}
    day = np.repeat([0, 1], 200)
    te["C"][:, 200:] += 3.0                       # day 1: C far above its threshold
    return tr, te, day


def test_guard_drops_a_member_only_on_its_bad_day():
    tr, te, day = two_day_case()
    md, scores, dec = g.guarded(tr, te, day, {"E": ["A", "B", "C"]}, 0.1, c=3)
    bad = md[(md.member == "C")].set_index("day")
    assert bad.loc[0, "passes"] and not bad.loc[1, "passes"]
    assert md[md.member != "C"].passes.all()
    d = dec.set_index("day")
    assert d.loc[0, "kept"] == "A+B+C" and not d.loc[0, "changed"]
    assert d.loc[1, "kept"] == "A+B" and d.loc[1, "dropped"] == "C" and d.loc[1, "changed"]
    # day 1 scores = fusion of A and B only, calibrators from their training scores
    cols = np.flatnonzero(day == 1)
    _, ab = g.fuse_members(tr, te, ["A", "B"], cols)
    assert np.array_equal(scores["E"][:, cols], ab) and scores["E"].shape == (2, 400)
    _, abc = g.fuse_members(tr, te, ["A", "B", "C"], np.flatnonzero(day == 0))
    assert np.array_equal(scores["E"][:, day == 0], abc)


def test_guard_falls_back_to_the_static_ensemble():
    tr, te, day = two_day_case()
    te["B"][:, 200:] += 3.0                       # day 1: B and C break, only A passes
    md, scores, dec = g.guarded(tr, te, day, {"E": ["A", "B", "C"]}, 0.1, c=3)
    row = dec.set_index("day").loc[1]
    assert row.fallback and row.kept == "A+B+C" and not row.changed
    _, abc = g.fuse_members(tr, te, ["A", "B", "C"], np.flatnonzero(day == 1))
    assert np.array_equal(scores["E"][:, day == 1], abc)


def test_guard_threshold_comes_from_training_scores_only():
    tr, te, day = two_day_case()
    md, _, _ = g.guarded(tr, te, day, {"E": ["A", "B", "C"]}, 0.1, c=3)
    thr = np.quantile(tr["C"], 0.9, axis=1, method="linear")
    expect = int((te["C"][:, day == 1] > thr[:, None]).sum())
    assert md.set_index(["member", "day"]).loc[("C", 1), "alerts"] == expect


def test_fuse_members_is_the_frozen_primary_fusion():
    tr, te, _ = two_day_case()
    f_tr, f_te = g.fuse_members(tr, te, ["A", "B"])
    for i in range(2):
        a, b = fusion.fuse(np.column_stack([tr["A"][i], tr["B"][0]]),
                           np.column_stack([te["A"][i], te["B"][0]]), "minmax", "mean")
        assert np.array_equal(f_tr[i], a) and np.array_equal(f_te[i], b)


def test_daily_fp_uses_a_k_per_day():
    s = np.array([[5, 4, 3, 2, 1, 9, 8, 7, 6, 5.5]], dtype=float)
    y = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0])
    day = np.repeat([30, 31], 5)
    t = g.daily_fp(s, y, day, 0.4).set_index("day")
    assert t.loc[30, "k"] == 2 and t.loc[30, "fp_at_k"] == 1.0    # top 2 of day 30: 5 (0), 4 (1)
    assert t.loc[31, "k"] == 2 and t.loc[31, "fp_at_k"] == 0.0    # top 2 of day 31: 9, 8


def test_hours_of_day_2():
    assert gu.hours(np.array([86400.0, 89999.9, 90000.0, 172799.0])).tolist() == [0, 0, 1, 23]
    with pytest.raises(ValueError):
        gu.hours(np.array([86399.0]))


def test_chosen_single_is_the_lowest_fp_on_day_1_with_pool_order_ties():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    good = np.array([[9, 1, 2, 3, 4, 5, 6, 7, 8, 0]], dtype=float)  # fraud on top
    bad = good[:, ::-1].copy()
    sl = {"A": (None, bad), "B": (None, good), "C": (None, good.copy())}
    best, tab = gu.chosen_single(sl, y, 0.1, ["A", "B", "C"])
    assert best == "B" and tab.set_index("member").loc["B", "chosen"]
    assert tab["chosen"].sum() == 1


def test_comparisons_cover_both_versions_and_guarded_vs_static():
    p = gu.comparisons(["E-bench", "E-all"], "HBOS")
    assert ("guarded:E-all", bootstrap.MEDIAN) in p and ("static:E-all", "HBOS") in p
    assert ("guarded:E-bench", "static:E-bench") in p and len(p) == 2 * (2 * 3 + 1)
