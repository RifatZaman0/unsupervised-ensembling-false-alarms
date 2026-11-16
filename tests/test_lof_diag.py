"""LOF diagnosis and stress test (thesis.lof_diag): slices, leakage, the r statistic."""
import numpy as np
import pytest

from thesis import data, detectors, lof_diag


def times():
    # 10 rows per 6 h over two days
    return np.repeat(np.arange(8) * 21_600.0, 10) + np.tile(np.arange(10.0), 8)


def test_l2b_fit_slice_is_strictly_earlier_and_inside_day_1():
    t = times()
    fit, late = lof_diag.control_slices(t, "L2b")
    assert t[fit].max() < lof_diag.HALF_DAY_SECONDS <= t[late].min()
    assert t[late].max() < data.ULB_DAY_SECONDS
    assert len(fit) == len(late) == 20 and not set(fit) & set(late)


def test_other_control_slices():
    t = times()
    day1 = set(np.flatnonzero(t < data.ULB_DAY_SECONDS))
    a_fit, a_late = lof_diag.control_slices(t, "L2a")
    assert set(a_fit) | set(a_late) == day1 and not set(a_fit) & set(a_late)
    assert len(a_fit) == len(a_late) == 20
    c_fit, c_late = lof_diag.control_slices(t, "L2c")
    d_fit, d_late = lof_diag.control_slices(t, "L2d")
    assert set(c_late) == set(d_fit) == day1 and set(c_fit) == set(d_late)
    with pytest.raises(ValueError):
        lof_diag.control_slices(t, "L2x")


def test_scaler_and_detectors_see_the_fit_slice_only():
    rng = np.random.default_rng(0)
    X_fit = rng.normal(5.0, 2.0, size=(300, 4))
    X_a, X_b = rng.normal(5.0, 2.0, size=(200, 4)), rng.normal(50.0, 9.0, size=(200, 4))
    seen = []

    def spy(det, Z_fit, Z_score, seed):
        seen.append((Z_fit.copy(), Z_score.copy()))
        return detectors.score(det, Z_fit, Z_score, seed)

    a = lof_diag.slice_scores(X_fit, X_a, ["PCA", "KNN", "IForest"], [0, 1], spy)
    b = lof_diag.slice_scores(X_fit, X_b, ["PCA", "KNN", "IForest"], [0, 1], spy)
    Z_fit = seen[0][0]
    assert np.allclose(Z_fit.mean(axis=0), 0) and np.allclose(Z_fit.std(axis=0), 1)
    # the scored slice is scaled with fit-slice statistics, not its own
    assert abs(seen[-1][1].mean()) > 10
    for det in a:
        assert np.array_equal(a[det][0], b[det][0])         # fit-slice scores do not move
    assert a["IForest"][0].shape == (2, 300) and a["PCA"][1].shape == (1, 200)


def test_r_is_the_later_alert_rate_over_the_target_with_fit_slice_thresholds():
    s_fit = np.arange(1000.0)[None, :]                      # q 0.9 threshold = 899.1
    s_late = np.concatenate([np.full(700, 10.0), np.full(300, 950.0)])[None, :]
    tab = lof_diag.stress_table({"A": (s_fit, s_late), "B": (s_fit, s_fit)}, [0.9], "L2b")
    a, b = tab[tab.member == "A"].iloc[0], tab[tab.member == "B"].iloc[0]
    assert a.alert_rate == pytest.approx(0.3) and a.r == pytest.approx(3.0)
    assert b.r == pytest.approx(1.0) and a.fit_alert_rate == pytest.approx(0.1)
    assert a.r_over_median == pytest.approx(3.0 / 2.0)      # median of r = (1, 3)
    assert (a.n_fit, a.n_scored, a.n_seeds) == (1000, 1000, 1)


def test_r_per_seed_uses_each_seeds_own_threshold():
    s_fit = np.vstack([np.arange(100.0), np.arange(100.0) * 10])    # q 0.5: 49.5 and 495
    s_late = np.vstack([np.full(100, 60.0), np.full(100, 60.0)])    # seed 0 all above, seed 1 none
    row = lof_diag.stress_table({"A": (s_fit, s_late)}, [0.5], "L2b").iloc[0]
    assert (row.r_min, row.r_max, row.r) == pytest.approx((0.0, 2.0, 1.0))


def test_agreement_counts_shared_top_rows():
    s = np.arange(50.0)
    same = lof_diag.agreement(s + 1, s + 1, 5)
    assert same["top_k_shared"] == 5 and same["max_abs_diff"] == 0.0
    assert same["spearman"] == pytest.approx(1.0) and same["n"] == 50
    rev = lof_diag.agreement(s[::-1] + 1, s + 1, 5)
    assert rev["top_k_shared"] == 0 and rev["spearman"] == pytest.approx(-1.0)


def test_recomputed_lof_equals_the_pipeline_lof():
    rng = np.random.default_rng(1)
    X1, X2 = rng.normal(size=(400, 3)), rng.normal(0.5, 1.5, size=(150, 3))
    s1, s2 = lof_diag.recompute_lof(X1, X2, 20)
    sc = data.Standardizer.fit(X1)
    p1, p2 = detectors.score("LOF", sc.transform(X1), sc.transform(X2))
    assert np.allclose(s1, p1, rtol=1e-12) and np.allclose(s2, p2, rtol=1e-12)
    far = lof_diag.recompute_far(sc.transform(X1), sc.transform(X2[:5]) + 50.0, 20)
    assert far.min() > s1.max()                             # higher = more anomalous
