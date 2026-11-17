"""IEEE-CIS runner (thesis.e3_ieee): resamples, RQ4 statistics, layouts and run guards.
No test reads a month 2-6 row of the real file: the real layout is tested on a small fake CSV."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis import bootstrap, e3_ieee as r, ieee_features


def small_layout(days_per_month=(2, 3), rows_per_day=4, seed=0):
    rng = np.random.default_rng(seed)
    day, month = [], []
    d = 30
    for m, k in zip((2, 3), days_per_month):
        for _ in range(k):
            day += [d] * rows_per_day
            month += [m] * rows_per_day
            d += 1
    day, month = np.array(day), np.array(month)
    n = len(day)
    return r.Layout(pd.DataFrame({"a": rng.normal(size=10)}), pd.DataFrame({"a": rng.normal(size=n)}),
                    np.zeros(10, int), day, month, day - 30, [2, 3], np.arange(5), np.arange(5, 10),
                    rng.integers(0, 2, n), "synthetic")


def test_day_picks_are_the_freeze_v1_draws():
    lay = small_layout((3, 4))
    days = bootstrap.day_rows(lay.day_test, lay.month_test, lay.months)
    a, b = np.random.default_rng(0), np.random.default_rng(0)
    for _ in range(5):
        rows = bootstrap.draw_days(a, days)
        picks = r.day_picks(b, days)
        again = [np.concatenate([days[i][j] for j in p]) for i, p in enumerate(picks)]
        assert all(np.array_equal(x, y) for x, y in zip(rows, again))


def test_counts_follow_sorted_days_and_months():
    lay = small_layout((2, 3))
    C, month_days = r.rq4_counts(lay, 50, 0)
    assert C.shape == (50, 5)
    assert np.all(C[:, :2].sum(axis=1) == 2) and np.all(C[:, 2:].sum(axis=1) == 3)
    assert np.concatenate(month_days).tolist() == sorted(np.unique(lay.day_test))


def test_rq4_deltas_sum_drawn_days_and_recompute_median_and_oracle():
    D = pd.DataFrame({"A": [1.0, 2, 3], "B": [2.0, 2, 2], "C": [0.0, 5, 7], "G": [1.0, 1, 1]},
                     index=[30, 31, 32])
    C = np.array([[1, 1, 1], [3, 0, 0], [0, 0, 3]])
    out = r.rq4_deltas(D, C, ["A", "B", "C"], [("G", bootstrap.MEDIAN), ("G", bootstrap.ORACLE),
                                               ("G", "B")], 0.02, {("G", "B"): "H4b"}).set_index("comparator")
    # point: sums A 6, B 6, C 12, G 3 -> median 6 (mean 8), oracle 6
    assert out.loc[bootstrap.MEDIAN, "d_fp"] == -3.0 and out.loc[bootstrap.ORACLE, "d_fp"] == -3.0
    # resample 2 (day 30 three times): A 3, B 6, C 0 -> median 3, oracle 0, G 3
    boot_oracle = C @ D.to_numpy()
    assert boot_oracle[1].tolist() == [3, 6, 0, 3] and boot_oracle[2].tolist() == [9, 6, 21, 3]
    lo, hi = bootstrap.percentile_ci(np.array([-3.0, 3.0, -3.0]))          # G - oracle: 3-6, 3-0, 3-6
    assert (out.loc[bootstrap.ORACLE, "ci_low"], out.loc[bootstrap.ORACLE, "ci_high"]) == (lo, hi)
    assert out.loc["B", "role"] == "H4b" and out.loc[bootstrap.MEDIAN, "role"] == "descriptive"


def test_spearman_inf_and_abs_log():
    assert r.abs_log(np.array([1.0, np.e, 0.0])).tolist() == [0.0, 1.0, np.inf]
    x = np.array([0.1, 0.5, 2.0, 0.0])
    assert r.spearman_inf(r.abs_log(x), np.array([2.0, 1.0, 0.5, 9.0])) == pytest.approx(
        np.corrcoef([3, 1.5, 1.5, 4], [3, 2, 1, 4])[0, 1])      # |log 0.5| = |log 2|: a tie
    assert np.isnan(r.spearman_inf(np.ones(3), np.arange(3.0)))


def member_days(lay):
    rows = []
    for m, scale in (("A", 1), ("B", 3)):
        for d in np.unique(lay.day_test):
            n = int((lay.day_test == d).sum())
            rows.append({"member": m, "day": int(d), "n_rows": n, "n_seeds": 1,
                         "alerts": int(scale * (d - 29))})
    return pd.DataFrame(rows)


def test_monthly_r_weights_drawn_days():
    lay = small_layout((2, 3))                     # days 30, 31 (month 2), 32-34 (month 3), 4 rows each
    md = member_days(lay)
    mr = r.monthly_r(md, lay, 0.5).set_index(["member", "month"]).r
    assert mr[("A", 2)] == pytest.approx((1 + 2) / (8 * 0.5))
    assert mr[("B", 3)] == pytest.approx(3 * (3 + 4 + 5) / (12 * 0.5))
    w = np.array([2, 0, 0, 0, 3])
    mw = r.monthly_r(md, lay, 0.5, w).set_index(["member", "month"]).r
    assert mw[("A", 2)] == pytest.approx(2 * 1 / (8 * 0.5)) and mw[("A", 3)] == pytest.approx(3 * 5 / (12 * 0.5))


def test_h4a_point_is_spearman_of_abs_log_over_members_x_months():
    lay = small_layout((2, 3))
    md = member_days(lay)
    pre = pd.DataFrame({"budget": 0.5, "member": ["A", "B"], "r": [1.1, 4.0]})
    C, _ = r.rq4_counts(lay, 30, 0)
    row, pairs = r.h4a(pre, md, lay, 0.5, C, ["A", "B"])
    mr = r.monthly_r(md, lay, 0.5)
    x = r.abs_log(np.where(mr.member == "A", 1.1, 4.0))
    assert row["estimate"] == pytest.approx(r.spearman_inf(x, r.abs_log(mr.r.to_numpy())))
    assert row["n_pairs"] == 4 and set(pairs.columns) >= {"x_abs_log_r_pre", "y_abs_log_r"}
    assert row["verdict"] == ("supported" if row["ci_low"] > 0 else "not supported")


def hyp_inputs(i_low, i_high, ii_low, ii_high, changed_all, changed_bench=False):
    v3 = {"hypotheses": {"H4r": {"supported_if": "both", "i_threshold": {"supported_if": "i"},
                                 "ii_budget": {"supported_if": "ii"}},
                         **{h: {"supported_if": "x", "status": "exploratory"} for h in ("H4a", "H4b", "H4c", "H4d")}}}
    row = lambda role, lo, hi, **k: {"role": role, "ci_low": lo, "ci_high": hi, "method": "m",
                                     "comparator": "c", "ci_entirely_below_0": hi < 0,
                                     "ci_entirely_above_0": lo > 0, **k}
    dl = pd.DataFrame([row("H4r (ii)", ii_low, ii_high, d_fp=0.0), row("H4b", -1.0, 1.0, d_fp=0.0),
                       row("H4c", -1.0, 1.0, d_fp=0.0)])
    thr = pd.DataFrame([row("H4r (i)", i_low, i_high, d_D=-0.1)])
    dec = pd.DataFrame({"ensemble": ["E-all", "E-bench"], "changed": [changed_all, changed_bench]})
    return r.rq4_hypothesis_rows(dl, thr, dec, {"hypothesis": "H4a"}, {"rho": 0.1}, v3).set_index("hypothesis")


def test_h4r_decision_when_a_member_breaks():
    t = hyp_inputs(-0.5, -0.1, -3.0, 2.0, True)
    assert t.loc["H4r (i)", "verdict"] == "supported" and t.loc["H4r (ii)", "verdict"] == "no harm shown"
    assert t.loc["H4r", "verdict"] == "supported" and t.loc["H4r", "status"] == "decided"
    t = hyp_inputs(-0.5, 0.1, -3.0, 2.0, True)                 # (i) CI crosses 0
    assert t.loc["H4r (i)", "verdict"] == "not supported" and t.loc["H4r", "verdict"] == "not supported"
    t = hyp_inputs(-0.5, -0.1, 0.5, 2.0, True)                 # (ii) CI entirely above 0: worse
    assert t.loc["H4r (ii)", "verdict"] == "worse" and t.loc["H4r", "verdict"] == "not supported"
    assert t.loc["H4b", "status"] == "decided" and t.loc["H4b", "verdict"] == "not supported"


def test_no_break_turns_h4r_i_into_a_no_harm_check():
    t = hyp_inputs(-0.5, 0.1, -3.0, 2.0, False)                # crosses 0: no harm shown
    assert t.loc["H4r (i)", "verdict"] == "no harm shown" and t.loc["H4r (i)", "status"].startswith("no member broke")
    assert t.loc["H4r", "verdict"] == "supported"
    t = hyp_inputs(0.1, 0.5, -3.0, 2.0, False)                 # entirely above 0: worse
    assert t.loc["H4r (i)", "verdict"] == "worse" and t.loc["H4r", "verdict"] == "not supported"
    assert t.loc["H4b", "status"].startswith("not testable") and t.loc["H4c", "status"].startswith("descriptive")
    assert "H4e" not in t.index                                # descriptive rows live in rq4_deltas


def test_month_deviation_with_weights_and_zero_alert_months():
    A = np.array([2.0, 0.0, 0.0])                              # day 0 (month a), days 1-2 (month b)
    N = np.array([100.0, 100.0, 100.0])
    M = np.array([[1, 0], [0, 1], [0, 1]], dtype=float)
    q = 0.98                                                    # target 0.02
    # month a: 2 / (100 * 0.02) = 1 -> 0; month b: 0 alerts -> 0.5 / (200 * 0.02) = 0.125
    assert r.month_deviation(A, N, M, q) == pytest.approx(np.mean([0.0, abs(np.log(0.125))]))
    w = np.array([2, 1, 0])                                     # day 0 twice: month a r = 4/(200*0.02) = 1
    assert r.month_deviation(A, N, M, q, w) == pytest.approx(np.mean([0.0, abs(np.log(0.5 / 2))]))


def test_h4r_threshold_uses_training_thresholds_and_the_resamples():
    lay = small_layout((2, 3))
    n = len(lay.day_test)
    tr = np.arange(100.0)[None, :]                              # q 0.9 -> threshold 89.1
    good = np.where(np.arange(n) % 10 == 0, 95.0, 0.0)[None, :]   # 10% of rows alert
    bad = np.full((1, n), 95.0)                                 # every row alerts
    C, _ = r.rq4_counts(lay, 40, 0)
    t = r.h4r_threshold({"g": (tr, good), "b": (tr, bad)}, lay, 0.9, C, [("g", "b")],
                        {("g", "b"): "H4r (i)"}).iloc[0]
    assert t.D_comparator == pytest.approx(abs(np.log(10.0)))      # rate 1 / 0.1
    assert t.d_D < 0 and t.ci_entirely_below_0 and t.role == "H4r (i)"

def test_budget_sums_take_baselines_on_the_month_sums():
    rows = []
    for m, fps in ((2, {"IForest": 3, "PCA": 1, "X": 2}), (3, {"IForest": 1, "PCA": 4, "X": 2})):
        for meth, fp in fps.items():
            rows.append({"method": meth, "budget": 0.02, "seed_index": 0, "month": m,
                         "fp_at_k": fp, "k": 5, "tp_at_k": 5 - fp})
    t = r.budget_sums(pd.DataFrame(rows), ["IForest", "PCA", "X"]).set_index("method")
    assert t.loc["IForest", "fp_sum"] == 4 and t.loc["PCA", "fp_sum"] == 5 and t.loc["X", "k_sum"] == 10
    assert t.loc[bootstrap.ORACLE, "fp_sum"] == 4 and t.loc[bootstrap.ORACLE, "oracle_member"] == "IForest"
    assert t.loc[bootstrap.MEDIAN, "fp_sum"] == 4 and t.loc["PCA", "regret"] == 1


def test_dry_layout_reads_month_1_only_and_no_label(monkeypatch):
    seen = []
    real = pd.read_csv

    def spy(*a, **k):
        seen.append(k)
        return real(*a, **k)
    monkeypatch.setattr(pd, "read_csv", spy)
    names = [f["name"] for f in ieee_features.load_frozen()["features"]]
    if not ieee_features.RAW.exists():
        pytest.skip("IEEE-CIS file not downloaded")
    lay = r.dry_layout(names)
    assert all(ieee_features.LABEL not in (k.get("usecols") or []) for k in seen)
    assert all(k.get("nrows") is not None or k.get("usecols") == [ieee_features.TIME] for k in seen)
    assert lay.labels == "synthetic" and set(lay.day_train) == set(range(21))
    assert set(lay.day_test) == set(range(21, 30)) and lay.months == [2, 3, 4, 5, 6]
    assert lay.day_train[lay.pre_fit].max() < lay.day_train[lay.pre_scored].min()
    assert len(lay.train) + len(lay.test) == 134_339


def test_real_layout_on_a_fake_file(tmp_path, monkeypatch):
    names = ["card4", "TransactionAmt"]
    day = np.repeat(np.arange(0, 60), 3)
    df = pd.DataFrame({"TransactionDT": 86_400 + day * 86_400 + np.tile([0, 10, 20], 60),
                       "isFraud": np.tile([0, 0, 1], 60), "card4": "visa", "TransactionAmt": 1.0})
    path = tmp_path / "fake.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(ieee_features, "RAW", path)
    lay = r.real_layout(names)
    assert len(lay.train) == 90 and len(lay.test) == 90 and set(lay.month_test) == {2}
    assert lay.day_train.max() == 29 and lay.day_test.min() == 30
    assert lay.day_train[lay.pre_fit].max() == 20 and set(lay.day_train[lay.pre_scored]) == set(range(21, 30))
    assert len(lay.pre_fit) == 21 * 3 and len(lay.pre_scored) == 9 * 3
    assert lay.labels == "isFraud" and len(lay.y_test) == len(lay.test)
    assert not any("pre" in f and f.startswith("y") for f in r.Layout.__dataclass_fields__)  # no month-1 label


def test_real_run_refuses_without_the_freeze_v3_tag(monkeypatch):
    called = []
    monkeypatch.setattr(r, "real_layout", lambda *a, **k: called.append(1))
    monkeypatch.setattr(r, "git_state", lambda: {"commit": "x", "dirty": False})
    monkeypatch.setattr(r, "frozen_unchanged", lambda: None)
    real_git = r.git

    def no_v3(*args):
        if args[:3] == ("rev-parse", "--verify", "--quiet") and args[3].endswith("freeze-v3"):
            return type("P", (), {"returncode": 1})()
        return real_git(*args)
    monkeypatch.setattr(r, "git", no_v3)
    with pytest.raises(SystemExit, match="tag freeze-v3 is missing"):
        r.main([])
    assert not called


def test_real_run_refuses_a_second_run(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "git", lambda *a: type("P", (), {"returncode": 0, "stdout": ""})())
    monkeypatch.setattr(r.frozen_v3, "load", lambda for_tag=False: {})
    (tmp_path / "run.meta.json").write_text("{}")
    with pytest.raises(SystemExit, match="run once"):
        r.real_run_allowed(tmp_path)


def test_dry_run_never_writes_into_results(monkeypatch):
    from thesis.paths import TABLES
    monkeypatch.setattr(r, "dry_layout", lambda *a: pytest.fail("must stop before loading"))
    with pytest.raises(SystemExit, match="results"):
        r.main(["--dry-run", "--out", str(TABLES / "ieee_cis" / "x")])


def test_dry_run_options_are_refused_in_the_real_run(monkeypatch):
    monkeypatch.setattr(r, "git_state", lambda: {"commit": "x", "dirty": False})
    with pytest.raises(SystemExit, match="dry run only"):
        r.main(["--resamples", "5"])
