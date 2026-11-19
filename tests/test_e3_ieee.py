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


def hyp_inputs(lo, hi, cost_hi, changed_all, changed_bench=False):
    v3 = {"hypotheses": {"H4r": {"supported_if": "main"}, "H4r_cost": {"passes_if": "cost"},
                         **{h: {"supported_if": "x", "status": "exploratory"} for h in ("H4a", "H4b", "H4c", "H4d")}}}
    row = lambda role, lo, hi, **k: {"role": role, "ci_low": lo, "ci_high": hi, "method": "m",
                                     "comparator": "c", "ci_entirely_below_0": hi < 0,
                                     "ci_entirely_above_0": lo > 0, **k}
    dl = pd.DataFrame([row("H4b", -1.0, 1.0, d_fp=0.0), row("H4c", -1.0, 1.0, d_fp=0.0)])
    main = {"estimate": -0.2, "ci_low": lo, "ci_high": hi, "ci_entirely_below_0": hi < 0,
            "ci_entirely_above_0": lo > 0}
    cost = {"estimate": 0.01, "ci_low": -0.02, "ci_high": cost_hi, "passes": cost_hi <= 0.05}
    dec = pd.DataFrame({"ensemble": ["E-all", "E-bench"], "changed": [changed_all, changed_bench]})
    return r.rq4_hypothesis_rows(dl, main, cost, dec, {"hypothesis": "H4a"}, {"rho": 0.1}, v3).set_index("hypothesis")


def test_h4r_is_decided_by_the_triple_mean_and_the_cost_check_is_separate():
    t = hyp_inputs(-0.5, -0.1, 0.03, True)
    assert t.loc["H4r", "verdict"] == "supported" and t.loc["H4r", "status"] == "decided"
    assert t.loc["H4r-cost", "verdict"] == "passes"
    t = hyp_inputs(-0.5, 0.1, 0.03, True)                       # CI crosses 0
    assert t.loc["H4r", "verdict"] == "not supported"
    t = hyp_inputs(-0.5, -0.1, 0.08, True)                      # cost fails, H4r still supported
    assert t.loc["H4r", "verdict"] == "supported" and t.loc["H4r-cost", "verdict"] == "fails"
    assert t.loc["H4b", "status"] == "decided" and t.loc["H4b", "verdict"] == "not supported"


def test_no_break_turns_h4r_into_a_no_harm_check_and_keeps_the_cost_check():
    t = hyp_inputs(-0.5, 0.1, 0.03, False)                      # crosses 0: no harm shown
    assert t.loc["H4r", "verdict"] == "no harm shown" and t.loc["H4r", "status"].startswith("no member broke")
    t = hyp_inputs(0.1, 0.5, 0.08, False)                       # entirely above 0: worse
    assert t.loc["H4r", "verdict"] == "worse" and t.loc["H4r-cost", "verdict"] == "fails"
    assert t.loc["H4r-cost", "status"] == "decided"
    assert t.loc["H4b", "status"].startswith("not testable") and t.loc["H4c", "status"].startswith("descriptive")
    assert "H4e" not in t.index                                 # descriptive rows live in rq4_deltas


def test_h4r_triples_pairs_the_fusions_and_averages_the_triples(monkeypatch):
    lay = small_layout((2, 3), rows_per_day=40)
    rng = np.random.default_rng(3)
    n = len(lay.day_test)
    from thesis import detectors, robust_fusion
    names = list(detectors.DETECTORS)
    train = {m: rng.normal(size=(1, 400)) for m in names}
    test = {m: rng.normal(size=(1, n)) for m in names}
    test["LOF"] = test["LOF"] + 5.0                             # one member breaks
    fitted = r.Fitted(train, test, tuple(names), {})
    y = (rng.random(n) < 0.05).astype(int)
    C, _ = r.rq4_counts(lay, 30, 0)
    tab, main, cost = r.h4r_triples(fitted, lay, 0.9, C, 0.1, y)
    assert len(tab) == 20 and set(tab.triple) == {"+".join(t) for t in robust_fusion.TRIPLES}
    assert main["estimate"] == pytest.approx(tab.dD.mean()) and np.allclose(tab.dD, tab.D_robust - tab.D_frozen)
    assert tab[tab.triple.str.contains("LOF")].dD.mean() < tab[~tab.triple.str.contains("LOF")].dD.mean()
    assert cost["estimate"] == pytest.approx(tab.fp_total_robust.mean() / tab.fp_total_frozen.mean() - 1)
    assert cost["passes"] == (cost["ci_high"] <= 0.05)


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
                       "isFraud": (np.arange(180) % 7 == 0).astype(int),    # month 1 and 2 differ
                       "card4": "visa", "TransactionAmt": 1.0})
    path = tmp_path / "fake.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(ieee_features, "RAW", path)
    seen, real = [], pd.read_csv

    def spy(*a, **k):
        seen.append(k)
        return real(*a, **k)
    monkeypatch.setattr(pd, "read_csv", spy)
    lay = r.real_layout(names)
    assert all(k.get("usecols") is not None for k in seen)             # never all columns
    label_reads = [k for k in seen if "isFraud" in k["usecols"]]
    assert len(label_reads) == 1 and list(label_reads[0]["skiprows"]) == list(range(1, 91))
    assert np.array_equal(lay.y_test, df["isFraud"].to_numpy()[90:])   # the test rows' own labels
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


def test_h4b_and_h4c_status_follow_the_frozen_no_break_rule():
    t = hyp_inputs(-0.5, 0.1, 0.03, False)                      # no member broke at all
    assert t.loc["H4b", "verdict"] == "not testable" and t.loc["H4b", "status"].startswith("not testable")
    assert t.loc["H4c", "status"].startswith("descriptive: no member broke")
    t = hyp_inputs(-0.5, 0.1, 0.03, True, changed_bench=False)  # a member broke, E-bench unchanged
    assert t.loc["H4b", "status"] == "decided" and t.loc["H4b", "verdict"] == "not supported"
    assert t.loc["H4c", "status"].startswith("decided; the guard changed no E-bench member")
    assert "no member broke" not in t.loc["H4c", "status"]
    t = hyp_inputs(-0.5, 0.1, 0.03, True, changed_bench=True)
    assert t.loc["H4b", "status"] == t.loc["H4c", "status"] == "decided"


def _git(rc_for=None):
    def fake(*args):
        rc = 1 if rc_for and args[0] == "diff" and args[-1] == rc_for else 0
        return type("P", (), {"returncode": rc, "stdout": ""})()
    return fake


def test_real_run_refuses_a_frozen_file_that_differs_from_its_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(r.frozen_v3, "load", lambda for_tag=False: {})
    for tag, path in r.TAGS.items():
        monkeypatch.setattr(r, "git", _git(path))
        with pytest.raises(SystemExit, match=f"{path} differs from {tag}"):
            r.real_run_allowed(tmp_path)


def test_real_run_allowed_applies_the_tag_time_check(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "git", _git())

    def load(for_tag=False):
        if for_tag:
            raise ValueError("draft_status must be removed before the tag")
        return {}
    monkeypatch.setattr(r.frozen_v3, "load", load)
    with pytest.raises(ValueError, match="draft_status"):
        r.real_run_allowed(tmp_path)


def test_real_run_refuses_a_dirty_tree(monkeypatch):
    monkeypatch.setattr(r, "git_state", lambda: {"commit": "x", "dirty": True})
    monkeypatch.setattr(r, "real_layout", lambda *a: pytest.fail("must stop before loading"))
    with pytest.raises(SystemExit, match="uncommitted"):
        r.main([])


def test_real_run_checks_freeze_v1_before_loading(monkeypatch):
    monkeypatch.setattr(r, "git_state", lambda: {"commit": "x", "dirty": False})
    monkeypatch.setattr(r, "real_layout", lambda *a: pytest.fail("must stop before loading"))

    def changed():
        raise SystemExit("frozen_config.json differs from freeze-v1")
    monkeypatch.setattr(r, "frozen_unchanged", changed)
    with pytest.raises(SystemExit, match="freeze-v1"):
        r.main([])


def test_fit_all_fits_features_on_training_rows_only_and_blinds_labels(monkeypatch):
    lay = small_layout()
    fitted_on = []

    class Rule:
        @classmethod
        def fit(cls, df, names):
            fitted_on.append(df)
            return cls()

        def transform(self, df):
            return df.to_numpy(dtype=float)
    monkeypatch.setattr(r.ieee_features, "FeatureRule", Rule)
    monkeypatch.setattr(r, "fit_stage", lambda split, cfg, fn: split)
    split, _ = r.fit_all(lay, ["a"], {})
    assert len(fitted_on) == 1 and fitted_on[0] is lay.train
    assert np.array_equal(split.X_train, lay.train.to_numpy(float))
    assert not split.y_train.any() and not split.y_test.any()


def test_pre_deployment_thresholds_come_from_the_fit_slice(monkeypatch):
    lay = small_layout()
    fit_s = np.arange(100.0)[None, :]            # q 0.98 -> 97.02 (linear)
    sc_s = np.array([[97.0, 97.5, 200.0, 0.0]])  # a threshold from these would be about 192
    monkeypatch.setattr(r.ieee_stress, "features_for", lambda *a: (None, None, None))
    monkeypatch.setattr(r.lof_diag, "slice_scores", lambda *a: {"A": (fit_s, sc_s), "B": (fit_s, sc_s)})
    pre, _ = r.pre_deployment(lay, ["a"], ["A", "B"], [0], [0.02], {"E": ["A", "B"]})
    row = pre[pre.member == "A"].iloc[0]
    assert row.alerts == 2 and row.n_rows == 4 and not row.passes   # r = 2 / 0.08 = 25


def test_rq4_counts_are_the_freeze_v1_resamples():
    lay = small_layout()
    days = bootstrap.day_rows(lay.day_test, lay.month_test, lay.months)
    flat = np.concatenate([m for month in days for m in month])
    day_of_row = dict(zip(flat, lay.day_test[flat]))
    order = sorted(np.unique(lay.day_test))
    n_rows = {d: (lay.day_test == d).sum() for d in order}
    for seed in (0, 7):
        C, _ = r.rq4_counts(lay, 20, seed)
        rng = np.random.default_rng(seed)
        for row in C:
            drawn = np.concatenate(bootstrap.draw_days(rng, days))
            counts = [sum(day_of_row[i] == d for i in drawn) / n_rows[d] for d in order]
            assert np.array_equal(row, counts)


def test_h4r_cost_ci_comes_from_the_resamples():
    from thesis import detectors
    lay = small_layout()
    rng = np.random.default_rng(3)
    day = np.repeat([30, 31, 32, 60, 61], 40)
    lay = r.Layout(lay.train, pd.DataFrame({"a": np.zeros(len(day))}), lay.day_train, day,
                   np.repeat([2, 2, 2, 3, 3], 40), day - 30, [2, 3], lay.pre_fit, lay.pre_scored,
                   (rng.random(len(day)) < 0.1).astype(int), "synthetic")
    names = list(detectors.DETECTORS)
    train = {m: rng.normal(size=(1, 400)) for m in names}
    test = {m: rng.normal(size=(1, len(day))) for m in names}
    test["LOF"] = test["LOF"] + 3.0
    C, _ = r.rq4_counts(lay, 50, 0)
    _, _, cost = r.h4r_triples(r.Fitted(train, test, tuple(names), {}), lay, 0.9, C, 0.1, lay.y_test)
    assert cost["ci_low"] < cost["ci_high"]          # a CI built from the point alone has width 0


def test_bound_check_treats_the_member_with_the_largest_month_r_as_broken():
    lay = small_layout()
    rng = np.random.default_rng(1)
    n = len(lay.day_test)
    train = {m: rng.normal(size=(1, 200)) for m in ("A", "B", "C")}
    test = {m: rng.normal(size=(1, n)) for m in ("A", "B", "C")}
    fitted = r.Fitted(train, test, ("A", "B", "C"), {})
    mr = pd.DataFrame([{"member": m, "month": mo, "r": v}
                       for mo in (2, 3) for m, v in (("A", 1.0), ("B", 7.0), ("C", 0.5))])
    b = r.bound_monthly(fitted, lay, {"E": ["A", "B", "C"]}, mr, 0.9)
    assert set(b[b.b == 1].broken) == {"B"} and set(b[b.b == 1].broken_r) == {7.0}
