"""Paired bootstrap of the frozen protocol: month-stratified day resampling keeps every month,
K follows the resample size, the median single and the oracle are recomputed per resample,
and the IEEE-CIS month / H3b week rules match frozen_config.json."""
import numpy as np
import pytest

from thesis import bootstrap as bs
from thesis import data, metrics

SINGLES = ("IForest", "PCA", "ECOD", "HBOS", "KNN", "LOF")


def ieee_like(seed=0, n_days=182):
    """Synthetic TransactionDT over n_days with an uneven number of rows per day."""
    rng = np.random.default_rng(seed)
    per_day = rng.integers(20, 60, n_days)
    t = np.concatenate([86_400 + d * 86_400 + np.sort(rng.uniform(0, 86_399, c))
                        for d, c in enumerate(per_day)])
    day = data.ieee_cis_day(t)
    return day, data.ieee_cis_month(day)


def test_ieee_cis_month_and_week_rules():
    day = np.arange(182)
    month = data.ieee_cis_month(day)
    assert (month[[0, 29, 30, 59, 149, 150, 181]] == [1, 1, 2, 2, 5, 6, 6]).all()
    assert np.bincount(month)[1:].tolist() == [30, 30, 30, 30, 30, 32]  # month 6 = day 150 on
    week = data.ieee_cis_week(day)
    assert (week[:30] == -1).all()                                      # month 1 = reference
    # days 30-181 = 152 days = 21 full weeks + 5 days, which join the last week
    counts = np.bincount(week[30:])
    assert len(counts) == 21 and (counts[:-1] == 7).all() and counts[-1] == 12
    assert data.ieee_cis_day(np.array([86_400, 86_400 + 86_399, 86_400 * 2]))[[0, 1, 2]].tolist() == [0, 0, 1]


def test_month_stratified_resample_keeps_every_month():
    day, month = ieee_like()
    days = bs.day_rows(day, month, data.IEEE_TEST_MONTHS)
    rng = np.random.default_rng(0)
    for _ in range(50):
        win = bs.draw_days(rng, days)
        assert len(win) == 5
        for m, rows, month_days in zip(data.IEEE_TEST_MONTHS, win, days):
            assert set(month[rows]) == {m}                       # only days of its own month
            # every drawn day brings all its rows: rows per day / day size = times drawn
            size = {int(day[r[0]]): len(r) for r in month_days}
            ds, c = np.unique(day[rows], return_counts=True)
            drawn = [cc / size[int(d)] for d, cc in zip(ds, c)]
            assert all(float(x).is_integer() for x in drawn)
            assert sum(drawn) == len(month_days)                 # the month keeps its day count


def test_k_follows_the_resample_size_on_ieee_cis():
    day, month = ieee_like()
    days = bs.day_rows(day, month, data.IEEE_TEST_MONTHS)
    y = np.zeros(len(day))
    scores = {n: np.random.default_rng(i).random((1, len(day))) for i, n in enumerate(SINGLES)}
    stat = bs.budget_statistic(scores, y, 0.02, SINGLES, [])
    rng = np.random.default_rng(0)
    k_sums = set()
    for _ in range(20):
        win = bs.draw_days(rng, days)
        out = stat(win)
        assert out["k_sum"] == sum(metrics.budget_k(0.02, len(w)) for w in win)
        k_sums.add(out["k_sum"])
    assert len(k_sums) > 1                                       # day sizes differ -> K differs
    full = stat(bs.day_windows(days))["k_sum"]
    assert full == sum(metrics.budget_k(0.02, int((month == m).sum())) for m in data.IEEE_TEST_MONTHS)


def test_ulb_resample_keeps_the_size_and_k():
    rng = np.random.default_rng(0)
    for _ in range(5):
        (idx,) = bs.draw_transactions(rng, 140_021)
        assert len(idx) == 140_021 and metrics.budget_k(0.002, len(idx)) == 281


def detector_frame(n=400, seed=3):
    """Six singles of similar quality (the median detector depends on the rows drawn) and an
    ensemble; IForest and the ensemble have 5 seeds."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(float)
    scores = {}
    for i, name in enumerate(SINGLES):
        seeds = 5 if name == "IForest" else 1
        scores[name] = y * (0.6 + 0.05 * i) + rng.normal(0, 1, (seeds, n))
    scores["E-bench"] = y * 0.8 + rng.normal(0, 1, (5, n))
    return scores, y


def test_median_single_and_oracle_are_recomputed_per_resample():
    scores, y = detector_frame()
    comps = [("E-bench", bs.MEDIAN), ("E-bench", bs.ORACLE), ("E-bench", bs.DEFAULT_IFOREST)]
    stat = bs.budget_statistic(scores, y, 0.05, SINGLES, comps)
    full = stat(bs.transaction_windows(len(y)))
    full_order = np.argsort([full[f"fp:{n}"] for n in SINGLES], kind="stable")
    fixed_median_pair, fixed_oracle = full_order[2:4], full_order[0]
    rng = np.random.default_rng(0)
    changed_median = changed_oracle = 0
    for _ in range(60):
        out = stat(bs.draw_transactions(rng, len(y)))
        fp = np.array([out[f"fp:{n}"] for n in SINGLES])
        assert out[f"fp:{bs.MEDIAN}"] == pytest.approx(np.median(fp))
        assert out[f"fp:{bs.ORACLE}"] == pytest.approx(fp.min())
        assert out[f"d:E-bench-{bs.MEDIAN}"] == pytest.approx(out["fp:E-bench"] - np.median(fp))
        assert out[f"d:E-bench-{bs.DEFAULT_IFOREST}"] == pytest.approx(out["fp:E-bench"] - fp[0])
        assert out[f"d100:E-bench-{bs.MEDIAN}"] == pytest.approx(
            out[f"d:E-bench-{bs.MEDIAN}"] / out["k_sum"] * 100)
        # the detector(s) holding the median / minimum in the full window, kept fixed
        changed_median += not np.isclose(fp[fixed_median_pair].mean(), np.median(fp))
        changed_oracle += not np.isclose(fp[fixed_oracle], fp.min())
    assert changed_median > 0 and changed_oracle > 0


def test_seed_mean_is_the_mean_of_per_seed_fp():
    scores, y = detector_frame()
    win = bs.draw_transactions(np.random.default_rng(1), len(y))
    k = metrics.budget_k(0.05, len(y))
    per_seed = [k - metrics.tp_at_k(s[win[0]], y[win[0]], k) for s in scores["IForest"]]
    assert bs.seed_mean_fp(scores["IForest"], y, 0.05, win)[0] == pytest.approx(np.mean(per_seed))


def test_run_is_paired_and_reproducible():
    scores, y = detector_frame()
    stat = bs.budget_statistic(scores, y, 0.05, SINGLES, [("E-bench", bs.MEDIAN)])
    seen = []

    def draw(rng):
        w = bs.draw_transactions(rng, len(y))
        seen.append(w[0])
        return w
    a = bs.run(draw, stat, resamples=30, seed=0)
    b = bs.run(lambda r: bs.draw_transactions(r, len(y)), stat, resamples=30, seed=0)
    assert len(seen) == 30 and a.equals(b)             # one draw per resample, same stream
    # the i-th resample of default_rng(0) is the i-th draw, whatever the methods are
    rng = np.random.default_rng(0)
    assert all(np.array_equal(s, rng.integers(0, len(y), len(y))) for s in seen)


def test_criterion_needs_the_whole_ci_below_zero():
    assert bs.fewer_false_alarms(-0.1) and not bs.fewer_false_alarms(0.0)
    point = {"d:a-b": -1.0, "fp:a": 3.0}
    boot = __import__("pandas").DataFrame({"d:a-b": np.linspace(-2, 0.5, 101), "fp:a": 3.0})
    s = bs.summary(point, boot).set_index("stat")
    assert s.loc["d:a-b", "ci_high"] > 0 and s.loc["d:a-b", "supported"] is False
    assert s.loc["fp:a", "supported"] is None
    assert bs.percentile_ci(np.arange(1001)) == (25.0, 975.0)


def test_alert_rate_deviation():
    q = 0.99
    alerts = {n: np.zeros((1, 200), bool) for n in SINGLES}
    alerts["E-bench"] = np.zeros((1, 200), bool)
    alerts["E-bench"][0, [0, 100]] = True                 # 1% in both windows -> deviation 0
    alerts["LOF"][0, :10] = True                          # 10% / 0%
    win = [np.arange(100), np.arange(100, 200)]
    out = bs.alert_rate_statistic(alerts, q, SINGLES, "E-bench")(win)
    assert out["dev:E-bench"] == pytest.approx(0.0)
    assert out["dev:LOF"] == pytest.approx((0.09 + 0.01) / 2)
    assert out["dev:median_single"] == pytest.approx(0.01)  # five singles never alert
    assert out["d:E-bench-median_single"] == pytest.approx(-0.01)


def test_alert_rate_deviation_is_taken_per_seed_then_averaged():
    """evaluation.seeds: the measure is computed for each IForest seed, then the seed mean.
    Seed 0 alerts 3% and seed 1 none: per seed |0.03 - 0.01| = 0.02 and |0 - 0.01| = 0.01,
    mean 0.015; the deviation of the seed-mean rate (1.5%) would be 0.005."""
    q = 0.99
    alerts = {n: np.zeros((1, 200), bool) for n in SINGLES}
    alerts["IForest"] = np.zeros((2, 200), bool)
    alerts["IForest"][0, [0, 1, 2, 100, 101, 102]] = True
    alerts["E-bench"] = np.zeros((2, 200), bool)
    win = [np.arange(100), np.arange(100, 200)]
    out = bs.alert_rate_statistic(alerts, q, SINGLES, "E-bench")(win)
    assert out["dev:IForest"] == pytest.approx((0.02 + 0.01) / 2)
    assert out["dev:E-bench"] == pytest.approx(0.01)


def test_moving_blocks_and_spearman():
    rng = np.random.default_rng(0)
    for _ in range(20):
        i = bs.moving_blocks(rng, 21, 4)
        assert len(i) == 21 and i.min() >= 0 and i.max() <= 20
        assert all(np.all(np.diff(b) == 1) for b in np.split(i[:20], 5))  # blocks of 4 in a row
    x = np.arange(21.0)
    r = bs.spearman_moving_block(x, x ** 2, resamples=50)
    assert r["rho"] == pytest.approx(1.0) and r["ci_low"] == pytest.approx(1.0)
    assert r["n_windows"] == 21
    with pytest.raises(ValueError):
        bs.moving_blocks(rng, 3, 4)
