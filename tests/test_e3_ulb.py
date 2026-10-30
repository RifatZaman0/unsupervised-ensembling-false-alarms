"""E3 on ULB: leakage (no day-2 row or label in any fitting step), temporal split, frozen file
unchanged since the tag, and the budget / threshold / hypothesis logic."""
from __future__ import annotations

import copy
import subprocess

import numpy as np
import pandas as pd
import pytest

from thesis import bootstrap, data, detectors, e3_ulb, frozen, fusion
from thesis.paths import REPO

needs_ulb = pytest.mark.skipif(not data.ULB_CSV.exists(), reason="ULB not downloaded")


@pytest.fixture(scope="module")
def cfg():
    return frozen.load()


def toy_split(n_train=240, n_test=160, d=4, seed=0) -> data.Split:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train + n_test, d))
    y = (rng.random(n_train + n_test) < 0.06).astype(np.int8)
    X[y == 1] += 3.0
    t = np.r_[np.sort(rng.uniform(0, 86_000, n_train)), np.sort(rng.uniform(86_400, 170_000, n_test))]
    tr, te = np.arange(n_train), np.arange(n_train, n_train + n_test)
    return data.Split("ulb", X[tr], X[te], y[tr], y[te], tr, te, t_train=t[tr], t_test=t[te])


def direct_scores(split: data.Split, det: str) -> dict:
    s_tr, s_te = detectors.score(det, split.X_train, split.X_test, split.seed)
    return {"train": s_tr, "test": s_te, "runtime_s": 0.0}


# ---------------------------------------------------------------- leakage

def test_fitting_stage_sees_no_label(cfg, monkeypatch):
    split = toy_split()
    seen = []

    def spy(s, det):
        seen.append((s.y_train.copy(), s.y_test.copy()))
        return direct_scores(s, det)
    a = e3_ulb.fit_stage(split, cfg, spy)
    assert seen and all(not ytr.any() and not yte.any() for ytr, yte in seen)
    flipped = e3_ulb.fit_stage(
        e3_ulb.replace(split, y_train=1 - split.y_train, y_test=1 - split.y_test), cfg, direct_scores)
    assert a.train.keys() == flipped.train.keys()
    for m in a.train:
        np.testing.assert_array_equal(a.train[m], flipped.train[m])
        np.testing.assert_array_equal(a.test[m], flipped.test[m])


def test_fitting_stage_fits_on_day1_rows_only(cfg, monkeypatch):
    split = toy_split()
    expected = data.Standardizer.fit(split.X_train).transform(split.X_train)
    fit_inputs, calib_inputs = [], []
    real_fit_scores, real_cal_fit = detectors.fit_scores, fusion.Calibrator.fit.__func__

    def fit_spy(name, X_fit, seed=None):
        fit_inputs.append(np.array(X_fit))
        return real_fit_scores(name, X_fit, seed)

    def cal_spy(cls, s_train, kind):
        calib_inputs.append(np.array(s_train))
        return real_cal_fit(cls, s_train, kind)
    monkeypatch.setattr(detectors, "fit_scores", fit_spy)
    monkeypatch.setattr(fusion.Calibrator, "fit", classmethod(cal_spy))
    fitted = e3_ulb.fit_stage(split, cfg, direct_scores)
    assert len(fit_inputs) == 10                                   # IForest x 5 seeds + 5 others
    for X in fit_inputs:
        np.testing.assert_array_equal(X, expected)                 # day-1 rows, day-1 scaling
    assert calib_inputs and all(len(s) == len(split.y_train) for s in calib_inputs)
    # changing every day-2 row changes no training score and no threshold
    moved = e3_ulb.replace(split, X_test=split.X_test * 10 + 5)
    other = e3_ulb.fit_stage(moved, cfg, direct_scores)
    for m in fitted.train:
        np.testing.assert_array_equal(fitted.train[m], other.train[m])
        for q in cfg["thresholds"]["ulb"]["training_score_quantiles"]:
            np.testing.assert_array_equal(e3_ulb.fit_thresholds(fitted.train[m], q),
                                          e3_ulb.fit_thresholds(other.train[m], q))


def test_thresholds_come_from_training_scores_and_alert_is_strict():
    tr = np.array([[0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [9.0, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
    thr = e3_ulb.fit_thresholds(tr, 0.8)
    np.testing.assert_allclose(thr, [np.quantile(tr[0], 0.8, method="linear")] * 2)
    te = np.array([[thr[0], thr[0] + 1e-9], [0.0, 100.0]])
    np.testing.assert_array_equal(e3_ulb.alerts(te, thr), [[False, True], [False, True]])


# ---------------------------------------------------------------- temporal split, freeze

@needs_ulb
def test_every_training_time_is_before_every_test_time():
    s = e3_ulb.load_split()
    assert s.t_train.max() < s.t_test.min()
    assert s.t_train.max() < data.ULB_DAY_SECONDS <= s.t_test.min()


def test_runner_rejects_a_split_with_a_late_training_row(monkeypatch):
    s = toy_split()
    t_train = s.t_train.copy()
    t_train[0] = s.t_test.max() + 1
    monkeypatch.setattr(data, "split_ulb", lambda: e3_ulb.replace(s, t_train=t_train))
    with pytest.raises(AssertionError):
        e3_ulb.load_split()


def test_frozen_config_unchanged_since_the_tag():
    r = subprocess.run(["git", "diff", "freeze-v1", "--", "frozen_config.json"], cwd=REPO,
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout == "", "frozen_config.json differs from freeze-v1"
    e3_ulb.frozen_unchanged()
    with pytest.raises(RuntimeError, match="not found"):
        e3_ulb.frozen_unchanged("no-such-tag")


def test_code_must_match_the_frozen_file(cfg):
    e3_ulb.check_code_matches(cfg)
    for brk in (lambda c: c["detectors"]["params"]["KNN"].update(n_neighbors=10),
                lambda c: c["thresholds"]["common"].update(rule="alert if score >= threshold"),
                lambda c: c["statistics"]["bootstrap"].update(ulb="day-block"),
                lambda c: c["detectors"]["knn_lof_fit_sample"].update(max_rows=10_000)):
        bad = copy.deepcopy(cfg)
        brk(bad)
        with pytest.raises(RuntimeError):
            e3_ulb.check_code_matches(bad)


def test_methods_come_from_the_file(cfg):
    assert e3_ulb.fusion_configs(cfg)[0] == fusion.PRIMARY
    assert sorted(e3_ulb.fusion_configs(cfg)) == sorted(fusion.FUSION_CONFIGS)
    assert set(e3_ulb.distinct_ensembles(cfg)) == {"E-bench", "E-bench-5%", "E-all"}
    assert e3_ulb.h2b_secondary_pair(cfg) == ("E-bench-5%:minmax+mean", "E-bench:minmax+mean")
    comps = e3_ulb.comparisons(cfg)
    assert len(comps) == 3 * 6 * 3 + 1 and len(set(comps)) == len(comps)
    assert e3_ulb.ci_level(cfg) == 0.95


# ---------------------------------------------------------------- budget mode

def hand_fitted(y):
    """Singles with known FP@K at K = 2 (n = 10, budget 0.2): IForest 2 seeds."""
    n = len(y)
    def s(top):                       # score 1 on the rows in `top`, 0 elsewhere (no ties at K)
        v = np.linspace(0, 0.1, n)
        v[list(top)] = [1.0, 0.9][: len(top)]
        return v
    pos = np.flatnonzero(y)               # rows 0, 1 are frauds
    neg = np.flatnonzero(y == 0)
    test = {"IForest": np.vstack([s([pos[0], neg[-1]]), s([neg[-1], neg[-2]])]),   # FP 1, 2 -> 1.5
            "PCA": s(pos[:2])[None], "ECOD": s([pos[0], neg[-1]])[None],           # 0, 1
            "HBOS": s(neg[-2:])[None], "KNN": s(neg[-2:])[None],                   # 2, 2
            "LOF": s([pos[1], neg[-1]])[None]}                                     # 1
    test["E-bench:minmax+mean"] = s([pos[0], neg[-1]])[None]                       # 1
    singles = ("IForest", "PCA", "ECOD", "HBOS", "KNN", "LOF")
    return e3_ulb.Fitted({k: v for k, v in test.items()}, test, singles, {})


def test_budget_table_baselines_and_regret(cfg):
    y = np.zeros(10, np.int8)
    y[:2] = 1
    f = hand_fitted(y)
    t = e3_ulb.budget_table(e3_ulb.budget_runs(f, y, [0.2]), cfg).set_index("method")
    assert t.loc["IForest", "fp_at_k"] == 1.5 and t.loc["IForest", "n_seeds"] == 2
    assert t.loc[bootstrap.MEDIAN, "fp_at_k"] == 1.25                # sorted 0 1 1 1.5 2 2
    assert t.loc[bootstrap.ORACLE, "name"] == "PCA" and t.loc[bootstrap.ORACLE, "fp_at_k"] == 0
    assert t.loc[bootstrap.DEFAULT_IFOREST, "fp_at_k"] == 1.5
    assert t.loc["E-bench:minmax+mean", "regret"] == 1.0
    assert t.loc["E-bench:minmax+mean", "recall_at_k"] == 0.5
    assert (t.k == 2).all()


def test_bootstrap_point_matches_the_budget_table(cfg, monkeypatch):
    y = np.zeros(10, np.int8)
    y[:2] = 1
    f = hand_fitted(y)
    pair = ("E-bench:minmax+mean", bootstrap.MEDIAN)
    monkeypatch.setattr(e3_ulb, "comparisons", lambda c: [pair])
    summ, boot = e3_ulb.bootstrap_budget(f, y, 0.2, cfg, resamples=30)
    s = summ.set_index("stat")
    assert s.loc["fp:IForest", "estimate"] == 1.5
    assert s.loc["d:E-bench:minmax+mean-median_single", "estimate"] == pytest.approx(-0.25)
    assert s.loc["d100:E-bench:minmax+mean-median_single", "estimate"] == pytest.approx(-12.5)
    assert len(boot) == 30
    again = e3_ulb.bootstrap_budget(f, y, 0.2, cfg, resamples=30)[1]
    pd.testing.assert_frame_equal(boot, again)                       # frozen seed, reproducible


# ---------------------------------------------------------------- hypotheses

def deltas_for(cfg, h1a_hi: float, h1b_hi: float) -> pd.DataFrame:
    rows = []
    for a, comp in e3_ulb.comparisons(cfg):
        hi = {bootstrap.MEDIAN: h1a_hi, bootstrap.DEFAULT_IFOREST: h1b_hi}.get(comp, 1.0)
        ens, calib, agg = e3_ulb.parse_key(a)
        rows.append({"budget": cfg["budgets"]["ulb"]["primary"], "method": a, "ensemble": ens,
                     "calib": calib, "agg": agg, "comparator": comp, "d_fp": hi - 1,
                     "d_fp_ci_low": hi - 2, "d_fp_ci_high": hi, "d_fp_per_100": 0.0,
                     "d_fp_per_100_ci_low": 0.0, "d_fp_per_100_ci_high": 0.0,
                     "verdict": e3_ulb.SUPPORT if hi < 0 else e3_ulb.NULL,
                     "ci_entirely_above_0": hi - 2 > 0})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("h1a_hi, h1b_hi, h1a, h1b", [
    (-0.5, -0.5, "pending IEEE-CIS", "supported (holds on ULB; needs at least one)"),
    (0.5, 0.5, "not supported (fails on ULB; needs both)", "pending IEEE-CIS"),
    (0.0, -1e-9, "not supported (fails on ULB; needs both)",                   # CI upper = 0
     "supported (holds on ULB; needs at least one)"),
])
def test_hypotheses_follow_the_frozen_rules(cfg, h1a_hi, h1b_hi, h1a, h1b):
    h = e3_ulb.hypotheses(deltas_for(cfg, h1a_hi, h1b_hi), cfg, 1000).set_index("hypothesis")
    assert h.loc["H1a", "status"] == h1a and h.loc["H1b", "status"] == h1b
    assert h.loc["H1a", "verdict"] == (e3_ulb.SUPPORT if h1a_hi < 0 else e3_ulb.NULL)
    assert h.loc["H1a", "comparison"] == "E-bench (= E-lit) vs median_single"
    assert h.loc["H2b (primary)", "verdict"] == "degenerate" and h.loc["H2b (primary)", "d_fp"] == 0
    assert h.loc["H2b (secondary)", "comparison"] == "E-bench-5% vs E-lit (= E-bench)"
    assert "non-inferiority is not tested" in h.loc["H2b (secondary)", "status"]


# ---------------------------------------------------------------- threshold mode

def test_threshold_table_counts(cfg):
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
    train = np.arange(100, dtype=float)[None]                        # q 0.9 -> threshold 89.1
    test = np.array([[95, 10, 90, 99, 0, 0, 0, 0, 0, 89.1]])          # alerts rows 0, 2, 3
    f = e3_ulb.Fitted({"PCA": train}, {"PCA": test}, ("PCA",), {})
    t = e3_ulb.threshold_table(f, y, [0.9]).iloc[0]
    assert t.alert_count == 3 and t.alert_rate == 0.3
    assert t.fpr == pytest.approx(2 / 8) and t.recall == 0.5 and t.precision == pytest.approx(1 / 3)
    assert t.train_alert_rate == pytest.approx(0.1) and t.target_rate == pytest.approx(0.1)


def test_delta_table_applies_the_frozen_criterion(cfg):
    """'fewer false alarms' only if the whole CI is below 0; a CI above 0 stays 'no difference
    shown' (the frozen criterion has two outcomes) and is only flagged, descriptively."""
    rows = []
    for i, (a, comp) in enumerate(e3_ulb.comparisons(cfg)):
        lo, hi = [(-2.0, -0.1), (0.5, 3.0), (-1.0, 0.0)][i % 3]
        for pre in ("d", "d100"):
            rows.append({"budget": 0.002, "stat": f"{pre}:{a}-{comp}", "estimate": (lo + hi) / 2,
                         "ci_low": lo, "ci_high": hi, "supported": bootstrap.fewer_false_alarms(hi)})
    d = e3_ulb.delta_table(pd.DataFrame(rows), cfg)
    assert list(d.verdict[:3]) == [e3_ulb.SUPPORT, e3_ulb.NULL, e3_ulb.NULL]
    assert list(d.ci_entirely_above_0[:3]) == [False, True, False]
    assert len(d) == len(e3_ulb.comparisons(cfg))
