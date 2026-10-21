"""Leakage and validity rules for loaders and splits (Master Plan § Leakage rules).

Each rule has a test on clean input and a test that breaks the rule on purpose, so the
check is shown to fail when it should. Real-data tests skip when data/raw is not downloaded.
"""
import dataclasses
import inspect

import numpy as np
import pandas as pd
import pytest

from thesis import data
from thesis.data import Split, Standardizer, assert_disjoint, assert_temporal

needs_adbench = pytest.mark.skipif(not any(data.ADBENCH_DIR.glob("*.npz")),
                                   reason="ADBench not downloaded")
needs_ulb = pytest.mark.skipif(not data.ULB_CSV.exists(), reason="ULB not downloaded")


def toy(n=1000, rate=0.05, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.int8)
    y[rng.choice(n, int(n * rate), replace=False)] = 1
    return rng.normal(size=(n, 4)), y


# ---------------------------------------------------------------- selection rule

def test_selection_rule_boundaries():
    cat = pd.DataFrame([
        ("lo", 1_000, 5, 0.05), ("too_few", 999, 5, 0.05),
        ("hi", 50_000, 5, 0.05), ("too_many", 50_001, 5, 0.05),
        ("d100", 2_000, 100, 0.05), ("d101", 2_000, 101, 0.05),
        ("rate10", 2_000, 5, 0.10), ("rate11", 2_000, 5, 0.11),
        ("fraud", 2_000, 5, 0.01),
    ], columns=["dataset", "n_rows", "n_features", "anomaly_rate"])
    assert data.select_adbench(cat) == ["d100", "hi", "lo", "rate10"]


@needs_adbench
def test_selection_rule_on_adbench_gives_the_planned_15():
    cat = data.adbench_catalogue()
    assert len(cat) == 47, "expected all 47 ADBench classical files"
    assert data.select_adbench(cat) == sorted(data.ADBENCH_15, key=str.lower)


# ---------------------------------------------------------------- ADBench split

def test_adbench_split_is_stratified_60_40_and_deterministic():
    X, y = toy()
    s = data.split_adbench("toy", 0, X, y)
    assert len(s.idx_test) == 400 and len(s.idx_train) == 600
    assert s.y_train.sum() == 30 and s.y_test.sum() == 20
    assert np.array_equal(s.idx_test, data.split_adbench("toy", 0, X, y).idx_test)
    assert not np.array_equal(s.idx_test, data.split_adbench("toy", 1, X, y).idx_test)
    assert np.array_equal(s.X_test, X[s.idx_test]) and np.array_equal(s.y_test, y[s.idx_test])


def test_assert_disjoint_catches_shared_rows():
    X, y = toy()
    s = data.split_adbench("toy", 0, X, y)
    leaky_tr = np.r_[s.idx_train, s.idx_test[:1]]
    leaky = dataclasses.replace(s, idx_train=leaky_tr, X_train=X[leaky_tr], y_train=y[leaky_tr])
    with pytest.raises(AssertionError, match="share rows"):
        assert_disjoint(leaky, len(y))


def test_assert_disjoint_catches_dropped_rows():
    X, y = toy()
    s = data.split_adbench("toy", 0, X, y)
    short = dataclasses.replace(s, idx_test=s.idx_test[1:], X_test=s.X_test[1:], y_test=s.y_test[1:])
    with pytest.raises(AssertionError, match="exactly once"):
        assert_disjoint(short, len(y))


@needs_adbench
@pytest.mark.parametrize("name", data.ADBENCH_15)
def test_adbench_real_splits(name):
    X, y = data.load_adbench(name)
    assert not any(np.array_equal(X[:, j], y) for j in range(X.shape[1])), "label copied into features"
    test_ids = set()
    for seed in data.ADBENCH_SEEDS:
        s = data.split_adbench(name, seed, X, y)  # runs assert_disjoint
        assert abs(s.y_test.mean() - y.mean()) < 1.5 / len(s.y_test) + 1e-12
        test_ids.add(s.idx_test.tobytes())
    assert len(test_ids) == len(data.ADBENCH_SEEDS), "seeds give identical splits"


# ---------------------------------------------------------------- temporal split (fraud data)

def test_temporal_split_on_unsorted_times():
    rng = np.random.default_rng(0)
    t = rng.permutation(np.linspace(0, 2 * data.ULB_DAY_SECONDS - 1, 500))
    X, y = toy(500)
    s = data.split_ulb(X, y, t)
    assert s.t_train.max() < data.ULB_DAY_SECONDS <= s.t_test.min()
    assert np.array_equal(s.X_train, X[t < data.ULB_DAY_SECONDS])


def test_assert_temporal_catches_late_training_row():
    X, y = toy(500)
    t = np.linspace(0, 2 * data.ULB_DAY_SECONDS - 1, 500)
    s = data.split_ulb(X, y, t)
    leaky = dataclasses.replace(s, t_train=np.r_[s.t_train[:-1], s.t_test.max()])
    with pytest.raises(AssertionError, match="not earlier"):
        assert_temporal(leaky)


def test_assert_temporal_rejects_equal_timestamps():
    s = Split("eq", np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1), np.zeros(1),
              np.array([0]), np.array([1]), t_train=np.array([5.0]), t_test=np.array([5.0]))
    with pytest.raises(AssertionError):
        assert_temporal(s)


@pytest.fixture(scope="module")
def ulb():
    return data.load_ulb()


@needs_ulb
def test_ulb_day1_day2_split_on_real_data(ulb):
    X, y, t = ulb
    s = data.split_ulb(X, y, t)  # runs assert_disjoint and assert_temporal
    assert X.shape == (284_807, 29), "V1–V28 + Amount; Time must not be a feature"
    assert s.t_train.min() == 0 and s.t_train.max() < 86_400 <= s.t_test.min()
    assert (len(s.idx_train), len(s.idx_test)) == (144_786, 140_021)
    assert not any(np.array_equal(X[:, j], y) for j in range(X.shape[1])), "label copied into features"


# ---------------------------------------------------------------- preprocessing

def test_standardizer_fit_takes_no_labels():
    assert list(inspect.signature(Standardizer.fit).parameters) == ["X_train"]


def test_standardizer_parameters_come_from_training_rows_only():
    X, y = toy()
    s = data.split_adbench("toy", 0, X, y)
    a = data.standardize(s)
    shifted = dataclasses.replace(s, X_test=s.X_test * 100 + 7)  # any change to test rows
    b = data.standardize(shifted)
    assert np.array_equal(a.X_train, b.X_train)
    np.testing.assert_allclose(a.X_train.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(a.X_train.std(axis=0), 1, atol=1e-12)
    sc = Standardizer.fit(s.X_train)
    np.testing.assert_allclose(b.X_test, sc.transform(shifted.X_test))


def test_standardizer_keeps_constant_columns_finite():
    X = np.c_[np.ones(10), np.arange(10.0)]
    assert np.isfinite(Standardizer.fit(X).transform(np.c_[np.full(3, 2.0), np.ones(3)])).all()


# ---------------------------------------------------------------- raw data integrity

@pytest.mark.skipif(not (any(data.ADBENCH_DIR.glob("*.npz")) and data.ULB_CSV.exists()),
                    reason="raw data not downloaded")
def test_raw_files_match_manifest():
    from thesis.download import load_manifest, verify
    assert verify(load_manifest()) == []
