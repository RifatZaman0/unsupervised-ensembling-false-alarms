"""Rules for score() (Master Plan § Leakage rules): no labels, higher = more anomalous,
ECOD and every other detector score new rows without batch effects, seeds are fixed.

Where a rule could pass by accident, a second test shows the same check failing on a
version that breaks it (PyOD's own ECOD, self-query KNN scores).
"""
import inspect

import numpy as np
import pytest
from pyod.models.ecod import ECOD

from thesis import data, detectors, scorecache
from thesis.detectors import DETECTORS, score

needs_adbench = pytest.mark.skipif(not any(data.ADBENCH_DIR.glob("*.npz")),
                                   reason="ADBench not downloaded")


def blobs(n=600, d=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.normal(size=(200, d))


def batch_diff(score_fn, X_test, cut=77):
    """Largest change in any row's score when the test rows are scored in two batches."""
    whole = score_fn(X_test)
    parts = np.r_[score_fn(X_test[:cut]), score_fn(X_test[cut:])]
    return np.abs(whole - parts).max()


# ---------------------------------------------------------------- no labels

def test_score_takes_no_labels():
    params = inspect.signature(score).parameters
    assert not {"y", "y_train", "labels"} & set(params)
    assert list(params) == ["name", "X_fit", "X_score", "seed"]


# ---------------------------------------------------------------- direction

@pytest.mark.parametrize("det", DETECTORS)
def test_higher_score_means_more_anomalous(det):
    X_fit, X_new = blobs()
    X_new[0] = 12.0  # far outside the training cloud in every feature
    s_fit, s_new = score(det, X_fit, X_new, seed=0)
    assert s_new[0] > s_new[1:].max(), f"{det}: planted outlier is not the top score"
    assert s_new[0] > np.median(s_fit)


# ---------------------------------------------------------------- batch effects

@pytest.mark.parametrize("det", DETECTORS)
def test_new_rows_are_scored_independently(det):
    X_fit, X_new = blobs()
    assert batch_diff(lambda Z: score(det, X_fit, Z, seed=0)[1], X_new) == 0.0


def test_pyod_ecod_is_batch_dependent_so_the_check_can_fail():
    X_fit, X_new = blobs()
    pyod_ecod = ECOD().fit(X_fit)
    assert batch_diff(pyod_ecod.decision_function, X_new) > 1e-6


def test_train_only_ecod_matches_pyod_on_training_rows():
    X_fit, _ = blobs()
    X_fit[:20, 0] = X_fit[0, 0]  # ties
    ours = score("ECOD", X_fit, X_fit[:5])[0]
    assert np.allclose(ours, ECOD().fit(X_fit).decision_scores_, rtol=0, atol=1e-10)


def test_train_only_ecod_is_finite_beyond_training_range():
    X_fit, X_new = blobs()
    X_new[:2] = [[1e6] * 5, [-1e6] * 5]
    assert np.isfinite(score("ECOD", X_fit, X_new)[1]).all()


@needs_adbench
def test_ecod_batch_check_on_real_data():
    """The Master Plan's 'ECOD batch check' on cardio: PyOD's ECOD changes scores, ours does not."""
    X, y = data.load_adbench("cardio")
    s = data.standardize(data.split_adbench("cardio", 0, X, y))
    assert batch_diff(ECOD().fit(s.X_train).decision_function, s.X_test, cut=300) > 0.1
    assert batch_diff(lambda Z: score("ECOD", s.X_train, Z)[1], s.X_test, cut=300) == 0.0


# ---------------------------------------------------------------- training scores

@pytest.mark.parametrize("det", ["KNN", "LOF"])
def test_neighbour_training_scores_are_leave_one_out(det):
    X_fit, _ = blobs()
    s_fit, _ = score(det, X_fit, X_fit[:3])
    # a row scored as a new point is its own nearest neighbour; the training score must not be
    self_query = detectors.make_detector(det).fit(X_fit).decision_function(X_fit)
    assert not np.allclose(s_fit, self_query)
    if det == "KNN":  # with itself among the neighbours, the k-th distance can only shrink
        assert (self_query <= s_fit).all() and (self_query < s_fit).all()


@pytest.mark.parametrize("det", ["KNN", "LOF"])
def test_subsampled_fit_scores_every_training_row(det, monkeypatch):
    monkeypatch.setattr(detectors, "MAX_FIT_ROWS", 250)
    X_fit, X_new = blobs()
    s_fit, s_new = score(det, X_fit, X_new, seed=3)
    assert s_fit.shape == (600,) and np.isfinite(s_fit).all()
    assert np.array_equal(s_fit, score(det, X_fit, X_new, seed=3)[0])
    assert not np.array_equal(s_new, score(det, X_fit, X_new, seed=4)[1])


def test_pca_survives_collinear_features():
    X_fit, X_new = blobs()
    X_fit = np.c_[X_fit, X_fit[:, 0] + X_fit[:, 1]]
    X_new = np.c_[X_new, X_new[:, 0] + X_new[:, 1]]
    s_fit, s_new = score("PCA", X_fit, X_new)
    assert np.isfinite(s_fit).all() and np.isfinite(s_new).all()
    pca = detectors.make_detector("PCA").fit(X_fit)
    assert pca.n_dropped_ == 1


# ---------------------------------------------------------------- seeds

def test_iforest_is_seeded():
    X_fit, X_new = blobs()
    with pytest.raises(ValueError):
        score("IForest", X_fit, X_new)
    a, b = score("IForest", X_fit, X_new, 0)[1], score("IForest", X_fit, X_new, 0)[1]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, score("IForest", X_fit, X_new, 1)[1])


# ---------------------------------------------------------------- cache

def test_cache_roundtrip_and_split_mismatch(tmp_path):
    X, y = blobs()[0], np.r_[np.zeros(570), np.ones(30)].astype(np.int8)
    split = data.split_adbench("toy", 0, X, y)
    first = scorecache.get_scores(split, "HBOS", root=tmp_path)
    again = scorecache.get_scores(split, "HBOS", root=tmp_path)
    assert np.array_equal(first["test"], again["test"])
    assert first["runtime_s"] == again["runtime_s"]  # second call read the file
    other = data.split_adbench("toy", 1, X, y)
    other = data.Split(**{**other.__dict__, "seed": 0})  # same cache file, different rows
    with pytest.raises(RuntimeError, match="do not match"):
        scorecache.get_scores(other, "HBOS", root=tmp_path)


def test_cache_key_changes_with_parameters(monkeypatch):
    key = scorecache.cache_key()
    monkeypatch.setitem(detectors.PARAMS, "KNN", {"n_neighbors": 10, "method": "largest"})
    assert scorecache.cache_key() != key
