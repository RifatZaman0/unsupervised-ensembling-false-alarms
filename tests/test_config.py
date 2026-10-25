"""Protocol switches (thesis.config): the defaults reproduce the pre-switch behaviour and cache
key, every switch changes what it should and nothing else."""
import hashlib
import json

import numpy as np
import pyod
import pytest

from thesis import config, data, detectors, e1, metrics, scorecache
from thesis.config import Protocol


@pytest.fixture
def protocol(monkeypatch):
    def set_(**kw):
        monkeypatch.setattr(config, "PROTOCOL", Protocol(**kw))
    return set_


def legacy_key():
    """The cache key as computed before thesis.config existed (E1 run @1cdc4be)."""
    spec = {"params": detectors.PARAMS, "max_fit_rows": detectors.MAX_FIT_ROWS,
            "preprocessing": "zscore-train", "pyod": pyod.__version__, "ecod": "train-only-v2"}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]


def test_defaults_keep_the_existing_cache_key():
    assert config.PROTOCOL == Protocol()
    assert scorecache.cache_key() == legacy_key()


@pytest.mark.parametrize("switch", [{"zscore_inputs": False}, {"pca_drop_zero_variance": False},
                                    {"adbench_dedup": True}])
def test_switches_that_change_scores_change_the_cache_key(protocol, switch):
    protocol(**switch)
    assert scorecache.cache_key() != legacy_key()


def test_k_rule_does_not_change_the_cache_key(protocol):
    protocol(k_rule="floor")
    assert scorecache.cache_key() == legacy_key()


def test_k_rules(protocol):
    assert metrics.budget_k(0.02, 1001) == 21  # ceil (default)
    assert metrics.budget_k(0.02, 1001, "round") == 20
    assert metrics.budget_k(0.025, 100, "round") == 3  # 2.5 rounds half up
    assert metrics.budget_k(0.02, 1001, "floor") == 20
    assert metrics.budget_k(0.05, 140, "ceil") == 7  # float noise does not add an alert
    assert metrics.budget_k(0.001, 10, "floor") == 1  # at least one alert
    protocol(k_rule="floor")
    assert metrics.budget_k(0.02, 1001) == 20
    with pytest.raises(ValueError):
        Protocol(k_rule="nearest")


def test_zscore_switch(protocol):
    rng = np.random.default_rng(0)
    X, y = rng.normal(5.0, 3.0, size=(200, 3)), np.r_[np.zeros(190), np.ones(10)].astype(int)
    split = data.split_adbench("toy", 0, X, y)
    assert abs(data.preprocess(split).X_train.mean()) < 1e-12
    protocol(zscore_inputs=False)
    assert data.preprocess(split) is split
    assert scorecache.preprocessing() == "none"


def test_pca_drop_switch(protocol):
    rng = np.random.default_rng(0)
    X = np.c_[rng.normal(size=(300, 3)), np.full(300, 2.0)]  # constant column: zero variance
    detectors.score("PCA", X, X[:5])  # finite with the drop (default)
    protocol(pca_drop_zero_variance=False)
    assert detectors.params()["PCA"]["min_variance_ratio"] is None
    assert detectors.PARAMS["PCA"]["min_variance_ratio"] == 1e-10  # the default is untouched
    with pytest.raises(detectors.NonFiniteScores):
        detectors.score("PCA", X, X[:5])


def test_dedup_rows_keeps_first_occurrences_and_label_conflicts():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0, 0, 0, 1, 0])
    Xd, yd = data.dedup_rows(X, y)
    assert np.array_equal(Xd, X[[0, 1, 3, 4]]) and np.array_equal(yd, y[[0, 1, 3, 4]])


def test_variant_outputs_never_overwrite_the_main_ones():
    assert Protocol().tag() == "" and e1.out_dir(Protocol()) == e1.OUT
    p = Protocol(zscore_inputs=False, k_rule="round", pca_drop_zero_variance=False,
                 adbench_dedup=True)
    assert p.tag() == "no-zscore_k-round_no-pca-drop_dedup"
    assert e1.out_dir(p) == e1.OUT / "variants" / p.tag()


def test_command_line_flags_map_to_the_protocol():
    import argparse
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    assert config.from_args(parser.parse_args([])) == Protocol()
    assert config.from_args(parser.parse_args(
        ["--no-zscore", "--k-rule", "floor", "--no-pca-drop", "--dedup"])) == Protocol(
        False, "floor", False, True)
