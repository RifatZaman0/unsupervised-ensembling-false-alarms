"""Budget metrics (FP@K, Precision@K, Recall@K) checked against hand-computed cases.
Ties at the K-th score must not be decided by row order."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from thesis.metrics import budget_k, budget_metrics, ranking_metrics, tp_at_k


def test_budget_k_rounds_up_and_is_at_least_one():
    assert budget_k(0.02, 1000) == 20
    assert budget_k(0.02, 1001) == 21
    assert budget_k(0.05, 140) == 7  # 7.000000000000001 in floating point must not become 8
    assert budget_k(0.001, 10) == 1


def test_hand_computed_case_without_ties():
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 1, 0])
    m = budget_metrics(s, y, 0.3)  # K = 3: rows 0, 1, 2 -> 2 TP
    assert (m["k"], m["tp_at_k"], m["fp_at_k"]) == (3, 2.0, 1.0)
    assert np.isclose(m["precision_at_k"], 2 / 3) and np.isclose(m["recall_at_k"], 2 / 3)


def test_ties_at_the_cut_are_split_in_expectation():
    s = np.array([5.0, 3.0, 3.0, 3.0, 3.0, 1.0])
    y = np.array([0, 1, 0, 0, 1, 1])
    # K = 3: row 0 is in; 2 of the 4 tied rows are picked, of which half are anomalies -> 1 TP
    assert tp_at_k(s, y, 3) == pytest.approx(1.0)


def test_row_order_does_not_change_the_result():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 5, 300).astype(float)  # heavy ties, like HBOS
    y = np.r_[np.zeros(270), np.ones(30)]  # anomalies stored in a block, as in ADBench
    perm = rng.permutation(300)
    for k in (1, 6, 15, 60):
        assert tp_at_k(s, y, k) == pytest.approx(tp_at_k(s[perm], y[perm], k))


def test_positional_tie_breaking_would_be_caught():
    """The same check fails for top-K by sort position, which rewards block-stored anomalies."""
    s, y = np.zeros(100), np.r_[np.zeros(90), np.ones(10)]
    positional = lambda s, y, k: y[np.argsort(-s, kind="stable")[:k]].sum()  # noqa: E731
    rev = np.arange(100)[::-1]
    assert positional(s, y, 10) != positional(s[rev], y[rev], 10)
    assert tp_at_k(s, y, 10) == tp_at_k(s[rev], y[rev], 10) == pytest.approx(1.0)


def test_fp_plus_tp_is_k_and_bounds():
    rng = np.random.default_rng(1)
    s, y = rng.normal(size=500), (rng.random(500) < 0.05).astype(int)
    for b in (0.02, 0.05):
        m = budget_metrics(s, y, b)
        assert m["tp_at_k"] + m["fp_at_k"] == pytest.approx(m["k"])
        assert 0 <= m["precision_at_k"] <= 1 and 0 <= m["recall_at_k"] <= 1


def test_ranking_metrics_match_sklearn_and_reward_higher_scores():
    y = np.array([0, 0, 1, 1])
    assert ranking_metrics(np.array([0.1, 0.2, 0.8, 0.9]), y) == {"pr_auc": 1.0, "roc_auc": 1.0}
    assert ranking_metrics(np.array([0.9, 0.8, 0.2, 0.1]), y)["roc_auc"] == 0.0
    s = np.array([0.3, 0.1, 0.3, 0.7])
    assert ranking_metrics(s, y)["pr_auc"] == average_precision_score(y, s)


@pytest.mark.parametrize("scores, y, k, match", [
    (np.arange(5.0), np.zeros(4), 2, "equal length"),
    (np.ones((3, 2)), np.zeros(3), 2, "1-D"),
    (np.arange(5.0), np.zeros(5), 0, "outside"),
    (np.arange(5.0), np.zeros(5), 6, "outside"),
    (np.arange(5.0), np.zeros(5), 2.0, "integer"),
    (np.arange(5.0), np.zeros(5), True, "integer"),
    (np.array([1.0, np.nan, 3.0]), np.zeros(3), 1, "scores contain 1 NaN"),
    (np.arange(3.0), np.array([0.0, np.nan, 1.0]), 1, "y contains 1 NaN"),
])
def test_tp_at_k_rejects_bad_input(scores, y, k, match):
    with pytest.raises(ValueError, match=match):
        tp_at_k(scores, y, k)


def test_tp_at_k_accepts_numpy_k_and_int_labels():
    assert tp_at_k(np.arange(4.0), np.array([0, 0, 1, 1]), np.int64(2)) == 2.0
