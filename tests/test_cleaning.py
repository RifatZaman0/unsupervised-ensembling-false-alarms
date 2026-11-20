"""Tests for committee cleaning (label-free row selection on the training split)."""
import inspect

import numpy as np
import pytest

from thesis import cleaning


def members(seed, n=2000, rate=0.05, strengths=(3.0, 2.0, 1.0)):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    S = rng.standard_normal((len(strengths), n)) + np.asarray(strengths)[:, None] * y
    return {f"m{i}": S[i] for i in range(len(strengths))}, y


def test_removal_mask_removes_the_top_k_without_ties():
    s = np.arange(100.0)
    m = cleaning.removal_mask(s, 0.05)
    assert m.sum() == 5 and m[-5:].all() and not m[:-5].any()


def test_rows_tied_at_the_cut_are_kept():
    s = np.array([9.0, 8.0, 7.0, 7.0, 7.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    m = cleaning.removal_mask(s, 0.3)  # k = 3, the 3rd largest is 7 (tied)
    assert m.tolist() == [True, True] + [False] * 8


def test_removal_never_exceeds_k_and_ignores_row_order():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 20, 500).astype(float)  # many ties
    m = cleaning.removal_mask(s, 0.05)
    assert m.sum() <= 25
    perm = rng.permutation(500)
    assert np.array_equal(cleaning.removal_mask(s[perm], 0.05), m[perm])


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1])
def test_removal_rejects_bad_fraction(q):
    with pytest.raises(ValueError):
        cleaning.removal_mask(np.arange(10.0), q)


@pytest.mark.parametrize("bad", [np.array([1.0, np.nan]), np.array([]), np.ones((2, 2))])
def test_removal_rejects_bad_scores(bad):
    with pytest.raises(ValueError):
        cleaning.removal_mask(bad, 0.1)


def test_minmax_train_range_and_constant_member():
    assert cleaning.minmax_train(np.array([2.0, 4.0, 6.0])).tolist() == [0.0, 0.5, 1.0]
    assert cleaning.minmax_train(np.full(4, 3.0)).tolist() == [0.0] * 4


def test_consensus_is_the_mean_of_minmax_scores():
    sc, _ = members(1)
    expected = np.mean([cleaning.minmax_train(v) for v in sc.values()], axis=0)
    assert cleaning.consensus_scores(sc) == pytest.approx(expected)


def test_consensus_invariant_to_positive_affine_rescaling_of_a_member():
    sc, _ = members(2)
    sc2 = dict(sc, m1=5.0 * sc["m1"] + 3.0)
    assert cleaning.consensus_scores(sc2) == pytest.approx(cleaning.consensus_scores(sc))


def test_committee_mask_is_permutation_equivariant_in_members():
    sc, _ = members(3)
    a = cleaning.committee_keep_mask(sc, 0.05)
    b = cleaning.committee_keep_mask(dict(reversed(list(sc.items()))), 0.05)
    assert np.array_equal(a, b)


def test_committee_removes_more_true_anomalies_than_chance():
    sc, y = members(4)
    keep = cleaning.committee_keep_mask(sc, 0.05)
    removed = ~keep
    assert y[removed].mean() > 5 * y.mean()


def test_self_mask_uses_only_its_own_scores():
    sc, _ = members(5)
    assert np.array_equal(cleaning.self_keep_mask(sc["m0"], 0.05), ~cleaning.removal_mask(sc["m0"], 0.05))


def test_masks_take_no_label_argument():
    for fn in (cleaning.committee_keep_mask, cleaning.self_keep_mask, cleaning.removal_mask):
        assert not any(p in ("y", "labels", "y_train", "y_test") for p in inspect.signature(fn).parameters)


def test_committee_needs_two_members_of_equal_length():
    with pytest.raises(ValueError):
        cleaning.consensus_scores({"a": np.arange(5.0)})
    with pytest.raises(ValueError):
        cleaning.consensus_scores({"a": np.arange(5.0), "b": np.arange(6.0)})
