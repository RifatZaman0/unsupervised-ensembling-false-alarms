"""Tests for spectral weighting (label-free) and the label-tuned ceiling (diagnostic)."""
import inspect
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from thesis import ceiling, data, metrics, spectral


def synthetic(seed, n=6000, rate=0.03, strengths=(3.0, 2.0, 1.2, 0.6), shared_noise=None):
    """Members score label * strength + noise. shared_noise=(i, j, rho) makes i, j share false alarms."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    noise = rng.standard_normal((len(strengths), n))
    if shared_noise is not None:
        i, j, rho = shared_noise
        z = rng.standard_normal(n)
        noise[i] = math.sqrt(rho) * z + math.sqrt(1 - rho) * noise[i]
        noise[j] = math.sqrt(rho) * z + math.sqrt(1 - rho) * noise[j]
    S = noise + np.asarray(strengths)[:, None] * y
    return {f"m{i}": S[i] for i in range(len(strengths))}, y


# --- spectral.py -----------------------------------------------------------------------------

def test_topk_membership_sums_to_k_and_splits_ties():
    a = spectral.topk_membership(np.array([5.0, 3.0, 3.0, 3.0, 1.0]), 2)
    assert a.sum() == pytest.approx(2.0)
    assert a.tolist() == pytest.approx([1.0, 1 / 3, 1 / 3, 1 / 3, 0.0])


@pytest.mark.parametrize("bad", [np.array([1.0, np.nan]), np.array([]), np.ones((2, 2))])
def test_topk_membership_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        spectral.topk_membership(bad, 1)


def test_excess_overlap_matches_brute_force_without_ties():
    scores, _ = synthetic(0)
    names, C, k, n = spectral.excess_overlap(scores, 0.02)
    tops = {nm: set(np.argsort(-scores[nm])[:k]) for nm in names}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                assert C[i, j] == pytest.approx(len(tops[a] & tops[b]) - k * k / n)


def test_rank_one_fit_recovers_exact_rank_one_offdiagonal():
    v_true = np.array([4.0, 2.0, 1.0, 0.5, 3.0])
    C = np.outer(v_true, v_true)
    np.fill_diagonal(C, 99.0)  # the diagonal must be ignored
    v, lam = spectral.rank_one_offdiag(C)
    assert v == pytest.approx(v_true, rel=1e-6)


def test_weights_rank_members_by_strength_under_independence():
    rhos, hits = [], 0
    for seed in range(20):
        scores, y = synthetic(seed)
        res = spectral.spectral_weights(scores, 0.02)
        assert not res.fallback
        k_te = metrics.budget_k(0.02, y.size)
        quality = [-ceiling.fp_at_k(scores[nm], y, k_te) for nm in res.names]
        rhos.append(spearmanr(res.v, quality).statistic)
        hits += res.selected == "m0"
    assert np.mean(rhos) >= 0.9
    assert hits >= 18


def test_weights_have_no_label_or_test_argument():
    params = list(inspect.signature(spectral.spectral_weights).parameters)
    assert params == ["train_scores", "budget", "k_rule"]


def test_invariant_to_monotone_transforms_of_each_member():
    scores, _ = synthetic(1)
    transformed = {nm: (np.exp(s) if i % 2 else 3 * s - 7) for i, (nm, s) in enumerate(scores.items())}
    a = spectral.spectral_weights(scores, 0.02)
    b = spectral.spectral_weights(transformed, 0.02)
    assert a.weights == pytest.approx(b.weights)


def test_permuting_members_permutes_weights():
    scores, _ = synthetic(2)
    a = spectral.spectral_weights(scores, 0.05)
    rev = dict(reversed(list(scores.items())))
    b = spectral.spectral_weights(rev, 0.05)
    assert dict(zip(a.names, a.weights)) == pytest.approx(dict(zip(b.names, b.weights)))


def test_weights_non_negative_and_sum_to_one():
    scores, _ = synthetic(3)
    res = spectral.spectral_weights(scores, 0.02)
    assert np.all(res.weights >= 0) and res.weights.sum() == pytest.approx(1.0)


def test_member_worse_than_random_gets_zero_weight():
    scores, _ = synthetic(8, strengths=(2.5, 2.0, 1.5, -2.0))
    res = spectral.spectral_weights(scores, 0.02)
    assert not res.fallback
    assert res.v[3] < 0 and res.weights[3] == 0.0


def test_single_positive_member_falls_back_to_equal_weights():
    scores, _ = synthetic(9, strengths=(3.0, -1.5, -1.5))
    res = spectral.spectral_weights(scores, 0.02)
    assert res.fallback and res.reason == "fewer than two positive weights"
    assert res.weights == pytest.approx([1 / 3] * 3)


def test_constant_members_fall_back_to_equal_weights():
    scores = {f"m{i}": np.zeros(500) for i in range(3)}
    res = spectral.spectral_weights(scores, 0.02)
    assert res.fallback and res.reason == "leading eigenvalue <= 0"
    assert res.weights == pytest.approx([1 / 3] * 3)


def test_two_members_are_refused():
    scores, _ = synthetic(4, strengths=(2.0, 1.0))
    with pytest.raises(ValueError):
        spectral.spectral_weights(scores, 0.02)


def test_shared_false_alarms_inflate_the_dependent_pair():
    """Documents the known failure mode: members that share false alarms get too much weight."""
    gaps = []
    for seed in range(10):
        ind, _ = synthetic(seed)
        dep, _ = synthetic(seed, shared_noise=(2, 3, 0.6))
        wi = spectral.spectral_weights(ind, 0.02).weights
        wd = spectral.spectral_weights(dep, 0.02).weights
        gaps.append((wd[2] + wd[3]) - (wi[2] + wi[3]))
    assert np.mean(gaps) > 0.05


def test_weighted_fusion_is_the_weighted_mean():
    scores, _ = synthetic(5)
    res = spectral.spectral_weights(scores, 0.02)
    fused = spectral.weighted_fusion(scores, res)
    expected = sum(w * scores[nm] for nm, w in zip(res.names, res.weights))
    assert fused == pytest.approx(expected)


# --- ceiling.py ------------------------------------------------------------------------------

def test_fp_at_k_splits_ties_in_expectation():
    s = np.array([3.0, 2.0, 2.0, 1.0])
    y = np.array([1, 0, 1, 0])
    assert ceiling.fp_at_k(s, y, 2) == pytest.approx(0.5)


@pytest.mark.parametrize("m,step,size", [(3, 0.05, 231), (6, 0.1, 3003), (2, 0.5, 3)])
def test_simplex_grid_size_and_rows(m, step, size):
    g = ceiling.simplex_grid(m, step)
    assert g.shape == (size, m)
    assert np.all(g >= 0) and g.sum(axis=1) == pytest.approx(np.ones(size))
    assert len({tuple(np.round(r, 9)) for r in g}) == size


def test_tuned_weights_never_worse_than_best_member_on_training():
    scores, y = synthetic(6, strengths=(2.0, 1.5, 1.0))
    X = np.vstack(list(scores.values()))
    k = metrics.budget_k(0.05, y.size)
    w = ceiling.tune_weights(X, y, k, ceiling.simplex_grid(3, 0.1))
    best_member = min(ceiling.fp_at_k(x, y, k) for x in X)
    assert ceiling.fp_at_k(w @ X, y, k) <= best_member + 1e-9


def test_tuning_prefers_a_single_member_when_tied():
    X = np.vstack([np.arange(100.0), np.arange(100.0), np.arange(100.0)])  # identical members
    y = (np.arange(100) >= 95).astype(int)
    grid = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # first row is not a corner
    w = ceiling.tune_weights(X, y, 5, grid)
    assert w.tolist() == [1.0, 0.0, 0.0]


def test_label_selected_member_is_picked_on_training_not_test():
    rng = np.random.default_rng(0)
    y_tr = np.r_[np.ones(10), np.zeros(190)].astype(int)
    y_te = y_tr.copy()
    good, noise = y_tr * 10.0 + rng.random(200), rng.random(200)
    cal_train = np.vstack([good, noise, noise[::-1]])  # member 0 is best on training rows
    cal_test = np.vstack([noise, good, noise[::-1]])  # member 1 is best on test rows
    out = ceiling.ceiling_run(cal_train, y_tr, cal_test, y_te, 0.05, 0.5)
    assert out["weights"].tolist() == [1.0, 0.0, 0.0]  # tuned on training labels, not test labels
    assert out["fp_label_selected_member"] == pytest.approx(ceiling.fp_at_k(noise, y_te, 10))
    assert out["fp_oracle_best_member"] == pytest.approx(0.0)


def test_ceiling_run_reports_all_comparators():
    scores, y = synthetic(7, strengths=(2.5, 1.0, 0.5))
    X = np.vstack(list(scores.values()))
    tr, te = slice(0, 3600), slice(3600, None)
    out = ceiling.ceiling_run(X[:, tr], y[tr], X[:, te], y[te], 0.02, 0.05)
    assert out["fp_oracle_best_member"] <= out["fp_median_member"]
    assert out["weights"].sum() == pytest.approx(1.0)


# --- real data (cached ADBench split) and the Step 1 runner ----------------------------------

needs_adbench = pytest.mark.skipif(not any(data.ADBENCH_DIR.glob("*.npz")),
                                   reason="ADBench not downloaded")


@needs_adbench
def test_real_split_weights_are_valid_and_deterministic():
    from thesis import detectors, scorecache, sw
    if not all(scorecache.cache_path("cardio", 0, d).exists() for d in detectors.DETECTORS):
        pytest.skip("score cache for cardio seed 0 missing")
    _, train, _ = sw.load_split("cardio", 0)
    for b in metrics.ADBENCH_BUDGETS:
        a = spectral.spectral_weights(train, b)
        again = spectral.spectral_weights({d: train[d].copy() for d in train}, b)
        assert a.names == tuple(detectors.DETECTORS)
        assert np.all(np.isfinite(a.weights)) and np.all(np.isfinite(a.v))
        assert np.all(a.weights >= 0) and a.weights.sum() == pytest.approx(1.0, abs=1e-12)
        assert a.k == metrics.budget_k(b, len(train["IForest"]))
        assert np.array_equal(a.weights, again.weights) and np.array_equal(a.v, again.v)


def _per_ds(outcomes, member_set="IForest+PCA+ECOD+HBOS+KNN+LOF", budget=0.02):
    from thesis import sw
    rows = []
    for i, (tuned, sel) in enumerate(outcomes):
        for seed, noise in enumerate((-0.9, 0.3, 0.6)):  # seed means = the given values; medians differ
            rows.append({"dataset": f"d{i}", "seed": seed, "member_set": member_set,
                         "size": len(member_set.split("+")), "budget": budget,
                         "k_test": 10, "max_weight": 1.0,
                         "fp_tuned_weights": tuned + noise, "fp_label_selected_member": sel - noise,
                         "fp_equal_weights": 5.0, "fp_oracle_best_member": 1.0,
                         "fp_median_member": 4.0})
    return sw.per_dataset(pd.DataFrame(rows))



def test_per_dataset_compares_seed_means_and_counts_ties():
    per = _per_ds([(3.0, 4.0), (4.0, 3.0), (2.0, 2.0), (1.0, 1.0 + 1e-12)])
    assert per["outcome"].tolist() == ["tuned fewer", "selected fewer", "tie", "tie"]
    assert per["fp_tuned_weights"].tolist() == pytest.approx([3.0, 4.0, 2.0, 1.0])
    assert per["n_seeds"].tolist() == [3] * 4


def test_calibration_is_minmax_fitted_on_training_rows():
    from thesis import sw
    train = {"a": np.array([2.0, 4.0, 6.0]), "b": np.array([-1.0, 0.0, 3.0]), "c": np.array([5.0, 5.0, 9.0])}
    test = {"a": np.array([8.0, 2.0]), "b": np.array([7.0, -1.0]), "c": np.array([1.0, 9.0])}
    cal_tr, cal_te = sw.calibrated(train, test, ("a", "b", "c"))
    assert cal_tr.tolist() == [[0.0, 0.5, 1.0], [0.0, 0.25, 1.0], [0.0, 0.0, 1.0]]
    assert cal_te.tolist() == [[1.5, 0.0], [2.0, 0.0], [-1.0, 1.0]]  # not clipped


@pytest.mark.parametrize("outcomes,passed", [
    ([(3, 4), (3, 4), (4, 3)], True),            # 2 vs 1
    ([(3, 4), (4, 3)], False),                   # 1 vs 1: not more
    ([(3, 4), (2, 2), (2, 2), (4, 3), (4, 3)], False),
    ([(3, 4), (2, 2), (2, 2)], True),            # ties are not counted
])
def test_gate_needs_more_datasets_for_the_tuned_weights(outcomes, passed):
    from thesis import sw
    gate = sw.gate_table(_per_ds(outcomes))
    assert gate["is_gate"].tolist() == [True]
    assert sw.gate_passes(gate) is passed


def test_gate_reads_only_e_all_at_two_percent():
    from thesis import sw
    per = pd.concat([_per_ds([(4, 3)]),                                    # gate row: fails
                     _per_ds([(3, 4)] * 3, budget=0.05),                   # E-all at 5%
                     _per_ds([(3, 4)] * 3, member_set="IForest+PCA+KNN")])  # a triple at 2%
    gate = sw.gate_table(per)
    assert len(gate) == 3 and gate["is_gate"].sum() == 1
    assert gate.loc[gate["is_gate"], "budget"].item() == 0.02
    assert gate.loc[gate["is_gate"], "member_set"].item() == sw.GATE["member_set"]
    assert sw.gate_passes(gate) is False


def test_step1_refuses_dirty_tree_missing_tag_and_foreign_head():
    from thesis import sw
    clean = {"commit": "abc", "dirty": False}
    with pytest.raises(SystemExit, match="uncommitted"):
        sw.check_can_write({**clean, "dirty": True}, "t", lambda c: True)
    with pytest.raises(SystemExit, match="does not exist"):
        sw.check_can_write(clean, None, lambda c: True)
    with pytest.raises(SystemExit, match="does not contain"):
        sw.check_can_write(clean, "t", lambda c: False)
    sw.check_can_write(clean, "t", lambda c: True)


def test_equal_weight_check_flags_any_difference(tmp_path):
    from thesis import sw
    e2 = pd.DataFrame({"dataset": ["a", "a"], "seed": [0, 0], "combo": ["x", "x"],
                       "size": [3, 3], "calib": ["minmax", "minmax"], "agg": ["mean", "mean"],
                       "budget": [0.02, 0.05], "fp_at_k": [0.1 + 0.2, 7.0]})
    e2.to_csv(tmp_path / "e2.csv", index=False)
    eq = e2[["dataset", "seed", "combo", "budget"]].assign(fp_equal_weights=[0.1 + 0.2, 7.0])
    assert sw.compare_with_e2(eq, tmp_path / "e2.csv") == {
        "n_compared": 2, "n_unmatched": 0, "n_not_identical": 0, "max_abs_diff": 0.0}
    off = eq.assign(fp_equal_weights=[0.3, 7.0])  # 1 ulp away from 0.1 + 0.2
    assert sw.compare_with_e2(off, tmp_path / "e2.csv")["n_not_identical"] == 1
    assert sw.compare_with_e2(eq.iloc[:1], tmp_path / "e2.csv")["n_unmatched"] == 1
