"""Rules for fuse(): calibration parameters come from training scores only, the 6 configs
behave as specified, and a test row's fused score depends on that row alone."""
import numpy as np
import pytest

from thesis.fusion import AGGREGATIONS, CALIBRATIONS, FUSION_CONFIGS, PRIMARY, Calibrator, fuse


def members(seed=0, n_train=500, n_test=200, m=3):
    rng = np.random.default_rng(seed)
    scale = np.array([1.0, 50.0, 0.01])[:m]  # members on very different raw scales
    return rng.gamma(2.0, size=(n_train, m)) * scale, rng.gamma(2.0, size=(n_test, m)) * scale


def leaky_fuse(S_train, S_test, calib, agg):
    """Deliberately wrong: calibrators fitted on train + test scores."""
    S_all = np.r_[S_train, S_test]
    cals = [Calibrator.fit(S_all[:, j], calib) for j in range(S_all.shape[1])]
    C = np.column_stack([c.transform(S_test[:, j]) for j, c in enumerate(cals)])
    return None, C.mean(axis=1) if agg == "mean" else C.max(axis=1)


def depends_on_other_test_rows(fuse_fn, calib, agg):
    """True if changing other test rows changes the fused score of test row 0."""
    S_train, S_test = members()
    changed = S_test.copy()
    changed[1:] *= 3.0
    return not np.isclose(fuse_fn(S_train, S_test, calib, agg)[1][0],
                          fuse_fn(S_train, changed, calib, agg)[1][0])


def test_six_configs_and_primary():
    assert len(FUSION_CONFIGS) == 6 and PRIMARY == ("minmax", "mean")
    assert set(FUSION_CONFIGS) == {(c, a) for c in CALIBRATIONS for a in AGGREGATIONS}


@pytest.mark.parametrize("calib,agg", FUSION_CONFIGS)
def test_calibration_uses_training_scores_only(calib, agg):
    assert not depends_on_other_test_rows(fuse, calib, agg)


@pytest.mark.parametrize("calib", CALIBRATIONS)
def test_leaky_calibration_is_caught(calib):
    assert depends_on_other_test_rows(leaky_fuse, calib, "mean")


def test_minmax_maps_training_range_to_unit_interval_without_clipping():
    c = Calibrator.fit(np.array([2.0, 4.0, 6.0]), "minmax")
    assert np.allclose(c.transform(np.array([2.0, 6.0, 8.0, 0.0])), [0.0, 1.0, 1.5, -0.5])


def test_zscore_uses_training_mean_and_std():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    c = Calibrator.fit(s, "zscore")
    assert np.allclose(c.transform(s).mean(), 0) and np.allclose(c.transform(s).std(), 1)
    assert np.isclose(c.transform(np.array([2.5 + s.std()]))[0], 1.0)


def test_percentile_is_share_of_training_scores_at_or_below():
    c = Calibrator.fit(np.array([1.0, 2.0, 2.0, 5.0]), "percentile")
    assert np.allclose(c.transform(np.array([0.0, 1.0, 2.0, 3.0, 5.0, 9.0])),
                       [0.0, 0.25, 0.75, 0.75, 1.0, 1.0])


@pytest.mark.parametrize("calib", CALIBRATIONS)
def test_constant_training_scores_stay_finite(calib):
    c = Calibrator.fit(np.full(10, 3.0), calib)
    assert np.isfinite(c.transform(np.array([1.0, 3.0, 7.0]))).all()


def test_mean_and_max_aggregate_calibrated_members():
    S_train = np.array([[0.0, 10.0], [1.0, 20.0]])
    S_test = np.array([[0.5, 20.0], [1.0, 10.0]])
    assert np.allclose(fuse(S_train, S_test, "minmax", "mean")[1], [0.75, 0.5])
    assert np.allclose(fuse(S_train, S_test, "minmax", "max")[1], [1.0, 1.0])
    f_train, _ = fuse(S_train, S_test, "minmax", "mean")
    assert np.allclose(f_train, [0.0, 1.0])


@pytest.mark.parametrize("calib", CALIBRATIONS)
def test_single_member_keeps_its_ranking(calib):
    S_train, S_test = members(m=1)
    fused = fuse(S_train, S_test, calib, "mean")[1]
    order = np.argsort(S_test[:, 0], kind="stable")
    assert (np.diff(fused[order]) >= 0).all()


def test_scale_does_not_let_one_member_dominate():
    """After min-max, multiplying one member's raw scores by 1000 changes nothing."""
    S_train, S_test = members()
    k = np.array([1.0, 1000.0, 1.0])
    assert np.allclose(fuse(S_train, S_test)[1], fuse(S_train * k, S_test * k)[1])


def test_rejects_mismatched_members():
    with pytest.raises(ValueError):
        fuse(np.zeros((5, 2)), np.zeros((5, 3)))
