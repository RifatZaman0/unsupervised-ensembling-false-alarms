"""Environment checks: fail loudly if a dependency upgrade breaks the pipeline."""
import numpy as np
import pandas as pd


def test_pandas_below_3_for_adtk():
    assert int(pd.__version__.split(".")[0]) < 3


def test_adtk_level_shift_detects_planted_shift():
    from adtk.data import validate_series
    from adtk.detector import LevelShiftAD

    idx = pd.date_range("2026-01-01", periods=120, freq="D")
    rng = np.random.default_rng(0)
    s = validate_series(pd.Series(np.r_[rng.normal(100, 5, 60), rng.normal(140, 5, 60)], index=idx))
    flagged = LevelShiftAD(c=6.0, side="both", window=5).fit_detect(s)
    assert flagged.eq(True).loc["2026-02-25":"2026-03-05"].any()


def test_pyod_higher_score_is_more_anomalous():
    from pyod.models.iforest import IForest

    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    clf = IForest(random_state=0).fit(X)
    assert clf.decision_function(np.array([[8.0, 8.0, 8.0]]))[0] > np.median(clf.decision_function(X))
