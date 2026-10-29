"""Zero-variance check: the criterion matches what fusion maps to 0, the ULB day-1 loader never
reads labels or day-2 rows, and fit_scores gives exactly the training scores of score()."""
import numpy as np
import pandas as pd
import pytest

from thesis import config, data, detectors, fusion, scorecache, zero_variance


def test_zero_variance_is_the_case_fusion_sets_to_zero():
    const, varied = np.full(50, 3.2), np.r_[np.full(49, 3.2), 3.3]
    assert zero_variance.is_zero_variance(const) and not zero_variance.is_zero_variance(varied)
    _, fused = fusion.fuse(np.column_stack([const, varied]), np.array([[9.0, 3.3]]), "zscore",
                           "mean")
    z_varied = (3.3 - varied.mean()) / varied.std()
    assert fused[0] == pytest.approx(z_varied / 2)          # the constant member contributes 0


def test_ulb_day1_loader_reads_no_labels_and_no_day2(tmp_path, monkeypatch):
    n = 10
    df = pd.DataFrame({"Time": np.r_[np.linspace(0, 86_399, n), 86_400.0, 90_000.0],
                       **{f"V{i}": np.arange(n + 2, dtype=float) for i in range(1, 29)},
                       "Amount": np.arange(n + 2, dtype=float), "Class": ["bad"] * (n + 2)})
    path = tmp_path / "creditcard.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(data, "ULB_CSV", path)
    seen = []
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: seen.append(k.get("usecols")) or real(*a, **k))
    X = data.load_ulb_day1_features()
    assert X.shape == (n, 29)                               # day-2 rows (Time >= 86,400) dropped
    assert X[:, -1] == pytest.approx(np.log1p(np.arange(n)))
    assert seen and all(cols is not None and "Class" not in cols for cols in seen)


@pytest.mark.parametrize("det", ["KNN", "LOF"])
def test_fit_scores_equals_score_training_part_with_sample(det, monkeypatch):
    monkeypatch.setattr(detectors, "MAX_FIT_ROWS", 60)
    X = np.random.default_rng(1).normal(size=(150, 4))
    fitted, s = detectors.fit_scores(det, X)
    s_fit, _ = detectors.score(det, X, X[:3])
    np.testing.assert_array_equal(s, s_fit)
    assert len(fitted.decision_scores_) == 60                # fitted on the sample only
    # the sample is drawn with the fixed seed when the split has none
    rng = np.random.default_rng(config.KNN_LOF_SAMPLE_SEED)
    sample = np.sort(rng.choice(150, 60, replace=False))
    ref = detectors.make_detector(det).fit(X[sample])
    np.testing.assert_allclose(s[sample], ref.decision_scores_)


def test_subsample_constants_keep_the_cache_key():
    assert detectors.MAX_FIT_ROWS == config.KNN_LOF_MAX_FIT_ROWS == 30_000
    assert scorecache.cache_key() == zero_variance.MAIN_CACHE_KEY
