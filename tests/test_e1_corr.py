"""Spearman agreement: matches scipy, seeds averaged before datasets, constant scores give NaN
and are left out, extreme pairs found, and the commit reaches the figure and the CSV."""
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy import stats

from thesis import e1_corr

DETS = ("A", "B", "C")


def test_spearman_matches_scipy():
    rng = np.random.default_rng(0)
    S = rng.normal(size=(50, 3))
    S[:, 1] = S[:, 0] + rng.normal(scale=0.5, size=50)
    m = e1_corr.spearman(S, DETS)
    assert m.loc["A", "B"] == pytest.approx(stats.spearmanr(S[:, 0], S[:, 1])[0])
    assert np.allclose(m.to_numpy(), m.to_numpy().T)


def test_constant_scores_give_nan():
    S = np.column_stack([np.arange(10.0), np.ones(10), np.arange(10.0)[::-1]])
    m = e1_corr.spearman(S, DETS)
    assert np.isnan(m.loc["A", "B"])
    assert m.loc["A", "C"] == pytest.approx(-1.0)


def runs_frame(values: dict) -> pd.DataFrame:
    """values: {(dataset, seed): {(a, b): rho}}"""
    return pd.DataFrame([{"dataset": d, "seed": s, "det_a": a, "det_b": b, "rho": r}
                         for (d, s), pairs in values.items() for (a, b), r in pairs.items()])


def test_seeds_averaged_within_dataset_before_datasets():
    # d1 has 3 seeds, d2 one; a pooled mean would weight d1 three times
    runs = runs_frame({("d1", 0): {("A", "B"): 0.9}, ("d1", 1): {("A", "B"): 0.9},
                       ("d1", 2): {("A", "B"): 0.9}, ("d2", 0): {("A", "B"): 0.1}})
    avg = e1_corr.average(e1_corr.per_dataset(runs), dets=("A", "B"))
    assert avg.loc["A", "B"] == pytest.approx(0.5)
    assert avg.loc["B", "A"] == avg.loc["A", "B"]
    assert avg.loc["A", "A"] == 1.0


def test_nan_runs_left_out_and_counted():
    runs = runs_frame({("d1", 0): {("A", "B"): 0.4}, ("d1", 1): {("A", "B"): np.nan}})
    per_ds = e1_corr.per_dataset(runs)
    assert per_ds.loc[0, "rho"] == pytest.approx(0.4)
    assert per_ds.loc[0, "n_runs"] == 1


def test_extreme_pairs():
    avg = pd.DataFrame([[1, 0.2, 0.8], [0.2, 1, -0.1], [0.8, -0.1, 1]],
                       index=DETS, columns=DETS)
    lo, hi = e1_corr.extreme_pairs(avg)
    assert lo == ("B", "C", -0.1)
    assert hi == ("A", "C", 0.8)


def test_write_records_commit(tmp_path):
    pairs = {("A", "B"): 0.3, ("A", "C"): 0.7, ("B", "C"): -0.2}
    runs = runs_frame({(d, s): pairs for d in ("d1", "d2") for s in (0, 1)})
    meta = {"commit": "abc1234", "dirty": False}
    avg, info = e1_corr.write(runs, tmp_path, meta, dets=DETS)
    assert info["most_correlated"] == {"pair": ("A", "C"), "rho": 0.7}
    assert json.loads((tmp_path / "spearman.meta.json").read_text())["commit"] == "abc1234"
    assert set(pd.read_csv(tmp_path / "spearman_per_dataset.csv")["commit"]) == {"abc1234"}
    assert "abc1234" in Image.open(tmp_path / "spearman_test_scores.png").info["Comment"]
    assert b"abc1234" in (tmp_path / "spearman_test_scores.pdf").read_bytes()
