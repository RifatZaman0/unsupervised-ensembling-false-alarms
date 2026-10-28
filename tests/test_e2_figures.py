"""E2 figures: gain sign and units, the H2a correlation, the symmetric pair matrix, the triple
LaTeX table, and that every output names the commit."""
import json

import numpy as np
import pandas as pd
import pytest

from thesis import detectors, e2, e2_figures, fusion


def synthetic_per_ds(n_datasets=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_datasets):
        train = {d: rng.normal(size=80) for d in detectors.DETECTORS}
        test = {d: rng.normal(size=60) for d in detectors.DETECTORS}
        y = np.zeros(60, dtype=int)
        y[rng.choice(60, 6, replace=False)] = 1
        rows += e2.run_split(train, test, y, {"dataset": f"d{i}", "seed": 0, "n_test": 60,
                                              "anomalies_test": 6})
    return e2.per_dataset(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def tables():
    per_ds = synthetic_per_ds()
    return per_ds, e2.summary(per_ds), e2.friedman(per_ds)


def test_gain_is_false_alarms_saved_per_100_alerts(tables):
    per_ds, _, _ = tables
    prim = e2_figures.primary(per_ds)
    assert set(prim.calib) == {fusion.PRIMARY[0]} and set(prim["agg"]) == {fusion.PRIMARY[1]}
    assert len(prim) == 35 * 4 * 2
    np.testing.assert_allclose(prim.gain_best, -100 * prim.delta_fp_best_rel)
    assert ((prim.gain_best > 0) == prim.beats_best).all()     # positive gain = fewer alarms


def test_overlap_gain_rho_sign_and_within_dataset():
    rows = []
    for ds in ("a", "b", "c"):
        for j in range(10):          # gain falls as overlap rises: the H2a direction
            rows.append({"dataset": ds, "budget": 0.05, "size": 2, "calib": "minmax",
                         "agg": "mean", "train_jaccard": j / 10, "train_spearman": j / 10,
                         "gain_best": -j + (ds == "b") * 50.0, "gain_median": float(j)})
    st = e2_figures.overlap_gain_stats(pd.DataFrame(rows)).set_index(["overlap", "reference"])
    assert st.loc[("train_jaccard", "best"), "rho_within_median"] == pytest.approx(-1)
    assert st.loc[("train_jaccard", "best"), "n_datasets_rho_negative"] == 3
    assert st.loc[("train_jaccard", "median"), "rho_pooled"] == pytest.approx(1)
    assert st.loc[("train_jaccard", "best"), "n_points"] == 30
    # dataset b sits higher: the pooled rho is weaker than the within-dataset one
    assert -1 < st.loc[("train_jaccard", "best"), "rho_pooled"] < 0


def test_pair_matrix_is_symmetric_with_empty_diagonal(tables):
    per_ds, _, _ = tables
    pg = e2_figures.pair_gain(e2_figures.primary(per_ds))
    assert len(pg) == 15 * 2 and (pg.n_datasets == 4).all()
    m = e2_figures.pair_matrix(pg[pg.budget == 0.05], "mean_gain_best")
    assert np.isnan(np.diag(m.to_numpy())).all()
    off = ~np.eye(len(m), dtype=bool)
    np.testing.assert_allclose(m.to_numpy()[off], m.T.to_numpy()[off])
    r = pg[pg.budget == 0.05].iloc[0]
    assert m.loc[r.det_a, r.det_b] == pytest.approx(r.mean_gain_best)
    prim = e2_figures.primary(per_ds)
    sel = prim[(prim.combo == r.combo) & (prim.budget == 0.05)]
    assert r.mean_gain_best == pytest.approx(sel.gain_best.mean())
    assert r.n_beats_best == sel.beats_best.sum()


def test_latex_triples(tables):
    _, summ, fried = tables
    tex = e2_figures.latex_triples(summ, fried, {"commit": "c0ffee1"})
    body = [ln for ln in tex.splitlines() if ln.count("+") == 2 and ln.endswith("\\\\")]
    assert len(body) == 20
    first = summ[(summ["size"] == 3) & (summ.budget == 0.02) & (summ.calib == "minmax")
                 & (summ["agg"] == "mean")].sort_values(["avg_rank_fp_at_k", "combo"])
    assert [ln.split(" & ")[0] for ln in body] == list(first.combo)
    assert "c0ffee1" in tex and "\\begin{tabular}" in tex and "Friedman" in tex


def test_write_names_the_commit(tables, tmp_path):
    per_ds, summ, fried = tables
    meta = {"commit": "c0ffee1", "dirty": False, "cache_key": "k", "created_utc": "t",
            "protocol": {}}
    e2_figures.write(per_ds, summ, fried, tmp_path / "fig", tmp_path, meta, "beef123")
    for stem in ("overlap_vs_gain", "pair_gain_heatmap", "triple_ranking"):
        assert (tmp_path / "fig" / f"{stem}.pdf").stat().st_size > 0
        assert (tmp_path / "fig" / f"{stem}.png").stat().st_size > 0
    info = json.loads((tmp_path / "fig" / "e2_figures.meta.json").read_text(encoding="utf-8"))
    assert info["commit"] == "c0ffee1" and info["tables_commit"] == "beef123"
    assert set(info["h2a_rho_jaccard_vs_gain_best"]) == {"2%", "5%"}
    for f in ("overlap_vs_gain.csv", "pair_gain.csv"):
        assert set(pd.read_csv(tmp_path / "fig" / f)["commit"]) == {"c0ffee1"}
    assert "c0ffee1" in (tmp_path / "e2_triples.tex").read_text(encoding="utf-8")
