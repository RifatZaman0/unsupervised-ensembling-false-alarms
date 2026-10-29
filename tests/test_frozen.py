"""frozen_config.json: it loads, every ensemble member is in the pool, budgets and quantiles are
valid, it agrees with the code that produced the results, and E-bench follows the declared rule."""
import copy

import pandas as pd
import pyod
import pytest

from thesis import config, data, detectors, e2, e2_figures, ebench, frozen, fusion, metrics, scorecache
from thesis.paths import TABLES


@pytest.fixture(scope="module")
def cfg():
    return frozen.load()


def test_loads_and_members_are_in_the_pool(cfg):
    assert set(cfg["ensembles"]) == {"E-lit", "E-bench", "E-bench-5%", "E-all"}
    for e in cfg["ensembles"].values():
        assert set(e["members"]) <= set(detectors.DETECTORS)
    assert tuple(cfg["ensembles"]["E-all"]["members"]) == detectors.DETECTORS
    assert e2.combo_name(cfg["ensembles"]["E-lit"]["members"]) == e2_figures.E_LIT
    assert cfg["ensembles"]["E-bench-5%"]["role"] == "secondary"


def test_budgets_and_quantiles(cfg):
    ad, ulb = cfg["budgets"]["adbench"], cfg["budgets"]["ulb"]
    assert tuple(ad["values"]) == metrics.ADBENCH_BUDGETS and ad["primary"] == 0.02
    assert ulb["values"] == [0.001, 0.002, 0.005] and ulb["primary"] == 0.002
    assert cfg["thresholds"]["ulb"]["training_score_quantiles"] == [0.998, 0.995]
    # a quantile threshold flags about (1 - q) of the rows: inside the range of the budgets
    for q in cfg["thresholds"]["ulb"]["training_score_quantiles"]:
        assert min(ulb["values"]) <= round(1 - q, 9) <= max(ulb["values"])


def test_matches_the_code(cfg):
    d = cfg["detectors"]
    assert d["params"] == detectors.PARAMS
    assert d["library"] == f"PyOD {pyod.__version__}" == f"PyOD {cfg['score_cache']['pyod']}"
    assert d["knn_lof_fit_sample"]["max_rows"] == config.KNN_LOF_MAX_FIT_ROWS
    assert d["iforest_seeds"] == list(data.ADBENCH_SEEDS)
    assert cfg["data"]["adbench"]["datasets"] == list(data.ADBENCH_15)
    assert cfg["data"]["adbench"]["split_seeds"] == list(data.ADBENCH_SEEDS)
    assert cfg["score_cache"]["key"] == scorecache.cache_key()
    assert cfg["rules"]["k_rule"] == config.Protocol().k_rule
    p = cfg["fusion"]["primary"]
    assert (p["calibration"], p["aggregation"]) == fusion.PRIMARY


def test_ebench_follows_the_rule(cfg):
    path = TABLES / "e2" / "e2_summary.csv"
    if not path.exists():
        pytest.skip("E2 results missing; run python -m thesis.e2")
    summ = pd.read_csv(path)
    assert ebench.select(summ, 0.02) == tuple(cfg["ensembles"]["E-bench"]["members"])
    assert ebench.select(summ, 0.05) == tuple(cfg["ensembles"]["E-bench-5%"]["members"])


@pytest.mark.parametrize("breaks", [
    lambda c: c["ensembles"]["E-lit"]["members"].append("OCSVM"),
    lambda c: c["ensembles"]["E-all"].update(members=["KNN"]),
    lambda c: c["budgets"]["ulb"].update(primary=0.003),
    lambda c: c["budgets"]["adbench"].update(values=[0.05, 0.02]),
    lambda c: c["thresholds"]["ulb"].update(training_score_quantiles=[0.998, 1.2]),
    lambda c: c["fusion"]["sensitivity"].pop(),
    lambda c: c["fusion"].update(primary={"calibration": "zscore", "aggregation": "mean"}),
])
def test_validate_rejects_broken_configs(cfg, breaks):
    bad = copy.deepcopy(cfg)
    breaks(bad)
    with pytest.raises(ValueError):
        frozen.validate(bad)


def summary_frame(ranks: dict[str, float], budget=0.02) -> pd.DataFrame:
    return pd.DataFrame([{"budget": budget, "size": 3, "calib": fusion.PRIMARY[0],
                          "agg": fusion.PRIMARY[1], "combo": c, "avg_rank_fp_at_k": r,
                          "mean_fp_at_k": 1.0, "n_beats_best": 0, "n_beats_median": 0,
                          "n_datasets": 15} for c, r in ranks.items()])


def test_select_takes_the_lowest_average_rank_and_refuses_a_tie():
    s = summary_frame({"A+B+C": 2.0, "A+B+D": 1.5, "B+C+D": 3.0})
    assert ebench.select(s) == ("A", "B", "D")
    # another config or size with a lower rank does not count
    other = s.assign(calib="zscore", avg_rank_fp_at_k=0.1, combo="X+Y+Z")
    assert ebench.select(pd.concat([s, other])) == ("A", "B", "D")
    with pytest.raises(ValueError, match="tie"):
        ebench.select(summary_frame({"A+B+C": 1.5, "A+B+D": 1.5 + 1e-12, "B+C+D": 3.0}))
