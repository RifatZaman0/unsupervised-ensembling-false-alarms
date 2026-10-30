"""frozen_config.json: it loads, every ensemble member is in the pool, budgets and quantiles are
valid, it agrees with the code that produced the results, and E-bench follows the declared rule."""
import copy

import pandas as pd
import pyod
import pytest

from thesis import (bootstrap, config, data, detectors, e2, e2_figures, ebench, frozen, fusion,
                    metrics, scorecache)
from thesis.paths import FROZEN_CONFIG, TABLES


@pytest.fixture(scope="module")
def cfg():
    return frozen.load()


def test_loads_and_members_are_in_the_pool(cfg):
    assert set(cfg["ensembles"]) == {"E-lit", "E-bench", "E-bench-5%", "E-all"}
    for e in cfg["ensembles"].values():
        assert set(e["members"]) <= set(detectors.DETECTORS)
    assert tuple(cfg["ensembles"]["E-all"]["members"]) == detectors.DETECTORS
    assert e2.combo_name(cfg["ensembles"]["E-lit"]["members"]) == e2_figures.E_LIT
    roles = {n: e["role"] for n, e in cfg["ensembles"].items()}
    assert roles == {"E-bench": "primary", "E-lit": "alias", "E-bench-5%": "secondary",
                     "E-all": "secondary"}
    assert cfg["ensembles"]["E-lit"]["alias_of"] == "E-bench"
    assert cfg["ensembles"]["E-lit"]["members"] == cfg["ensembles"]["E-bench"]["members"]


def test_budgets_and_quantiles(cfg):
    ad, ulb = cfg["budgets"]["adbench"], cfg["budgets"]["ulb"]
    assert tuple(ad["values"]) == metrics.ADBENCH_BUDGETS and ad["primary"] == 0.02
    assert ulb["values"] == [0.001, 0.002, 0.005] and ulb["primary"] == 0.002
    ieee = cfg["budgets"]["ieee_cis"]
    assert ieee["values"] == [0.005, 0.01, 0.02] and ieee["primary"] == 0.02
    assert cfg["thresholds"]["ulb"]["training_score_quantiles"] == [0.998, 0.995]
    assert cfg["thresholds"]["ieee_cis"]["training_score_quantiles"] == [0.99, 0.98]
    # a quantile threshold flags about (1 - q) of the rows: inside the range of the budgets
    for data in ("ulb", "ieee_cis"):
        b = cfg["budgets"][data]["values"]
        for q in cfg["thresholds"][data]["training_score_quantiles"]:
            assert min(b) <= round(1 - q, 9) <= max(b)


def test_statistics_and_hypotheses(cfg):
    boot = cfg["statistics"]["bootstrap"]
    assert boot["resamples"] == 1000 and boot["seed"] == 0
    assert "transaction-level" in boot["ulb"] and "day-block" in boot["ieee_cis"]
    assert "95%" in boot["ci"] and "entirely below 0" in cfg["statistics"]["criterion"]
    h = cfg["hypotheses"]
    assert h["primary_ensemble"] == "E-bench"
    assert (h["primary_fusion"]["calibration"], h["primary_fusion"]["aggregation"]) == fusion.PRIMARY
    assert h["H1a"]["comparator"] == "median_single" and h["H1b"]["comparator"] == "default_iforest"
    assert "degenerate" in h["H2b"]["primary"] and "E-bench-5%" in h["H2b"]["secondary"]
    assert "month-stratified day-block" in boot["ieee_cis"]
    assert "median single" in boot["per_resample"] and "oracle" in boot["per_resample"]
    assert "default_rng(seed)" in boot["rng"] and boot["code"] == "thesis.bootstrap"
    assert "adbench" not in cfg["statistics"]                  # not fraud data; still open
    assert "at most 5%" in cfg["statistics"]["multiplicity"]
    assert "sum over m = 2..6 of FP@K_m" in cfg["evaluation"]["ieee_cis_windows"]
    assert "all six detectors" in cfg["evaluation"]["pool_on_fraud_data"]
    assert "non-inferiority is not tested" in h["H2b"]["secondary"]
    assert h["H3b"]["status"].startswith("exploratory") and "freeze-v2 features" in h["H3b"]["drift_share"]
    assert "three distinct ensembles" in cfg["evaluation"]["budget_mode_runs"]
    assert "both KNN and LOF" in cfg["detectors"]["knn_lof_fallback"]
    adtk = cfg["context"]["adtk"]["detectors"]
    assert adtk["LevelShiftAD"] == {"window": 5, "c": 6.0, "side": "both",
                                    "fit": "fit_detect on days 30 onwards"}
    assert (adtk["QuantileAD"]["low"], adtk["QuantileAD"]["high"]) == (0.01, 0.99)
    assert "alerts / transactions that day" in cfg["context"]["adtk"]["series"]
    assert "months 2-6" in cfg["context"]["score_drift"]
    assert "no calibration window" in cfg["data"]["ieee_cis"]["split"]
    assert "freeze-v2" in cfg["data"]["ieee_cis"]["features_freeze_v2"]


def test_no_wording_that_goes_stale_after_the_tag():
    text = FROZEN_CONFIG.read_text(encoding="utf-8").lower()
    for word in ("candidate", "pending", "awaiting", "to be decided", "tbd", "[student]"):
        assert word not in text, word


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
    boot, h3b = cfg["statistics"]["bootstrap"], cfg["hypotheses"]["H3b"]
    assert (boot["resamples"], boot["seed"]) == (bootstrap.RESAMPLES, bootstrap.SEED)
    assert boot["ci"].startswith(f"{bootstrap.CI_LEVEL:.0%}")
    assert (h3b["ci"]["block_weeks"], h3b["ci"]["resamples"]) == (bootstrap.H3B_BLOCK_WEEKS,
                                                                  bootstrap.RESAMPLES)
    assert f"from day {data.IEEE_WEEK_START_DAY}" in h3b["windows"]
    assert f"month 6 = day {(data.IEEE_LAST_MONTH - 1) * data.IEEE_MONTH_DAYS} onwards"         in cfg["data"]["ieee_cis"]["time"]
    import adtk, evidently
    assert cfg["context"]["adtk"]["version"].startswith(f"adtk {adtk.__version__} ")
    assert f"evidently {evidently.__version__} " in h3b["drift_share"]


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
    # primary swapped with a sensitivity config (hypotheses moved along): the set of configs is
    # still complete, so only the check on the primary itself catches it
    lambda c: (c["fusion"]["sensitivity"].__setitem__(1, c["fusion"]["primary"]),
               c["fusion"].update(primary={"calibration": "zscore", "aggregation": "mean"}),
               c["hypotheses"].update(primary_fusion={"calibration": "zscore",
                                                      "aggregation": "mean"})),
    lambda c: c["ensembles"]["E-all"].update(role="main"),
    lambda c: c["ensembles"]["E-bench-5%"].update(members=["IForest", "IForest", "KNN"]),
    lambda c: c["ensembles"]["E-all"].update(role="primary"),
    lambda c: c["ensembles"]["E-lit"].update(members=["IForest", "PCA", "LOF"]),
    lambda c: c["budgets"].pop("ieee_cis"),
    lambda c: c["thresholds"].pop("ieee_cis"),
    lambda c: c["statistics"]["bootstrap"].update(resamples=0),
    lambda c: c["hypotheses"]["H1a"].update(ensemble="E-all"),
    lambda c: c.pop("hypotheses"),
    # one break per check (re-review 2026-10-29, H9)
    lambda c: c["ensembles"].update({"E-x": {"members": ["IForest", "PCA", "KNN"],
                                             "role": "alias", "alias_of": "E-lit"}}),
    lambda c: c["ensembles"]["E-lit"].update(alias_of="E-unknown"),
    lambda c: c["data"].pop("ieee_cis"),
    lambda c: c["statistics"]["bootstrap"].update(seed="0"),
    lambda c: c["statistics"]["bootstrap"].pop("ieee_cis"),
    lambda c: c["hypotheses"].update(primary_ensemble="E-all"),
    lambda c: c["hypotheses"].update(primary_fusion={"calibration": "zscore", "aggregation": "mean"}),
    lambda c: c["hypotheses"]["H1b"].update(ensemble="E-all"),
    lambda c: c["hypotheses"]["H3a"].update(ensemble="E-all"),
    lambda c: c["statistics"]["bootstrap"].update(resamples=1000.0),
    lambda c: c["ensembles"]["E-bench"].update(role="secondary"),
    # checks added with H1-H4 and the H3b / E5 decisions (2026-10-30)
    lambda c: c.pop("context"),
    lambda c: c["hypotheses"].pop("H3b"),
    lambda c: c["hypotheses"]["H3b"].update(status="confirmatory"),
    lambda c: c["hypotheses"]["H3b"]["ci"].update(block_weeks=4.0),
    lambda c: c["hypotheses"]["H3b"]["ci"].update(resamples=0),
    lambda c: c["statistics"]["bootstrap"].pop("per_resample"),
    lambda c: c["statistics"]["bootstrap"].pop("rng"),
    lambda c: c["evaluation"].pop("ieee_cis_windows"),
    lambda c: c["evaluation"].pop("pool_on_fraud_data"),
    lambda c: c["statistics"].pop("multiplicity"),
    lambda c: c["context"]["adtk"]["detectors"].pop("QuantileAD"),
    lambda c: c["context"]["adtk"]["detectors"]["QuantileAD"].update(low=0.99, high=0.01),
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
