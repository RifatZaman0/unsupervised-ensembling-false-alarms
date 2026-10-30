"""Load and check frozen_config.json (README § Layout).

`load()` returns the parsed config after `validate()`; experiments on fraud data read their
ensembles, budgets, thresholds, statistics and hypotheses from here, so they cannot drift from
what was frozen.
"""
from __future__ import annotations

import json
from pathlib import Path

from thesis import detectors, fusion
from thesis.paths import FROZEN_CONFIG

FRAUD_DATA = ("ulb", "ieee_cis")
REQUIRED = ("detectors", "ensembles", "fusion", "budgets", "thresholds", "rules", "data",
            "metrics", "evaluation", "statistics", "hypotheses", "context", "baselines")
HYPOTHESES = ("H1a", "H1b", "H2b", "H3a", "H3b")
BOOTSTRAP_KEYS = ("resamples", "ci", "seed", "rng", "paired", "scores", "per_resample")
EVALUATION_KEYS = ("seeds", "median_single", "regret", "ieee_cis_windows", "pool_on_fraud_data")
ROLES = ("primary", "secondary", "alias")


def validate(cfg: dict) -> None:
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"missing sections {missing}")
    pool = cfg["detectors"]["pool"]
    if tuple(pool) != detectors.DETECTORS:
        raise ValueError(f"pool {pool} is not the detector pool {detectors.DETECTORS}")
    ens = cfg["ensembles"]
    for name, e in ens.items():
        m = e["members"]
        if not set(m) <= set(pool):
            raise ValueError(f"{name}: members {sorted(set(m) - set(pool))} not in the pool")
        if len(set(m)) != len(m) or len(m) < 2:
            raise ValueError(f"{name}: needs at least 2 distinct members, got {m}")
        if e["role"] not in ROLES:
            raise ValueError(f"{name}: unknown role {e['role']!r}")
        if e["role"] == "alias":
            target = ens.get(e.get("alias_of"))
            if target is None or target["role"] == "alias" or set(target["members"]) != set(m):
                raise ValueError(f"{name}: alias of {e.get('alias_of')!r} with other members")
    primary = [n for n, e in ens.items() if e["role"] == "primary"]
    if len(primary) != 1:
        raise ValueError(f"exactly one primary ensemble needed, got {primary}")
    configs = [cfg["fusion"]["primary"], *cfg["fusion"]["sensitivity"]]
    pairs = [(c["calibration"], c["aggregation"]) for c in configs]
    if pairs[0] != fusion.PRIMARY or sorted(pairs) != sorted(fusion.FUSION_CONFIGS):
        raise ValueError(f"fusion configs {pairs} do not match {fusion.FUSION_CONFIGS}")
    for data, b in cfg["budgets"].items():
        vals = b["values"]
        if not all(0 < v < 1 for v in vals) or vals != sorted(set(vals)):
            raise ValueError(f"{data}: budgets {vals} must be distinct, ascending, in (0, 1)")
        if b["primary"] not in vals:
            raise ValueError(f"{data}: primary budget {b['primary']} not among {vals}")
    for data in FRAUD_DATA:
        if data not in cfg["budgets"] or data not in cfg["thresholds"] or data not in cfg["data"]:
            raise ValueError(f"{data}: budgets, thresholds and data must all be frozen")
        q = cfg["thresholds"][data]["training_score_quantiles"]
        if not all(0.5 < v < 1 for v in q) or len(set(q)) != len(q):
            raise ValueError(f"{data}: quantiles {q} must be distinct and in (0.5, 1)")
    boot = cfg["statistics"]["bootstrap"]
    if not (isinstance(boot["resamples"], int) and boot["resamples"] > 0
            and isinstance(boot["seed"], int) and all(d in boot for d in FRAUD_DATA)):
        raise ValueError("bootstrap needs integer resamples and seed and a method per dataset")
    hyp = cfg["hypotheses"]
    p = hyp["primary_fusion"]
    if hyp["primary_ensemble"] != primary[0] or (p["calibration"], p["aggregation"]) != pairs[0]:
        raise ValueError("hypotheses must use the primary ensemble and the primary fusion")
    missing = [h for h in HYPOTHESES if h not in hyp]
    if missing:
        raise ValueError(f"hypotheses {missing} missing")
    for h in ("H1a", "H1b", "H3a"):
        if hyp[h]["ensemble"] != primary[0]:
            raise ValueError(f"{h} must test the primary ensemble {primary[0]}")
    if not hyp["H3b"]["status"].startswith("exploratory"):
        raise ValueError("H3b is exploratory: no hypothesis decision")
    ci = hyp["H3b"]["ci"]
    if not (isinstance(ci["block_weeks"], int) and ci["block_weeks"] >= 1
            and isinstance(ci["resamples"], int) and ci["resamples"] > 0):
        raise ValueError("H3b CI needs an integer block length and number of resamples")
    for key, need in (("bootstrap", BOOTSTRAP_KEYS), ("evaluation", EVALUATION_KEYS)):
        block = boot if key == "bootstrap" else cfg["evaluation"]
        missing = [k for k in need if k not in block]
        if missing:
            raise ValueError(f"{key}: {missing} missing")
    if "multiplicity" not in cfg["statistics"]:
        raise ValueError("statistics: the multiplicity rule must be stated")
    adtk = cfg["context"]["adtk"]["detectors"]
    if set(adtk) != {"LevelShiftAD", "QuantileAD"}:
        raise ValueError(f"context.adtk: detectors {sorted(adtk)}")
    qa = adtk["QuantileAD"]
    if not 0 < qa["low"] < qa["high"] < 1:
        raise ValueError(f"QuantileAD bounds {qa['low']}, {qa['high']}")


def load(path: Path = FROZEN_CONFIG) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    validate(cfg)
    return cfg
