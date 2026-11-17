"""Load and check frozen_stress_v3.json (freeze-v3: RQ4 for the single IEEE-CIS run).

`validate(cfg, v1)` checks the file on its own and against freeze-v1 (frozen_config.json);
`check_code_matches(cfg)` checks that the code uses the frozen values (the runner calls both).
`validate(..., for_tag=True)` also refuses a draft: the draft_status block and any status word
that would go stale after the tag.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from thesis import data, fragility, frozen, fusion, guard, ieee_stress, robust_fusion
from thesis.paths import REPO

FROZEN_V3 = REPO / "frozen_stress_v3.json"
REQUIRED = ("name", "scope", "builds_on", "development_data", "rq4", "robust_fusion", "statistic",
            "guard", "pre_deployment", "evaluation", "hypotheses", "breaks", "if_no_member_breaks",
            "removed", "runner")
HYPOTHESES = ("H4r", "H4r_cost", "H4a", "H4b", "H4c", "H4d", "H4e", "bound_check", "drift_slope", "multiplicity")
DECIDED = ("H4r", "H4a", "H4b", "H4c")
STALE = re.compile(r"\b(pending|candidate|proposed|draft|awaiting|tentative|to be (?:decided|confirmed))\b",
                   re.IGNORECASE)
RQ4_TEXT = ("When one ensemble member breaks under temporal drift, can a label-free ensemble keep "
            "its false alarms under control through its choice of calibration and fusion, without "
            "detecting the break, and how does this compare with detecting and excluding the member "
            "(run-time guard)?")


def validate(cfg: dict, v1: dict, for_tag: bool = False) -> None:
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"missing sections {missing}")
    if cfg["name"] != "freeze-v3":
        raise ValueError(f"name {cfg['name']!r}")
    if cfg["builds_on"] != {"freeze-v1": "frozen_config.json", "freeze-v2": "frozen_features_v2.json"}:
        raise ValueError("builds_on must name the freeze-v1 and freeze-v2 files")
    if cfg["rq4"]["text"] != RQ4_TEXT:
        raise ValueError("rq4.text is not the agreed RQ4 wording")
    g = cfg["guard"]
    if g["side"] != "one-sided" or g["basis"] != "absolute":
        raise ValueError(f"guard must be one-sided and absolute, got {g['side']}, {g['basis']}")
    if not (isinstance(g["c"], int) and g["c"] > 1):
        raise ValueError(f"guard.c must be an integer > 1 (exact comparison), got {g['c']!r}")
    if not (isinstance(g["min_members"], int) and g["min_members"] >= 1):
        raise ValueError(f"guard.min_members {g['min_members']!r}")
    if "r = c passes" not in cfg["statistic"]["ties"] or "Fraction" not in cfg["statistic"]["ties"]:
        raise ValueError("statistic.ties must say r = c passes and how the tie is computed")
    if not cfg["statistic"]["q"].startswith("1 - budget"):
        raise ValueError("statistic.q must be 1 - budget")
    ens = v1["ensembles"]
    if not set(g["ensembles"]) <= set(ens) or any(ens[e]["role"] == "alias" for e in g["ensembles"]):
        raise ValueError(f"guard.ensembles {g['ensembles']} must be distinct freeze-v1 ensembles")
    if not {"E-all", "E-bench"} <= set(g["ensembles"]):
        raise ValueError("guard.ensembles must include E-all (H4b) and E-bench (H4c)")
    if "developed on ulb" not in cfg["development_data"].lower():
        raise ValueError("development_data must say what was developed on ULB")
    rf = cfg["robust_fusion"]
    if not (rf["calibration"].startswith("train-percentile") and rf["aggregation"].startswith("median")):
        raise ValueError("robust_fusion is train-percentile + median")
    if rf["label"] != "developed on ULB":
        raise ValueError("robust_fusion must be labelled developed on ULB")
    p = cfg["pre_deployment"]
    fit, scored = p["fit_days"], p["scored_days"]
    if not (0 <= fit[0] <= fit[1] < scored[0] <= scored[1] < data.IEEE_MONTH_DAYS):
        raise ValueError(f"pre_deployment days {fit} -> {scored} must be ordered inside month 1")
    hyp = cfg["hypotheses"]
    missing = [h for h in HYPOTHESES if h not in hyp]
    if missing:
        raise ValueError(f"hypotheses {missing} missing")
    for h in DECIDED:
        if "supported_if" not in hyp[h]:
            raise ValueError(f"{h}: supported_if missing")
    for h in ("H4d", "drift_slope"):
        if not hyp[h]["status"].startswith("exploratory"):
            raise ValueError(f"{h} is exploratory: no decision")
    if "appendix only" not in hyp["H4d"]["status"]:
        raise ValueError("H4d is reported in the appendix only")
    for h in ("H4e", "bound_check"):
        if not hyp[h]["status"].startswith("descriptive"):
            raise ValueError(f"{h} is descriptive: no decision")
    if "no month-1 labels" not in hyp["H4e"]["status"]:
        raise ValueError("H4e reads no month-1 label")
    r = hyp["H4r"]
    if r["method"] != "train-percentile + median" or not r["comparator"].startswith("min-max + mean"):
        raise ValueError("H4r compares train-percentile + median with min-max + mean")
    if not r["ensembles"].startswith("all 20 three-member ensembles"):
        raise ValueError("H4r averages over all 20 three-member ensembles")
    if "paired per triple" not in r["statistic"] or "|log r_month|" not in r["statistic"]:
        raise ValueError("H4r: paired per-triple change in mean monthly |log r|")
    if "entirely below 0" not in r["supported_if"]:
        raise ValueError("H4r: CI of dD_bar entirely below 0")
    if not all(e in r["descriptive"] for e in ("E-bench", "E-bench-5%", "E-all")):
        raise ValueError("H4r: E-bench, E-bench-5% and E-all are descriptive rows")
    ieee = v1["budgets"]["ieee_cis"]
    if r["q"] != round(1 - ieee["primary"], 12):
        raise ValueError("H4r uses q = 1 - primary budget")
    cost = hyp["H4r_cost"]
    if "budget 0.02, daily K" not in cost["statistic"] or "mean over the 20 triples" not in cost["statistic"]:
        raise ValueError("H4r_cost: 2% primary budget, daily K, mean over the 20 triples")
    if cost["passes_if"] != "the upper bound of the 95% day-block CI of R is <= 0.05":
        raise ValueError("H4r_cost passes if the upper CI bound of R is <= +5%")
    if "set before IEEE-CIS" not in cost["margin"] or "ADBench" not in cost["margin"]:
        raise ValueError("H4r_cost: the margin is justified from ADBench and set before IEEE-CIS")
    for h in ("H4b", "H4c"):
        if hyp[h]["budget"] != ieee["primary"]:
            raise ValueError(f"{h}: budget {hyp[h]['budget']} is not the IEEE-CIS primary {ieee['primary']}")
    if hyp["H4b"]["method"] != "E-guard(E-all)" or hyp["H4b"]["comparator"] != "static E-all":
        raise ValueError("H4b compares E-guard(E-all) with static E-all")
    if hyp["H4c"]["method"] != "E-guard(E-bench)" or hyp["H4c"]["comparator"] != "static E-bench":
        raise ValueError("H4c compares E-guard(E-bench) with static E-bench")
    if "entirely below 0" not in hyp["H4b"]["supported_if"]:
        raise ValueError("H4b uses the freeze-v1 criterion (CI entirely below 0)")
    if "entirely above 0" not in hyp["H4a"]["supported_if"]:
        raise ValueError("H4a: CI of rho entirely above 0")
    q_primary = round(1 - ieee["primary"], 12)
    for h in ("H4a", "H4d"):
        if f"q = {q_primary}" not in hyp[h]["statistic"]:
            raise ValueError(f"{h} must use q = {q_primary} (1 - primary budget)")
    ev = cfg["evaluation"]
    if "K_d = ceil(budget x n_d)" not in ev["daily_budget"] or "default_rng(0)" not in ev["bootstrap"]:
        raise ValueError("evaluation must state the daily K and the freeze-v1 resamples")
    boot = v1["statistics"]["bootstrap"]
    if boot["resamples"] != 1000 or boot["seed"] != 0 or not boot["ieee_cis"].startswith("month-stratified day-block"):
        raise ValueError("freeze-v1 IEEE-CIS bootstrap is not the one this file reuses")
    for k in ("H4r", "H4r_cost", "H4b", "H4c", "H4d", "H4e"):
        if k not in cfg["if_no_member_breaks"]:
            raise ValueError(f"if_no_member_breaks must say what is reported for {k}")
    if not {"relative_to_median", "sead"} <= set(cfg["removed"]):
        raise ValueError("removed must list the relative-to-median rules and SEAD")
    if for_tag:
        if "draft_status" in cfg:
            raise ValueError("draft_status must be removed before the tag")
        text = json.dumps(cfg)
        m = STALE.search(text)
        if m:
            raise ValueError(f"status word {m.group(0)!r} would go stale after the tag")


def check_code_matches(cfg: dict) -> None:
    """The constants the code uses are the frozen ones."""
    g, p = cfg["guard"], cfg["pre_deployment"]
    checks = {
        "guard c": g["c"] == guard.C,
        "guard min_members": g["min_members"] == guard.MIN_MEMBERS,
        "guard fusion": g["fusion"].startswith("min-max + mean") and (guard.CALIB, guard.AGG) == ("minmax", "mean"),
        "pre-deployment fit days": p["fit_days"] == [0, ieee_stress.FIT_LAST_DAY],
        "pre-deployment scored days": p["scored_days"] == list(ieee_stress.SCORED_DAYS),
        "fragility delta": f"t - {fragility.DELTA} (t - median)" in cfg["hypotheses"]["H4d"]["fragility"],
        "guard code": g["code"] == "thesis.guard",
        "median outside the frozen configs": "median" in fusion.EXTRA_AGGREGATIONS and "median" not in fusion.AGGREGATIONS,
        "robust configuration": robust_fusion.ROBUST == ("percentile", "median"),
        "robust code": cfg["robust_fusion"]["code"] == "thesis.robust_fusion.fuse_set",
        "20 triples": len(robust_fusion.TRIPLES) == 20,
        "cost margin": robust_fusion.COST_MARGIN == 0.05,
        "zero-alert month": robust_fusion.ZERO_ALERTS == 0.5 and "0.5 alert" in cfg["hypotheses"]["H4r"]["zero_alert_month"],
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise RuntimeError(f"code does not match frozen_stress_v3.json: {bad}")


def load(path: Path = FROZEN_V3, for_tag: bool = False) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    validate(cfg, frozen.load(), for_tag=for_tag)
    return cfg
