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

from thesis import data, fragility, frozen, guard, ieee_stress
from thesis.paths import REPO

FROZEN_V3 = REPO / "frozen_stress_v3.json"
REQUIRED = ("name", "scope", "builds_on", "rq4", "statistic", "guard", "pre_deployment",
            "evaluation", "hypotheses", "if_no_member_breaks", "removed", "runner")
HYPOTHESES = ("H4a", "H4b", "H4c", "H4d", "H4e", "drift_slope", "multiplicity")
DECIDED = ("H4a", "H4b", "H4c", "H4e")
STALE = re.compile(r"\b(pending|candidate|proposed|draft|awaiting|tentative|to be (?:decided|confirmed))\b",
                   re.IGNORECASE)
RQ4_TEXT = ("Can label-free information about how each ensemble member's alert rate changes over "
            "time (a) be measured on the training window to predict, before deployment, which "
            "members will break, and (b) be used during deployment to exclude breaking members "
            "and reduce false alarms at a fixed daily budget?")


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
        raise ValueError("guard.ensembles must include E-all (H4b) and E-bench (H4c, H4e)")
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
    ieee = v1["budgets"]["ieee_cis"]
    for h in ("H4b", "H4c", "H4e"):
        if hyp[h]["budget"] != ieee["primary"]:
            raise ValueError(f"{h}: budget {hyp[h]['budget']} is not the IEEE-CIS primary {ieee['primary']}")
    if hyp["H4b"]["method"] != "E-guard(E-all)" or hyp["H4b"]["comparator"] != "static E-all":
        raise ValueError("H4b compares E-guard(E-all) with static E-all")
    if hyp["H4c"]["method"] != "E-guard(E-bench)" or hyp["H4c"]["comparator"] != "static E-bench":
        raise ValueError("H4c compares E-guard(E-bench) with static E-bench")
    if not hyp["H4e"]["comparator"].startswith("chosen single"):
        raise ValueError("H4e compares with the single chosen on month 1")
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
    for k in ("H4b", "H4c", "H4d", "H4e"):
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
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise RuntimeError(f"code does not match frozen_stress_v3.json: {bad}")


def load(path: Path = FROZEN_V3, for_tag: bool = False) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    validate(cfg, frozen.load(), for_tag=for_tag)
    return cfg
