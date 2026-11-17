"""freeze-v3 file (frozen_stress_v3.json) and its checks (thesis.frozen_v3)."""
import copy
import json

import pytest

from thesis import frozen, frozen_v3 as f3, guard


@pytest.fixture(scope="module")
def v1():
    return frozen.load()


@pytest.fixture(scope="module")
def cfg():
    return f3.load()


def test_loads_and_matches_the_code(cfg):
    f3.check_code_matches(cfg)
    assert cfg["guard"]["c"] == guard.C == 3 and cfg["guard"]["min_members"] == 2
    assert cfg["rq4"]["text"] == f3.RQ4_TEXT


def test_hypotheses_follow_the_agreed_design(cfg, v1):
    h = cfg["hypotheses"]
    assert set(f3.HYPOTHESES) <= set(h)
    assert h["H4b"]["budget"] == h["H4c"]["budget"] == h["H4e"]["budget"] == v1["budgets"]["ieee_cis"]["primary"]
    assert h["H4d"]["status"].startswith("exploratory") and "developed after seeing ULB" in h["H4d"]["status"]
    assert "members x months" in h["H4a"]["statistic"] and "|log r_m|" in h["H4a"]["statistic"]
    assert cfg["pre_deployment"]["use"].endswith("not used for H4b")


def test_draft_is_refused_for_the_tag(cfg, v1):
    with pytest.raises(ValueError, match="draft_status"):
        f3.validate(cfg, v1, for_tag=True)
    clean = copy.deepcopy(cfg)
    del clean["draft_status"]
    f3.validate(clean, v1, for_tag=True)           # nothing else in the file goes stale
    stale = copy.deepcopy(clean)
    stale["guard"]["pass_rule"] += " (pending)"
    with pytest.raises(ValueError, match="stale"):
        f3.validate(stale, v1, for_tag=True)


def test_freeze_v1_and_v2_files_are_not_touched():
    import subprocess
    from thesis.paths import REPO
    for tag, path in (("freeze-v1", "frozen_config.json"), ("freeze-v2", "frozen_features_v2.json")):
        r = subprocess.run(["git", "diff", "--quiet", tag, "--", path], cwd=REPO)
        assert r.returncode == 0, f"{path} differs from {tag}"


def _set(path, value):
    def apply(c):
        node = c
        for k in path[:-1]:
            node = node[k]
        if value is DELETE:
            del node[path[-1]]
        else:
            node[path[-1]] = value(node[path[-1]]) if callable(value) else value
    return apply


DELETE = object()
BREAKS = {
    "missing section": _set(("runner",), DELETE),
    "wrong name": _set(("name",), "freeze-v2"),
    "builds on another file": _set(("builds_on", "freeze-v1"), "other.json"),
    "rq4 reworded": _set(("rq4", "text"), "Does the guard work?"),
    "two-sided guard": _set(("guard", "side"), "two-sided"),
    "relative guard": _set(("guard", "basis"), "relative"),
    "float c": _set(("guard", "c"), 3.0),
    "c not above 1": _set(("guard", "c"), 1),
    "min_members 0": _set(("guard", "min_members"), 0),
    "ties without r = c": _set(("statistic", "ties"), "exact, with fractions.Fraction"),
    "ties without the method": _set(("statistic", "ties"), "r = c passes"),
    "q not 1 - budget": _set(("statistic", "q"), "0.99"),
    "alias ensemble": _set(("guard", "ensembles"), ["E-lit", "E-all"]),
    "unknown ensemble": _set(("guard", "ensembles"), ["E-bench", "E-all", "E-new"]),
    "E-all missing": _set(("guard", "ensembles"), ["E-bench", "E-bench-5%"]),
    "pre-deployment overlap": _set(("pre_deployment", "scored_days"), [20, 29]),
    "pre-deployment outside month 1": _set(("pre_deployment", "scored_days"), [21, 30]),
    "hypothesis missing": _set(("hypotheses", "H4e"), DELETE),
    "no supported_if": _set(("hypotheses", "H4c", "supported_if"), DELETE),
    "H4d decided": _set(("hypotheses", "H4d", "status"), "decided"),
    "drift slope decided": _set(("hypotheses", "drift_slope", "status"), "decided"),
    "H4b other budget": _set(("hypotheses", "H4b", "budget"), 0.01),
    "H4b other method": _set(("hypotheses", "H4b", "method"), "E-guard(E-bench)"),
    "H4c other comparator": _set(("hypotheses", "H4c", "comparator"), "static E-all"),
    "H4e other comparator": _set(("hypotheses", "H4e", "comparator"), "oracle_best_single"),
    "H4b other criterion": _set(("hypotheses", "H4b", "supported_if"), "the CI does not lie above 0"),
    "H4a other criterion": _set(("hypotheses", "H4a", "supported_if"), "rho > 0"),
    "H4a other q": _set(("hypotheses", "H4a", "statistic"), lambda s: s.replace("q = 0.98", "q = 0.99")),
    "H4d other q": _set(("hypotheses", "H4d", "statistic"), lambda s: s.replace("q = 0.98", "q = 0.99")),
    "monthly K": _set(("evaluation", "daily_budget"), "K = ceil(budget x n) per month"),
    "other resamples": _set(("evaluation", "bootstrap"), "transaction-level"),
    "no-break case incomplete": _set(("if_no_member_breaks",), "H4b is not testable"),
    "removed incomplete": _set(("removed", "sead"), DELETE),
}


@pytest.mark.parametrize("name", list(BREAKS))
def test_validate_rejects_broken_configs(cfg, v1, name):
    bad = copy.deepcopy(cfg)
    BREAKS[name](bad)
    with pytest.raises(ValueError):
        f3.validate(bad, v1)


V1_BREAKS = {
    "v1 resamples": lambda v: v["statistics"]["bootstrap"].__setitem__("resamples", 999),
    "v1 seed": lambda v: v["statistics"]["bootstrap"].__setitem__("seed", 1),
    "v1 transaction bootstrap": lambda v: v["statistics"]["bootstrap"].__setitem__("ieee_cis", "transaction-level"),
}


@pytest.mark.parametrize("name", list(V1_BREAKS))
def test_validate_rejects_a_changed_freeze_v1(cfg, v1, name):
    bad = copy.deepcopy(v1)
    V1_BREAKS[name](bad)
    with pytest.raises(ValueError):
        f3.validate(cfg, bad)


def test_code_check_catches_a_changed_constant(cfg, monkeypatch):
    monkeypatch.setattr(guard, "C", 5)
    with pytest.raises(RuntimeError, match="guard c"):
        f3.check_code_matches(cfg)
    monkeypatch.setattr(guard, "C", 3)
    monkeypatch.setattr(guard, "MIN_MEMBERS", 1)
    with pytest.raises(RuntimeError, match="min_members"):
        f3.check_code_matches(cfg)


def test_the_file_is_valid_json_with_ascii_text():
    from thesis.paths import REPO
    raw = (REPO / "frozen_stress_v3.json").read_bytes()
    json.loads(raw)
    assert b"\xc3\x83" not in raw and b"\xc3\xa2\xe2\x82\xac" not in raw
