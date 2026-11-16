"""E-stress and the run-time guard (thesis.e_stress): rules, member sets, fusion, fallback."""
import numpy as np
import pandas as pd
import pytest

from thesis import e3_ulb, e_stress as e


def r_tab(r):
    t = pd.DataFrame({"member": list(r), "r": list(r.values())})
    t["r_over_median"] = t.r / t.r.median()
    return t


def test_passes_boundaries():
    assert e.passes(2.0, "one-sided", 2) and not e.passes(2.01, "one-sided", 2)
    assert e.passes(0.0, "one-sided", 2)
    assert e.passes(0.5, "two-sided", 2) and e.passes(2.0, "two-sided", 2)
    assert not e.passes(0.49, "two-sided", 2) and not e.passes(2.01, "two-sided", 2)
    with pytest.raises(ValueError):
        e.passes(1.0, "three-sided", 2)


def test_rules_cover_every_candidate():
    rs = e.rules()
    assert len(rs) == 2 * 2 * 3 and len({r["rule"] for r in rs}) == 12
    assert {(r["side"], r["basis"], r["c"]) for r in rs} == {
        (s, b, c) for s in e.SIDES for b in e.BASES for c in e.CS}


def test_passing_absolute_and_relative():
    # median r = (0.9 + 1.25) / 2 = 1.075; mean differs (7.4)
    t = r_tab({"IForest": 0.2, "PCA": 0.9, "KNN": 1.25, "LOF": 41.0})
    rule = {"side": "one-sided", "basis": "absolute", "c": 3}
    assert e.passing(t, rule) == {"IForest", "PCA", "KNN"}
    assert e.passing(t, {**rule, "side": "two-sided"}) == {"PCA", "KNN"}       # 0.2 < 1/3
    rel = {"side": "two-sided", "basis": "relative", "c": 5}
    assert e.passing(t, rel) == {"PCA", "KNN"}                                  # 0.2/1.075 < 0.2


def test_kept_members_keeps_order_and_falls_back():
    assert e.kept_members(["IForest", "KNN", "LOF"], {"LOF", "IForest"}) == (["IForest", "LOF"], False)
    assert e.kept_members(["IForest", "KNN"], {"PCA"}) == (["IForest", "KNN"], True)


def test_decisions_table():
    tabs = {"guard": r_tab({"IForest": 0.9, "PCA": 0.9, "KNN": 1.2, "LOF": 41.0})}
    ens = {"E-bench": ["IForest", "PCA", "KNN"], "E-bench-5%": ["IForest", "KNN", "LOF"]}
    d = e.decisions(tabs, ens)
    assert len(d) == 12 * 2
    one = d[(d.rule == "one-sided absolute c=2") & (d.ensemble == "E-bench-5%")].iloc[0]
    assert (one.kept, one.dropped, one.changed, one.fallback) == ("IForest+KNN", "LOF", True, False)
    same = d[(d.rule == "one-sided absolute c=2") & (d.ensemble == "E-bench")].iloc[0]
    assert not same.changed and same.dropped == ""


def synthetic_fitted(seed=0):
    rng = np.random.default_rng(seed)
    tr = {"IForest": rng.normal(size=(3, 200)), "PCA": rng.normal(size=(1, 200)),
          "KNN": rng.normal(size=(1, 200)), "LOF": rng.normal(size=(1, 200))}
    te = {k: rng.normal(size=(len(v), 80)) for k, v in tr.items()}
    return e3_ulb.Fitted(tr, te, tuple(tr), {})


def test_fuse_members_equals_the_frozen_fit_stage_pairing():
    f = synthetic_fitted()
    tr, te = e.fuse_members(f.train, f.test, ["IForest", "PCA", "KNN"])
    assert tr.shape == (3, 200) and te.shape == (3, 80)
    from thesis import fusion
    for i in range(3):                                      # IForest seed i, others seed 0
        S_tr = np.column_stack([f.train["IForest"][i], f.train["PCA"][0], f.train["KNN"][0]])
        S_te = np.column_stack([f.test["IForest"][i], f.test["PCA"][0], f.test["KNN"][0]])
        a, b = fusion.fuse(S_tr, S_te, "minmax", "mean")
        assert np.array_equal(tr[i], a) and np.array_equal(te[i], b)
    one = e.fuse_members(f.train, f.test, ["KNN"])[1]
    assert one.shape == (1, 80)


def test_fusion_uses_training_scores_only():
    f = synthetic_fitted()
    _, te1 = e.fuse_members(f.train, f.test, ["PCA", "KNN"])
    shifted = {k: v + (100.0 if k == "KNN" else 0.0) for k, v in f.test.items()}
    _, te2 = e.fuse_members(f.train, shifted, ["PCA", "KNN"])
    # min-max from training scores: a shift of the test scores moves the fused test score by 100/range/2
    rng = f.train["KNN"][0].max() - f.train["KNN"][0].min()
    assert np.allclose(te2 - te1, 100.0 / rng / 2)


def test_variant_fitted_has_every_set_once_and_comparisons_point_to_the_static_ensemble():
    f = synthetic_fitted()
    ens = {"E-bench-5%": ["IForest", "KNN", "LOF"]}
    tabs = {"guard": r_tab({"IForest": 0.9, "PCA": 0.9, "KNN": 1.2, "LOF": 41.0})}
    d = e.decisions(tabs, ens)
    v = e.variant_fitted(f, d, ens)
    assert set(v.test) == ({e.set_name(k.split("+")) for k in d.kept} | {"{IForest+KNN+LOF}"}
                           | set(f.singles))
    assert v.test["KNN"] is f.test["KNN"]                   # singles kept unchanged
    pairs = e.comparisons(d, ens)
    assert pairs and all(c == "{IForest+KNN+LOF}" for _, c in pairs)
    assert len(pairs) == len(set(pairs))
    assert ":" not in "".join(v.test)                       # read as plain methods by e3_ulb
