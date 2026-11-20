"""Tests for the committee cleaning runner (prereg_cc.md): refit pipeline, Step 1 and guards."""
import dataclasses
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from thesis import cc, config, data, detectors, fusion, metrics, scorecache


def synthetic_split(seed=0, n_train=240, n_test=160, d=4, rate=0.08):
    rng = np.random.default_rng(seed)
    y_tr = (rng.random(n_train) < rate).astype(np.int8)
    y_te = (rng.random(n_test) < rate).astype(np.int8)
    scale = rng.uniform(0.5, 30.0, d)  # unequal feature scales, so the z-score matters for KNN
    X_tr = (rng.standard_normal((n_train, d)) + 4.0 * y_tr[:, None]) * scale
    X_te = (rng.standard_normal((n_test, d)) + 4.0 * y_te[:, None]) * scale
    return data.Split("synthetic", X_tr, X_te, y_tr, y_te, np.arange(n_train),
                      np.arange(n_train, n_train + n_test), seed=seed)


# ------------------------------------------------------------------ refit pipeline

@pytest.mark.parametrize("det", detectors.DETECTORS)
def test_refit_with_every_row_kept_equals_the_e1_pipeline(det):
    split = synthetic_split(1)
    tr, te = cc.refit_member(split, np.ones(len(split.y_train), bool), det)
    pre = data.preprocess(split)
    tr0, te0 = detectors.score(det, pre.X_train, pre.X_test, split.seed)
    assert np.array_equal(tr, tr0) and np.array_equal(te, te0)


@pytest.mark.parametrize("det", detectors.DETECTORS)
def test_refit_ignores_the_removed_training_rows(det):
    split = synthetic_split(2)
    keep = split.y_train == 0
    tr, te = cc.refit_member(split, keep, det)
    X2 = split.X_train.copy()
    X2[~keep] = 1e6  # removed rows changed beyond recognition: nothing may move
    s2 = data.Split(split.dataset, X2, split.X_test, split.y_train, split.y_test,
                    split.idx_train, split.idx_test, seed=split.seed)
    tr2, te2 = cc.refit_member(s2, keep, det)
    assert np.array_equal(tr, tr2) and np.array_equal(te, te2)
    assert len(tr) == keep.sum() and len(te) == len(split.y_test)


def test_refit_zscore_is_fitted_on_the_kept_rows():
    split = synthetic_split(3)
    keep = split.y_train == 0
    sc = data.Standardizer.fit(split.X_train[keep])
    expected = detectors.score("KNN", sc.transform(split.X_train[keep]),
                               sc.transform(split.X_test), split.seed)
    got = cc.refit_member(split, keep, "KNN")
    assert np.array_equal(got[0], expected[0]) and np.array_equal(got[1], expected[1])
    sc_all = data.Standardizer.fit(split.X_train)
    wrong = detectors.score("KNN", sc_all.transform(split.X_train[keep]),
                            sc_all.transform(split.X_test), split.seed)
    assert not np.array_equal(got[1], wrong[1])


def test_refit_follows_the_zscore_switch(monkeypatch):
    split = synthetic_split(3)
    keep = split.y_train == 0
    monkeypatch.setattr(config, "PROTOCOL", config.Protocol(zscore_inputs=False))
    got = cc.refit_member(split, keep, "KNN")
    expected = detectors.score("KNN", split.X_train[keep], split.X_test, split.seed)
    assert np.array_equal(got[1], expected[1])


def test_refit_uses_the_split_seed():
    a = synthetic_split(4)
    b = dataclasses.replace(a, seed=7)
    keep = np.ones(len(a.y_train), bool)
    ta, tb = cc.refit_member(a, keep, "IForest")[1], cc.refit_member(b, keep, "IForest")[1]
    assert not np.array_equal(ta, tb)
    pre = data.preprocess(b)
    assert np.array_equal(tb, detectors.score("IForest", pre.X_train, pre.X_test, 7)[1])


def test_refit_never_reads_labels():
    split = synthetic_split(5)
    keep = split.y_train == 0
    blind = data.Split(split.dataset, split.X_train, split.X_test, None, None,
                       split.idx_train, split.idx_test, seed=split.seed)
    for det in detectors.DETECTORS:
        a, b = cc.refit_member(split, keep, det), cc.refit_member(blind, keep, det)
        assert np.array_equal(a[1], b[1])
    for fn in (cc.refit_member, cc.refit_all):
        assert not set(inspect.signature(fn).parameters) & {"y", "y_train", "y_test", "labels"}


@pytest.mark.parametrize("keep, msg", [(np.ones(240, int), "boolean mask"),
                                       (np.ones(239, bool), "boolean mask"),
                                       (np.zeros(240, bool), "every training row")])
def test_refit_rejects_bad_masks(keep, msg):
    with pytest.raises(ValueError, match=msg):
        cc.refit_member(synthetic_split(6), keep, "HBOS")


def test_refits_never_touch_the_score_cache(monkeypatch, tmp_path):
    def forbidden(*a, **k):
        raise AssertionError("cache used")
    monkeypatch.setattr(scorecache, "get_scores", forbidden)
    monkeypatch.setattr(scorecache, "cache_path", forbidden)
    monkeypatch.setattr(np, "savez", forbidden)
    split = synthetic_split(7)
    train, test = cc.refit_all(split, split.y_train == 0)
    assert list(train) == list(detectors.DETECTORS) == list(test)


def test_oracle_keep_mask_keeps_exactly_the_normal_rows():
    y = np.array([0, 1, 0, 0, 1])
    assert cc.oracle_keep_mask(y).tolist() == [True, False, True, True, False]
    with pytest.raises(ValueError):
        cc.oracle_keep_mask(np.array([0, 2]))


def test_fused_test_is_minmax_on_training_scores_then_mean():
    train = {"a": np.array([0.0, 2.0, 4.0]), "b": np.array([10.0, 20.0, 30.0])}
    test = {"a": np.array([1.0, 6.0]), "b": np.array([30.0, 0.0])}
    # a: (s - 0) / 4 -> [0.25, 1.5]; b: (s - 10) / 20 -> [1.0, -0.5]; mean -> [0.625, 0.5]
    assert cc.fused_test(train, test, ("a", "b")).tolist() == [0.625, 0.5]


def test_fused_test_matches_fusion_fuse_for_eall():
    split = synthetic_split(8)
    train, test = cc.refit_all(split, np.ones(len(split.y_train), bool))
    S_tr = np.column_stack([train[d] for d in detectors.DETECTORS])
    S_te = np.column_stack([test[d] for d in detectors.DETECTORS])
    assert np.array_equal(cc.fused_test(train, test), fusion.fuse(S_tr, S_te)[1])


def test_refit_matches_cache_flags_a_difference():
    split = synthetic_split(9)
    keep = np.ones(len(split.y_train), bool)
    train, test = cc.refit_all(split, keep)
    rows = cc.refit_matches_cache(split, train, test)
    assert all(r["train_identical"] and r["test_identical"] for r in rows)
    bad_test = dict(test, PCA=test["PCA"] + 1e-12)
    bad_train = dict(train, LOF=np.r_[train["LOF"][:-1], train["LOF"][-1] + 1e-12])
    rows = {r["detector"]: r for r in cc.refit_matches_cache(split, bad_train, bad_test)}
    assert not rows["PCA"]["test_identical"] and rows["PCA"]["train_identical"]
    assert not rows["LOF"]["train_identical"] and rows["LOF"]["test_identical"]
    assert rows["PCA"]["max_abs_diff"] > 0


def test_refit_ok_needs_every_member_identical():
    good = {"n_members": 6, "n_train_identical": 6, "n_test_identical": 6}
    assert cc.refit_ok(good, 6)
    assert not cc.refit_ok({**good, "n_test_identical": 5}, 6)
    assert not cc.refit_ok({**good, "n_train_identical": 5}, 6)
    assert not cc.refit_ok({**good, "n_members": 5, "n_train_identical": 5,
                            "n_test_identical": 5}, 6)
    assert not cc.refit_ok({**good, "n_members": 7}, 6)  # a member checked twice


def test_refit_reproduces_one_cached_adbench_split():
    """Section 10, item 3 on one real split (all six members, bit for bit)."""
    name, seed = "vowels", 2
    path = scorecache.cache_path(name, seed, "LOF")
    if not path.exists() or scorecache.cache_key() != cc.CACHE_KEY:
        pytest.fail(f"cached split {name} seed {seed} missing (key {cc.CACHE_KEY})")
    raw = data.split_adbench(name, seed)
    rows = cc.refit_matches_cache(raw, *cc.cached_scores(raw))
    assert len(rows) == 6 and all(r["train_identical"] and r["test_identical"] for r in rows)


# ------------------------------------------------------------------ Step 1

def hand_scores():
    """Six members on 10 test rows, anomalies at rows 0 and 1; K = 1 at 2%, 1 at 5%."""
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
    good = np.array([9.0, 8, 1, 1, 1, 1, 1, 1, 1, 0])
    bad = np.array([0.0, 0, 1, 1, 1, 1, 1, 1, 1, 9])
    plain = {d: (good if d in ("PCA", "KNN") else bad) for d in detectors.DETECTORS}
    oc = {d: good for d in detectors.DETECTORS}
    train = {d: np.array([0.0, 10.0]) for d in detectors.DETECTORS}
    return y, plain, oc, train


def hand_split(y, seed=0):
    return data.Split("hand", np.zeros((5, 1)), np.zeros((len(y), 1)),
                      np.array([0, 1, 0, 0, 1], np.int8), y, np.arange(5),
                      np.arange(5, 5 + len(y)), seed=seed)


def test_step1_rows_columns():
    y, plain, oc, train = hand_scores()
    raw = hand_split(y)
    keep = cc.oracle_keep_mask(raw.y_train)
    rows = cc.step1_rows(raw, train, plain, train, oc, keep)
    assert [r["budget"] for r in rows] == list(cc.BUDGETS)
    r = rows[0]
    assert (r["n_train"], r["n_train_anomalies"], r["n_kept_oc"], r["k_test"]) == (5, 2, 3, 1)
    assert r["fp_plain_iforest"] == 1.0 and r["fp_plain_PCA"] == 0.0 and r["fp_oc_IForest"] == 0.0
    assert r["fp_best_plain_member"] == 0.0 and r["best_plain_member"] == "PCA"
    assert r["fp_median_plain_member"] == 1.0  # four members at 1, two at 0
    assert r["fp_best_oc_member"] == 0.0 and r["fp_oc_eall"] == 0.0
    # plain E-all: mean of 2 good + 4 bad min-max scores; row 9 gets (4 * 0.9 + 0) / 6 = 0.6,
    # row 0 gets 2 * 0.9 / 6 = 0.3, so the top row is a normal one
    assert r["fp_plain_eall"] == 1.0


def test_step1_rows_iforest_is_the_iforest_member():
    y, plain, oc, train = hand_scores()
    plain = dict(plain, IForest=plain["PCA"], HBOS=plain["PCA"])  # IForest good, HBOS good
    plain["ECOD"] = np.array([0.0, 9, 1, 1, 1, 1, 1, 1, 1, 8])  # tie 9 at row 1 (anomaly)
    r = cc.step1_rows(hand_split(y), train, plain, train, oc, np.ones(5, bool))[0]
    assert r["fp_plain_iforest"] == 0.0 and r["fp_plain_ECOD"] == 0.0
    assert r["fp_plain_LOF"] == 1.0


def test_step1_rows_oc_uses_the_oc_refits_and_their_training_scores():
    y, plain, _, train = hand_scores()
    good, bad = plain["PCA"], plain["IForest"]
    oc = {d: (good if d == "IForest" else bad) for d in detectors.DETECTORS}
    oc_train = dict(train, IForest=np.array([0.0, 1.0]))
    # OC calibration: IForest / 1 puts row 0 at 9 / 6 = 1.5, above row 9 at 5 * 0.9 / 6 = 0.75;
    # with the plain calibration ([0, 10] for all) row 9 (normal) would be on top
    r = cc.step1_rows(hand_split(y), train, plain, oc_train, oc, np.ones(5, bool))[0]
    assert r["fp_oc_eall"] == 0.0
    assert r["fp_best_oc_member"] == 0.0 and r["best_oc_member"] == "IForest"
    oc_bad = {d: bad for d in detectors.DETECTORS}
    r = cc.step1_rows(hand_split(y), train, plain, train, oc_bad, np.ones(5, bool))[0]
    assert r["fp_best_oc_member"] == 1.0 and r["fp_best_plain_member"] == 0.0
    assert r["fp_oc_eall"] == 1.0 and r["fp_oc_PCA"] == 1.0


def runs_frame(values, budget=0.02):
    """values: {dataset: [(oc, plain, iforest), ...] one tuple per seed}."""
    rows = []
    for ds, seeds in values.items():
        for s, (oc, plain, ifo) in enumerate(seeds):
            rows.append({"dataset": ds, "seed": s, "budget": budget, "k_test": 10,
                         "fp_oc_eall": oc, "fp_plain_eall": plain, "fp_plain_iforest": ifo,
                         "fp_best_plain_member": min(plain, ifo),
                         "fp_median_plain_member": plain, "fp_best_oc_member": oc})
    return pd.DataFrame(rows)


def test_per_dataset_takes_rounded_seed_means():
    runs = runs_frame({"a": [(1.0, 2.0, 3.0), (2.0, 2.0, 3.0), (6.0, 2.0, 3.0)],
                       "b": [(0.1 + 0.2, 0.3, 1.0), (0.3, 0.3, 1.0), (0.3, 0.3, 1.0)]})
    pd_ = cc.per_dataset(runs).set_index("dataset")
    assert pd_.loc["a", "fp_oc_eall"] == 3.0  # mean, not the median 2.0
    assert pd_.loc["a", "outcome"] == "plain fewer" and pd_.loc["a", "n_seeds"] == 3
    assert pd_.loc["b", "outcome"] == "tie"  # 0.1 + 0.2 != 0.3 before rounding
    assert pd_.loc["b", "delta_oc_vs_plain"] == 0.0


def test_per_dataset_stores_the_rounded_means():
    pd_ = cc.per_dataset(runs_frame({"a": [(0.1, 1.0, 1.0), (0.2, 1.0, 1.0), (0.4, 1.0, 1.0)]}))
    assert pd_["fp_oc_eall"].tolist() == [0.233333333]
    assert pd_["delta_oc_vs_plain"].tolist() == [0.233333333 - 1.0]


def test_per_dataset_outcome_oc_fewer():
    pd_ = cc.per_dataset(runs_frame({"a": [(1.0, 2.0, 3.0), (1.0, 4.0, 3.0), (1.0, 3.0, 3.0)]}))
    assert pd_["outcome"].tolist() == ["oc fewer"] and pd_["delta_oc_vs_plain"].tolist() == [-2.0]


def gate_row(values):
    g = cc.gate_table(cc.per_dataset(runs_frame(values)))
    return g.iloc[0]


def test_gate_passes_when_all_three_hold():
    r = gate_row({"a": [(8.0, 10.0, 12.0)], "b": [(9.0, 10.0, 12.0)], "c": [(10.0, 9.0, 12.0)]})
    assert (r["n_oc_fewer"], r["n_plain_fewer"], r["n_ties"]) == (2, 1, 0)
    assert r["cond1_oc_fewer_on_more_datasets"] and r["cond2_mean_at_least_5pct_below_plain"]
    assert r["cond3_mean_below_iforest"] and r["all_three"] and r["is_gate"]
    assert r["mean_fp_oc_eall"] == 9.0 and r["rel_reduction_oc_vs_plain"] == pytest.approx(1 - 27 / 29)


def test_gate_condition1_does_not_count_ties():
    # one fewer, one more, two ties: not "more datasets than the reverse"
    r = gate_row({"a": [(1.0, 20.0, 50.0)], "b": [(21.0, 20.0, 50.0)],
                  "c": [(5.0, 5.0, 50.0)], "d": [(5.0, 5.0, 50.0)]})
    assert (r["n_oc_fewer"], r["n_plain_fewer"], r["n_ties"]) == (1, 1, 2)
    assert not r["cond1_oc_fewer_on_more_datasets"]
    assert r["cond2_mean_at_least_5pct_below_plain"] and r["cond3_mean_below_iforest"]
    assert not r["all_three"]


def test_gate_condition2_alone_fails():
    # OC fewer on 2 of 3, below IForest, but only 2.9% below plain E-all on the mean
    r = gate_row({"a": [(99.0, 100.0, 200.0)], "b": [(99.0, 100.0, 200.0)],
                  "c": [(103.0, 102.0, 200.0)]})
    assert r["cond1_oc_fewer_on_more_datasets"] and r["cond3_mean_below_iforest"]
    assert not r["cond2_mean_at_least_5pct_below_plain"] and not r["all_three"]


def test_gate_condition2_is_exact_at_5_percent():
    # 31.635 is exactly 95% of 33.3, but 0.95 * 33.3 = 31.634999999999994 in floats
    r = gate_row({"a": [(31.635, 33.3, 40.0)]})
    assert r["cond2_mean_at_least_5pct_below_plain"]
    r = gate_row({"a": [(31.635000001, 33.3, 40.0)]})
    assert not r["cond2_mean_at_least_5pct_below_plain"]


def test_gate_condition3_alone_fails_and_is_strict():
    r = gate_row({"a": [(8.0, 10.0, 8.0)], "b": [(8.0, 10.0, 8.0)]})  # equal to IForest
    assert r["cond1_oc_fewer_on_more_datasets"] and r["cond2_mean_at_least_5pct_below_plain"]
    assert not r["cond3_mean_below_iforest"] and not r["all_three"]


def test_gate_uses_the_2pct_row_only():
    a = cc.per_dataset(runs_frame({"a": [(1.0, 10.0, 20.0)]}, budget=0.02))
    b = cc.per_dataset(runs_frame({"a": [(30.0, 10.0, 20.0)]}, budget=0.05))
    g = cc.gate_table(pd.concat([b, a], ignore_index=True))
    assert g["budget"].tolist() == [0.02, 0.05] and g["is_gate"].tolist() == [True, False]
    assert cc.gate_passes(g)
    a2 = cc.per_dataset(runs_frame({"a": [(30.0, 10.0, 20.0)]}, budget=0.02))
    b2 = cc.per_dataset(runs_frame({"a": [(1.0, 10.0, 20.0)]}, budget=0.05))
    assert not cc.gate_passes(cc.gate_table(pd.concat([a2, b2], ignore_index=True)))
    with pytest.raises(ValueError):
        cc.gate_passes(cc.gate_table(b))


def test_gate_means_are_over_rounded_seed_means():
    r = gate_row({"a": [(1.0, 4.0, 9.0), (2.0, 4.0, 9.0), (9.0, 4.0, 9.0)],
                  "b": [(0.0, 2.0, 1.0), (0.0, 2.0, 1.0), (0.0, 2.0, 1.0)]})
    assert r["mean_fp_oc_eall"] == 2.0 and r["mean_fp_plain_eall"] == 3.0
    assert r["mean_fp_plain_iforest"] == 5.0


def test_members_match_e1_detects_one_stale_value(tmp_path):
    rows, e1 = [], []
    for s in (0, 1):
        for b in cc.BUDGETS:
            row = {"dataset": "a", "seed": s, "budget": b}
            for i, d in enumerate(detectors.DETECTORS):
                v = 0.1 * i + s + b
                row[f"fp_plain_{d}"] = v
                e1.append({"dataset": "a", "seed": s, "detector": d, "budget": b, "fp_at_k": v})
            rows.append(row)
    runs = pd.DataFrame(rows)
    p = tmp_path / "e1_runs.csv"
    pd.DataFrame(e1).to_csv(p, index=False)
    assert cc.members_match_e1(runs, p) == {"n_compared": 24, "n_unmatched": 0,
                                            "n_not_identical": 0, "max_abs_diff": 0.0}
    runs.loc[3, "fp_plain_KNN"] += 1e-9
    res = cc.members_match_e1(runs, p)
    assert res["n_not_identical"] == 1
    res = cc.members_match_e1(runs.assign(seed=runs["seed"] + 5), p)
    assert res["n_unmatched"] == 24


# ------------------------------------------------------------------ guards

CLEAN = {"commit": "c0ffee", "dirty": False}


def yes(*a):
    return True


def no(*a):
    return False


def test_check_can_write_accepts_a_clean_tagged_unchanged_tree(tmp_path):
    cc.check_can_write(CLEAN, "abc", yes, yes, cc.STEP1_FILES, tmp_path)


@pytest.mark.parametrize("state, prereg, contains, unchanged, msg", [
    ({"commit": "c", "dirty": True}, "abc", yes, yes, "uncommitted"),
    (CLEAN, None, yes, yes, "does not exist"),
    (CLEAN, "abc", no, yes, "does not descend"),
    (CLEAN, "abc", yes, no, "differs from"),
])
def test_check_can_write_refuses(tmp_path, state, prereg, contains, unchanged, msg):
    with pytest.raises(SystemExit, match=msg):
        cc.check_can_write(state, prereg, contains, unchanged, cc.STEP1_FILES, tmp_path)


@pytest.mark.parametrize("name", cc.STEP1_FILES)
def test_check_can_write_refuses_existing_outputs(tmp_path, name):
    (tmp_path / name).write_text("x")
    with pytest.raises(SystemExit, match="already exist"):
        cc.check_can_write(CLEAN, "abc", yes, yes, cc.STEP1_FILES, tmp_path)


def test_cache_listing_sees_a_new_or_changed_file(tmp_path):
    (tmp_path / "k").mkdir()
    (tmp_path / "k" / "a.npz").write_bytes(b"1")
    before = cc.cache_listing(tmp_path)
    assert cc.cache_listing(tmp_path) == before
    (tmp_path / "k" / "b.npz").write_bytes(b"2")
    assert cc.cache_listing(tmp_path) != before
    del before
    before = cc.cache_listing(tmp_path)
    (tmp_path / "k" / "a.npz").write_bytes(b"11")
    assert cc.cache_listing(tmp_path) != before


def fake_step1(n_datasets=3):
    """Step 1 rows and refit checks for n datasets x 5 seeds, from hand-made scores."""
    y, plain, oc, train = hand_scores()
    runs, chk = [], []
    for i in range(n_datasets):
        for seed in data.ADBENCH_SEEDS:
            raw = dataclasses.replace(hand_split(y, seed), dataset=f"d{i}")
            runs += cc.step1_rows(raw, train, plain, train, oc, cc.oracle_keep_mask(raw.y_train))
            chk += [{"dataset": raw.dataset, "seed": seed, "detector": d, "train_identical": True,
                     "test_identical": True, "max_abs_diff": 0.0} for d in detectors.DETECTORS]
    return pd.DataFrame(runs), pd.DataFrame(chk)


@pytest.fixture
def step1_env(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "ADBENCH_15", ("d0", "d1", "d2"))
    monkeypatch.setattr(cc, "OUT", tmp_path / "cc")
    monkeypatch.setattr(cc, "git_state", lambda: dict(CLEAN))
    monkeypatch.setattr(cc, "tag_commit", lambda: "abc")
    monkeypatch.setattr(cc, "head_contains", yes)
    monkeypatch.setattr(cc, "prereg_unchanged", yes)
    monkeypatch.setattr(cc, "cache_listing", lambda: {"x": (1, 1)})
    monkeypatch.setattr(cc, "run_step1", lambda datasets: fake_step1(len(datasets)))
    monkeypatch.setattr(cc, "members_match_e1", lambda runs: {
        "n_compared": 1, "n_unmatched": 0, "n_not_identical": 0, "max_abs_diff": 0.0})
    return monkeypatch, tmp_path / "cc"


def test_step1_main_writes_outputs_and_meta(step1_env):
    mp, out = step1_env
    cc.step1_main(["step1"])
    assert sorted(p.name for p in out.iterdir()) == sorted(cc.STEP1_FILES)
    meta = json.loads((out / "cc_step1.meta.json").read_text(encoding="utf-8"))
    assert meta["commit"] == "c0ffee" and meta["dirty"] is False
    assert meta["prereg_tag"] == "prereg-cc" and meta["prereg_tag_commit"] == "abc"
    assert meta["refit_all_rows_vs_cache"]["n_test_identical"] == 90
    assert meta["score_cache_unchanged"] is True and meta["gate_rebuilt_from_csv_identical"]
    # hand scores: OC E-all 0 FP, plain E-all 1 FP, IForest 1 FP on every split -> gate passes
    assert meta["gate"]["passed"] is True
    gate = pd.read_csv(out / "cc_step1_gate.csv")
    assert gate["n_oc_fewer"].tolist() == [3, 3] and gate["commit"].eq("c0ffee").all()


def test_step1_main_refuses_before_running(step1_env):
    mp, out = step1_env
    mp.setattr(cc, "git_state", lambda: {"commit": "c", "dirty": True})
    mp.setattr(cc, "run_step1", lambda d: pytest.fail("ran on a dirty tree"))
    with pytest.raises(SystemExit, match="uncommitted"):
        cc.step1_main(["step1"])
    assert not out.exists()


def test_step1_main_refuses_existing_outputs(step1_env):
    mp, out = step1_env
    out.mkdir()
    (out / "cc_step1_gate.csv").write_text("x")
    mp.setattr(cc, "run_step1", lambda d: pytest.fail("ran twice"))
    with pytest.raises(SystemExit, match="already exist"):
        cc.step1_main(["step1"])


def test_step1_main_writes_nothing_if_the_refit_differs(step1_env):
    mp, out = step1_env

    def broken(datasets):
        runs, chk = fake_step1(len(datasets))
        chk.loc[17, "test_identical"] = False
        return runs, chk
    mp.setattr(cc, "run_step1", broken)
    with pytest.raises(SystemExit, match="does not reproduce the cache"):
        cc.step1_main(["step1"])
    assert not out.exists()


def test_step1_main_writes_nothing_if_a_refit_check_is_missing(step1_env):
    mp, out = step1_env
    mp.setattr(cc, "run_step1", lambda d: (fake_step1(3)[0], fake_step1(3)[1].iloc[:-1]))
    with pytest.raises(SystemExit, match="does not reproduce the cache"):
        cc.step1_main(["step1"])
    assert not out.exists()


def test_step1_main_writes_nothing_if_e1_differs(step1_env):
    mp, out = step1_env
    mp.setattr(cc, "members_match_e1", lambda runs: {
        "n_compared": 1, "n_unmatched": 0, "n_not_identical": 1, "max_abs_diff": 1.0})
    with pytest.raises(SystemExit, match="e1_runs.csv"):
        cc.step1_main(["step1"])
    assert not out.exists()


def test_step1_main_writes_nothing_if_the_cache_changed(step1_env):
    mp, out = step1_env
    calls = iter([{"x": (1, 1)}, {"x": (1, 2)}])
    mp.setattr(cc, "cache_listing", lambda: next(calls))
    with pytest.raises(SystemExit, match="score cache changed"):
        cc.step1_main(["step1"])
    assert not out.exists()


def test_main_refuses_a_real_data_subset_for_step1():
    with pytest.raises(SystemExit, match="all 15"):
        cc.main(["step1", "cardio"])


def test_main_needs_the_default_protocol(monkeypatch):
    monkeypatch.setattr(config, "PROTOCOL", config.Protocol(k_rule="round"))
    with pytest.raises(SystemExit, match="default protocol"):
        cc.main(["check-refit", "cardio"])


def test_step1_main_runs_the_guard_with_the_tag_diff(step1_env):
    mp, out = step1_env
    mp.setattr(cc, "prereg_unchanged", no)
    with pytest.raises(SystemExit, match="differs from"):
        cc.step1_main(["step1"])
    mp.setattr(cc, "prereg_unchanged", yes)
    mp.setattr(cc, "head_contains", no)
    with pytest.raises(SystemExit, match="does not descend"):
        cc.step1_main(["step1"])


def test_guards_point_at_this_prereg():
    assert cc.PREREG_FILE == "prereg_cc.md" and cc.PREREG_TAG == "prereg-cc"
    p = inspect.signature(cc.prereg_unchanged).parameters
    assert p["path"].default == "prereg_cc.md" and p["tag"].default == "prereg-cc"
    assert inspect.signature(cc.tag_commit).parameters["tag"].default == "prereg-cc"


def test_budget_k_is_the_k_rule_of_cleaning():
    from thesis import cleaning
    for fn in (cleaning.removal_mask, cleaning.committee_keep_mask, cleaning.self_keep_mask):
        assert inspect.signature(fn).parameters["k_rule"].default is metrics.budget_k
