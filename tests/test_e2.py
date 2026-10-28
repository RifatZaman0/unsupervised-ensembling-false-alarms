"""E2: all 35 combos, training-only overlap (labels and test rows cannot reach it), the tie rule
of the top-K Jaccard matches thesis.metrics, gains against the best / median member, and the
per-dataset ranks, summary and headline shares."""
import json

import numpy as np
import pandas as pd
import pytest

from thesis import data, e2, fusion, metrics, scorecache


def test_combos_are_the_15_pairs_and_20_triples():
    sizes = pd.Series([len(c) for c in e2.COMBOS]).value_counts()
    assert sizes[2] == 15 and sizes[3] == 20
    assert len(set(e2.COMBOS)) == 35


def test_topk_weights_follow_the_metrics_tie_rule():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 6, size=40).astype(float)      # many ties
    y = rng.integers(0, 2, size=40)
    for k in (1, 3, 7, 20, 40):
        w = e2.topk_weights(s, k)
        assert w.sum() == pytest.approx(k)
        assert y @ w == pytest.approx(metrics.tp_at_k(s, y, k))


def test_jaccard_topk():
    a = np.arange(10.0)
    assert e2.jaccard_topk(a, a, 3) == pytest.approx(1.0)
    assert e2.jaccard_topk(a, -a, 3) == pytest.approx(0.0)
    # top-2 of a = {8, 9}; b ties rows 7, 8, 9 for 2 slots -> E|A & B| = 2 * 2/3
    b = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1.0])
    inter = 4 / 3
    assert e2.jaccard_topk(a, b, 2) == pytest.approx(inter / (4 - inter))


def test_spearman_constant_scores_give_nan():
    assert np.isnan(e2.spearman(np.ones(5), np.arange(5.0)))
    assert e2.spearman(np.arange(5.0), np.arange(5.0) ** 2) == pytest.approx(1.0)


def scores(seed=0, n_train=60, n_test=50, dets=("A", "B", "C")):
    rng = np.random.default_rng(seed)
    train = {d: rng.normal(size=n_train) for d in dets}
    test = {d: rng.normal(size=n_test) for d in dets}
    y = np.zeros(n_test, dtype=int)
    y[rng.choice(n_test, 5, replace=False)] = 1
    return train, test, y


BASE = {"dataset": "d", "seed": 0, "n_test": 50, "anomalies_test": 5}
COMBOS = (("A", "B"), ("A", "B", "C"))


def test_overlap_ignores_labels_and_test_scores():
    train, test, y = scores()
    ref = pd.DataFrame(e2.run_split(train, test, y, BASE, COMBOS))
    _, test2, _ = scores(seed=9)                         # other test scores
    other = pd.DataFrame(e2.run_split(train, test2, 1 - y, BASE, COMBOS))   # flipped labels
    cols = ["train_spearman", "train_jaccard", "k_train"]
    pd.testing.assert_frame_equal(ref[cols], other[cols])
    assert not ref.fp_at_k.equals(other.fp_at_k)        # the test side did change
    train2 = {**train, "A": train["A"] + np.linspace(0, 3, 60)}
    moved = pd.DataFrame(e2.run_split(train2, test, y, BASE, COMBOS))
    assert not ref.train_spearman.equals(moved.train_spearman)


def test_triple_overlap_is_the_mean_of_its_pairs():
    train, _, _ = scores()
    tri = e2.overlap(train, ("A", "B", "C"), 0.05)
    pairs = [e2.overlap(train, p, 0.05) for p in (("A", "B"), ("A", "C"), ("B", "C"))]
    assert tri["train_spearman"] == pytest.approx(np.mean([p["train_spearman"] for p in pairs]))
    assert tri["train_jaccard"] == pytest.approx(np.mean([p["train_jaccard"] for p in pairs]))
    assert tri["k_train"] == metrics.budget_k(0.05, 60)


def test_gain_against_best_and_median_member():
    train, test, y = scores()
    rows = pd.DataFrame(e2.run_split(train, test, y, BASE, COMBOS))
    assert len(rows) == len(COMBOS) * 6 * len(metrics.ADBENCH_BUDGETS)
    for r in rows.itertuples():
        fps = [metrics.budget_metrics(test[d], y, r.budget)["fp_at_k"] for d in r.combo.split("+")]
        assert r.fp_best_member == min(fps)
        assert r.fp_median_member == pytest.approx(np.median(fps))
        _, fused = fusion.fuse(np.column_stack([train[d] for d in r.combo.split("+")]),
                               np.column_stack([test[d] for d in r.combo.split("+")]),
                               r.calib, r.agg)
        assert r.fp_at_k == pytest.approx(metrics.budget_metrics(fused, y, r.budget)["fp_at_k"])
        assert r.delta_fp_best == pytest.approx(r.fp_at_k - min(fps))
        assert r.delta_fp_median_rel == pytest.approx((r.fp_at_k - np.median(fps)) / r.k)


def test_ensemble_of_a_detector_with_itself_has_zero_gain():
    train, test, y = scores()
    train["B"], test["B"] = train["A"].copy(), test["A"].copy()
    rows = pd.DataFrame(e2.run_split(train, test, y, BASE, (("A", "B"),)))
    # min-max and z-score are strictly increasing maps; percentile is not (test scores above
    # the training maximum all become 1.0, so it can create ties at the top)
    monotone = rows[rows.calib != "percentile"]
    assert (monotone.delta_fp_best == 0).all() and (rows.train_jaccard == 1).all()
    assert set(rows.best_member) == {"A/B"}


def runs_frame():
    """Two datasets, one seed each; d1 favours combo X, d2 combo Y."""
    rows = []
    for ds, fp in (("d1", {"X": 1.0, "Y": 3.0, "T": 2.0}), ("d2", {"X": 4.0, "Y": 2.0, "T": 5.0})):
        for combo, v in fp.items():
            rows.append({"dataset": ds, "seed": 0, "budget": 0.05, "size": 3 if combo == "T" else 2,
                         "calib": "minmax", "agg": "mean", "combo": combo, "fp_at_k": v,
                         "precision_at_k": 0.0, "recall_at_k": 0.0, "pr_auc": 0.5, "roc_auc": 0.5,
                         "delta_fp_best": v - 2.5, "delta_fp_median": v - 3.5,
                         "delta_fp_best_rel": (v - 2.5) / 10, "delta_fp_median_rel": (v - 3.5) / 10,
                         "train_spearman": 0.3, "train_jaccard": 0.2, "k": 10, "n_test": 100})
    return pd.DataFrame(rows)


def test_per_dataset_ranks_within_size_and_summary_counts():
    per_ds = e2.per_dataset(runs_frame())
    pairs = per_ds[per_ds["size"] == 2].set_index(["dataset", "combo"])
    assert pairs.rank_fp_at_k.to_dict() == {("d1", "X"): 1, ("d1", "Y"): 2,
                                            ("d2", "X"): 2, ("d2", "Y"): 1}
    assert (per_ds[per_ds["size"] == 3].rank_fp_at_k == 1).all()   # alone in its size
    summ = e2.summary(per_ds).set_index("combo")
    assert summ.loc["X", "avg_rank_fp_at_k"] == 1.5
    assert summ.loc["X", "n_beats_best"] == 1 and summ.loc["T", "n_beats_best"] == 1
    assert summ.loc["Y", "n_beats_median"] == 2
    head = e2.headline(per_ds).set_index("size")
    assert head.loc[2, "share_beats_best"] == 0.5 and head.loc[2, "n_cells"] == 4


def test_members_match_e1():
    train, test, y = scores()
    rows = []
    for seed in (0, 1):
        rows += e2.run_split(train, test, y, {**BASE, "seed": seed}, (("A", "B"),))
    runs = pd.DataFrame(rows)
    e1 = pd.DataFrame([{"dataset": "d", "seed": s, "budget": b, "detector": d,
                        "fp_at_k": metrics.budget_metrics(test[d], y, b)["fp_at_k"]}
                       for s in (0, 1) for b in metrics.ADBENCH_BUDGETS for d in ("A", "B")])
    assert e2.members_match_e1(runs, e1) == 0
    e1.loc[0, "fp_at_k"] -= 3
    assert e2.members_match_e1(runs, e1) > 0


def test_write_names_the_commit_everywhere(tmp_path):
    train, test, y = scores()
    runs = pd.DataFrame(e2.run_split(train, test, y, BASE, COMBOS))
    meta = {"commit": "c0ffee1", "created_utc": "t", "cache_key": "k",
            "protocol": {"zscore_inputs": True, "k_rule": "ceil", "pca_drop_zero_variance": True,
                         "adbench_dedup": False}}
    e2.write(runs, tmp_path, meta)
    for f in ("e2_per_dataset.csv", "e2_summary.csv", "e2_headline.csv"):
        assert set(pd.read_csv(tmp_path / f)["commit"]) == {"c0ffee1"}
    assert "c0ffee1" in (tmp_path / "e2_summary.md").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "e2_runs.meta.json").read_text())["commit"] == "c0ffee1"
    assert "Triple ranking" in (tmp_path / "e2_summary.md").read_text(encoding="utf-8")


@pytest.mark.skipif(not any(data.ADBENCH_DIR.glob("*.npz")), reason="ADBench not downloaded")
def test_real_cardio_seed_from_cache():
    X, y = data.load_adbench("cardio")
    split = data.preprocess(data.split_adbench("cardio", 0, X, y))
    sc = {d: scorecache.get_scores(split, d) for d in ("IForest", "PCA", "ECOD")}
    rows = pd.DataFrame(e2.run_split({d: s["train"] for d, s in sc.items()},
                                     {d: s["test"] for d, s in sc.items()}, split.y_test,
                                     {**BASE, "dataset": "cardio"}, (("IForest", "PCA", "ECOD"),)))
    assert len(rows) == 12 and rows[["fp_at_k", "train_jaccard"]].notna().all().all()
    assert ((rows.train_jaccard > 0) & (rows.train_jaccard <= 1)).all()
    assert rows.train_spearman.between(-1, 1).all()
