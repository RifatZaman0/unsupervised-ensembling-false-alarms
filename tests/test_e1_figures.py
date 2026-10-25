"""CD diagram statistics: Nemenyi CD matches Demšar's table, Friedman matches the textbook
formula, groups are the maximal non-significant runs, and the commit reaches the figures."""
import math

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from thesis import e1_figures


def per_ds(values: dict, budget=0.05) -> pd.DataFrame:
    """values: {detector: [metric per dataset]} -> e1_summary.per_dataset-like frame."""
    rows = [{"dataset": f"d{i}", "budget": budget, "detector": det, "fp_at_k": v}
            for det, vs in values.items() for i, v in enumerate(vs)]
    return pd.DataFrame(rows)


def test_nemenyi_cd_matches_demsar_table():
    # Demšar (2006) Table 5(a): q_0.05 = 2.850 for k = 6, 2.569 for k = 4
    assert e1_figures.nemenyi_cd(6, 15) == pytest.approx(2.850 * math.sqrt(6 * 7 / (6 * 15)), abs=1e-3)
    assert e1_figures.nemenyi_cd(4, 10) == pytest.approx(2.569 * math.sqrt(4 * 5 / (6 * 10)), abs=1e-3)


def test_friedman_matches_textbook_formula_without_ties():
    rng = np.random.default_rng(0)
    vals = {d: list(rng.permutation(4) + 10 * j) for j, d in enumerate("ABCD")}
    vals["B"] = list(rng.normal(15, 20, size=4))
    res = e1_figures.friedman_nemenyi(per_ds(vals))
    df = per_ds(vals).pivot(index="dataset", columns="detector", values="fp_at_k")
    R = df.rank(axis=1).mean()
    n, k = df.shape
    chi2 = 12 * n / (k * (k + 1)) * ((R ** 2).sum() - k * (k + 1) ** 2 / 4)
    assert res["friedman_chi2"] == pytest.approx(chi2)
    assert res["avg_ranks"] == pytest.approx(R.sort_values().to_dict())


def test_consistent_order_is_significant_and_lower_fp_ranks_first():
    vals = {"A": [1, 2, 3, 4, 5, 6, 7, 8], "B": [11, 12, 13, 14, 15, 16, 17, 18],
            "C": [21, 22, 23, 24, 25, 26, 27, 28]}
    res = e1_figures.friedman_nemenyi(per_ds(vals))
    assert res["friedman_p"] < 0.01
    assert list(res["avg_ranks"]) == ["A", "B", "C"] and res["avg_ranks"]["A"] == 1.0


def test_groups_are_maximal_runs_within_cd():
    avg = pd.Series({"a": 1.0, "b": 1.5, "c": 3.0, "d": 3.2})
    assert e1_figures.groups(avg, 1.0) == [["a", "b"], ["c", "d"]]
    assert e1_figures.groups(avg, 2.1) == [["a", "b", "c"], ["b", "c", "d"]]
    assert e1_figures.groups(avg, 0.1) == []


def test_missing_values_are_rejected():
    vals = {"A": [1.0, np.nan, 3.0], "B": [2.0, 3.0, 4.0]}
    with pytest.raises(ValueError, match="missing"):
        e1_figures.friedman_nemenyi(per_ds(vals))


def test_write_records_commit_in_meta_and_figures(tmp_path):
    rng = np.random.default_rng(1)
    frames = [per_ds({d: list(rng.normal(j, 1, 15)) for j, d in enumerate("ABCDEF")}, b)
              for b in (0.02, 0.05)]
    meta = {"commit": "abc1234def", "dirty": False, "cache_key": "x", "protocol": {},
            "created_utc": "t"}
    res = e1_figures.write(pd.concat(frames), tmp_path, meta, (0.02, 0.05))
    assert [r["figure"] for r in res] == ["cd_fp_at_k_2pct", "cd_fp_at_k_5pct"]
    assert Image.open(tmp_path / "cd_fp_at_k_5pct.png").info["Comment"] == "commit abc1234def"
    assert b"abc1234def" in (tmp_path / "cd_fp_at_k_2pct.pdf").read_bytes()
    assert '"commit": "abc1234def"' in (tmp_path / "cd.meta.json").read_text()
