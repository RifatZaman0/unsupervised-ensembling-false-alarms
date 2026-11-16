"""IEEE-CIS month-1 stress test (thesis.ieee_stress): split, fit-window features, rules."""
import numpy as np
import pandas as pd
import pytest

from thesis import ieee_features, ieee_stress as s


def test_split_is_weeks_1_to_3_against_the_rest_of_month_1():
    day = np.repeat(np.arange(30), 3)
    fit, scored = s.split_days(day)
    assert set(day[fit]) == set(range(21)) and set(day[scored]) == set(range(21, 30))
    assert day[fit].max() < day[scored].min() and len(fit) + len(scored) == len(day)


def test_split_refuses_rows_outside_month_1():
    with pytest.raises(ValueError):
        s.split_days(np.arange(31))
    with pytest.raises(ValueError):
        s.split_days(np.array([-1, 0, 25]))


def test_feature_parameters_come_from_the_fit_window_only():
    day = np.repeat(np.arange(30), 4)
    n = len(day)
    df = pd.DataFrame({"TransactionAmt": np.where(day <= 20, 10.0, 1e5),
                       "ProductCD": np.where(day <= 20, "W", "NEW"),
                       "C1": np.where(day <= 20, np.arange(n) % 3, 900).astype(float)})
    df.loc[0, "C1"] = np.nan
    df.loc[n - 1, "C1"] = np.nan
    fit, scored = s.split_days(day)
    X_fit, X_sc, rule = s.features_for(df, fit, scored, ["TransactionAmt", "ProductCD", "C1"])
    assert rule.medians["TransactionAmt"] == pytest.approx(np.log1p(10.0))
    assert rule.frequencies["ProductCD"] == {"W": 1.0}
    assert rule.medians["C1"] == pytest.approx(np.nanmedian(df.C1.iloc[fit]))
    assert (X_sc[:, 1] == 0.0).all()                        # "NEW" unseen in the fit window
    assert X_sc[-1, 2] == rule.medians["C1"]                # scored row imputed with fit median
    assert X_fit.shape == (len(fit), 3) and X_sc.shape == (len(scored), 3)


def test_rule_table_one_and_two_sided():
    r = [0.3, 0.45, 1.0, 2.0, 2.5, 4.0, 6.0]
    tab = pd.DataFrame({"run": "x", "member": list("ABCDEFG"), "q": 0.99, "r": r})
    tab["r_over_median"] = tab.r / tab.r.median()          # median 2.0, mean 2.32
    rules = s.rule_table(tab)
    ab = rules[rules.basis == "absolute"]

    def passing(c, col):
        g = ab[(ab.c == c) & ab[col]]
        return set(g.member)

    assert passing(2, "one_sided_pass") == set("ABCD")     # r = c passes
    assert passing(3, "one_sided_pass") == set("ABCDE")
    assert passing(5, "one_sided_pass") == set("ABCDEF")
    assert passing(2, "two_sided_pass") == set("CD")       # [0.5, 2]
    assert passing(3, "two_sided_pass") == set("BCDE")     # [1/3, 3]
    assert passing(5, "two_sided_pass") == set("ABCDEF")   # [0.2, 5]
    rel = rules[(rules.basis == "relative to median") & (rules.c == 2)]
    assert set(rel[rel.two_sided_pass].member) == set("CDEF")    # r/2 in [0.5, 2]
    assert set(rel.value.round(3)) == set((tab.r / 2.0).round(3))
    assert len(rules) == 7 * 3 * 2


@pytest.mark.skipif(not ieee_features.RAW.exists(), reason="IEEE-CIS raw file not downloaded")
def test_real_month_1_split_sizes():
    month1, info = ieee_features.load_month1()
    from thesis import data
    day = data.ieee_cis_day(month1["TransactionDT"].to_numpy())
    fit, scored = s.split_days(day)
    assert len(fit) + len(scored) == info["rows_month1"]
    assert month1["TransactionDT"].iloc[fit].max() < month1["TransactionDT"].iloc[scored].min()
