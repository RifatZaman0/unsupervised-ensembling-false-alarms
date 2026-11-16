"""IEEE-CIS feature rule (thesis.ieee_features): month 1 only, no labels, the frozen list."""
import copy

import numpy as np
import pandas as pd
import pytest

from thesis import data, ieee_features as f

DAY = data.IEEE_DAY_SECONDS


def frame(n_month1=100, n_later=40):
    """Month 1 (days 0-29) then later rows whose values would change every fitted parameter."""
    n = n_month1 + n_later
    t = np.concatenate([86_400 + np.linspace(0, 30 * DAY - 1, n_month1),
                        86_400 + np.linspace(30 * DAY, 100 * DAY, n_later)]).astype(np.int64)
    amt = np.concatenate([np.arange(1.0, n_month1 + 1), np.full(n_later, 1e6)])
    c1 = np.concatenate([np.arange(n_month1) % 7, np.full(n_later, 500)]).astype(float)
    c1[:10] = np.nan                                        # 10% missing in month 1
    d4 = np.concatenate([np.arange(n_month1, dtype=float), np.zeros(n_later)])
    d4[:20] = np.nan                                        # exactly 20% missing: dropped
    prod = np.array(["W"] * 60 + ["C"] * 30 + ["H"] * 10 + ["NEW"] * n_later, dtype=object)
    dom = np.array(["a.com"] * 50 + [None] * 15 + ["b.com"] * 35 + ["a.com"] * n_later,
                   dtype=object)
    return pd.DataFrame({
        "TransactionID": np.arange(n), "isFraud": np.arange(n) % 2, "TransactionDT": t,
        "TransactionAmt": amt, "ProductCD": prod, "card1": np.arange(n) % 5 + 1000,
        "P_emaildomain": dom, "C1": c1, "C3": np.zeros(n), "D4": d4,
        "V1": np.ones(n), "V200": np.arange(n, dtype=float)})


@pytest.fixture
def csv(tmp_path):
    path = tmp_path / "train_transaction.csv"
    frame().to_csv(path, index=False)
    return path


def test_month_1_is_read_without_the_label_and_without_later_rows(csv):
    month1, info = f.load_month1(csv)
    assert "isFraud" not in month1.columns and len(month1) == 100
    assert info == {"rows_month1": 100, "rows_file": 140, "days_month1": 30,
                    "transaction_dt_first": 86_400,
                    "transaction_dt_last_month1": 86_400 + 30 * DAY - 1}
    assert month1["TransactionAmt"].max() == 100.0          # the 1e6 rows are later rows


def test_month_1_count_needs_a_sorted_file():
    t = 86_400 + np.array([0, 5, 29 * DAY, 30 * DAY, 31 * DAY])
    assert f.month1_row_count(t) == 3
    with pytest.raises(ValueError):
        f.month1_row_count(t[::-1])


def test_candidates_leave_out_id_label_time_and_v_columns():
    cols = list(frame().columns) + ["Vendor"]
    assert f.candidates(cols) == ["TransactionAmt", "ProductCD", "card1", "P_emaildomain",
                                  "C1", "C3", "D4", "Vendor"]


def test_selection_rule(csv):
    month1, _ = f.load_month1(csv)
    tab = f.selection_table(month1).set_index("column")
    assert list(tab.index[tab.kept]) == ["TransactionAmt", "ProductCD", "card1",
                                         "P_emaildomain", "C1"]
    assert tab.loc["D4", "reason_dropped"] == "missing share >= 0.2"      # exactly 20%
    assert tab.loc["C3", "reason_dropped"] == "constant in month 1"
    assert tab.loc["C1", "missing_share_month1"] == pytest.approx(0.10)
    assert dict(tab["type"]) == {"TransactionAmt": "numerical", "ProductCD": "categorical",
                                 "card1": "categorical", "P_emaildomain": "categorical",
                                 "C1": "numerical", "C3": "numerical", "D4": "numerical"}
    assert tab.loc["TransactionAmt", "transform"] == "log1p, median imputation"


def test_undeclared_text_column_is_an_error(csv):
    month1, _ = f.load_month1(csv)
    with pytest.raises(ValueError):
        f.selection_table(month1.assign(Vendor=["x", "y"] * 50))


def test_parameters_come_from_month_1_only(csv):
    full = frame()
    month1, _ = f.load_month1(csv)
    feats = ["TransactionAmt", "ProductCD", "card1", "P_emaildomain", "C1"]
    rule = f.FeatureRule.fit(month1, feats)
    # medians: log1p first, month-1 rows only (the later rows are 1e6 and 500)
    assert rule.medians["TransactionAmt"] == pytest.approx(np.median(np.log1p(np.arange(1.0, 101))))
    assert rule.medians["C1"] == pytest.approx(np.nanmedian(full["C1"][:100]))
    assert rule.frequencies["ProductCD"] == {"W": 0.6, "C": 0.3, "H": 0.1}
    assert rule.frequencies["P_emaildomain"] == {"a.com": 0.5, "b.com": 0.35, f.MISSING: 0.15}
    X = rule.transform(full)
    assert X.shape == (140, 5) and np.isfinite(X).all()
    later = X[100:]
    assert (later[:, 1] == 0.0).all()                       # ProductCD "NEW": not seen in month 1
    assert (later[:, 0] == np.log1p(1e6)).all()
    assert (later[:, 3] == 0.5).all()                       # a.com keeps its month-1 share
    assert (X[50:65, 3] == 0.15).all()                      # missing is its own value
    assert (X[:10, 4] == rule.medians["C1"]).all()          # imputed with the month-1 median
    assert (later[:, 2] == 0.2).all()                       # card1 as float or int: same level
    with pytest.raises(AssertionError):
        f.FeatureRule.fit(full, feats)                      # a frame with the label is refused


def test_validate_rejects_broken_files(csv):
    month1, info = f.load_month1(csv)
    cfg = f.frozen_dict(f.selection_table(month1), info)
    f.validate(cfg)
    assert cfg["n_features"] == 5 and cfg["n_candidates"] == 7
    assert [d["name"] for d in cfg["dropped"]] == ["C3", "D4"]

    def broken(change):
        c = copy.deepcopy(cfg)
        change(c)
        with pytest.raises(ValueError):
            f.validate(c)

    broken(lambda c: c.update(name="freeze-v1"))
    broken(lambda c: c.update(n_features=4))
    broken(lambda c: c["features"].append(dict(c["features"][0])))
    broken(lambda c: c["features"][0].update(name="isFraud"))
    broken(lambda c: c["features"][0].update(name="TransactionDT"))
    broken(lambda c: c["features"][0].update(name="V12"))
    broken(lambda c: c["features"][1].update(type="numerical"))
    broken(lambda c: c["features"][4].update(type="text"))
    broken(lambda c: c["features"][4].update(missing_share_month1=0.2))
    broken(lambda c: c["dropped"].append({"name": "C1"}))


@pytest.mark.skipif(not f.RAW.exists(), reason="IEEE-CIS raw file not downloaded")
def test_frozen_file_equals_the_rule_on_the_real_month_1():
    if not f.FROZEN_FEATURES.exists():
        pytest.fail("frozen_features_v2.json is missing; run python -m thesis.ieee_features")
    cfg = f.load_frozen()
    month1, info = f.load_month1()
    assert "isFraud" not in month1.columns
    assert month1["TransactionDT"].max() < month1["TransactionDT"].min() + 30 * DAY
    fresh = f.frozen_dict(f.selection_table(month1), info)
    assert fresh == cfg
    X = f.FeatureRule.fit(month1, [x["name"] for x in cfg["features"]]).transform(month1)
    assert X.shape == (info["rows_month1"], cfg["n_features"]) and np.isfinite(X).all()
