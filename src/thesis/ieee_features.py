"""IEEE-CIS feature rule: fixed from month 1 only and without labels (freeze-v2).

    python -m thesis.ieee_features    # writes frozen_features_v2.json (repo root) and
                                      # results/tables/ieee_cis/features_month1.{csv,meta.json}

The rule (Master Plan, "How to keep IEEE-CIS easy"), applied to train_transaction.csv:
1. Candidates: every column except TransactionID, isFraud, TransactionDT (the split variable)
   and the V columns (V1-V339).
2. Keep a candidate if less than 20% of its month-1 values are missing and it takes more than
   one value in month 1.
3. Categorical columns (the ones the data description lists as categorical): frequency encoding,
   value -> share of month-1 rows with that value; missing is its own value; a value not seen in
   month 1 -> 0.
4. Numerical columns: missing -> month-1 median. TransactionAmt -> log1p first.
The z-score of every detector input (frozen in freeze-v1) follows afterwards.

What is read: TransactionDT of all rows (to find where month 1 ends; a timestamp, no label),
then the first rows of the file up to the end of month 1, without the isFraud column. No
feature value of months 2-6 and no label is loaded here.
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from thesis import data
from thesis.export_datasets import git_state
from thesis.paths import DATA_RAW, REPO, TABLES

RAW = DATA_RAW / "ieee_cis" / "train_transaction.csv.zip"
FROZEN_FEATURES = REPO / "frozen_features_v2.json"
OUT = TABLES / "ieee_cis"
LABEL, TIME = "isFraud", "TransactionDT"
NOT_FEATURES = ("TransactionID", LABEL, TIME)
V_COLUMN = re.compile(r"V\d+")
MAX_MISSING = 0.20                      # keep if the month-1 missing share is strictly below
# categorical columns of the transaction table according to the competition's data description
CATEGORICAL = ("ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1",
               "addr2", "P_emaildomain", "R_emaildomain",
               "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9")
LOG1P = ("TransactionAmt",)
MISSING = "(missing)"                   # the level a missing categorical value is counted as


# ---------------------------------------------------------------- month 1, no labels

def month1_row_count(transaction_dt: np.ndarray) -> int:
    """Number of leading rows that belong to month 1; the file must be sorted by time."""
    t = np.asarray(transaction_dt, dtype=np.float64)
    if (np.diff(t) < 0).any():
        raise ValueError("TransactionDT is not sorted; month 1 is not a leading block")
    return int((data.ieee_cis_month(data.ieee_cis_day(t)) == 1).sum())


def load_month1(path=None) -> tuple[pd.DataFrame, dict]:
    """Month-1 rows of the transaction table without the label column."""
    path = RAW if path is None else path
    t = pd.read_csv(path, usecols=[TIME])[TIME].to_numpy()
    n1 = month1_row_count(t)
    header = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(path, usecols=[c for c in header if c != LABEL], nrows=n1)
    assert LABEL not in df.columns and len(df) == n1
    assert np.array_equal(df[TIME].to_numpy(), t[:n1])
    day = data.ieee_cis_day(t)[:n1]
    assert day.max() < data.IEEE_MONTH_DAYS and (n1 == len(t) or t[n1] > t[n1 - 1])
    return df, {"rows_month1": n1, "rows_file": int(len(t)), "days_month1": int(day.max()) + 1,
                "transaction_dt_first": int(t[0]), "transaction_dt_last_month1": int(t[n1 - 1])}


# ---------------------------------------------------------------- the rule

def candidates(columns) -> list[str]:
    return [c for c in columns if c not in NOT_FEATURES and not V_COLUMN.fullmatch(c)]


def kind(column: str) -> str:
    return "categorical" if column in CATEGORICAL else "numerical"


def selection_table(month1: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate: type, month-1 missing share, distinct values, kept or why not."""
    rows = []
    for c in candidates(month1.columns):
        s = month1[c]
        missing = float(s.isna().mean())
        n_values = int(s.nunique(dropna=True)) + int(missing > 0)     # missing counts as a value
        if missing >= MAX_MISSING:
            reason = f"missing share >= {MAX_MISSING:g}"
        elif n_values <= 1:
            reason = "constant in month 1"
        else:
            reason = ""
        rows.append({"column": c, "type": kind(c), "missing_share_month1": missing,
                     "distinct_values_month1": n_values, "kept": reason == "",
                     "reason_dropped": reason,
                     "transform": ("frequency encoding" if kind(c) == "categorical" else
                                   ("log1p, median imputation" if c in LOG1P
                                    else "median imputation"))})
    tab = pd.DataFrame(rows)
    if kind_mismatch := [c for c in tab.column[tab.type == "numerical"]
                         if not pd.api.types.is_numeric_dtype(month1[c])]:
        raise ValueError(f"non-numeric columns not declared categorical: {kind_mismatch}")
    return tab


@dataclass(frozen=True)
class FeatureRule:
    """Imputation medians and frequency maps; every parameter comes from month 1 only."""
    features: tuple[str, ...]
    medians: dict[str, float]
    frequencies: dict[str, dict]

    @staticmethod
    def _levels(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):            # card1 = 1000 and 1000.0 are one level
            s = s.astype(np.float64)
        return s.astype(object).where(s.notna(), MISSING)

    @staticmethod
    def _numeric(s: pd.Series, column: str) -> pd.Series:
        s = s.astype(np.float64)
        return np.log1p(s) if column in LOG1P else s

    @classmethod
    def fit(cls, month1: pd.DataFrame, features) -> "FeatureRule":
        assert LABEL not in month1.columns
        medians, freqs = {}, {}
        for c in features:
            if kind(c) == "categorical":
                freqs[c] = cls._levels(month1[c]).value_counts(normalize=True).to_dict()
            else:
                medians[c] = float(cls._numeric(month1[c], c).median())
        return cls(tuple(features), medians, freqs)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.features:
            if c in self.frequencies:
                x = self._levels(df[c]).map(self.frequencies[c]).fillna(0.0)
            else:
                x = self._numeric(df[c], c).fillna(self.medians[c])
            cols.append(x.to_numpy(dtype=np.float64))
        X = np.column_stack(cols)
        assert np.isfinite(X).all()
        return X


# ---------------------------------------------------------------- the frozen file

def frozen_dict(tab: pd.DataFrame, info: dict) -> dict:
    kept = tab[tab.kept]
    return {
        "name": "freeze-v2",
        "scope": "IEEE-CIS feature preprocessing only (frozen_config.json data.ieee_cis."
                 "features_freeze_v2). Nothing else of freeze-v1 changes.",
        "source": "Kaggle ieee-fraud-detection, train_transaction.csv",
        "fitted_on": "month 1 (days 0-29) only, without labels: the isFraud column is not read "
                     "and no feature value of months 2-6 is loaded when the rule is fixed",
        "rule": {
            "candidates": "all columns except TransactionID, isFraud, TransactionDT and V1-V339",
            "keep_if": f"month-1 missing share < {MAX_MISSING:g} and more than one value in "
                       "month 1 (missing counts as a value)",
            "categorical": "the columns the data description lists as categorical: "
                           + ", ".join(CATEGORICAL),
            "categorical_encoding": "frequency encoding: value -> share of fit-window rows "
                                    "with that value; missing is its own value; a value not "
                                    "seen in the fit window -> 0",
            "numerical": "missing -> fit-window median",
            "log1p": list(LOG1P),
            "then": "z-score with fit-window statistics (freeze-v1 preprocessing)",
            "parameters": "the fit window is month 1 for the evaluation of months 2-6; a "
                          "run that fits on a part of month 1 uses that part only. Medians and "
                          "frequency maps are computed from the fit window by "
                          "thesis.ieee_features.FeatureRule.fit and applied unchanged to every "
                          "later row. The feature list is the same in every run",
            "rows": "every row of every month is kept: no row filter, no deduplication",
            "drift_columns": "H3b (frozen_config.json hypotheses.H3b.drift_share) uses the raw "
                             "columns of the feature list, before encoding and imputation, "
                             "with the column types listed here",
        },
        "month1": info,
        "n_candidates": int(len(tab)), "n_features": int(len(kept)),
        "features": [{"name": r.column, "type": r.type, "transform": r.transform,
                      "missing_share_month1": round(r.missing_share_month1, 6)}
                     for r in kept.itertuples()],
        "dropped": [{"name": r.column, "type": r.type, "reason": r.reason_dropped,
                     "missing_share_month1": round(r.missing_share_month1, 6)}
                    for r in tab[~tab.kept].itertuples()],
    }


def validate(cfg: dict) -> None:
    """Structure checks of frozen_features_v2.json; raises ValueError."""
    names = [f["name"] for f in cfg["features"]]
    if cfg["name"] != "freeze-v2" or cfg["n_features"] != len(names) or not names:
        raise ValueError("freeze-v2: name or feature count")
    if len(set(names)) != len(names):
        raise ValueError("freeze-v2: duplicate feature")
    bad = [n for n in names if n in NOT_FEATURES or V_COLUMN.fullmatch(n)]
    if bad:
        raise ValueError(f"freeze-v2: not allowed as features: {bad}")
    for f in cfg["features"]:
        if f["type"] not in ("numerical", "categorical") or f["type"] != kind(f["name"]):
            raise ValueError(f"freeze-v2: column type of {f['name']}")
        if not f["missing_share_month1"] < MAX_MISSING:
            raise ValueError(f"freeze-v2: {f['name']} breaks the missing-share rule")
    if set(names) & {d["name"] for d in cfg["dropped"]}:
        raise ValueError("freeze-v2: a feature is also listed as dropped")


def load_frozen(path=None) -> dict:
    cfg = json.loads((FROZEN_FEATURES if path is None else path).read_text(encoding="utf-8"))
    validate(cfg)
    return cfg


def main(argv: list[str]) -> None:
    if argv:
        raise SystemExit("usage: python -m thesis.ieee_features")
    state = git_state()
    if state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    t0 = time.perf_counter()
    month1, info = load_month1()
    tab = selection_table(month1)
    cfg = frozen_dict(tab, info)
    validate(cfg)
    rule = FeatureRule.fit(month1, [f["name"] for f in cfg["features"]])
    X = rule.transform(month1)
    FROZEN_FEATURES.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    tab.assign(commit=state["commit"]).to_csv(OUT / "features_month1.csv", index=False)
    meta = {**state, "command": "python -m thesis.ieee_features",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "labels_read": False, "rows_read": "month 1 only (TransactionDT of all rows to find "
            "the end of month 1)", **info, "n_candidates": int(len(tab)),
            "n_features": cfg["n_features"], "matrix_shape_month1": list(X.shape),
            "constant_after_transform": [f for j, f in enumerate(rule.features)
                                         if np.ptp(X[:, j]) == 0],
            "runtime_s": round(time.perf_counter() - t0, 1)}
    (OUT / "features_month1.meta.json").write_text(json.dumps(meta, indent=1) + "\n",
                                                   encoding="utf-8")
    pd.set_option("display.width", 200, "display.max_rows", 100)
    print(tab.to_string(index=False))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
