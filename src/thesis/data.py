"""Dataset loaders and train/test splits (Master Plan § Dataset decision).

Every split runs its leakage checks before returning, so a split that breaks a rule never
reaches a detector. Labels are returned separately and only for evaluation; nothing in this
module fits anything on them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from thesis import config
from thesis.paths import DATA_RAW

ADBENCH_DIR = DATA_RAW / "adbench"
ULB_CSV = DATA_RAW / "ulb" / "creditcard.csv"

# ADBench selection rule, stated before any result is shown.
ADBENCH_MIN_ROWS, ADBENCH_MAX_ROWS = 1_000, 50_000
ADBENCH_MAX_FEATURES = 100
ADBENCH_MAX_ANOMALY_RATE = 0.10
ADBENCH_EXCLUDED = ("fraud",)  # 13_fraud is the ULB dataset; keeping it would leak into E3
ADBENCH_15 = ("ALOI", "annthyroid", "cardio", "letter", "mammography", "mnist", "optdigits",
              "PageBlocks", "pendigits", "satimage-2", "shuttle", "thyroid", "vowels",
              "Waveform", "Wilt")
ADBENCH_SEEDS = (0, 1, 2, 3, 4)
ADBENCH_TEST_SIZE = 0.4

ULB_DAY_SECONDS = 86_400
ULB_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


@dataclass(frozen=True)
class Split:
    dataset: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray  # evaluation only, never passed to fit()
    y_test: np.ndarray
    idx_train: np.ndarray  # row positions in the full dataset
    idx_test: np.ndarray
    t_train: np.ndarray | None = None  # timestamps, fraud data only
    t_test: np.ndarray | None = None
    seed: int | None = None


# ---------------------------------------------------------------- leakage checks

def assert_disjoint(split: Split, n_rows: int) -> None:
    """Train and test rows are disjoint and together cover the dataset exactly once."""
    tr, te = split.idx_train, split.idx_test
    assert np.intersect1d(tr, te).size == 0, f"{split.dataset}: train and test share rows"
    assert np.array_equal(np.sort(np.r_[tr, te]), np.arange(n_rows)), \
        f"{split.dataset}: split does not cover every row exactly once"
    assert len(split.X_train) == len(split.y_train) == len(tr)
    assert len(split.X_test) == len(split.y_test) == len(te)


def assert_temporal(split: Split) -> None:
    """Fraud data: every training row is strictly earlier than every test row."""
    assert split.t_train is not None and split.t_test is not None, \
        f"{split.dataset}: temporal split without timestamps"
    assert split.t_train.max() < split.t_test.min(), \
        f"{split.dataset}: training row at t={split.t_train.max()} is not earlier than " \
        f"test row at t={split.t_test.min()}"


# ---------------------------------------------------------------- ADBench

def adbench_files() -> dict[str, Path]:
    """Map ADBench dataset name (e.g. 'cardio') to its .npz file."""
    files = {p.stem.split("_", 1)[1]: p for p in ADBENCH_DIR.glob("*.npz")}
    if not files:
        raise FileNotFoundError(f"No ADBench files in {ADBENCH_DIR}; run python -m thesis.download")
    return files


def adbench_catalogue() -> pd.DataFrame:
    """Size, dimension and anomaly rate of every downloaded ADBench classical dataset."""
    rows = []
    for name, path in adbench_files().items():
        with np.load(path, allow_pickle=False) as d:
            n, dim = d["X"].shape
            rows.append({"dataset": name, "file": path.name, "n_rows": n, "n_features": dim,
                         "n_anomalies": int(d["y"].sum()), "anomaly_rate": float(d["y"].mean())})
    return pd.DataFrame(rows).sort_values("dataset", key=lambda s: s.str.lower()).reset_index(drop=True)


def select_adbench(catalogue: pd.DataFrame) -> list[str]:
    """Apply the selection rule; returns dataset names in case-insensitive alphabetical order."""
    keep = (catalogue.n_rows.between(ADBENCH_MIN_ROWS, ADBENCH_MAX_ROWS)
            & (catalogue.n_features <= ADBENCH_MAX_FEATURES)
            & (catalogue.anomaly_rate <= ADBENCH_MAX_ANOMALY_RATE)
            & ~catalogue.dataset.isin(ADBENCH_EXCLUDED))
    return sorted(catalogue.dataset[keep], key=str.lower)


def load_adbench(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Features as float64 and labels as int8 (1 = anomaly).

    Duplicate rows are kept unless config.PROTOCOL.adbench_dedup is on.
    """
    with np.load(adbench_files()[name], allow_pickle=False) as d:
        X, y = d["X"].astype(np.float64), d["y"].astype(np.int8)
    assert np.isfinite(X).all(), f"{name}: non-finite feature values"
    assert set(np.unique(y)) <= {0, 1}, f"{name}: labels are not binary"
    if config.PROTOCOL.adbench_dedup:
        X, y = dedup_rows(X, y)
    return X, y


def dedup_rows(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first occurrence of each exact (features, label) row, in file order.

    Rows with equal features but different labels are both kept (they are not duplicates).
    """
    _, first = np.unique(np.column_stack([X, y]), axis=0, return_index=True)
    keep = np.sort(first)
    return X[keep], y[keep]


def duplicate_stats(X: np.ndarray, y: np.ndarray) -> dict:
    """How much exact duplication a dataset has (for the duplicate table and the dedup check).

    rows_in_dup_groups: rows whose (features, label) occurs more than once
    rows_removed / anomalies_removed: what dedup_rows drops
    label_conflict_rows: rows whose features also occur with the other label (kept by dedup)
    """
    _, inv, counts = np.unique(np.column_stack([X, y]), axis=0, return_inverse=True,
                               return_counts=True)
    _, xinv = np.unique(X, axis=0, return_inverse=True)
    inv, xinv = inv.ravel(), xinv.ravel()
    labels_per_x = pd.Series(y).groupby(xinv).nunique().to_numpy()
    in_group = counts[inv] > 1
    first = np.zeros(len(y), dtype=bool)
    first[np.unique(inv, return_index=True)[1]] = True
    _, yd = dedup_rows(X, y)
    return {"n_rows": len(y), "n_anomalies": int(y.sum()),
            "rows_in_dup_groups": int(in_group.sum()),
            "anomalies_in_dup_groups": int((in_group & (y == 1)).sum()),
            "rows_removed": int((~first).sum()),
            "anomalies_removed": int(((~first) & (y == 1)).sum()),
            "label_conflict_rows": int((labels_per_x[xinv] > 1).sum()),
            "n_rows_dedup": len(yd), "anomaly_rate": float(y.mean()),
            "anomaly_rate_dedup": float(yd.mean())}


def split_adbench(name: str, seed: int, X: np.ndarray | None = None,
                  y: np.ndarray | None = None) -> Split:
    """Random stratified 60/40 split. Training rows keep their anomalies (unsupervised setting)."""
    if X is None or y is None:
        X, y = load_adbench(name)
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=ADBENCH_TEST_SIZE,
                                           stratify=y, random_state=seed)
    idx_train, idx_test = np.sort(idx_train), np.sort(idx_test)
    split = Split(name, X[idx_train], X[idx_test], y[idx_train], y[idx_test],
                  idx_train, idx_test, seed=seed)
    assert_disjoint(split, len(y))
    return split


# ---------------------------------------------------------------- ULB credit card

def load_ulb() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features (V1–V28, log1p(Amount)), labels, and Time in seconds since the first transaction.

    Time is the split variable and is not a feature. log1p is parameter-free, so it cannot leak.
    """
    df = pd.read_csv(ULB_CSV)
    X = df[ULB_FEATURES].to_numpy(dtype=np.float64)
    X[:, -1] = np.log1p(X[:, -1])
    y, t = df["Class"].to_numpy(dtype=np.int8), df["Time"].to_numpy(dtype=np.float64)
    assert np.isfinite(X).all() and set(np.unique(y)) <= {0, 1}
    return X, y, t


def load_ulb_unlabelled() -> tuple[np.ndarray, np.ndarray]:
    """All ULB features (same columns and log1p(Amount) as load_ulb) and Time, without labels.

    The Class column is never read (usecols), so nothing computed from this can use a label.
    For label-free diagnostics of day 1 against day 2.
    """
    df = pd.read_csv(ULB_CSV, usecols=ULB_FEATURES + ["Time"])
    assert "Class" not in df.columns
    X = df[ULB_FEATURES].to_numpy(dtype=np.float64)
    X[:, -1] = np.log1p(X[:, -1])
    assert np.isfinite(X).all()
    return X, df["Time"].to_numpy(dtype=np.float64)


def load_ulb_day1_features() -> np.ndarray:
    """Day-1 features only (same columns and log1p(Amount) as load_ulb), without labels.

    The Class column is never read, and day-2 rows are dropped right after reading Time, so
    nothing here can see a label or a test row. For training-score checks before the freeze.
    """
    df = pd.read_csv(ULB_CSV, usecols=ULB_FEATURES + ["Time"])
    assert "Class" not in df.columns
    df = df[df["Time"] < ULB_DAY_SECONDS]
    X = df[ULB_FEATURES].to_numpy(dtype=np.float64)
    X[:, -1] = np.log1p(X[:, -1])
    assert np.isfinite(X).all()
    return X


def split_ulb(X: np.ndarray | None = None, y: np.ndarray | None = None,
              t: np.ndarray | None = None) -> Split:
    """Day 1 (Time < 86,400 s) fit, day 2 test."""
    if X is None or y is None or t is None:
        X, y, t = load_ulb()
    is_train = t < ULB_DAY_SECONDS
    idx_train, idx_test = np.flatnonzero(is_train), np.flatnonzero(~is_train)
    split = Split("ulb", X[idx_train], X[idx_test], y[idx_train], y[idx_test],
                  idx_train, idx_test, t_train=t[idx_train], t_test=t[idx_test])
    assert_disjoint(split, len(y))
    assert_temporal(split)
    return split


# ---------------------------------------------------------------- IEEE-CIS time windows
# The rules of frozen_config.json data.ieee_cis.time and hypotheses.H3b.windows. They use
# TransactionDT only, never a label.

IEEE_DAY_SECONDS = 86_400
IEEE_MONTH_DAYS = 30
IEEE_LAST_MONTH = 6          # month 6 = day 150 onwards (absorbs the tail)
IEEE_TEST_MONTHS = (2, 3, 4, 5, 6)
IEEE_WEEK_START_DAY = 30     # weekly H3b windows start with the first test day
IEEE_WEEK_DAYS = 7


def ieee_cis_day(transaction_dt: np.ndarray) -> np.ndarray:
    """Day 0, 1, ... counted in 24-hour blocks from the first TransactionDT."""
    t = np.asarray(transaction_dt, dtype=np.float64)
    return np.floor((t - t.min()) / IEEE_DAY_SECONDS).astype(np.int64)


def ieee_cis_month(day: np.ndarray) -> np.ndarray:
    """Month 1 = days 0-29, month m = days 30(m-1) to 30m-1, month 6 = day 150 onwards."""
    return np.minimum(np.asarray(day) // IEEE_MONTH_DAYS + 1, IEEE_LAST_MONTH)


def ieee_cis_week(day: np.ndarray) -> np.ndarray:
    """H3b week 0, 1, ...: 7-day blocks from day 30; a final partial week joins the last block.

    Month-1 days (before day 30) get -1: they are the reference, not a window.
    """
    day = np.asarray(day)
    n_full = (int(day.max()) - IEEE_WEEK_START_DAY + 1) // IEEE_WEEK_DAYS
    if n_full < 1:
        raise ValueError("fewer than 7 test days: no complete H3b week")
    w = np.minimum((day - IEEE_WEEK_START_DAY) // IEEE_WEEK_DAYS, n_full - 1)
    return np.where(day < IEEE_WEEK_START_DAY, -1, w)


# ---------------------------------------------------------------- preprocessing

@dataclass(frozen=True)
class Standardizer:
    """Z-score scaler whose parameters come from the training split only.

    `fit` takes no labels by design. Constant training columns get scale 1 so they stay finite.
    """
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X_train: np.ndarray) -> "Standardizer":
        mean, std = X_train.mean(axis=0), X_train.std(axis=0)
        return cls(mean, np.where(std > 0, std, 1.0))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


def preprocess(split: Split) -> Split:
    """Detector input: z-scored with training statistics, or unchanged if zscore_inputs is off."""
    return standardize(split) if config.PROTOCOL.zscore_inputs else split


def standardize(split: Split) -> Split:
    """Return a copy of the split with both parts scaled by training-split statistics."""
    sc = Standardizer.fit(split.X_train)
    return Split(split.dataset, sc.transform(split.X_train), sc.transform(split.X_test),
                 split.y_train, split.y_test, split.idx_train, split.idx_test,
                 split.t_train, split.t_test, split.seed)
