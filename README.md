# Unsupervised Model Ensembling for False Alarm Reduction in Anomaly Detection

BSc thesis code (BME VIK). Planning documents, `results/` and writing live in the OneDrive project folder
(`C:\Users\rifat\OneDrive\Documents\thesis-docs`, see `src/thesis/paths.py`).

## Setup

```
"C:\Program Files\Python311\python.exe" -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pytest
```

Python 3.11. pandas stays below 3 because adtk 0.6.2 does not run on pandas 3.

## Data

```
set PYTHONPATH=src            (PowerShell: $env:PYTHONPATH="src")
.venv\Scripts\python.exe -m thesis.download          # ADBench (pinned commit) + ULB (Kaggle), then checksum check
.venv\Scripts\python.exe -m thesis.export_datasets   # dataset table -> results/tables/datasets/
```

- ADBench: all 47 classical `.npz` files from `Minqi824/ADBench` at the commit in `src/thesis/download.py`.
  The selection rule in `src/thesis/data.py` picks the 15 sets; `tests/test_data.py` checks it.
- ULB: `creditcard.csv` from Kaggle. Features V1–V28 + log1p(Amount); `Time` is the split variable only.
  Day 1 = `Time < 86400` (fit), day 2 = test.
- `data/manifest.json` holds SHA-256 checksums of every raw file.

## Detectors, fusion, E1

```
.venv\Scripts\python.exe -m thesis.e1          # all E1 outputs: e1_runs.csv + summary tables in results/tables/e1/
                                               # (scores from cache/scores/; missing ones are computed)
.venv\Scripts\python.exe -m thesis.e1 cardio   # quick check on one dataset, writes nothing to results/
.venv\Scripts\python.exe -m thesis.e1 --no-zscore --k-rule round --no-pca-drop --dedup
                                               # provisional protocol switches (src/thesis/config.py);
                                               # non-default runs go to results/tables/e1/variants/<switches>/
```

- `detectors.score(name, X_fit, X_score, seed)` → (training scores, new-row scores); PyOD defaults, inputs
  z-scored with training statistics. Deviations: ECOD is a train-only re-implementation (PyOD's ECOD scores new
  rows together with the training rows, so scores depend on the batch); PCA drops zero-variance components
  (PyOD's weighted PCA gives infinite scores otherwise, e.g. on cardio). KNN/LOF use leave-one-out training scores.
- `fusion.fuse(S_train, S_test, calib, agg)`: min-max / z-score / train-percentile × mean / max, calibrators fitted
  on training scores only. Primary: min-max + mean.
- `metrics`: FP@K, Precision@K, Recall@K with K = ceil(budget × n_test); ties at the K-th score are split in
  expectation, so row order never decides an alert.
- The cache folder name is a hash of detector parameters, preprocessing and PyOD version (`scorecache.cache_key`).

## Layout

- `src/thesis/`: loaders, detectors (`score`), fusion (`fuse`), metrics, splits
- `tests/`: leakage rules as tests
- `data/raw/`, `cache/scores/`: not in git; regenerable
- `frozen_config.json`: written and tagged `freeze-v1` before any fraud-data result; never edited afterwards
