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
- ADBench files are used as distributed, duplicates included (this is not a full reproduction of the
  ADBench protocol). `python -m thesis.export_datasets` writes the duplicate table
  `results/tables/datasets/duplicates.csv`.

## Detectors, fusion, E1

```
.venv\Scripts\python.exe -m thesis.e1          # all E1 outputs: e1_runs.csv + summary tables in results/tables/e1/,
                                               # CD diagrams (Friedman + Nemenyi) in results/figures/e1/
                                               # (scores from cache/scores/; missing ones are computed)
.venv\Scripts\python.exe -m thesis.e1 cardio   # quick check on one dataset, writes nothing to results/
.venv\Scripts\python.exe -m thesis.e1_corr     # Spearman heatmap of detector test scores + per-dataset CSV
                                               # in results/figures/e1/ (from the score cache)
.venv\Scripts\python.exe -m thesis.e1 --no-zscore --k-rule round --no-pca-drop --dedup
                                               # provisional protocol switches (src/thesis/config.py);
                                               # non-default runs go to results/tables/e1/variants/<switches>/
.venv\Scripts\python.exe -m thesis.e1 --dedup  # duplicate robustness check: E1 without exact duplicates,
.venv\Scripts\python.exe -m thesis.e1_dedup    # then ranks compared with the main E1 (variants/dedup/dedup_vs_main.*)
.venv\Scripts\python.exe -m thesis.e2          # E2: 15 pairs + 20 triples x 6 fusion configs from the score cache
                                               # -> results/tables/e2/ (runs, per dataset, summary, headline,
                                               # Friedman tests, triple ranking in e2_summary.md) and the figures
                                               # in results/figures/e2/; same switches as thesis.e1
.venv\Scripts\python.exe -m thesis.e2_figures  # rebuild only the E2 figures and e2_triples.tex from the E2 tables
.venv\Scripts\python.exe -m thesis.ebench      # E-bench rule applied to the E2 tables -> results/tables/e2/ebench.md
```

## Before the freeze

```
.venv\Scripts\python.exe -m thesis.zero_variance  # constant training scores: ADBench score cache + ULB day 1
                                                  # (no labels read, day 2 not scored) -> results/tables/setup/
.venv\Scripts\python.exe -m thesis.export_setup   # Experimental Setup tables (datasets, hyperparameters,
                                                  # versions, runtimes; CSV + booktabs) -> results/tables/setup/
```

- `frozen_config.json` holds the ensembles (E-lit, E-bench, E-bench-5%, E-all), fusion configs, budgets,
  thresholds, rules, data handling, metrics and baselines. `thesis.frozen.load()` reads and checks it;
  `tests/test_frozen.py` checks that it agrees with the code and that E-bench follows its rule.
- KNN / LOF fit sample: `config.KNN_LOF_MAX_FIT_ROWS` (30,000) rows, drawn with the split seed or
  `config.KNN_LOF_SAMPLE_SEED` (0) when the split has none; it never applies to ADBench.

- `detectors.score(name, X_fit, X_score, seed)` → (training scores, new-row scores); PyOD defaults, inputs
  z-scored with training statistics. Deviations: ECOD is a train-only re-implementation (PyOD's ECOD scores new
  rows together with the training rows, so scores depend on the batch); PCA drops zero-variance components
  (PyOD's weighted PCA gives infinite scores otherwise, e.g. on cardio). KNN/LOF use leave-one-out training scores.
- `fusion.fuse(S_train, S_test, calib, agg)`: min-max / z-score / train-percentile × mean / max, calibrators fitted
  on training scores only. Primary: min-max + mean.
- `metrics`: FP@K, Precision@K, Recall@K with K = ceil(budget × n_test); ties at the K-th score are split in
  expectation, so row order never decides an alert.
- E2 overlap (Spearman rho, top-K Jaccard) uses training scores only; the best member of a combo is chosen
  with test labels (an oracle reference, ADBench only). Percentile calibration maps every test score above the
  training maximum to 1.0, so it can create ties at the top of the fused score. "Beats" is strict; exact ties
  with a member are frequent, so the headline reports beats and ties (loses = the rest).
- The cache folder name is a hash of detector parameters, preprocessing and PyOD version (`scorecache.cache_key`).

## Layout

- `src/thesis/`: loaders, detectors (`score`), fusion (`fuse`), metrics, splits
- `tests/`: leakage rules as tests
- `data/raw/`, `cache/scores/`: not in git; regenerable
- `frozen_config.json`: written and tagged `freeze-v1` before any fraud-data result; never edited afterwards
