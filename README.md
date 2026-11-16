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
                                                  # versions, runtimes; CSV + booktabs; duplicates.tex from
                                                  # tables/datasets/duplicates.csv) -> results/tables/setup/
```

- Every exported `.tex` table must compile in the TDK preamble (`tdk/main.tex` in the project folder) and fit
  its 16 cm text block: `tests/test_tdk_exports.py` (needs XeLaTeX; `thesis.texcheck`). Wide tables use a
  narrower `\tabcolsep` (`booktabs(..., colsep=...)`), never different content.

- `frozen_config.json` holds the ensembles (E-bench primary, E-lit = E-bench, E-bench-5% and E-all secondary),
  fusion configs, budgets and thresholds for ULB and IEEE-CIS, rules, data handling, metrics, seed handling,
  bootstrap, the hypotheses with their criterion (H3b and the E5 context exploratory), and the baselines.
  IEEE-CIS feature preprocessing is the only part fixed later (freeze-v2, from month-1 data without labels).
  It is `frozen_features_v2.json`: the rule and the list of 26 features with their column types, written by
  `python -m thesis.ieee_features` from the month-1 rows of `train_transaction.csv` (the `isFraud` column and
  the feature values of months 2-6 are not read). `thesis.ieee_features.FeatureRule` fits the medians and
  frequency maps on the fit window and applies them to later rows; `tests/test_ieee_features.py` checks that
  the file equals the rule applied to the real month 1.
  `thesis.frozen.load()` reads and checks it; `tests/test_frozen.py` checks that it agrees with the code and
  that E-bench follows its rule.
- `thesis.bootstrap`: the paired bootstrap of the frozen config. ULB resamples day-2 rows; IEEE-CIS resamples
  days within each of months 2-6 (month-stratified) and sums FP@K over the months, each month with its own K.
  In every resample K, the seed means, the median single and the oracle are recomputed from fixed scores.
  `numpy.random.default_rng(0)`, one stream. IEEE-CIS day / month / H3b week rules: `data.ieee_cis_day`,
  `ieee_cis_month`, `ieee_cis_week`.
- KNN / LOF fit sample: `config.KNN_LOF_MAX_FIT_ROWS` (30,000) rows, drawn with the split seed or
  `config.KNN_LOF_SAMPLE_SEED` (0) when the split has none; it never applies to ADBench.

## E3 on ULB (after freeze-v1)

```
python -m thesis.e3_ulb             # ~15 min; results/{tables,figures}/e3_ulb/
python -m thesis.e3_ulb --quick     # 20 resamples, prints the hypotheses, writes nothing
python -m thesis.e3_ulb_figures     # redraw the two figures (16 cm wide, text >= 8 pt) and rewrite the
                                    # three .tex tables from the e3_ulb CSVs; nothing recomputed
python -m thesis.e3_ulb_lof         # label-free LOF diagnosis (day-1 vs day-2 k-NN distances, feature
                                    # mean shift; Class never read) -> results/tables/e3_ulb/lof_diagnosis.*
```

- Refuses to run unless tag `freeze-v1` exists and `git diff freeze-v1 -- frozen_config.json` is empty, and
  checks that the code's settings equal the frozen ones (`e3_ulb.check_code_matches`). Ensembles, fusion
  configs, budgets, quantiles, seeds, bootstrap and hypotheses are all read from the file.
- Fitting stage (`e3_ulb.fit_stage`): gets the day-1 / day-2 split with its labels replaced by zeros; z-score,
  detectors, calibrators and fused training scores use day-1 rows only. Thresholds come from each method's own
  day-1 training scores. Day-2 labels are used only by the metrics. ULB scores are cached under the same
  cache key as ADBench (`cache/scores/<key>/ulb/`).
- Budget mode: seed means, median single, default IForest, oracle-best single (chosen with test labels: an
  upper bound) and regret; paired transaction-level bootstrap CIs (`thesis.bootstrap`) for every ensemble x
  fusion config against the three comparators and for the H2b secondary check. H1a / H1b report the ULB part
  and what it implies for the two-dataset rule. The frozen criterion has two outcomes; a CI entirely above 0
  is only flagged (`ci_entirely_above_0`), descriptively. In `e3_ulb_deltas.csv` only the hypothesis rows
  (`role` H1a, H1b, H2b secondary, primary budget) carry the criterion's verdict; every other row is
  "descriptive, no criterion applied" (`statistics.descriptive_cis`) with `ci_entirely_below_0` /
  `ci_entirely_above_0` as plain booleans.
- Threshold mode: day-1 quantile thresholds applied unchanged to day 2 (alert count, alert rate vs 1 - q,
  FPR, recall, precision).

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
