# Unsupervised Model Ensembling for False Alarm Reduction in Anomaly Detection

Code and results of a BSc thesis at the Budapest University of Technology and Economics, Faculty of
Electrical Engineering and Informatics (BME VIK), 2026. The thesis asks whether small ensembles of
classical unsupervised anomaly detectors, fixed in advance and run without labels, raise fewer false
alarms at a fixed alert budget than a single detector, and what happens when one member of an ensemble
breaks over time. Six detectors (Isolation Forest, PCA, KNN, LOF, ECOD, HBOS) are combined in pairs and
triples, the ensembles are compared on 15 ADBench datasets, the settings are frozen in tagged files, and
the frozen settings are then tested on two fraud datasets in time order (ULB credit-card fraud and
IEEE-CIS). The repository contains the code, the tests, the frozen configuration files and the tables
and figures of the thesis.

The version behind the TDK paper (November 2026) is the tag `tdk-submitted` in this repository and the
archived repository
[unsupervised-ensembling-false-alarms-tdk](https://github.com/RifatZaman0/unsupervised-ensembling-false-alarms-tdk).

## Contents

| Path | What |
|---|---|
| `src/thesis/` | loaders and splits, detectors, fusion, metrics, bootstrap, experiments and exporters |
| `tests/` | unit tests, including the leakage rules (labels never reach fitting, scaling, calibration or thresholds) |
| `frozen_config.json` | pre-registered ensembles, fusion configurations, budgets, thresholds and hypotheses (tag `freeze-v1`) |
| `frozen_features_v2.json` | IEEE-CIS feature rule and feature list, fixed from month 1 without labels (tag `freeze-v2`) |
| `frozen_stress_v3.json` | settings for the broken-member analyses on IEEE-CIS (tag `freeze-v3`) |
| `data/manifest.json` | SHA-256 checksums of the raw ADBench and ULB files |
| `results/` | tables (CSV, LaTeX, Markdown, JSON metadata) and figures (PDF, PNG) |
| `COMMIT_MAP.md` | old to new commit hashes (see "Freeze protocol") |
| `requirements-lock.txt` | exact package versions |

No raw data is included. `data/raw/` and the score cache `cache/scores/` are created locally.

## Setup

Python 3.11 is required.

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt      (Linux/macOS: .venv/bin/python)
```

pandas stays below version 3 because ADTK 0.6.2 does not run on pandas 3. Install only from
`requirements-lock.txt`.

All commands below run from the repository root with `src` on the Python path:

```
set PYTHONPATH=src            (PowerShell: $env:PYTHONPATH="src"; bash: export PYTHONPATH=src)
```

Outputs go to `results/` in the repository. Set `THESIS_PROJECT_DIR` to write them to another folder.

## Data

```
python -m thesis.download            # ADBench + ULB: download what is missing, then check the checksums
python -m thesis.download --verify   # check the checksums only
```

- **ADBench:** the 47 classical `.npz` files from
  [Minqi824/ADBench](https://github.com/Minqi824/ADBench) at the commit pinned in
  `src/thesis/download.py` (`3dac822`). A selection rule in `src/thesis/data.py` keeps 15 datasets.
  The files are used as distributed, duplicates included, so the setup is not a full reproduction of
  the ADBench protocol.
- **ULB:** `creditcard.csv` from Kaggle (`mlg-ulb/creditcardfraud`). The download needs a Kaggle API
  token (see the Kaggle API documentation). Day 1 (`Time < 86400`) is used for fitting, day 2 for testing.
- **IEEE-CIS:** `train_transaction.csv` from the Kaggle competition `ieee-fraud-detection`. You must
  accept the competition rules on Kaggle first. The download script does not fetch this file; download
  it with

  ```
  kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv -p data/raw/ieee_cis
  ```

  and keep it zipped as `data/raw/ieee_cis/train_transaction.csv.zip` (pandas reads it directly).
  SHA-256 of the zip used in the thesis:
  `d426943b810094085d572023df5e7d52e57d67f6e40d087bd88a3ebeae0b6c4a` (Kaggle may re-pack the zip, so
  compare the unzipped CSV if the zip hash differs). Month 1 (the first 134,339 rows) is used for
  fitting, months 2-6 for testing.

## Tests

```
python -m pytest
```

Tests that need the raw data skip when `data/raw/` is missing; tests that compile LaTeX tables skip
without XeLaTeX and the TDK paper preamble (`tdk/main.tex` under `THESIS_PROJECT_DIR`). Check the skip
count in the summary.

## Running the experiments

The modules record the git commit next to their outputs and refuse to write results from a working tree
with uncommitted changes. Scores are cached in `cache/scores/<key>/`; the key changes with detector
parameters, preprocessing or the PyOD version.

ADBench (benchmark steps):

```
python -m thesis.export_datasets   # dataset table and duplicate table -> results/tables/datasets/
python -m thesis.e1                # E1: six detectors, 15 datasets x 5 seeds -> results/{tables,figures}/e1/
python -m thesis.e1 cardio         # quick check on one dataset, writes nothing
python -m thesis.e1_corr           # Spearman correlation of detector scores
python -m thesis.e1 --dedup        # robustness check without exact duplicates, then:
python -m thesis.e1_dedup          #   ranks compared with the main E1
python -m thesis.e2                # E2: all pairs and triples x 6 fusion configurations -> results/{tables,figures}/e2/
python -m thesis.e2_figures        # redraw the E2 figures from the E2 tables
python -m thesis.ebench            # the E-bench selection rule -> results/tables/e2/ebench.md
python -m thesis.zero_variance     # constant training scores (fusion rule check)
python -m thesis.export_setup      # setup tables (datasets, hyperparameters, versions, runtimes)
python -m thesis.robust_fusion     # robust fusion on development data -> results/tables/robust_fusion/
```

Protocol switches (non-default runs go to `results/tables/e1/variants/<switches>/`):
`python -m thesis.e1 --no-zscore --k-rule round --no-pca-drop --dedup`.

ULB (after `freeze-v1`):

```
python -m thesis.e3_ulb            # budget and threshold mode, bootstrap CIs -> results/{tables,figures}/e3_ulb/
python -m thesis.e3_ulb --quick    # 20 resamples, prints, writes nothing
python -m thesis.e3_ulb_figures    # redraw the ULB figures and tables from the CSVs
python -m thesis.e3_ulb_lof        # label-free LOF diagnosis
python -m thesis.lof_diag l0       # LOF diagnosis steps (also l2b, l2)
python -m thesis.e_stress          # broken-member analyses on ULB -> results/tables/e3_ulb/stress/
python -m thesis.guard_ulb         # run-time guard on ULB
python -m thesis.export_stress     # tables of the ULB stress analyses
```

IEEE-CIS:

```
python -m thesis.ieee_features     # feature rule from month 1 (wrote frozen_features_v2.json)
python -m thesis.ieee_stress       # label-free checks inside month 1
python -m thesis.fragility         # fragility index, ULB day 1 and IEEE-CIS month 1
python -m thesis.e3_ieee --dry-run # month 1 only, synthetic labels, writes nothing to results/
python -m thesis.e3_ieee           # the single run on months 2-6 (after freeze-v3)
python -m thesis.export_ch5        # result tables and figures from the run -> results/tables/ieee_cis/ch5/
```

The fraud-data runs refuse to start when a frozen file differs from its tag.

## Freeze protocol

Every setting used on the fraud data was written to a file and tagged before the data it governs were
used for testing:

| Tag | File | Fixed before |
|---|---|---|
| `freeze-v1` | `frozen_config.json` | any fraud-data result (ULB and IEEE-CIS) |
| `freeze-v2` | `frozen_features_v2.json` | any IEEE-CIS output after month 1; built from month 1 without labels |
| `freeze-v3` | `frozen_stress_v3.json` | the IEEE-CIS run on months 2-6, which was run once |

The frozen files are never edited after their tag, so these commands print nothing:

```
git diff freeze-v1 -- frozen_config.json
git diff freeze-v2 -- frozen_features_v2.json
git diff freeze-v3 -- frozen_stress_v3.json
```

The broken-member methods (robust fusion, run-time guard, pre-deployment check) were developed on ULB
after `freeze-v1` and fixed in `freeze-v3` before the IEEE-CIS run.

The tags `prereg-sw` and `prereg-cc` mark two further pre-registered side tests (spectral weighting and
committee cleaning). They are not part of the thesis results; their code is at those tags, and their
outputs are in `results/tables/sw`, `results/tables/cc` and the matching figure folders.

The results were produced before the git history was rewritten to a single author identity. The rewrite
changed only author, committer and tagger names and e-mails (trees, messages and dates are identical),
but every commit hash changed. The commit hashes recorded in the `*.meta.json` files are the old ones;
`COMMIT_MAP.md` maps each of them to the commit in this repository.

## Results layout

```
results/
  tables/
    datasets/       dataset and duplicate tables
    setup/          hyperparameters, versions, runtimes, zero-variance counts
    e1/             E1 (single detectors on ADBench); variants/ holds the protocol-switch runs
    e2/             E2 (pairs and triples) and the E-bench selection
    robust_fusion/  robust fusion on development data
    e3_ulb/         ULB: budget mode, thresholds, lof/ (LOF diagnosis), stress/ (broken-member analyses)
    ieee_cis/       IEEE-CIS: month-1 features and checks, run/ (the single run), ch5/ (thesis tables)
    sw/, cc/        side tests (see "Freeze protocol")
  figures/          figures for the same steps (PDF for print, PNG preview)
```

Outputs come with a `*.meta.json` file that records the command, the commit and the inputs.

Two note lines in results/tables/e2/ebench.md and robust_fusion.meta.json were reworded for publication; no number changed.

## License

MIT, see [LICENSE](LICENSE).
