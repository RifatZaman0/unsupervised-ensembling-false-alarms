# E2 summary: pairs and triples of detectors on the 15 ADBench sets

Commit `8b25943a3f5a3eed83a3f7252f8063cd10d7fe4f` · cache key `21ba8c24b9` · source `e2_runs.csv` · created 2026-10-29T17:15:58+00:00

Protocol: z-score inputs on, K = ceil(budget × n_test), PCA zero-variance drop on, ADBench duplicates kept. Primary fusion config: min-max + mean.
Seed means per dataset. Beats best / median = fewer false alarms at K than the best / median member (best chosen with test labels, an oracle reference). Ties = exactly as many false alarms; loses = 1 − beats − ties. Δ/K = (FP@K ensemble − FP@K member) / K; negative = fewer false alarms.

## How often an ensemble beats its members (share of combo × dataset cells)

| Budget | Size | Calibration | Aggregation | beats best | ties best | beats median | ties median | median Δ/K vs best | median Δ/K vs median |
|---|---|---|---|---|---|---|---|---|---|
| 2% | 2 | minmax | max | 0.084 | 0.231 | 0.644 | 0.044 | +0.0131 | -0.0133 |
| 2% | 2 | minmax | mean | 0.107 | 0.071 | 0.564 | 0.076 | +0.0323 | -0.0065 |
| 2% | 2 | percentile | max | 0.080 | 0.067 | 0.760 | 0.040 | +0.0354 | -0.0145 |
| 2% | 2 | percentile | mean | 0.098 | 0.071 | 0.436 | 0.053 | +0.0500 | +0.0025 |
| 2% | 2 | zscore | max | 0.089 | 0.084 | 0.627 | 0.071 | +0.0333 | -0.0121 |
| 2% | 2 | zscore | mean | 0.107 | 0.058 | 0.547 | 0.071 | +0.0414 | -0.0036 |
| 2% | 3 | minmax | max | 0.067 | 0.103 | 0.697 | 0.073 | +0.0286 | -0.0222 |
| 2% | 3 | minmax | mean | 0.060 | 0.037 | 0.660 | 0.087 | +0.0593 | -0.0154 |
| 2% | 3 | percentile | max | 0.080 | 0.020 | 0.617 | 0.053 | +0.0671 | -0.0182 |
| 2% | 3 | percentile | mean | 0.023 | 0.030 | 0.450 | 0.090 | +0.1011 | +0.0000 |
| 2% | 3 | zscore | max | 0.080 | 0.033 | 0.620 | 0.090 | +0.0621 | -0.0193 |
| 2% | 3 | zscore | mean | 0.067 | 0.030 | 0.557 | 0.113 | +0.0655 | -0.0071 |
| 5% | 2 | minmax | max | 0.089 | 0.213 | 0.591 | 0.044 | +0.0069 | -0.0043 |
| 5% | 2 | minmax | mean | 0.124 | 0.058 | 0.573 | 0.049 | +0.0145 | -0.0035 |
| 5% | 2 | percentile | max | 0.133 | 0.027 | 0.747 | 0.044 | +0.0174 | -0.0130 |
| 5% | 2 | percentile | mean | 0.098 | 0.067 | 0.422 | 0.031 | +0.0263 | +0.0021 |
| 5% | 2 | zscore | max | 0.102 | 0.049 | 0.653 | 0.031 | +0.0159 | -0.0067 |
| 5% | 2 | zscore | mean | 0.129 | 0.044 | 0.618 | 0.071 | +0.0190 | -0.0056 |
| 5% | 3 | minmax | max | 0.087 | 0.090 | 0.633 | 0.043 | +0.0190 | -0.0105 |
| 5% | 3 | minmax | mean | 0.093 | 0.050 | 0.713 | 0.060 | +0.0261 | -0.0125 |
| 5% | 3 | percentile | max | 0.107 | 0.010 | 0.670 | 0.023 | +0.0300 | -0.0120 |
| 5% | 3 | percentile | mean | 0.040 | 0.020 | 0.493 | 0.057 | +0.0711 | +0.0000 |
| 5% | 3 | zscore | max | 0.110 | 0.023 | 0.660 | 0.037 | +0.0290 | -0.0153 |
| 5% | 3 | zscore | mean | 0.063 | 0.043 | 0.693 | 0.063 | +0.0299 | -0.0116 |

## Triple ranking, min-max + mean, budget 2%

| # | Triple | avg rank FP@K (of 20) | mean FP@K | mean PR-AUC | #beats best | #beats median | train ρ | train Jaccard |
|---|---|---|---|---|---|---|---|---|
| 1 | IForest+PCA+KNN | 8.57 | 47.7 | 0.342 | 2/15 | 11/15 | 0.684 | 0.381 |
| 2 | IForest+HBOS+KNN | 8.67 | 48.5 | 0.332 | 1/15 | 11/15 | 0.680 | 0.281 |
| 3 | IForest+ECOD+KNN | 9.10 | 47.8 | 0.352 | 1/15 | 12/15 | 0.704 | 0.329 |
| 4 | IForest+HBOS+LOF | 9.67 | 48.9 | 0.311 | 1/15 | 10/15 | 0.511 | 0.186 |
| 5 | IForest+PCA+ECOD | 9.83 | 47.9 | 0.334 | 0/15 | 11/15 | 0.854 | 0.454 |
| 6 | ECOD+HBOS+KNN | 9.93 | 49.0 | 0.326 | 2/15 | 13/15 | 0.607 | 0.250 |
| 7 | IForest+PCA+HBOS | 9.97 | 48.5 | 0.315 | 2/15 | 8/15 | 0.760 | 0.381 |
| 8 | PCA+ECOD+KNN | 10.33 | 48.3 | 0.341 | 2/15 | 9/15 | 0.646 | 0.331 |
| 9 | IForest+ECOD+LOF | 10.37 | 48.3 | 0.328 | 0/15 | 10/15 | 0.559 | 0.235 |
| 10 | ECOD+KNN+LOF | 10.43 | 50.3 | 0.306 | 0/15 | 7/15 | 0.533 | 0.234 |
| 11 | IForest+PCA+LOF | 10.53 | 48.0 | 0.322 | 0/15 | 11/15 | 0.538 | 0.259 |
| 12 | IForest+ECOD+HBOS | 10.67 | 49.1 | 0.317 | 1/15 | 9/15 | 0.774 | 0.389 |
| 13 | IForest+KNN+LOF | 10.87 | 49.5 | 0.308 | 1/15 | 9/15 | 0.624 | 0.265 |
| 14 | PCA+HBOS+KNN | 10.87 | 48.5 | 0.315 | 2/15 | 9/15 | 0.588 | 0.280 |
| 15 | PCA+ECOD+HBOS | 11.33 | 49.2 | 0.308 | 1/15 | 11/15 | 0.759 | 0.356 |
| 16 | PCA+ECOD+LOF | 11.60 | 48.8 | 0.311 | 0/15 | 10/15 | 0.519 | 0.234 |
| 17 | PCA+HBOS+LOF | 11.63 | 49.1 | 0.290 | 1/15 | 11/15 | 0.436 | 0.181 |
| 18 | PCA+KNN+LOF | 11.70 | 49.8 | 0.295 | 1/15 | 7/15 | 0.507 | 0.295 |
| 19 | HBOS+KNN+LOF | 11.93 | 51.0 | 0.276 | 0/15 | 9/15 | 0.521 | 0.206 |
| 20 | ECOD+HBOS+LOF | 12.00 | 49.8 | 0.293 | 0/15 | 10/15 | 0.456 | 0.179 |

Friedman test over the 20 triples (15 datasets): χ² = 8.81, p = 0.976 (Iman-Davenport p = 0.979); Nemenyi CD = 7.66 at α = 0.05; 20 of 20 triples lie within CD of the lowest average rank. Not significant means the triples cannot be told apart on these datasets: the top row has the lowest average rank, it is not shown to be better.

## Triple ranking, min-max + mean, budget 5%

| # | Triple | avg rank FP@K (of 20) | mean FP@K | mean PR-AUC | #beats best | #beats median | train ρ | train Jaccard |
|---|---|---|---|---|---|---|---|---|
| 1 | IForest+KNN+LOF | 7.53 | 129.1 | 0.308 | 2/15 | 8/15 | 0.624 | 0.303 |
| 2 | IForest+ECOD+KNN | 8.77 | 131.2 | 0.352 | 0/15 | 13/15 | 0.704 | 0.382 |
| 3 | IForest+PCA+LOF | 8.77 | 129.4 | 0.322 | 2/15 | 11/15 | 0.538 | 0.293 |
| 4 | IForest+HBOS+KNN | 8.97 | 131.1 | 0.332 | 1/15 | 11/15 | 0.680 | 0.353 |
| 5 | IForest+ECOD+LOF | 9.07 | 131.1 | 0.328 | 0/15 | 12/15 | 0.559 | 0.272 |
| 6 | IForest+HBOS+LOF | 9.13 | 131.1 | 0.311 | 0/15 | 13/15 | 0.511 | 0.240 |
| 7 | IForest+PCA+KNN | 9.33 | 130.1 | 0.342 | 1/15 | 11/15 | 0.684 | 0.425 |
| 8 | ECOD+KNN+LOF | 9.57 | 134.6 | 0.306 | 3/15 | 8/15 | 0.533 | 0.267 |
| 9 | PCA+ECOD+KNN | 10.43 | 133.1 | 0.341 | 1/15 | 11/15 | 0.646 | 0.382 |
| 10 | HBOS+KNN+LOF | 10.63 | 132.2 | 0.276 | 3/15 | 8/15 | 0.521 | 0.246 |
| 11 | PCA+ECOD+LOF | 10.83 | 132.7 | 0.311 | 2/15 | 12/15 | 0.519 | 0.274 |
| 12 | ECOD+HBOS+KNN | 11.20 | 132.1 | 0.326 | 1/15 | 13/15 | 0.607 | 0.314 |
| 13 | PCA+HBOS+KNN | 11.20 | 131.7 | 0.315 | 3/15 | 10/15 | 0.588 | 0.347 |
| 14 | PCA+KNN+LOF | 11.20 | 129.9 | 0.295 | 2/15 | 8/15 | 0.507 | 0.303 |
| 15 | IForest+PCA+ECOD | 11.50 | 131.6 | 0.334 | 1/15 | 11/15 | 0.854 | 0.516 |
| 16 | IForest+PCA+HBOS | 11.60 | 131.3 | 0.315 | 2/15 | 12/15 | 0.760 | 0.470 |
| 17 | ECOD+HBOS+LOF | 12.17 | 132.3 | 0.293 | 2/15 | 12/15 | 0.456 | 0.225 |
| 18 | PCA+HBOS+LOF | 12.23 | 131.5 | 0.290 | 2/15 | 11/15 | 0.436 | 0.236 |
| 19 | IForest+ECOD+HBOS | 12.57 | 132.3 | 0.317 | 0/15 | 10/15 | 0.774 | 0.452 |
| 20 | PCA+ECOD+HBOS | 13.30 | 132.2 | 0.308 | 0/15 | 9/15 | 0.759 | 0.448 |

Friedman test over the 20 triples (15 datasets): χ² = 19.61, p = 0.419 (Iman-Davenport p = 0.422); Nemenyi CD = 7.66 at α = 0.05; 20 of 20 triples lie within CD of the lowest average rank. Not significant means the triples cannot be told apart on these datasets: the top row has the lowest average rank, it is not shown to be better.

## Friedman tests over the combos of one size, min-max + mean

| Budget | Size | combos | χ² | p | Iman-Davenport p | Nemenyi CD | lowest avg rank | within CD |
|---|---|---|---|---|---|---|---|---|
| 2% | 2 | 15 | 8.16 | 0.881 | 0.889 | 5.54 | IForest+KNN (6.77) | 15/15 |
| 2% | 3 | 20 | 8.81 | 0.976 | 0.979 | 7.66 | IForest+PCA+KNN (8.57) | 20/20 |
| 5% | 2 | 15 | 14.10 | 0.442 | 0.447 | 5.54 | IForest+LOF (5.83) | 15/15 |
| 5% | 3 | 20 | 19.61 | 0.419 | 0.422 | 7.66 | IForest+KNN+LOF (7.53) | 20/20 |
