# E1 summary: single detectors on the 15 ADBench sets

Commit `ef2d2855935d624e330e04ba1a04c2e1b693e4a7` · cache key `21ba8c24b9` · source `e1_runs.csv` · created 2026-10-28T01:26:05+00:00

Protocol: z-score inputs on, K = ceil(budget × n_test), PCA zero-variance drop on, ADBench duplicates kept.
Means are over datasets of seed means (5 seeds). Ranks per dataset, 1 = best; #best = datasets with the lowest FP@K (ties count for each tied detector).

## Budget 2% of test rows

| Detector | P@K | R@K | FP@K | PR-AUC | ROC-AUC | avg rank FP@K | avg rank PR-AUC | #best FP@K |
|---|---|---|---|---|---|---|---|---|
| IForest | 0.370 | 0.201 | 48.8 | 0.333 | 0.803 | 3.07 | 2.53 | 3/15 |
| ECOD | 0.358 | 0.196 | 49.8 | 0.309 | 0.763 | 3.17 | 3.53 | 4/15 |
| KNN | 0.337 | 0.161 | 63.3 | 0.253 | 0.774 | 3.50 | 3.40 | 3/15 |
| PCA | 0.360 | 0.186 | 49.1 | 0.311 | 0.759 | 3.53 | 3.67 | 0/15 |
| LOF | 0.266 | 0.113 | 68.0 | 0.170 | 0.683 | 3.80 | 3.87 | 2/15 |
| HBOS | 0.323 | 0.179 | 51.1 | 0.281 | 0.744 | 3.93 | 4.00 | 3/15 |

## Budget 5% of test rows

| Detector | P@K | R@K | FP@K | PR-AUC | ROC-AUC | avg rank FP@K | avg rank PR-AUC | #best FP@K |
|---|---|---|---|---|---|---|---|---|
| IForest | 0.280 | 0.323 | 131.6 | 0.333 | 0.803 | 2.70 | 2.53 | 4/15 |
| KNN | 0.263 | 0.294 | 175.7 | 0.253 | 0.774 | 3.27 | 3.40 | 4/15 |
| ECOD | 0.268 | 0.316 | 136.5 | 0.309 | 0.763 | 3.40 | 3.53 | 3/15 |
| LOF | 0.196 | 0.209 | 185.3 | 0.170 | 0.683 | 3.87 | 3.87 | 3/15 |
| PCA | 0.266 | 0.296 | 132.2 | 0.311 | 0.759 | 3.87 | 3.67 | 0/15 |
| HBOS | 0.244 | 0.292 | 135.7 | 0.281 | 0.744 | 3.90 | 4.00 | 2/15 |
