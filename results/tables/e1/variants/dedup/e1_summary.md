# E1 summary: single detectors on the 15 ADBench sets

Commit `b12667f5989f7b6145a3fe126ff0e08d0f4d07e3` · cache key `2ef4e6d580` · source `e1_runs.csv` · created 2026-10-28T01:36:52+00:00

Protocol: z-score inputs on, K = ceil(budget × n_test), PCA zero-variance drop on, ADBench duplicates dropped.
Means are over datasets of seed means (5 seeds). Ranks per dataset, 1 = best; #best = datasets with the lowest FP@K (ties count for each tied detector).

## Budget 2% of test rows

| Detector | P@K | R@K | FP@K | PR-AUC | ROC-AUC | avg rank FP@K | avg rank PR-AUC | #best FP@K |
|---|---|---|---|---|---|---|---|---|
| IForest | 0.367 | 0.194 | 47.3 | 0.335 | 0.800 | 3.17 | 2.47 | 3/15 |
| ECOD | 0.363 | 0.194 | 48.4 | 0.311 | 0.763 | 3.20 | 3.67 | 5/15 |
| PCA | 0.365 | 0.182 | 47.3 | 0.310 | 0.754 | 3.40 | 3.60 | 0/15 |
| KNN | 0.319 | 0.147 | 62.2 | 0.246 | 0.771 | 3.53 | 3.33 | 3/15 |
| HBOS | 0.326 | 0.182 | 49.2 | 0.286 | 0.744 | 3.83 | 3.93 | 2/15 |
| LOF | 0.258 | 0.105 | 66.8 | 0.166 | 0.678 | 3.87 | 4.00 | 2/15 |

## Budget 5% of test rows

| Detector | P@K | R@K | FP@K | PR-AUC | ROC-AUC | avg rank FP@K | avg rank PR-AUC | #best FP@K |
|---|---|---|---|---|---|---|---|---|
| IForest | 0.285 | 0.318 | 126.9 | 0.335 | 0.800 | 2.77 | 2.47 | 4/15 |
| KNN | 0.266 | 0.295 | 171.0 | 0.246 | 0.771 | 3.27 | 3.33 | 4/15 |
| ECOD | 0.269 | 0.307 | 132.7 | 0.311 | 0.763 | 3.37 | 3.67 | 3/15 |
| PCA | 0.272 | 0.293 | 127.6 | 0.310 | 0.754 | 3.73 | 3.60 | 0/15 |
| LOF | 0.191 | 0.195 | 181.4 | 0.166 | 0.678 | 3.90 | 4.00 | 3/15 |
| HBOS | 0.247 | 0.294 | 130.8 | 0.286 | 0.744 | 3.97 | 3.93 | 2/15 |
