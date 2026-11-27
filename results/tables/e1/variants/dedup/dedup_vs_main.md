# E1 duplicate robustness check: duplicates kept (main) vs removed (--dedup)

Main E1 commit `ef2d285` · dedup E1 commit `b12667f` · comparison commit `f937461` · created 2026-10-28T02:20:30+00:00

Ranks per dataset (seed means), 1 = fewest false alarms; averaged over the 15 sets. FP@K values are not compared one to one, because removing rows changes n_test and K.

## Budget 2%

Kendall τ between the two average-rank orders: 0.733. Top detector: IForest (main), IForest (dedup). Friedman p: 0.771 (main), 0.845 (dedup). Datasets changed by dedup: 8; best detector changed on: annthyroid, thyroid.

| Detector | avg rank FP@K main | dedup | Δ | avg rank PR-AUC main | dedup | #best FP@K main | dedup |
|---|---|---|---|---|---|---|---|
| IForest | 3.07 | 3.17 | +0.10 | 2.53 | 2.47 | 3 | 3 |
| ECOD | 3.17 | 3.20 | +0.03 | 3.53 | 3.67 | 4 | 5 |
| KNN | 3.50 | 3.53 | +0.03 | 3.40 | 3.33 | 3 | 3 |
| PCA | 3.53 | 3.40 | -0.13 | 3.67 | 3.60 | 0 | 0 |
| LOF | 3.80 | 3.87 | +0.07 | 3.87 | 4.00 | 2 | 2 |
| HBOS | 3.93 | 3.83 | -0.10 | 4.00 | 3.93 | 3 | 2 |

## Budget 5%

Kendall τ between the two average-rank orders: 0.966. Top detector: IForest (main), IForest (dedup). Friedman p: 0.422 (main), 0.472 (dedup). Datasets changed by dedup: 8; best detector changed on: none.

| Detector | avg rank FP@K main | dedup | Δ | avg rank PR-AUC main | dedup | #best FP@K main | dedup |
|---|---|---|---|---|---|---|---|
| IForest | 2.70 | 2.77 | +0.07 | 2.53 | 2.47 | 4 | 4 |
| KNN | 3.27 | 3.27 | +0.00 | 3.40 | 3.33 | 4 | 4 |
| ECOD | 3.40 | 3.37 | -0.03 | 3.53 | 3.67 | 3 | 3 |
| LOF | 3.87 | 3.90 | +0.03 | 3.87 | 4.00 | 3 | 3 |
| PCA | 3.87 | 3.73 | -0.13 | 3.67 | 3.60 | 0 | 0 |
| HBOS | 3.90 | 3.97 | +0.07 | 4.00 | 3.93 | 2 | 2 |
