# E3 on ULB (day 1 fit, day 2 test)

Commit `83ddbda7067442a2b689281c379eb832246dc0a0` · frozen config = tag `freeze-v1` · created 2026-10-30T02:25:52+00:00 · `python -m thesis.e3_ulb`

Day 1: 144,786 rows (Time 86398 s at most); day 2: 140,021 rows (Time 86400 s at least), **211 frauds** (0.151%). K at 0.1%, 0.2%, 0.5%: 141, 281, 701. Bootstrap: 1000 paired transaction-level resamples, seed 0, 95% percentile CI.

## Hypotheses (criterion: 'fewer false alarms' only if the 95% CI of dFP@K lies entirely below 0; otherwise 'no difference shown')

| Hypothesis | Comparison | dFP@K [95% CI] | per 100 alerts [95% CI] | ULB result | Status |
|---|---|---|---|---|---|
| H1a | E-bench (= E-lit) vs median_single | -0.8 [-2.7, 2.4] | -0.3 [-1.0, 0.9] | no difference shown | not supported (fails on ULB; needs both) |
| H1b | E-bench (= E-lit) vs default_iforest | 2.4 [1.2, 5.8] | 0.9 [0.4, 2.1] | no difference shown (CI entirely above 0: more false alarms; descriptive) | pending IEEE-CIS |
| H2b (primary) | E-bench vs E-lit | 0.0 |  | degenerate | degenerate: the 2% rule selected E-lit, so E-bench vs E-lit has dFP@K = 0 by identity (benchmark selection confirms the literature default); reported as such |
| H2b (secondary) | E-bench-5% vs E-lit (= E-bench) | 30.4 [21.6, 40.0] | 10.8 [7.7, 14.2] | no difference shown (CI entirely above 0: more false alarms; descriptive) | descriptive only (a superiority check; non-inferiority is not tested) |

The frozen criterion has two outcomes. 'CI entirely above 0' is a descriptive note, not a hypothesis outcome.

## Budget mode at the primary budget 0.2% (seed means; regret = FP@K − FP@K of the oracle-best single)

| Method | FP@K | P@K | R@K | PR-AUC | ROC-AUC | Regret |
|---|---|---|---|---|---|---|
| IForest | 235.6 | 0.162 | 0.215 | 0.114 | 0.960 | 21.6 |
| PCA | 242.0 | 0.139 | 0.185 | 0.105 | 0.962 | 28.0 |
| ECOD | 231.0 | 0.178 | 0.237 | 0.145 | 0.959 | 17.0 |
| HBOS | 214.0 | 0.238 | 0.318 | 0.180 | 0.959 | 0.0 |
| KNN | 255.0 | 0.093 | 0.123 | 0.089 | 0.969 | 41.0 |
| LOF | 281.0 | 0.000 | 0.000 | 0.002 | 0.379 | 67.0 |
| E-bench (= E-lit), min-max + mean | 238.0 | 0.153 | 0.204 | 0.100 | 0.962 | 24.0 |
| E-bench-5%, min-max + mean | 268.4 | 0.045 | 0.060 | 0.055 | 0.957 | 54.4 |
| E-all, min-max + mean | 231.8 | 0.175 | 0.233 | 0.126 | 0.965 | 17.8 |
| median single | 238.8 |  |  |  |  | 24.8 |
| default IForest | 235.6 | 0.162 | 0.215 | 0.114 | 0.960 | 21.6 |
| oracle-best single (HBOS) | 214.0 | 0.238 | 0.318 | 0.180 | 0.959 | 0.0 |

All budgets, fusion configs and CIs: `e3_ulb_budget.csv`, `e3_ulb_deltas.csv`.

## Threshold mode (day-1 quantile threshold applied unchanged to day 2; min-max + mean for ensembles; seed means)

| Method | q | Alerts | Alert rate | 1 − q | FPR | Recall | Precision |
|---|---|---|---|---|---|---|---|
| IForest | 0.998 | 250.8 | 0.1791% | 0.2% | 0.1481% | 0.208 | 0.176 |
| IForest | 0.995 | 718.8 | 0.5134% | 0.5% | 0.4611% | 0.352 | 0.104 |
| PCA | 0.998 | 249.0 | 0.1778% | 0.2% | 0.1502% | 0.185 | 0.157 |
| PCA | 0.995 | 682.0 | 0.4871% | 0.5% | 0.4385% | 0.327 | 0.101 |
| ECOD | 0.998 | 247.0 | 0.1764% | 0.2% | 0.1423% | 0.227 | 0.194 |
| ECOD | 0.995 | 775.0 | 0.5535% | 0.5% | 0.4807% | 0.488 | 0.133 |
| HBOS | 0.998 | 230.0 | 0.1643% | 0.2% | 0.1237% | 0.270 | 0.248 |
| HBOS | 0.995 | 765.0 | 0.5463% | 0.5% | 0.4578% | 0.592 | 0.163 |
| KNN | 0.998 | 349.0 | 0.2492% | 0.2% | 0.2232% | 0.175 | 0.106 |
| KNN | 0.995 | 926.0 | 0.6613% | 0.5% | 0.5887% | 0.488 | 0.111 |
| LOF | 0.998 | 11510.0 | 8.2202% | 0.2% | 8.2111% | 0.142 | 0.003 |
| LOF | 0.995 | 23853.0 | 17.0353% | 0.5% | 17.0288% | 0.213 | 0.002 |
| E-bench (= E-lit), min-max + mean | 0.998 | 262.2 | 0.1873% | 0.2% | 0.1582% | 0.194 | 0.157 |
| E-bench (= E-lit), min-max + mean | 0.995 | 722.0 | 0.5156% | 0.5% | 0.4602% | 0.373 | 0.109 |
| E-bench-5%, min-max + mean | 0.998 | 662.4 | 0.4731% | 0.2% | 0.4450% | 0.191 | 0.061 |
| E-bench-5%, min-max + mean | 0.995 | 1290.0 | 0.9213% | 0.5% | 0.8623% | 0.400 | 0.066 |
| E-all, min-max + mean | 0.998 | 252.6 | 0.1804% | 0.2% | 0.1473% | 0.221 | 0.185 |
| E-all, min-max + mean | 0.995 | 753.4 | 0.5381% | 0.5% | 0.4631% | 0.502 | 0.141 |

The oracle-best single is chosen with day-2 labels: an upper bound, not a deployable method. E-lit is the same triple as E-bench and is evaluated once.
