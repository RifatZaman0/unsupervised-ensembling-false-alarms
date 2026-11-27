# IEEE-CIS month 1: temporal stress test (label-free, training window only)

Commit `1d1c7f421df2194786f9ce8a85405580eb9524c8` · `python -m thesis.ieee_stress` · created 2026-11-16T23:19:13+00:00

Fit window days 0-20 (91,760 rows); scored slice days 21-29 (42,579 rows); sensitivity: days 21-27 (34,906 rows). No label read; no row of months 2-6 loaded. Feature parameters, z-score, detectors and thresholds fitted on the fit window only. Development output; c is not chosen.

## days 0-20 -> 21-29

| member | q | alert rate | r | r / median r |
|---|---|---|---|---|
| IForest | 0.99 | 0.002% | 0.00 (0.00-0.01 over seeds) | 0.05 |
| IForest | 0.98 | 0.404% | 0.20 (0.18-0.26 over seeds) | 0.71 |
| PCA | 0.99 | 0.063% | 0.06 | 1.64 |
| PCA | 0.98 | 0.679% | 0.34 | 1.19 |
| ECOD | 0.99 | 0.014% | 0.01 | 0.36 |
| ECOD | 0.98 | 0.458% | 0.23 | 0.81 |
| HBOS | 0.99 | 0.002% | 0.00 | 0.06 |
| HBOS | 0.98 | 0.378% | 0.19 | 0.67 |
| KNN | 0.99 | 0.613% | 0.61 | 15.82 |
| KNN | 0.98 | 1.214% | 0.61 | 2.14 |
| LOF | 0.99 | 1.271% | 1.27 | 32.79 |
| LOF | 0.98 | 2.377% | 1.19 | 4.18 |

## days 0-20 -> 21-27

| member | q | alert rate | r | r / median r |
|---|---|---|---|---|
| IForest | 0.99 | 0.002% | 0.00 (0.00-0.01 over seeds) | 0.09 |
| IForest | 0.98 | 0.402% | 0.20 (0.19-0.23 over seeds) | 0.72 |
| PCA | 0.99 | 0.034% | 0.03 | 1.33 |
| PCA | 0.98 | 0.653% | 0.33 | 1.18 |
| ECOD | 0.99 | 0.017% | 0.02 | 0.67 |
| ECOD | 0.98 | 0.458% | 0.23 | 0.82 |
| HBOS | 0.99 | 0.003% | 0.00 | 0.11 |
| HBOS | 0.98 | 0.352% | 0.18 | 0.63 |
| KNN | 0.99 | 0.556% | 0.56 | 21.56 |
| KNN | 0.98 | 1.112% | 0.56 | 2.00 |
| LOF | 0.99 | 1.344% | 1.34 | 52.11 |
| LOF | 0.98 | 2.478% | 1.24 | 4.46 |

## Rules on days 0-20 -> 21-29 (pass = P, fail = F)

| member | q | basis | one-sided c=2 | two-sided c=2 | one-sided c=3 | two-sided c=3 | one-sided c=5 | two-sided c=5 |
|---|---|---|---|---|---|---|---|---|
| IForest | 0.99 | absolute | P | F | P | F | P | F |
| IForest | 0.99 | relative to median | P | F | P | F | P | F |
| IForest | 0.98 | absolute | P | F | P | F | P | P |
| IForest | 0.98 | relative to median | P | P | P | P | P | P |
| PCA | 0.99 | absolute | P | F | P | F | P | F |
| PCA | 0.99 | relative to median | P | P | P | P | P | P |
| PCA | 0.98 | absolute | P | F | P | P | P | P |
| PCA | 0.98 | relative to median | P | P | P | P | P | P |
| ECOD | 0.99 | absolute | P | F | P | F | P | F |
| ECOD | 0.99 | relative to median | P | F | P | P | P | P |
| ECOD | 0.98 | absolute | P | F | P | F | P | P |
| ECOD | 0.98 | relative to median | P | P | P | P | P | P |
| HBOS | 0.99 | absolute | P | F | P | F | P | F |
| HBOS | 0.99 | relative to median | P | F | P | F | P | F |
| HBOS | 0.98 | absolute | P | F | P | F | P | F |
| HBOS | 0.98 | relative to median | P | P | P | P | P | P |
| KNN | 0.99 | absolute | P | P | P | P | P | P |
| KNN | 0.99 | relative to median | F | F | F | F | F | F |
| KNN | 0.98 | absolute | P | P | P | P | P | P |
| KNN | 0.98 | relative to median | F | F | P | P | P | P |
| LOF | 0.99 | absolute | P | P | P | P | P | P |
| LOF | 0.99 | relative to median | F | F | F | F | F | F |
| LOF | 0.98 | absolute | P | P | P | P | P | P |
| LOF | 0.98 | relative to median | F | F | F | F | P | P |

Fit-window and scored alerts per day: `stress_month1_alert_days.csv`. Per-feature tails, mean shift and SD ratio: `stress_month1_feature_tails.csv`.
