# ULB: label-free LOF diagnosis, day 1 against day 2

Commit `ecde099b42287b5ab96ad43d93933caaa00bc126` · `python -m thesis.e3_ulb_lof` · created 2026-10-30T21:14:28+00:00

No label is read (`Class` column never loaded). Inputs z-scored with day-1 statistics; k = 20 (PyOD default, as in E3); fit sample 30,000 of 144,786 day-1 rows (seed 0), sampled rows leave-one-out; 140,021 day-2 rows. Descriptive, added after `freeze-v1`, not part of the frozen evaluation.

## k-nearest-neighbour distances (z-scored units)

| statistic | day 1 | day 2 | day 2 / day 1 |
|---|---|---|---|
| distance to the 20th neighbour, quantile 0.5 | 2.329 | 3.363 | 1.44 |
| distance to the 20th neighbour, quantile 0.9 | 4.173 | 5.001 | 1.20 |
| distance to the 20th neighbour, quantile 0.99 | 9.101 | 9.578 | 1.05 |
| distance to the 20th neighbour, mean | 2.645 | 3.727 | 1.41 |
| mean distance to 20 neighbours, quantile 0.5 | 1.997 | 3.168 | 1.59 |
| mean distance to 20 neighbours, quantile 0.9 | 3.793 | 4.698 | 1.24 |
| mean distance to 20 neighbours, quantile 0.99 | 7.939 | 8.644 | 1.09 |
| mean distance to 20 neighbours, mean | 2.288 | 3.507 | 1.53 |

Share of rows whose 20th-neighbour distance exceeds the day-1 0.99 quantile: day 1 1.00%, day 2 1.14%; above the day-1 0.998 quantile: day 1 0.20%, day 2 0.21%.

## Feature means of day 2 against day-1 statistics

3 of 29 features shift by more than 0.5 day-1 standard deviations. Largest shifts:

| feature | shift (day-1 SD) | SD ratio day 2 / day 1 |
|---|---|---|
| V3 | -1.078 | 1.134 |
| V25 | -0.607 | 1.291 |
| V15 | -0.507 | 0.910 |
| V5 | +0.438 | 1.067 |
| V11 | -0.401 | 0.902 |

Reading: the median 20th-neighbour distance is 2.329 on day 1 and 3.363 on day 2 (0.99 quantile 9.101 and 9.578). This describes how far day-2 rows lie from the day-1 fit sample; it does not by itself show why LOF ranks frauds low.
