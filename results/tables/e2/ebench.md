# E-bench: the triple chosen on ADBench

Commit `20c84136b6b924c872ba569c56d5a85b9fb71b9e` · source `e2_summary.csv`, `e2_friedman.csv` (E2 commit `8b25943`) · created 2026-10-29T18:29:49+00:00

## Rule (declared before any fraud-data result)

E-bench is the triple with the **lowest average FP@K rank** among the 20 triples of the 6-detector pool, over the 15 ADBench sets (seed means), under the primary fusion config **min-max calibration + mean fusion**, at the primary ADBench budget **2%**. FP@K needs test labels, so this choice uses labels on ADBench only; no fraud data enters it. The rule was fixed before any fraud-data result.

## Top 5 triples at 2%

| Pos. | Triple | Avg. FP@K rank (of 20) | Mean FP@K | Beats best member (of 15) | Beats median member (of 15) |
|---|---|---|---|---|---|
| 1 | IForest+PCA+KNN | 8.57 | 47.7 | 2 | 11 |
| 2 | IForest+HBOS+KNN | 8.67 | 48.5 | 1 | 11 |
| 3 | IForest+ECOD+KNN | 9.10 | 47.8 | 1 | 12 |
| 4 | IForest+HBOS+LOF | 9.67 | 48.9 | 1 | 10 |
| 5 | IForest+PCA+ECOD | 9.83 | 47.9 | 0 | 11 |

Friedman test over the 20 triples (15 sets): χ² = 8.81, p = 0.976; Nemenyi CD = 7.66 at α = 0.05; 20 of 20 triples lie within CD of the lowest average rank.

## Top 5 triples at 5%

| Pos. | Triple | Avg. FP@K rank (of 20) | Mean FP@K | Beats best member (of 15) | Beats median member (of 15) |
|---|---|---|---|---|---|
| 1 | IForest+KNN+LOF | 7.53 | 129.1 | 2 | 8 |
| 2 | IForest+ECOD+KNN | 8.77 | 131.2 | 0 | 13 |
| 2 | IForest+PCA+LOF | 8.77 | 129.4 | 2 | 11 |
| 4 | IForest+HBOS+KNN | 8.97 | 131.1 | 1 | 11 |
| 5 | IForest+ECOD+LOF | 9.07 | 131.1 | 0 | 12 |

Friedman test over the 20 triples (15 sets): χ² = 19.61, p = 0.419; Nemenyi CD = 7.66 at α = 0.05; 20 of 20 triples lie within CD of the lowest average rank.

## Reading

- **The triples cannot be told apart on these 15 sets** (Friedman p = 0.976 at 2%, 0.419 at 5%; every triple lies within the critical difference of the first). E-bench has the lowest average rank under the declared rule; it is not shown to be better than the other triples.
- **E-bench = IForest+PCA+KNN. This is the same triple as E-lit (IForest+PCA+KNN), the published default.** The coincidence must be stated in the paper: the benchmark rule confirms the literature default. E-bench (= E-lit) is the one primary ensemble; E-all is a secondary, descriptive control.
- **E-bench-5% = IForest+KNN+LOF** (the same rule at the 5% budget) is declared a **secondary** ensemble and frozen with the others before any fraud-data result. It is reported next to the primary ensembles, never chosen after seeing fraud results.
- E2 finding, reported as found: the ensembles rarely beat their best member but usually beat their median member, and the members' training overlap is only weakly associated with the gain.
