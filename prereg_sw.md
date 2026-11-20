# Pre-registration: spectral weighting of ensemble members (exploratory test)

**Status.** Written before any result of this method exists. Commit this file together with the code and tag the commit `prereg-sw` before anything is run on real data. Nothing below changes after the first real-data result; a change is a new, disclosed version. If the method enters the thesis, every number listed in Section 7 is reported, whatever it shows.

**Time box.** Steps 1–3 must be finished by Mon 23 Nov 2026. If they are not, the test is dropped and thesis writing continues.

---

## 1. Question

Can label-free weights, estimated from the members' top-K overlaps on the training split, make a score-level ensemble raise fewer false alarms than

- (a) the same ensemble with equal weights, and
- (b) the single detector one would deploy without labels (default Isolation Forest)?

## 2. Why it might work

- **Equal weights dilute the strongest member.** If members carry independent evidence of strength d′₁ … d′ₘ, an equal-weight average reaches Σd′/√m, while weights proportional to d′ reach √(Σd′²), which is never below the best member. Example: 3, 1 and 0.5 give 2.6 with equal weights (below the best member's 3.0) and 3.2 with proportional weights (above it).
- **Member quality can be estimated without labels.** If members make their mistakes independently given the true class, the off-diagonal part of the covariance matrix of their binary decisions is rank one, and its leading eigenvector is proportional to each member's Youden index (balanced accuracy minus chance) (Dawid and Skene, 1979; Parisi et al., PNAS 2014).
- **This fits the budget protocol exactly.** At a fixed budget every member flags K rows. Then n · Cov(top-K_i, top-K_j) = |T_i ∩ T_j| − K²/n, and the Youden index equals TP@K · (1/n_anom + 1/n_norm) − K/n_norm, an increasing linear function of TP@K. Under the independence assumption, the eigenvector therefore ranks the members by FP@K.
- **The same model explains H2a.** It predicts that the extra top-K overlap of two members is proportional to the product of their qualities. High overlap then means both members are good, which is the positive overlap–gain relation found in E2.

**Known risk.** Members that raise the same false alarms (for example KNN and LOF) violate the independence assumption. Their overlap is then inflated and they receive too much weight. This test measures whether that happens on real data.

## 3. Method SW (fixed)

For one member set M (|M| ≥ 3), one dataset split and one budget b. Implemented in `spectral.py`:

1. Inputs: the members' **training** scores from the score cache (key `21ba8c24b9`). No labels, no test rows.
2. K_tr = the repository's K rule applied to n_train (ceil(b · n_train), at least 1, as for E2's training overlap).
3. Fractional top-K membership a_i(r) ∈ [0, 1] with the expected-value tie rule (Σ_r a_i(r) = K_tr).
4. Excess overlap C_ij = Σ_r a_i(r) a_j(r) − K_tr²/n_train for i ≠ j.
5. Rank-one fit of the off-diagonal of C by iterative diagonal completion: start with a zero diagonal; repeat (leading eigenpair (λ, u), diagonal := λu²) until the diagonal changes by less than 1e-12 (relative), at most 500 times. v = √λ · u, signed so that Σv > 0.
6. Weights w_i = max(v_i, 0) / Σ_j max(v_j, 0).
7. Fallback to equal weights if λ is not positive (below 1e-9 · max(1, max|C_ij|), i.e. rounding noise), any value is not finite, or fewer than two weights are positive. The fallback rate is reported.
8. **SW fusion:** the primary calibration (min-max fitted on training scores, not clipped), then the weighted mean Σ w_i · c_i(s). Only the weights differ from the existing primary configuration.
9. **SW selection:** the member with the largest v (label-free model selection).

## 4. Data, member sets, budgets

| Item | Choice |
|---|---|
| Main test data | The 15 ADBench sets × 5 seeds (stratified 60/40 splits), as in E1/E2 |
| Budgets | 2% (primary), 5% (secondary) |
| Primary member set | E-all (all six detectors) |
| Secondary member sets | The 20 triples, including E-bench |
| Not possible | Pairs (one overlap cannot separate two qualities) |
| Fraud data | ULB (day 1 → day 2) and IEEE-CIS (month 1 → months 2–6): post-hoc only (Step 3), because both have already been evaluated for other methods |

**Comparators.** EQ: the same members with equal weights (min-max + mean). IForest: default parameters, seed = split seed. Oracle-best and median member: test labels, references only.

## 5. Step 1: ceiling gate (training labels; diagnostic only)

For E-all (grid step 0.1, 3,003 weight vectors) and each triple (step 0.05, 231 vectors) at 2% and 5%:

- tune the weights to minimise FP@K on the **training** split, using **training** labels (ties go to the vector with the largest single weight, then grid order), and evaluate them on test;
- compare with the member that has the fewest training false alarms (also training labels), evaluated on test.

Implemented in `ceiling.py`. **Gate:** continue to Step 2 only if, for E-all at 2%, label-tuned weights raise fewer test false alarms than the label-selected member on **more datasets than the reverse** (seed means; ties not counted).

If the gate fails, stop: even with labels, weighting does not beat picking one member, so a label-free weighting cannot be expected to. The ceiling table can still be reported as a result.

## 6. Step 2: main test (ADBench)

Hypotheses, tested in fixed order (fixed-sequence procedure: H-SW2 is tested only if H-SW1 is rejected; each at α = 0.05). Unit: seed-mean FP@K per dataset, 15 datasets, E-all, budget 2%. Test: Wilcoxon signed-rank, one-sided (`scipy.stats.wilcoxon(sw, other, alternative="less", zero_method="wilcox")`), as recommended for two methods over many datasets (Demšar, 2006).

- **H-SW1 (primary):** FP@K(SW fusion) < FP@K(EQ).
- **H-SW2:** FP@K(SW fusion) < FP@K(IForest).

**Descriptive (no decision attached):**

- the same comparisons at 5% and for the 20 triples;
- beats / ties / loses against the oracle-best and the median member, for SW and EQ (share of dataset cells, as in E2);
- Spearman correlation between v and the members' −FP@K, per split, summarised by the median;
- the share of splits where the SW selection is the oracle-best member, and the FP@K of the SW selection against IForest;
- the fallback rate;
- one figure: ΔFP@K (SW − EQ) per dataset at 2% for E-all.

## 7. Step 3: fraud data (only if H-SW1 is rejected)

SW weights from the training window only (ULB day 1; IEEE-CIS month 1), applied unchanged to the test period. ΔFP@K (SW − EQ) and (SW − IForest) at the frozen budgets, with the existing paired bootstrap (ULB) and day-block bootstrap (IEEE-CIS). These are reported whatever their sign, and labelled post-hoc.

## 8. Decision rule

| Outcome | Thesis |
|---|---|
| Gate fails | SW is not added. The ceiling result may be reported as evidence that weighting cannot beat member selection here. |
| Gate passes, H-SW1 not rejected | SW is not added (at most one sentence in Future Work). |
| H-SW1 rejected, H-SW2 not | SW is added as an exploratory extension: "label-free weighting beats equal weighting". No claim that it beats the single detector. |
| H-SW1 and H-SW2 rejected | SW is added as an exploratory extension, and the thesis may state that it beats the single detector one would deploy without labels (on ADBench; fraud data as post-hoc support). |

## 9. Leakage rules for this test

- `spectral_weights` takes only training scores and the budget; a test asserts that its signature has no label argument.
- Calibrators are fitted on training scores only (unchanged fusion code).
- Labels are used only in: Step 1 (training labels, diagnostic), the evaluation metrics on the test split, and the oracle references.
- No setting in Section 3 is changed after a real-data result has been seen.

## 10. Implementation checklist

1. New branch `exp/sw` in the code repo. The files start in `docs/notes/spectral_weighting_2026-11-20/`. Copy this pre-registration to the repo root as `prereg_sw.md` (next to the `frozen_*.json` files, so the tag covers it). Add `spectral.py` and `ceiling.py` to `src/thesis/`, and `test_spectral_weighting.py` to `tests/`. Change the imports to `from thesis.spectral import ...` and `from thesis import ceiling, spectral`.
2. Replace `spectral.default_k_rule` with `metrics.budget_k`, and `ceiling.fp_at_k` with k − `metrics.tp_at_k`. Assert once, on one cached split, that the replaced functions give identical results.
3. **Reproduction check:** EQ for the 20 triples (min-max + mean) must reproduce the FP@K in `e2_runs.csv` exactly, before SW is run.
4. One real-data test: on a cached split (for example cardio, seed 0), SW weights are finite, non-negative, sum to 1, and the run is deterministic.
5. Commit, then tag `prereg-sw`. Only then run Step 1, then Step 2.
6. Outputs go to `results/tables/sw/` and `results/figures/sw/`, each with a `meta.json` (commit, dirty flag, cache key, protocol), as for E2.
7. A fresh review session checks the leakage rules and reproduces the headline numbers before they are used.

## References

- A. P. Dawid and A. M. Skene, "Maximum likelihood estimation of observer error-rates using the EM algorithm," *Journal of the Royal Statistical Society, Series C*, 28(1), 20–28, 1979.
- F. Parisi, F. Strino, B. Nadler and Y. Kluger, "Ranking and combining multiple predictors without labeled data," *PNAS*, 111(4), 1253–1258, 2014.
- J. Demšar, "Statistical comparisons of classifiers over multiple data sets," *JMLR*, 7, 1–30, 2006.
