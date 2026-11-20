# Pre-registration: committee cleaning of the training data (exploratory test)

**Status.** Written before any result of this method exists. Commit this file with the code and tag the commit `prereg-cc` before anything is run on real data. Nothing below changes after the first real-data result; a change is a new, disclosed version. If the method enters the thesis, every number listed in Sections 6 and 7 is reported, whatever it shows. Smoke tests run on synthetic data only.

**Time box.** Steps 1–2 must be finished by Tue 24 Nov 2026. If they are not, the test is dropped and thesis writing continues.

---

## 1. Question

The training splits contain unlabelled anomalies, and every detector learns from them. If the ensemble acts as a committee that removes the training rows it jointly scores as most anomalous, and every member is refitted on the remaining rows, does the resulting ensemble raise fewer false alarms than

- (a) the same ensemble fitted on all training rows, and
- (b) the single detector one would deploy without labels (default Isolation Forest, fitted on all training rows)?

## 2. Why it might work

- **Contamination hides anomalies.** Anomalies in the training data make similar test anomalies look normal (masking): a neighbour-based detector finds close training neighbours for them, and distribution-based detectors get heavier tails. ADBench notes that duplicated anomalies hurt detectors such as KNN.
- **It changes the members, not only their combination.** Step 1 of the spectral weighting test showed that no weighting of the six fixed scores beats picking the best member, even with labels. Cleaning refits the members, so that ceiling does not apply.
- **Related evidence.** Self-supervise, Refine, Repeat (SRR; Yoon et al., TMLR 2022) refines training data with a committee of one-class models and reports large gains, including on tabular data; Latent Outlier Exposure (Qiu et al., ICML 2022) reports gains on 30 tabular datasets. Both use deep one-class models. This test uses heterogeneous classical detectors at a fixed alert budget, with a self-cleaning control that separates the effect of the committee from the effect of cleaning.

## 3. Method CC (fixed)

For one ADBench split (training rows X_train, test rows X_test). Row selection is implemented in `cleaning.py`:

1. **Committee score:** the six members' cached training scores (cache key `21ba8c24b9`, fitted on all training rows), each min-max scaled on its training scores, then averaged (the primary fusion, on training rows).
2. **Removal:** with q = 0.05 and k = the repository's K rule applied to (q, n_train), remove the k training rows with the highest committee score. If rows tied with the k-th largest score would push the count above k, the whole tied block is kept. No labels, no test rows.
3. **Refit:** for each of the six detectors, rebuild the pipeline on the kept training rows only: z-score fitted on the kept rows, the detector fitted on the kept rows with the same parameters and seed (seed = split seed), and the test rows scored. Training scores of the kept rows are the fit-time scores, exactly as in E1 (leave-one-out for KNN and LOF).
4. **CC ensemble:** min-max fitted on the kept rows' training scores, then the mean over the six members (E-all). The test rows are never changed or removed.

**Self-cleaning control (SC):** each member removes the k training rows it scores highest itself (same q, same tie rule), is refitted on its own kept rows, and the six members are fused as in step 4.

**Oracle cleaning (OC, Step 1 only):** all training anomalies are removed using training labels; everything else as in steps 3–4. Diagnostic only, never a method.

## 4. Data, budgets, comparators

| Item | Choice |
|---|---|
| Main data | The 15 ADBench sets × 5 seeds (stratified 60/40 splits), as in E1/E2 |
| Budgets | 2% (primary), 5% (secondary) |
| Primary ensemble | E-all (six detectors), min-max + mean |
| Secondary | E-bench (IForest + PCA + KNN) with the same cleaning; the single detectors after cleaning |
| Fraud data | ULB and IEEE-CIS: post-hoc only (Step 3), because both have already been evaluated for other methods |

**Comparators.** Plain E-all (all training rows; equal to the existing E-all on cached scores). Plain IForest (E1). Self-cleaned E-all (SC). Oracle-best and median plain member (test labels; references only).

## 5. Step 1: oracle ceiling (training labels; diagnostic only)

Compute, per dataset (seed means), at 2% and 5%: FP@K of OC E-all, plain E-all and plain IForest; and the oracle-best member before and after oracle cleaning.

**Gate:** continue to Step 2 only if, for E-all at 2%, **all three** hold:
1. OC E-all has fewer false alarms than plain E-all on more datasets than the reverse (ties not counted);
2. the mean FP@K of OC E-all over the 15 datasets is at least 5% below that of plain E-all;
3. the mean FP@K of OC E-all is below that of plain IForest.

If the gate fails, stop: even perfect cleaning would not give the label-free version enough room. The Step 1 table can still be reported as a result about contamination.

## 6. Step 2: main test (ADBench)

Fixed-sequence hypotheses (each at α = 0.05; a hypothesis is tested only if the previous one is rejected). Unit: seed-mean FP@K per dataset, 15 datasets, E-all, 2%. Test: Wilcoxon signed-rank, one-sided (`scipy.stats.wilcoxon(cc, other, alternative="less", zero_method="wilcox")`; Demšar, 2006).

- **H-CC1:** FP@K(CC E-all) < FP@K(plain E-all).
- **H-CC2:** FP@K(CC E-all) < FP@K(plain IForest).
- **H-CC3:** FP@K(CC E-all) < FP@K(SC E-all).

**Descriptive (no decision attached):**
- the same comparisons at 5% and for E-bench;
- beats / ties / loses of CC E-all against the oracle-best and the median plain member (share of datasets);
- each detector after committee cleaning, after self-cleaning and plain (mean FP@K, per-dataset table);
- cleaning quality, using labels for description only: share of removed rows that are anomalies, share of training anomalies removed;
- E-all at 2% with q = 0.02 and q = 0.10 (sensitivity only);
- one figure: ΔFP@K (CC − plain E-all) per dataset at 2%.

## 7. Step 3: fraud data (only if H-CC1 and H-CC2 are rejected)

ULB: committee cleaning of the day-1 training rows with the same rule (KNN/LOF then sample their 30,000 fitting rows from the kept rows, as in E3), refit, score day 2; ΔFP@K (CC − plain E-all) and (CC − plain IForest) at the frozen budgets with the existing paired bootstrap. IEEE-CIS only if time allows, with the day-block bootstrap. Reported whatever the sign, labelled post-hoc.

## 8. Decision rule

| Outcome | Thesis |
|---|---|
| Gate fails | CC is not added. The Step 1 table may be reported as a result about contamination. |
| H-CC1 or H-CC2 not rejected | CC is not added (at most one sentence in Future Work). |
| H-CC1 and H-CC2 rejected, H-CC3 not | CC is added as an exploratory extension: "a committee-cleaned ensemble beats the single detector one would deploy", with the caveat that each member cleaning its own data does as well. |
| H-CC1, H-CC2 and H-CC3 rejected | As above, and the thesis may state that the committee itself, not cleaning alone, is what helps. |

In every case the thesis does not claim that the ensemble beats the oracle-best plain detector unless the descriptive beats / ties / loses table shows it.

## 9. Leakage rules for this test

- `cleaning.py` takes only training scores and q; a test asserts that no function has a label argument.
- After removal, the z-score, the detectors and the calibrators are fitted on the kept training rows only.
- Test rows are never removed, reordered or used for fitting.
- Labels are used only in: oracle cleaning (Step 1, diagnostic), the evaluation metrics on the test split, the oracle references and the descriptive cleaning-quality numbers.
- No setting in Section 3 is changed after a real-data result has been seen.

## 10. Implementation checklist

1. New branch `exp/cc` from `main` (not from `exp/sw`). The files start in `docs/notes/committee_cleaning_2026-11-20/`. Copy this pre-registration to the repo root as `prereg_cc.md`; add `cleaning.py` to `src/thesis/` and `test_cleaning.py` to `tests/` (imports: `from thesis import cleaning`).
2. Replace `cleaning.default_k_rule` with `metrics.budget_k` and check once that both agree.
3. **Reproduction check:** the refit pipeline with all training rows kept must reproduce the cached training and test scores of all 450 ADBench members (and therefore `e1_runs.csv`) before any cleaned run. The runner refuses to write results if it fails.
4. A runner `src/thesis/cc.py` with the same guards as `sw.py`: clean tree, HEAD descends from `prereg-cc`, `prereg_cc.md` unchanged since the tag, refuses to overwrite existing outputs. Steps run separately (`step1`, `step2`).
5. Cleaned refits are not written into the main score cache (key `21ba8c24b9` stays untouched); use a separate cache folder or keep them in memory.
6. Tests: the provided ones, the reproduction check on one cached split, and mutation checks of the new runner code.
7. Commit, tag `prereg-cc`, run Step 1, report, stop. Step 2 only after the gate passes.
8. Outputs in `results/tables/cc/` and `results/figures/cc/` with `meta.json` (commit, dirty flag, cache key, protocol).
9. A fresh review session checks the leakage rules and reproduces the headline numbers before anything is used.

## References

- J. Yoon, K. Sohn, C.-L. Li, S. O. Arik and T. Pfister, "Self-supervise, Refine, Repeat: Improving Unsupervised Anomaly Detection," *Transactions on Machine Learning Research*, 2022.
- C. Qiu, A. Li, M. Kloft, M. Rudolph and S. Mandt, "Latent Outlier Exposure for Anomaly Detection with Contaminated Data," *ICML*, PMLR 162, 2022.
- J. Demšar, "Statistical comparisons of classifiers over multiple data sets," *JMLR*, 7, 1–30, 2006.
