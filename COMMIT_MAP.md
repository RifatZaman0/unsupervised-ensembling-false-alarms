# Commit map

The history of this repository was rewritten once, before publication, so that every commit and tag carries one author identity. Only the author, committer and tagger name and e-mail changed; every tree, commit message and date is identical to the original, so every file at every commit is unchanged. Because a commit hash covers the author fields, every hash changed. Results files produced before the rewrite record the old hashes; this table maps them to the commits in this repository.

## Hashes cited in the thesis and the TDK paper

| Old (short) | Old (full) | New (full) | What |
|---|---|---|---|
| `bd15630` | `bd15630d81967b215bf55da750f0e4b0c0954198` | `5283a207eb66e9daff596c75c31be81c3f205d32` | tag `freeze-v1` (commit) |
| `dfda3f0` | `dfda3f0d8d0c0b80eb01b949c4a550b275be2329` | `5e655639f4871d7a217f201ae9c1325648b3cb84` | tag `tdk-submitted` (commit) |
| `233cfe1` | `233cfe17ba74cca8b559d1a9703d4dd17a6ec75a` | `2193e973c9b02577013dcdfd7a3e63c7dfca0258` | tag `freeze-v2` (commit) |
| `4f3b58f` | `4f3b58f392d05e785500c673bdc86ece81cfe182` | `2647cf9942bdd3133efb75d8ba180d3c3ce7989d` | tag `freeze-v3` (commit) |
| `625bb29` | `625bb29467dcbc28314e1ea7a7c7c2173be0fcab` | `89c1cd6e47d30550e0ad95ef3e235b4616f2e221` | merged branch `exp/ch5-exports` (Chapter 5 exports) |
| `ef25b6d` | `ef25b6dfe73102495c286c32cf6a4f441ea39cfc` | `d4b72484d8ea6e280d27a1638162d8cbe49abf67` | `main` before the publication commit |
| `9f92a42` | `9f92a42f91785add388d1ba606b74dedc72ece80` | `640cbdad85997e0df3910e0f1b430a7d156924fa` | tag `prereg-sw` (commit) |
| `bfaa7e5` | `bfaa7e5f5a344e482b07d882e1b12e3c428f21af` | `053e3f26e7dc8d9762fbca2b902905f0e6b647cb` | tag `prereg-cc` (commit) |

## Annotated tag objects

`run.meta.json` files record the tag object hash (`git rev-parse <tag>`), not the commit.

| Tag | Old tag object | New tag object | Old commit | New commit |
|---|---|---|---|---|
| `freeze-v1` | `c2c7560ce212b2f17d3ea98269bb213783e9198a` | `5461e1d97797240f1f8dc8359d5717cca7d0bcc3` | `bd15630` | `5283a20` |
| `freeze-v2` | `e81efc1f7ae60e2c6f86a311aaaf8e971bee7e6a` | `b1180497aeac5e12b720e8b133566c9e2d8089fa` | `233cfe1` | `2193e97` |
| `freeze-v3` | `53bbd50fa5f83f5b58e4cac1c87078c7049fde1f` | `f47e7d9e9eefe3055e1728fd7a4c7240c32be618` | `4f3b58f` | `2647cf9` |
| `prereg-cc` | `381d46a43e259d3bc2ac6aeb176ed47291641b5b` | `7c9d27de86ec140ea50192fcebe19ee5c1333c4e` | `bfaa7e5` | `053e3f2` |
| `prereg-sw` | `c5781610b1525ce780381023f4baee0c6b3aeec2` | `cf591351b363a5fd5931767ab86c646bdea07952` | `9f92a42` | `640cbda` |
| `tdk-submitted` | `2d3385b932f75354baa1892bee6fa433be0b8f03` | `b98b04c676c9848cb7cacde24b867057aef0d428` | `dfda3f0` | `5e65563` |

## Relation to the TDK repository

The archived repository `unsupervised-ensembling-false-alarms-tdk` already used the same author identity. 40 commits have identical hashes in both repositories: every commit up to and including the tag `tdk-submitted` (`5e65563`). Its last two commits (the public-release preparation of the TDK version) exist only there.

## All commits (oldest first)

| Date | Old | New | Same hash in TDK repo | Subject |
|---|---|---|---|---|
| 2026-10-20 | `e72f9841e32b3cf28c2da4c8833dc723aa6d8fb3` | `82b1ce90ea943bd9dc0d333cbb824d32788d951e` | yes | Scaffold repo: pinned env (pandas<3 for adtk), paths, env tests |
| 2026-10-20 | `6660e4670566d7d1d29c4351ac8f73a397be29e2` | `d00919fb428e8b852aa20e1f6310d873f949f27d` | yes | Add Kaggle API client for dataset downloads |
| 2026-10-20 | `58525900bbc605f0135926c00b16db5be93eab3a` | `c63e8403a750011d4f40635cb32a9902831d6d04` | yes | Point PROJECT path to renamed thesis-docs folder |
| 2026-10-21 | `54a3de9395083c5af07653166fe6b13b01758e70` | `641257508d372790e31a55fd503dfec28c4ffd68` | yes | Add ADBench + ULB loaders, splits and leakage tests |
| 2026-10-25 | `ed9e486f46c9de4d3611e1ddce5b128ed42f014e` | `31682bee4cfcb25fb60fcc46966fda12204d4d76` | yes | Merge exp/data-loaders (review: PASS) |
| 2026-10-25 | `6801c1c5d70ad95735034c67afebade9b4c6fd9d` | `3c1e99deaffff78bbd7013f3e1680798dd69405f` | yes | Add score() with train-only ECOD, fuse(), budget metrics, score cache, E1 runner |
| 2026-10-25 | `69881980ab378575e2a22863b6459afe081a2e9e` | `3b4a717668cd975f3f8255e7093dbd602b86ee68` | yes | ECOD: constant columns get skewness 0 (as in PyOD); README section on detectors, fusion, E1 |
| 2026-10-25 | `4ca42995e32db0a4a24efc4efcf0e2ba8d77cf0c` | `b5a7ebca8755fdd89dc582baa64b486f7dae9a1d` | yes | Merge exp/score-fuse (review: PASS) |
| 2026-10-25 | `8757d2b7e187cda82eaf1540ccd067249ccdcd5a` | `0f84cd0b3016e01d0092172197519fc4b9740f00` | yes | fusion: zscore maps a zero-variance member to 0 and warns (provisional) |
| 2026-10-25 | `e5322f3c5a4ab40c72f38d46170e235537df5ea9` | `d5a98771aee6513d55f3960200616d16e56cdb89` | yes | metrics: validate tp_at_k inputs (shapes, k, NaN) |
| 2026-10-25 | `aac47dbc6b5ebf53b8bde6ece4cae30b059ac77a` | `aa7f0b5f0afb93e61b02b4bfa3de3aed05161626` | yes | E1 summary table; provisional protocol choices as switches |
| 2026-10-25 | `f262171c61b1d47dcacd69c0a8463cb5a6e3134e` | `8354433805fe6cc9650489d99651de06ba53bf2c` | yes | E1 critical-difference diagrams (Friedman + Nemenyi) |
| 2026-10-25 | `fe87b901c54da8aab9ad3feec01350c095648b40` | `ad611c08c66e9b5c6b3f274bb0a07f601ede5a37` | yes | E1 CD diagram: tighter layout; README lists the CD outputs |
| 2026-10-28 | `70666c4d74cedbfddae026e5310e004e88d9ae08` | `e452e39d450a0bd8ec64e3d3a5790440a0d93efa` | yes | E1 Spearman heatmap of detector test scores (per dataset x seed, seed mean, dataset mean) |
| 2026-10-28 | `ef2d2855935d624e330e04ba1a04c2e1b693e4a7` | `8840e3dbe63900d45d33e315e064f0f54d235165` | yes | Merge exp/e1-outputs (review: PASS) |
| 2026-10-28 | `cf116a7d41aec2776d08f15102953ea3234c00f9` | `3ba30ed4c2d38075710514d26fe31537b5a3cec3` | yes | paths: neutral docstring |
| 2026-10-28 | `b204b33419bd6fe5c0a30537cb54d1c5d35b0a14` | `37b5cd790ab9848720bf67df231149f7c712df7f` | yes | Duplicate table per ADBench set and E1 duplicate robustness check (--dedup vs main ranks) |
| 2026-10-28 | `b12667f5989f7b6145a3fe126ff0e08d0f4d07e3` | `d4102ac2b7fae45c6cdf7055070880cde9fc0e8d` | yes | e1_dedup: a dataset is changed when dedup removes rows, not when n_test changes (letter) |
| 2026-10-28 | `c9e4ecd59a7f2ea0a005e1faa0a4e305335048ed` | `f798a6ee33f8290467401d2ec8b0d3afaf5b9c0f` | yes | Merge exp/e1-dedup (review: PASS) |
| 2026-10-28 | `3875513bb0e265a3f2b931401de2175fd4dc9e95` | `e4e36f96a7ddc4b8107833a731e9669ad4dc08ed` | yes | e1_dedup: identity check includes PR-AUC; rows_removed_per_dataset restores the protocol |
| 2026-10-28 | `897141a40b7af5e01522fe8b47af4e3a74f84f07` | `c80d0bb0cb2dd377422b2a7ed5071bdd688db7de` | yes | E2: pair and triple fusion runner (35 combos x 6 configs, training-only overlap, gain vs best and median member) |
| 2026-10-28 | `f9374615a80e5a5b280e85850cc1e7329af2e649` | `4849c5365f62d88ae98bf013b83ce897cf35ce83` | yes | README: E2 command and notes |
| 2026-10-28 | `7a4958bdf289d89d4e0304c0a17f049f7f00b66f` | `adda6d082df4a66a985e7855f5469d49774bf48f` | yes | Merge exp/e2 (review: PASS) |
| 2026-10-28 | `30459df56310d6b233644855741220e645f4257e` | `23f729c872a5bca54b6de43d065727e71ef2e9e6` | yes | E2 outputs: overlap-vs-gain scatter, pair-gain heatmap, triple ranking figure and LaTeX table; Friedman tests over combos; tie shares in the headline; member check covers both pair members |
| 2026-10-28 | `db409b37bea8d396b2a1e53b8de1c2874efac082` | `b870a30413a35e25e4fd3621470144902b302c37` | yes | E2 figures: round overlap and gain before ranking so exact ties survive a CSV round trip |
| 2026-10-28 | `c6d844a633310f511d625b4b2aa3ed208b457a56` | `271f8f10880087715d1392fdb684bb32024144bf` | yes | Merge exp/e2-figures (review: PASS) |
| 2026-10-29 | `d09525c05a52dde50cb0b18edd0358d884686493` | `7d6690e37cc57bf626a269d51f5f9cdb73ee0d7e` | yes | E2 review notes: a win is not a tie, n_within_cd tested, tied triple ranks share a position, min-max in reader-facing text |
| 2026-10-29 | `3ff44d5b171878388c7f59bd63da9ca6e0bfbc38` | `8f7c24da6a0fcddf9aba611c77123d64deae7a17` | yes | Zero-variance check of training scores (ADBench cache, ULB day 1, no labels); fit_scores; KNN/LOF sample size and seed in config; z-score treats a constant member as constant even when its float std is not 0 |
| 2026-10-29 | `8b25943a3f5a3eed83a3f7252f8063cd10d7fe4f` | `1598114c0e29e8d84e1ff5e9ab6e6fe79e1f057f` | yes | E-bench rule and note, frozen_config.json candidate with loader and checks, Experimental Setup tables (datasets, hyperparameters, versions, runtimes) |
| 2026-10-29 | `3b9ba9a5b4f42a81b19760096fd7976b709ec280` | `a67b3d5d229c3c7c3a151776dc5fe10f84b31165` | yes | Setup tables: Windows 11 name, leave-one-out training scores listed as PyOD behaviour |
| 2026-10-29 | `20c84136b6b924c872ba569c56d5a85b9fb71b9e` | `db1db9db18595173e772c06891fa9c480fa9af11` | yes | Frozen config completed: IEEE-CIS budgets, thresholds and split, seed handling, bootstrap, hypotheses and criterion, one primary ensemble, wording that stays true after the tag; stricter checks |
| 2026-10-30 | `decbfc5aeaf1ae8c7b8ecc18978fc52412f69589` | `75cdded7cbcdee854e10020e3f832b133d1d9a2e` | yes | Frozen config: IEEE-CIS decision over months 2-6 (sum of monthly FP@K), month-stratified day bootstrap, per-resample recomputation, H3b and E5 context as exploratory, multiplicity, six-detector pool; paired bootstrap module and IEEE-CIS window rules; a break test for every check |
| 2026-10-30 | `bd15630d81967b215bf55da750f0e4b0c0954198` | `5283a207eb66e9daff596c75c31be81c3f205d32` | yes | Merge exp/e-bench (review: PASS) |
| 2026-10-30 | `d2aa3329a6c39ee59ce698d2c103838a7181f02c` | `b523c433547a1ea53633170ef28c308058266ddb` | yes | E3 on ULB: budget and threshold mode from the frozen config, paired bootstrap CIs, H1a/H1b/H2b; leakage, temporal and freeze tests; two-seed test for the H3a seed handling |
| 2026-10-30 | `83ddbda7067442a2b689281c379eb832246dc0a0` | `fb92c28a3e999a8d1acd3c93f416fe35b2e6d042` | yes | E3 ULB: verdict only on the hypothesis rows of the deltas table (others descriptive); break tests for median single, ensemble seed pairing, paired budgets, primary fusion in H1a/H1b and every code check |
| 2026-10-30 | `c79b4a76cc41e8c4534858a4a59513e20d74981b` | `2d779ccf47782013a3b9efdbaa0126b94ac9e105` | yes | E3 ULB tests: hypothesis roles on the frozen pairs only, verdict recomputed from the CI, a CI crossing 0 for the above-0 flag |
| 2026-10-30 | `ae58e70db4c740ac6df4d51fe1076f0b06bcaebf` | `6a14dc53d17d081936d0bee09b10428498709650` | yes | Merge exp/ulb-e3 (review: PASS) |
| 2026-10-30 | `ecde099b42287b5ab96ad43d93933caaa00bc126` | `c53bd4c6d7dd6244793d47210d54551872d58233` | yes | TDK exports: tables fit the 16 cm text block (hyperparameters column count, narrower tabcolsep), duplicate table as LaTeX, ULB figures at 16 cm with text >= 8 pt and a standalone rebuild, label-free LOF diagnosis, LaTeX compile tests |
| 2026-10-30 | `1c6ad9229d4c12d400707e3090d848fd71a02f17` | `56c150742ca9399892df66b0c37a201b08923ebd` | yes | TDK exports, review R1-R4: height check in texcheck (Float too large), hyperparameters table in footnotesize, asserts for the dedup rate and in-group counts and for sd_ratio, no left spine next to the 0 line in the delta figure |
| 2026-10-30 | `dfda3f0d8d0c0b80eb01b949c4a550b275be2329` | `5e655639f4871d7a217f201ae9c1325648b3cb84` | yes | Merge exp/tdk-exports (review: PASS) |
| 2026-11-16 | `c1d33f1f31a4cbeb7edcede8c88d064122c9dd52` | `3611b98daa6d8509f09cf34441ccb84d4db6b41a` |  | LOF diagnosis on ULB: scikit-learn recomputation (L0), half-day stress test and within-day controls (L2), score quantiles and histograms (L1); label-free; slice, leakage and r tests |
| 2026-11-16 | `cb9efdd4a37fee3ccd4317c130a1db2d7aebd355` | `3aa02a844c9fab39171654a75e3236a283cfb041` |  | LOF L1 figure: legend below the panels, x-axis from the pooled 0.1% quantile |
| 2026-11-16 | `7475821a1b93528dc2fc5bf1675a699ca56ec23f` | `baadb2851bfed705c28641496309a831a0a70b67` |  | LOF L1 figure: x-axis floor from the positive scores (KNN has exact zeros) |
| 2026-11-17 | `33ecf84a20e7d314de9ac3f4d4fe9906a6513b5b` | `452dfda7cc600877844b6098a7b0a8526de05e70` |  | LOF diagnosis test: three members so the median and the mean of r differ |
| 2026-11-17 | `3d7d82b6900e475ccfdd0abde825177b04a0d45b` | `d8e144ede36aed064803e039760fa20a1fbbdcf8` |  | Merge exp/lof-diagnosis (review: PASS) |
| 2026-11-16 | `49c043508dd535226c2d0b6607a75982f9c9c314` | `862672fa15c234c51f06bac4f58dd38efd555412` |  | IEEE-CIS feature rule from month 1 without labels: candidates, missing-share and constant filter, frequency encoding, median imputation, log1p of TransactionAmt; month-1 loader that never reads the label; checks of the frozen feature file |
| 2026-11-16 | `22826760b328d43b576ec2e45edf08bcfdbc3a41` | `26785ae587d461c84091b83c79f5d96d9b9689af` |  | IEEE-CIS feature file: parameters fitted on the fit window of each run, no row filter, raw typed columns for the H3b drift share |
| 2026-11-16 | `1ac75bec28957c57e2ee755e5f2b0ff162494f46` | `e9badc8b436d56271614995545ecfffdf9b8530f` |  | IEEE-CIS feature file from month 1 |
| 2026-11-16 | `f19e78de15223e15880bc62657427ebc4e716072` | `6b313c12c1f20dcce8cdd821166557741b63a6ed` |  | IEEE-CIS feature file: 26 features with column types, fit-window, row and drift-column rules; README |
| 2026-11-16 | `17525fcdf7e1d9bbc619fba792387a9e3b71c280` | `b5d6e24642c91e7c1a3a3e28faca84fef94441c3` |  | IEEE-CIS feature file: fit-window wording |
| 2026-11-16 | `233cfe17ba74cca8b559d1a9703d4dd17a6ec75a` | `2193e973c9b02577013dcdfd7a3e63c7dfca0258` |  | IEEE-CIS feature file regenerated (fit-window wording) |
| 2026-11-17 | `fe4f197b657c116f4a196863423452e550fb1d55` | `35b9cfb4d8b22adcba00bf653b46b0e6763afbb0` |  | IEEE-CIS feature tests: no read of the label column (read_csv spy), int and float levels, duplicate-feature break on its own, load_frozen rejects a broken file |
| 2026-11-17 | `9ef0bf7a0a2914246bbc099f91c12b012ac99419` | `50dd36a7c875c1b1b03808388ac871eb36bea849` |  | Merge exp/ieee-features (review: PASS) |
| 2026-11-17 | `c8688f47b7f2fe610d7bd5ef7b7ccb18fb6639fb` | `3c8dca21605a6070118b58ebf53910c328341a41` |  | IEEE-CIS month-1 stress test: fit on days 0-20, score days 21-29 (and 21-27), month-1 rows only, no labels; r absolute and relative to the median member; one- and two-sided rules for several c |
| 2026-11-17 | `1d1c7f421df2194786f9ce8a85405580eb9524c8` | `14f38ab6c8b364c43c4dcdd59adc32de7b0f834f` |  | IEEE-CIS month-1 stress test: alerts per day at the fit-window thresholds and per-feature tails of the scored slice |
| 2026-11-17 | `f74140e7a56669aa982a01d28f50890ad5a5462b` | `63c1d2f18c94d3a1285b0219b1e709f1fdbe4bee` |  | Merge exp/ieee-stress (review: PASS WITH NOTES) |
| 2026-11-17 | `8e8bace5ad72c7568d9c40d731e47c03f8c34514` | `aceeac32708362fd70279aed7bac5df8316fdb63` |  | E-stress and run-time guard on ULB: label-free member decisions for every candidate rule (one- or two-sided, absolute or relative, c 2/3/5), static fallback, min-max + mean fusion of the kept members, FP@K, threshold mode and paired bootstrap against the static ensemble |
| 2026-11-17 | `98d9db253eb27facf66a79856b3a7f740cc67e00` | `6768b819b1654a994168fb2df9ecad32d657c429` |  | E-stress: singles in the variant scores (bootstrap input and reference rows) |
| 2026-11-17 | `ab0918120a23f83634653453b49b56133a6227d1` | `fe09753d8484bfd7d6130ab04546ca447037c2eb` |  | E-stress test: each variant compared with its own static ensemble (two ensembles) |
| 2026-11-17 | `e0fef8f7be9764a29330bbaf4a5d2ffb98863a11` | `a1af23453e6a994079ff4eaa562843a98877b711` |  | LaTeX tables from the committed CSVs: LOF L2 controls, IEEE-CIS features and month-1 stress test, ULB development table for the guard and the rules |
| 2026-11-17 | `a0fa4b087cd398f3b4c5ec2b4ea2f9c738981465` | `b9afe4b9bd5522eab93a35e1df47403cbb815094` |  | L2 control table: run codes in the header, run descriptions in the caption (fits the text block) |
| 2026-11-17 | `e8c0a8edd9669d46324fc726c5810fb6e9239a3e` | `010d8e2ed672c259da77db59d6be11ec46018be1` |  | ULB development table: wrapping columns for the rules and the kept members (fits the text block) |
| 2026-11-17 | `1932feda2934e378952371dc54534a5c8b3c85c3` | `28a97bd49a2a8ba32e2c58903a2f66a7a223d8d9` |  | ULB development table: narrower wrapping columns |
| 2026-11-17 | `bd2ad047525630a3ca1139ee374623c77bd989fc` | `45af22a0839e2d28bc0297b741ba9fb3dd8595d3` |  | Feature table: read the transform column by name, not as the DataFrame.transform attribute; test asserts every cell |
| 2026-11-17 | `c3f340db591f8b269bd2de1be5e0f4bfc06ae653` | `0bc984f64d3ed6a585f6577725a91f8b82306fe4` |  | IEEE-CIS stress test: alert_days threshold test with a score on the threshold and one between the linear and lower quantiles (review N1) |
| 2026-11-19 | `844438e6e1365ce016788352e65ab7c8eb69f36d` | `810b45f3c8036b684af272d1ab1b9ace58a702d5` |  | Merge exp/e-stress (review: PASS) |
| 2026-11-17 | `79766294caa7afe8dca5f6fb09d9f4708ae1cc3f` | `3f9b6a8b5cccb6d606bf7d2f422df99e7daec788` |  | Run-time guard: exact r <= c rule on integer alert counts (r = c passes), daily decisions, static fallback below two members, min-max + mean of the kept members; ULB comparison of guarded and static ensembles with the median, oracle and day-1-chosen single, transaction and hour-block bootstrap |
| 2026-11-17 | `e3f94827baaefbfaddf425c4951d7170e51f7c7a` | `f3adf686ea1eeff7a501e3cdf0da17f90d9d9bd9` |  | Threshold fragility index per member (exploratory): share of training scores above the threshold moved 10% of the way to the median, over 1 - q; affine-invariant; ULB day 1 and IEEE-CIS month 1 training scores only |
| 2026-11-17 | `43dae3c17e7df52cdb48ad58f51c160ce248dfe2` | `d69f038dcd998868e873c869eefea8ca299a600e` |  | freeze-v3 file (draft for review, not tagged): RQ4 wording, run-time guard as the main mechanism, pre-deployment check as a secondary test, H4a-H4e, drift slope, daily budget, no-break case, removed rules; validate() with a break test per check and a tag-time check that refuses the draft block and stale status words |
| 2026-11-17 | `e433581860ff122fe1c559cf6706bb81cebe7510` | `38504ae73c3b16bf41f00cc7c67533c96e030a73` |  | IEEE-CIS runner for the single run (freeze-v1 E3/E4/H3a/H3b/context and freeze-v3 RQ4 together), refuses to start without the freeze-v3 tag or a second time; dry run on month 1 only with synthetic labels; tests on a fake file, no month 2-6 row read; freeze-v3 file: H4e stays decided when no member breaks |
| 2026-11-17 | `2675c676e9cb186826b15f718ea3da3c646d71d5` | `ae6d550e0127a9ee568eac53460e07c054046d4b` |  | Robust fusion: median as an extra aggregation outside the six frozen fusion configs; ULB budget and threshold tables and the ADBench static check for min-max and train-percentile with mean and median; label-free bound check of the median proposition; tests including the frozen primary outputs |
| 2026-11-17 | `de2cc598b3b9fe80b5a0c282cd4fb178d94ccd8a` | `3cde6066689cbc3df87cdacced95ab91d12a8bc9` |  | freeze-v3 draft: robust fusion (train-percentile + median, developed on ULB) as the main RQ4 hypothesis H4r in threshold and budget mode, monthly label-free bound check, guard as detect-and-exclude comparison, H4d appendix only, H4e descriptive without month-1 labels; runner and validator updated, tests for every new check |
| 2026-11-17 | `15c8a577e31f1b8c986316d9faf172bc0bc7ad02` | `7716325ff6b3625be31b7b9e276e9846d8d81c0f` |  | Robust fusion over the 20 three-member ensembles: block alerts and the /log r/ deviation with the 0.5-alert rule, paired dD per triple, cost ratio of mean totals with the +5% margin, exact FP@K for block resamples from sorted scores; ULB day-2 development check (--triples); proposition test with the accepted even-m count |
| 2026-11-17 | `b56acd3c613b6f837b6874afd4bdab6417e53180` | `a09ed504d8ae10de1f280d4b1e20c1f41d1c672d` |  | freeze-v3 draft: H4r decided by threshold mode over the 20 three-member ensembles (paired mean change in monthly /log r/), H4r-cost as a separate budget-mode check with a +5% margin from the ADBench development cost, E-bench / E-bench-5% / E-all descriptive, every choice confirmed; runner computes both over the 20 triples; validator and tests |
| 2026-11-19 | `e855a3a62a13f5eaf15e5cf31906947f71303b24` | `7ee7cac023f29aaa4b16e0e8a025fef0bf8a638d` |  | freeze-v3: draft_status removed for the tag; IEEE-CIS labels read for months 2-6 only; H4b/H4c status follows the no-break rule; CI and edge-case wording for H4a, H4d, H4e, bound check and drift slope; tests for the run guards, fit windows, resamples, cost CI and bound check |
| 2026-11-19 | `4f3b58f392d05e785500c673bdc86ece81cfe182` | `2647cf9942bdd3133efb75d8ba180d3c3ce7989d` |  | Merge exp/freeze-v3 (review: PASS) |
| 2026-11-19 | `67866c7ae0f43e37b19161fb3fefc8f20fe0b23a` | `4dbb85bff1b1b13095602a0a66503d1913e800df` |  | Docstrings of guard, fragility and guard_ulb: the freeze-v3 rules are frozen, not proposed; the month-1 chosen-single comparator was left out of H4e (wording only) |
| 2026-11-19 | `b742c0084315afb4734ca9507b5a257b54920005` | `a1b850b048697db0037b19c9697c6b99006500e2` |  | Merge exp/n8-wording (review: PASS) |
| 2026-11-19 | `625bb29467dcbc28314e1ea7a7c7c2173be0fcab` | `89c1cd6e47d30550e0ad95ef3e235b4616f2e221` |  | Chapter 5 exports: IEEE-CIS hypothesis summary, H4r tables, guard events, appendix tables, ADBench static cost table, trade-off and H4a figures, ULB LOF control figure; read from the committed results only |
| 2026-11-19 | `ef25b6dfe73102495c286c32cf6a4f441ea39cfc` | `d4b72484d8ea6e280d27a1638162d8cbe49abf67` |  | Merge exp/ch5-exports (review: PASS) |
| 2026-11-20 | `9f92a42f91785add388d1ba606b74dedc72ece80` | `640cbdad85997e0df3910e0f1b430a7d156924fa` |  | Spectral weighting test: pre-registration, label-free weights, label-tuned ceiling, Step 1 runner and tests |
| 2026-11-20 | `62f20da0396b1a3ba83f9808fed096105d3b21cb` | `cc720b7aea8d47b92b69cd3352ab2e5228d40ecb` |  | Spectral weighting Step 2: SW fusion against EQ and IForest, H-SW1/H-SW2 fixed sequence, descriptive tables and delta figure |
| 2026-11-20 | `bfaa7e5f5a344e482b07d882e1b12e3c428f21af` | `053e3f26e7dc8d9762fbca2b902905f0e6b647cb` |  | Committee cleaning test: pre-registration, row selection, refit runner with reproduction check and Step 1 |
| 2026-11-20 | `c1f93114929baee40433a72327503bc2ec5659f7` | `452981a825d5a0cff8919a1793896c27a11c0860` |  | Committee cleaning Step 2: committee and self-cleaning refits, H-CC1 to H-CC3 fixed sequence, descriptive tables and delta figure |
