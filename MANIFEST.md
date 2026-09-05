# Numerical provenance manifest

All paths are repository-relative; line numbers below are one-based. Frozen logs preserve original numerical text. Documented redactions change only private path strings. Original and public SHA-256 hashes are in docs/FILE_INVENTORY.json.

| Manuscript claim | Exact source | Check scope |
|---|---|---|
| 29/34 positive Breslow differences | results_raw/experiments/protocol_decomp.txt, FULL COHORT cox:efron+breslow rows | Recomputed from all 34 keyed rows |
| 31/34 non-positive profile differences | Same file, FULL COHORT cox:efron+kalbfleisch-prentice rows | Recomputed at frozen printed precision |
| 0/34 resolved positive profile differences | Same profile rows and N-B SE | D_KP > 1.96 SE_KP count |
| SPARCS T=6: 0.29336, 0.00261, 0.00135, 0.00128 | Same file, sparcs/drg302 @ T=6, full-cohort Efron arms | Exact keyed values checked |
| E8 rho(all)=-0.1422; rho(events)=+0.8742 | results_raw/experiments/simulations_e8.txt, 27-cell table and final ordering block | Recomputed using original ordinal-rank algorithm; tie-aware Spearman differs |
| SUPPORT2 and MUSIC 20-split grid results | protocol_decomp.txt, support2/slos and drsa/music full-cohort blocks | Exported as results/main/support2_music.csv |
| All 34 supplement S1/S2 rows | protocol_decomp.txt plus ordering_variable.txt | All 170 values extracted independently from supplied supplement.pdf and compared |
| Independent SPARCS 12 rows; 3 resolved residuals | results_raw/experiments/kp_out_of_sample.txt | All rows exported, resolved count recomputed |
| Weibull shape=.8, n=10000, censor=0, five T values | results_raw/experiments/simulations_e5.txt | Five rows drive F3; full 135-row grid exported |
| Supplement S3/S4 | results_raw/experiments/flexibility_<cohort>.txt, PAIRED DIFFERENCES / KAN(matched) | All 25 differences and corrected SEs compared with PDF |
| Sensitivity percentages and likelihood differences | results_raw/experiments/robustness_e24e20e21e23e22e19.txt, E20/E21 | Source located; per-line scientific review pending |
| Theory and proof constants | theory scripts and results_raw/theory/ logs | Preserved; this preparation is not an independent theorem proof |

Rounding tolerance is half the printed final decimal unit plus a small floating-point margin. Corrected SE consistency propagates rounding in both printed columns. Missing split-level results prevent an independent raw sample-variance computation.

## Exact row provenance for S1/S2 and F1

| Cohort | T | Efron+Breslow line | Efron+KP line | Event-mass line in ordering_variable.txt |
|---|---:|---:|---:|---:|
| metabric | 4 | 25 | 26 | 37 |
| metabric | 6 | 61 | 62 | 38 |
| metabric | 10 | 97 | 98 | 39 |
| metabric | 20 | 133 | 134 | 40 |
| nwtco | 4 | 169 | 170 | 33 |
| nwtco | 6 | 205 | 206 | 34 |
| nwtco | 10 | 241 | 242 | 35 |
| nwtco | 20 | 277 | 278 | 36 |
| flchain | 4 | 313 | 314 | 21 |
| flchain | 6 | 349 | 350 | 22 |
| flchain | 10 | 385 | 386 | 23 |
| flchain | 20 | 421 | 422 | 24 |
| support-pycox | 4 | 457 | 458 | 25 |
| support-pycox | 6 | 493 | 494 | 26 |
| support-pycox | 10 | 529 | 530 | 27 |
| support-pycox | 20 | 565 | 566 | 28 |
| drsa/clinic | 5 | 599 | 600 | 17 |
| drsa/clinic | 10 | 663 | 664 | 18 |
| drsa/clinic | 25 | 699 | 700 | 19 |
| drsa/clinic | 50 | 735 | 736 | 20 |
| support2/slos | 6 | 769 | 770 | 7 |
| support2/slos | 10 | 831 | 832 | 8 |
| support2/slos | 15 | 893 | 894 | 9 |
| support2/slos | 20 | 957 | 958 | 10 |
| support2/slos | 30 | 993 | 994 | 11 |
| support2/slos | 60 | 1029 | 1030 | 12 |
| sparcs/drg302 | 6 | 1063 | 1064 | 13 |
| sparcs/drg302 | 10 | 1125 | 1126 | 14 |
| sparcs/drg302 | 15 | 1188 | 1189 | 15 |
| sparcs/drg302 | 30 | 1250 | 1251 | 16 |
| drsa/music | 6 | 1312 | 1313 | 29 |
| drsa/music | 12 | 1374 | 1375 | 30 |
| drsa/music | 30 | 1436 | 1437 | 31 |
| drsa/music | 60 | 1498 | 1499 | 32 |

## Claims not fully independently traced

See docs/NUMBER_AUDIT.csv for every digit-bearing manuscript source line, including equation labels, citations and numerical prose. A source map is not the same as numeric validation. Rows marked REVIEW_REQUIRED need author/scientific review; the package does not claim that all manuscript numbers have passed. Missing original supplement TeX, per-split metrics, original simulation hash seed, source-cache builders and original figure plotting code are listed in docs/RELEASE_BLOCKERS.md. All original scientific text remains unchanged.
