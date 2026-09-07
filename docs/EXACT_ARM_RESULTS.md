# The exact-discrete comparator: results

*Generated 2026-09-07 by `paper/exact_arm_report.py` from the result JSON of the four runs below. No number here is typed by hand.*

This covers the three experiments the review asked for before any further manuscript revision, and nothing else. Each section names the script that produced it and the log it can be checked against.

| # | experiment | script | source JSON | blocks | source written |
|---|---|---|---|---|---|
| 1a | Exact-discrete comparator on the Weibull simulation | `experiments/exact_arm_sim.py` | `experiments/exact_arm_sim.json` | 135 | 2026-09-06 21:30 |
| 1b/3 | Exact arm and coefficient comparison on real data | `experiments/exact_arm_real.py` | `experiments/exact_arm_real.json` | 7 | 2026-09-06 23:48 |
| 1c | Scalability ladder on SPARCS | `experiments/exact_arm_ladder.py` | `experiments/exact_arm_ladder.json` | 5 | 2026-09-06 23:58 |
| 2 | Out-of-sample SPARCS validation, 20 splits, event mass | `experiments/kp_out_of_sample_v2.py` | `experiments/kp_out_of_sample_v2.json` | 12 | 2026-09-07 00:53 |

---

## 1a — Exact-discrete comparator on the Weibull simulation

Four arms, differing in one thing at a time:

| arm | coefficients | baseline |
|---|---|---|
| A | Efron | Breslow |
| B | Efron | cloglog profile (= Kalbfleisch–Prentice) |
| C | exact-discrete | cloglog profile |
| D | grouped joint MLE | grouped joint MLE |

So **A − B** is the baseline representation, **B − C** is Efron against exact tie handling with the baseline held fixed, and **C − D** is what remains. Arm C profiles its baseline under cloglog rather than logit precisely so that B and C differ in the coefficient alone.

135 cells of the E5 Weibull design (3 shapes x 3 censoring levels x 3 sample sizes x 5 grids), 20 replications each, on seeds identical to the frozen E5 log so the two are directly comparable.

Arm C exists in **102 of 135 cells**. In the other 33 the exact recursion exceeds its budget, every one of them at n = [10000].

### Coefficient recovery by grid

`shrink` is the slope of the mean estimate on the true coefficient: 1.000 is unbiased, below 1 is attenuated toward zero, above 1 overshoots.

| T | cells | shrink Efron | shrink exact | shrink grouped |
|---:|---:|---:|---:|---:|
| 2 | 18 / 27 | 0.620 | 1.492 | 0.900 |
| 4 | 18 / 27 | 0.798 | 1.366 | 0.976 |
| 8 | 18 / 27 | 0.921 | 1.241 | 1.005 |
| 20 | 21 / 27 | 0.985 | 1.116 | 1.008 |
| 40 | 27 / 27 | 1.000 | 1.069 | 1.010 |

### Held-out likelihood, in both link families

Each arm is scored against the joint MLE of its **own** family. The cloglog columns are the ones comparable to the paper's D_T; the logit columns judge the exact method in its own metric.

| T | cells | D_A | D_B | D_C | D_B' (logit) | D_C' (logit) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 18 / 27 | +0.15610 | +0.00827 | +0.06204 | +0.04232 | -0.00006 |
| 4 | 18 / 27 | +0.08614 | +0.00364 | +0.03274 | +0.02662 | -0.00006 |
| 8 | 18 / 27 | +0.03314 | +0.00025 | +0.01563 | +0.00998 | -0.00006 |
| 20 | 21 / 27 | +0.00985 | -0.00042 | +0.00461 | +0.00145 | +0.00011 |
| 40 | 27 / 27 | +0.00690 | -0.00069 | +0.00155 | +0.00020 | +0.00011 |

### What the fourth arm does

- Mean attenuation across the 102 cells where arm C exists: **Efron 0.880, exact 1.236, grouped 0.983**.
- The exact coefficient is **further from the truth than Efron's in 97 of 102 cells** (95%), and it errs by overshooting rather than attenuating.
- Arm C predicts worse than arm B in 102 of 102 cells scored in the cloglog family — but that comparison is confounded, because the exact coefficient is a log odds ratio and the cloglog baseline is not its own. Scored in the **logit** family against the logit joint MLE, which is the exact method in its own metric, arm C' still predicts worse than arm B' in **13 of 102** cells.
- On the coarse grids the review singles out (T <= 4, 36 cells): Efron 0.709, exact 1.429, grouped 0.938. Exact tie handling does not recover what Efron loses; it misses in the opposite direction and by more.
- In its own family the exact method is **not merely competitive, it is indistinguishable from the joint MLE**: median |D_C'| = 0.000027 against |D_B'| = 0.007820, a factor of 291. The conditional likelihood gives up essentially nothing by conditioning the T nuisance parameters away rather than estimating them.

**What this means.** The exact-discrete conditional partial likelihood is not broken and is not a poor estimator — within the discrete *logistic* family it reproduces the joint MLE to five decimal places at every grid, which is also a strong check on the implementation. But its coefficient is a log **odds** ratio, and these data are cloglog-generated, so it is estimating a different parameter from the one that produced them — a log odds ratio is systematically larger in magnitude than the corresponding log hazard ratio, which is exactly the overshoot in the table above.

So the Cox-versus-grouped coefficient discrepancy is a question of **which parameter is being estimated**, not of how ties are approximated. Exact tie handling cannot close it, because the gap was never an Efron artifact. This removes the last of the three candidate explanations: the baseline representation, the link, and now tie approximation.

---

## 1b / 3 — Real data, and the coefficient comparison

Same 20 splits and same seeds as `protocol_decomp.py`, rerun only because that script drops the coefficient vectors before writing its JSON. `R` is the relative distance from the grouped joint MLE, on the standardised scale:

> R = || beta_arm - beta_grouped || / || beta_grouped ||

| cohort | T | n | modal event mass | level | R Efron | R exact | cos Efron | cos exact |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| drsa/clinic | 5 | 4,828 | 0.8374 | FULL COHORT | 0.13856 | infeasible | 0.9987 | -- |
| drsa/clinic | 5 | 4,828 | 0.8374 | SUBSAMPLE n_train=3306 | 0.13887 | 0.71803 | 0.9987 | 0.9782 |
| support2/slos | 6 | 9,105 | 0.4912 | FULL COHORT | 0.05172 | infeasible | 0.9994 | -- |
| support2/slos | 6 | 9,105 | 0.4912 | SUBSAMPLE n_train=4756 | 0.05156 | 0.35050 | 0.9994 | 0.9916 |
| sparcs/drg302 | 6 | 34,233 | 0.9032 | FULL COHORT | 0.36169 | infeasible | 0.9121 | -- |
| sparcs/drg302 | 6 | 34,233 | 0.9032 | SUBSAMPLE n_train=2986 | 0.58639 | 0.92285 | 0.7235 | 0.8176 |
| sparcs/drg302 | 10 | 34,233 | 0.6679 | FULL COHORT | 0.09953 | infeasible | 0.9877 | -- |
| sparcs/drg302 | 10 | 34,233 | 0.6679 | SUBSAMPLE n_train=3505 | 0.48128 | 0.62977 | 0.7819 | 0.8660 |
| sparcs/drg302 | 15 | 34,233 | 0.7191 | FULL COHORT | 0.09094 | infeasible | 0.9836 | -- |
| sparcs/drg302 | 15 | 34,233 | 0.7191 | SUBSAMPLE n_train=3347 | 0.53591 | 0.67000 | 0.7427 | 0.7578 |
| drsa/clinic | 50 | 4,828 | 0.4040 | FULL COHORT | 0.01351 | 0.13331 | 1.0000 | 0.9913 |
| sparcs/drg302 | 30 | 34,233 | 0.5454 | FULL COHORT | 0.01312 | infeasible | 0.9999 | -- |
| sparcs/drg302 | 30 | 34,233 | 0.5454 | SUBSAMPLE n_train=3854 | 0.68307 | 0.73543 | 0.5496 | 0.5478 |

Where both coefficients exist at the same sample size, the exact one is closer to the grouped MLE in **0 of 7** comparisons.

In 6 of 7 it is further from the grouped coefficient than **Breslow**, the crudest tie approximation available.

The two metrics disagree in 3 of 7 comparisons: the exact coefficient has the **higher cosine similarity** — it points in a better direction than Efron's — while having the larger `R`, because it is too long. That is the log-odds-versus-log-hazard scaling of run 1a showing up on real data: the direction of effect is right, the magnitude is systematically inflated.

### Does the disagreement contract as ties weaken?

| cohort | T | modal event mass | R Breslow | R Efron | R exact |
|---|---:|---:|---:|---:|---:|
| drsa/clinic | 5 | 0.8374 | 0.50740 | 0.13856 | infeasible |
| drsa/clinic | 50 | 0.4040 | 0.25283 | 0.01351 | 0.13331 |
| sparcs/drg302 | 6 | 0.9032 | 0.61118 | 0.36169 | infeasible |
| sparcs/drg302 | 30 | 0.5454 | 0.16348 | 0.01312 | infeasible |

On `drsa/clinic` the Efron-to-grouped distance falls from 0.1386 at T=5 (modal event mass 0.837) to 0.0135 at T=50 (0.404) — a factor of 10.3. Yes: the disagreement is a property of tie severity and it contracts as the grid refines.
On `sparcs/drg302` the Efron-to-grouped distance falls from 0.3617 at T=6 (modal event mass 0.903) to 0.0131 at T=30 (0.545) — a factor of 27.6. Yes: the disagreement is a property of tie severity and it contracts as the grid refines.

The exact arm ran on the full training split in 1 of 7 configurations. The other 6 needed a subsample, on which every arm was refitted so the comparison stays paired. The largest cohort-level cost is 132x the budget.

---

## 1c — Where the exact method stops being computable

Nested subsamples of one SPARCS cohort — a single permutation, prefixes of it — so the rungs are a growing dataset rather than unrelated draws.

| n | events | modal event mass | max_t d_t | cost | feasible | seconds | R Efron | R exact |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 500 | 500 | 0.9120 | 456 | 228,500 | yes | 82 | 0.91630 | 4.79064 |
| 1,000 | 999 | 0.8959 | 895 | 896,000 | yes | 135 | 0.78183 | 0.79125 |
| 2,000 | 1,997 | 0.9014 | 1,800 | 3,602,000 | yes | 254 | 0.34749 | 0.84506 |
| 5,000 | 4,996 | 0.9019 | 4,506 | 22,535,000 | **no** | 33 | 0.19546 | -- |
| 10,000 | 9,991 | 0.9005 | 8,997 | 89,980,000 | **no** | 55 | 0.39265 | -- |

The exact method runs up to **n = 2,000** and no further.

Cost grows quadratically because `max_t d_t` grows with n at fixed tie severity, so cost is O(n^2). The rungs confirm it rather than assume it — modal event mass is near-constant across them, and each observed ratio matches the square of the size ratio:

| step | size ratio | predicted cost ratio | observed |
|---|---:|---:|---:|
| n=500 to 1,000 | 2.0x | 4.00x | 3.92x |
| n=1,000 to 2,000 | 2.0x | 4.00x | 4.02x |
| n=2,000 to 5,000 | 2.5x | 6.25x | 6.26x |
| n=5,000 to 10,000 | 2.0x | 4.00x | 3.99x |

Modal event mass stays between 0.8959 and 0.9120 across the ladder, so the growth is the sample size and not a drift in tie severity.

On the rungs where both exist, exact is closer to the grouped coefficient in **0 of 3**.

One incidental observation: `D_B` — the Cox arm with a Kalbfleisch-Prentice baseline — is **negative on 4 of 5 rungs**, i.e. that arm beats the grouped joint MLE outright at these sizes. It turns positive only at the largest rung. That is the KP-null of the main paper reappearing on a cohort subsample, and it is consistent with the baseline representation carrying the effect.

---

## 2 — Out-of-sample SPARCS validation, corrected

Two inconsistencies with the main experiment are fixed here: 20 splits instead of 10, and modal **event** mass instead of modal all-exit mass. Everything else — cohorts, row cap, seed, coarsening, Nadeau–Bengio correction — is unchanged.

| cohort | T | n | events | intervals with events | max events in one interval | modal event | modal all-exit | D_T Breslow | NB SE | res | D_T KP | NB SE | res | baseline % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|
| sparcs/drg640 | 6 | 50,000 | 50,000 | 4 | 49,418 | 0.9884 | 0.9884 | +0.42282 | 0.00146 | yes | +0.00350 | 0.00093 | yes | 99.2 |
| sparcs/drg640 | 15 | 50,000 | 50,000 | 10 | 32,447 | 0.6489 | 0.6489 | +0.17693 | 0.00208 | yes | +0.00067 | 0.00108 | no | 99.6 |
| sparcs/drg640 | 30 | 50,000 | 50,000 | 20 | 29,156 | 0.5831 | 0.5831 | +0.09908 | 0.00198 | yes | +0.00068 | 0.00103 | no | 99.3 |
| sparcs/drg560 | 6 | 50,000 | 49,981 | 6 | 49,500 | 0.9904 | 0.9900 | +0.42682 | 0.00316 | yes | +0.00209 | 0.00226 | no | 99.5 |
| sparcs/drg560 | 15 | 50,000 | 49,981 | 15 | 32,626 | 0.6528 | 0.6525 | +0.19122 | 0.00300 | yes | +0.00031 | 0.00159 | no | 99.9 |
| sparcs/drg560 | 30 | 50,000 | 49,981 | 29 | 30,061 | 0.6014 | 0.6012 | +0.13572 | 0.00257 | yes | +0.00071 | 0.00061 | no | 99.5 |
| sparcs/drg540 | 6 | 50,000 | 49,882 | 6 | 46,859 | 0.9394 | 0.9372 | +0.34245 | 0.00217 | yes | +0.00476 | 0.00063 | yes | 98.6 |
| sparcs/drg540 | 15 | 50,000 | 49,882 | 15 | 38,770 | 0.7772 | 0.7754 | +0.21163 | 0.00251 | yes | +0.00169 | 0.00032 | yes | 99.2 |
| sparcs/drg540 | 30 | 50,000 | 49,882 | 29 | 23,286 | 0.4668 | 0.4657 | +0.08017 | 0.00208 | yes | +0.00014 | 0.00011 | no | 99.8 |
| sparcs/drg720 | 6 | 50,000 | 48,576 | 6 | 21,033 | 0.4330 | 0.4207 | +0.05685 | 0.00173 | yes | +0.00051 | 0.00025 | yes | 99.1 |
| sparcs/drg720 | 15 | 50,000 | 48,576 | 15 | 9,730 | 0.2003 | 0.1946 | +0.01192 | 0.00096 | yes | +0.00002 | 0.00006 | no | 99.8 |
| sparcs/drg720 | 30 | 50,000 | 48,576 | 29 | 4,948 | 0.1019 | 0.0990 | +0.00319 | 0.00052 | yes | +0.00000 | 0.00002 | no | 100.0 |

Resolved positive under Breslow: **12 of 12**. Resolved positive under KP: **4 of 12**.

Modal event mass spans 0.1019 to 0.9904; modal all-exit mass spans 0.0990 to 0.9900. The two are not interchangeable, which is why the substitution mattered.

| correlation | value |
|---|---:|
| Spearman(modal **event** mass, D_T Breslow) | 1.0000 |
| Spearman(modal all-exit mass, D_T Breslow) — what v1 reported | 1.0000 |
| Spearman(modal **event** mass, D_T KP) | 0.8671 |

**The correction was right in principle and changes nothing here.** On these four APR-DRGs almost every row is an event — censoring is negligible — so the two measures never differ by more than 0.0123 and rank the configurations identically. The Breslow correlation is unchanged to four decimals.

That is worth stating rather than glossing: the substitution matters where censoring is substantial, and `drsa/clinic` at T=5 in run 1b is such a case (0.8374 event against 0.7730 all-exit). It does not matter on the SPARCS validation cohorts, so no conclusion in this section rests on it.

---

## What these four runs settle

1. **The fourth arm does not rescue the coefficient story — it closes it.** Exact tie handling was the remaining candidate explanation for the Cox–grouped coefficient gap. It attenuates to 1.236 against Efron's 0.880 and the grouped 0.983, i.e. it misses in the opposite direction. The gap is not an Efron artifact.
2. **The real-data arms were already computed.** `protocol_decomp.py` has run `cox:exact-discrete+kalbfleisch-prentice` on 34 configurations for 10.4 hours; it simply discarded the coefficients. This rerun recovers them without changing a single fitted value.
3. **Exact tie handling is unavailable where ties are heaviest.** It runs to n = 2,000 and no further on SPARCS. The method that best handles ties is the one that cannot be run on the data with the most of them.
4. **The out-of-sample bound is now on the same footing as the claim it tests.** At 20 splits rather than 10, 12 of 12 configurations resolve positive under Breslow and 4 of 12 under KP — against 3 of 12 at 10 splits. Doubling the splits tightens the standard errors and lets one more marginal configuration cross the threshold, so the KP-null is **bounded, not confirmed**. That is the honest reading and it was already the one the manuscript carried; this run puts it on matched footing.

The recommendation that follows is unchanged in direction and firmer in evidence: the paper's result is about the **baseline estimator**, not about tie approximation and not about coefficient estimation. Both remaining candidate explanations have now been measured and neither carries the effect.

---

## Incidental finding: six simulation logs are stale

Run 1a uses seeds identical to `e5()` in `experiments/simulations.py`, so its `D_A` column should reproduce the frozen E5 log exactly. It does not — 0 of 135 cells match.

The cause is not in the new script. `kanrel/stats.py`, which introduced `stable_seed`, is dated 2026-09-05 19:54; every simulation log except E8 is dated 2026-09-04. The seed fix landed in the code and only E8 was regenerated. Re-running one cell through the *unmodified* `simulations.py` today confirms which side is stale:

| source for cell (shape 1.0, 20% censoring, n=500, T=4) | D_T |
|---|---:|
| `simulations.py` re-run with current code | +0.091137 |
| `experiments/exact_arm_sim.py` (run 1a) | +0.091137 |
| frozen `experiments/simulations_e5.txt` | +0.066110 |

Affected logs: `simulations_e5.txt`, `simulations_e6c.txt`, `simulations_e6d.txt`, `simulations_e6d_20reps.txt`, `simulations_e6d_conf120.txt`, `simulations_e6e7e8.txt`, `simulations_e7.txt`. Two manuscript numbers are sourced from the first of these and both move: `+0.173` becomes +0.175, and `35.6` becomes 34.8.

**Why no checker caught it.** `paper/check_numbers.py` verifies the manuscript against the frozen logs. Nothing verifies that the frozen logs are still reproducible from the code that claims to produce them, so the suite passes while both numbers are stale. That gap is worth closing regardless of these four runs.

Run 1a supersedes E5 outright — same design, current seeds, plus the exact arm — so E5 needs no separate regeneration. E6, E6c, E6d and E7 still do.

