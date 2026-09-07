# Changelog

## The exact-discrete fourth arm, and a freshness check - 2026-09-07

Four new experiment runs answering a review request: add Cox's exact discrete conditional partial
likelihood as a fourth prediction arm, so the Efron tie approximation is separated from the
baseline representation rather than confounded with it.

- `experiments/exact_arm_sim.py` -- arms A/B/C/D on the 135-cell grouped-Weibull design, 20 reps,
  seeds identical to E5. Arm C computable in 102 cells. Attenuation 0.880 (Efron), 1.236 (exact),
  0.983 (grouped); exact lies further from the truth than Efron in 97 of 102.
- `experiments/exact_arm_real.py` -- the same 20 splits as `protocol_decomp.py`, rerun only to
  retain the fitted coefficients, which the original discarded before writing its JSON. Exact is
  closer to the grouped coefficient in 0 of 7 paired comparisons.
- `experiments/exact_arm_ladder.py` -- nested SPARCS subsamples. Exact runs to n=2,000 and no
  further; cost confirmed O(n^2) to within 2% at near-constant tie severity.
- `experiments/kp_out_of_sample_v2.py` -- the out-of-sample test at 20 splits rather than 10, and
  modal EVENT mass rather than modal all-exit mass. 12/12 resolve positive under Breslow, 4/12
  under KP, against 3/12 at 10 splits.

Conclusion: the Cox-grouped coefficient difference is not the tie approximation. The exact estimator
targets a log odds ratio rather than a log hazard ratio, so it overshoots where Efron attenuates.
Scored in its own link family it is indistinguishable from the joint MLE (median absolute difference
2.7e-5), which also checks the implementation.

Added `scripts/check_freshness.py`. `check_numbers.py` validates the manuscript against the frozen
logs; nothing validated the frozen logs against the code. Six simulation logs predated the
`stable_seed` change and no longer reproduced, and two manuscript numbers were wrong throughout
while all checks passed. The new script separates SUSPECT (older than code it imports, a heuristic)
from STALE (a recompute probe disagreed, a verdict), using each script's real import closure.

Result values changed: the section 3.3 R1 simulation numbers now come from `exact_arm_sim.txt`
rather than the superseded `simulations_e5.txt` (+0.173 becomes +0.175, 35.6% becomes 34.8%, and
the grouped-arm bias bound moves from 0.5% to 1.3%), and the out-of-sample counts move from 3/12 to
4/12 at matched splits. E6, E6c, E6d and E7 regenerated under current seeding.

## 1.0.0 - 2026-09-05

Prepared from existing frozen outputs and supplied manuscript. Added result parsers, provenance checks, figures, tables, dataset documentation and release instructions. No experiments rerun. Original logs preserved privately; only local path strings redacted in release copies. No scientific values changed.

Owner selected CC BY 4.0, confirmed author details and waived editable supplement source. Supplied PDF retained; numerical outputs unchanged.

## Post-release citation update on main

Added the verified v1.0.0 Zenodo DOI, actual publication date, BibTeX citation and manuscript availability links. Verified archive checksum and all 93 frozen result hashes. No result values, release tag or archived files changed.
