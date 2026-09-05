"""A/B: does ``clamp_inputs=True`` cause the Theorem 4 Part-2 slope regression?

status.md 8b records that after ``clamp_inputs`` became the DEFAULT in
kanrel/kan.py, the smooth-target log-log slope moved from -1.005 to -0.873,
crossing the minimax bound of -0.889.  The suspected mechanism: clamping
extends every edge function as a CONSTANT outside the knot range, which biases
the tails of ghat.  That bias does not shrink with n at the same rate as the
variance, so it flattens the fitted slope -- exactly the observed direction.

This script is the controlled test the status file says is "being confirmed":
identical seeds, identical ladder, identical epochs, ONE bit changed.

Why it matters for the paper.  Theorem 4 Part 2 is a statement about the SIEVE
MLE -- the unconstrained maximiser over the spline sieve.  Clamping is an extra
constraint on the function class, so a clamped fit is a DIFFERENT estimator and
is not the object the theorem describes.  This is methodological lesson 4
("deep-learning defaults change the estimand") in a new costume.  If the A/B
confirms the mechanism, the fix is not to revert the clamp -- the clamp is right
for the empirical pipeline, where unbounded off-support predictions are
indefensible (4.3f) -- but to pin the THEORY gate to clamp_inputs=False and say
so in one line.

Run:  python -u theory/theorem4_clamp_ab.py [--quick]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kanrel.fit import fit
from kanrel.hazard import DiscreteHazardKAN
from theory.verify_theorem4 import (MINIMAX_BOUND, _TEST_SETS, rate_holds,
                                    simulate)


def g_mse_clamp(n, n_bins, seed, *, clamp, hidden=(), grid_scale=True,
                epochs=3000, target="smooth"):
    """verify_theorem4.g_mse with clamp_inputs threaded through to the KAN.

    Everything else -- seed, knot scaling G_n ~ n^{1/9}, epoch budget, absence of
    early stopping and of a validation split -- is copied verbatim so the only
    difference between the two arms is the clamp.
    """
    d = simulate(n, n_bins, seed, linear=False, target=target)
    grid_size = int(max(5, round(5 * (n / 500) ** (1 / 9)))) if grid_scale else 8
    m = DiscreteHazardKAN(d.X.shape[1], d.n_bins, hidden=hidden, mode="baseline",
                          link="cloglog", grid_size=grid_size,
                          clamp_inputs=clamp)
    m, _ = fit(m, d, epochs=epochs, lr=0.03, val_frac=0.0, patience=10**9,
               grid_update_epochs=(30, 80), seed=seed)
    Xte, gt = _TEST_SETS[target]
    with torch.no_grad():
        eta = m(torch.as_tensor(Xte))
        gh = (eta[:, 0] - m.alpha[0]).numpy()
    gh = gh - gh.mean()
    return float(np.mean((gh - gt) ** 2))


# The verdict predicate is IMPORTED from verify_theorem4 (rate_holds), never
# restated here.  Restating it is exactly how this script's first version
# inverted the comparison and called a passing run a failure; see
# verify_theorem4.rate_holds for the direction and the history.
holds = rate_holds


def ladder(ns, reps, clamp, target, n_boot=2000, seed=0):
    raw = {}
    for n in ns:
        t0 = time.time()
        raw[n] = np.array([g_mse_clamp(n, 20, s, clamp=clamp, target=target)
                           for s in range(reps)])
        print(f"    n={n:>6}  MSE={raw[n].mean():.6f}"
              f"  se={raw[n].std(ddof=1) / np.sqrt(reps):.6f}"
              f"  [{time.time() - t0:.0f}s]", flush=True)
    logn = np.log(np.array(ns, dtype=float))
    slope = float(np.polyfit(logn, np.log([raw[n].mean() for n in ns]), 1)[0])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        means = [raw[n][rng.integers(0, len(raw[n]), len(raw[n]))].mean() for n in ns]
        if min(means) <= 0:
            continue
        boots.append(np.polyfit(logn, np.log(means), 1)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return raw, slope, (float(lo), float(hi))


def main():
    quick = "--quick" in sys.argv
    ns = [500, 1000, 2000] if quick else [500, 1000, 2000, 4000, 8000, 16000]
    reps = 3 if quick else 12
    # smooth is the arm that regressed (8b); kinked stayed inside its
    # bound at -0.938 vs -0.667, so it carries no diagnostic information
    # about the clamp.  --both runs it anyway.
    targets = ("smooth", "kinked") if "--both" in sys.argv else ("smooth",)

    print("=" * 82)
    print("THEOREM 4 PART 2 -- clamp_inputs A/B")
    print("=" * 82)
    print(f"  ns={ns}  reps={reps}  epochs=3000  grid_scale=True")
    print("  Only difference between arms: KANLayer(clamp_inputs=...).")
    print()

    res = {}
    for target in targets:
        bound = MINIMAX_BOUND[target]
        for clamp in (False, True):
            print(f"  target={target}  clamp_inputs={clamp}   (bound {bound:+.3f})",
                  flush=True)
            raw, slope, ci = ladder(ns, reps, clamp, target)
            res[(target, clamp)] = (raw, slope, ci)
            print(f"    slope = {slope:+.3f}   95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]"
                  f"   -> {'OK' if holds(ci, bound) else 'VIOLATES BOUND'}")
            print()

    print("=" * 82)
    print("SUMMARY")
    print("=" * 82)
    print(f"  {'target':<10}{'clamp':<8}{'slope':>9}{'95% CI':>22}{'bound':>9}  verdict")
    ok = True
    for target in targets:
        bound = MINIMAX_BOUND[target]
        for clamp in (False, True):
            _, slope, ci = res[(target, clamp)]
            good = holds(ci, bound)
            print(f"  {target:<10}{str(clamp):<8}{slope:>+9.3f}"
                  f"   [{ci[0]:+.3f}, {ci[1]:+.3f}]{bound:>9.3f}"
                  f"  {'OK' if good else 'VIOLATED'}")
            if clamp is False and not good:
                ok = False
    print()
    for target in targets:
        _, s_off, _ = res[(target, False)]
        _, s_on, _ = res[(target, True)]
        print(f"  {target}: clamp moves the slope by {s_on - s_off:+.3f}"
              f"  ({s_off:+.3f} -> {s_on:+.3f})")
    print()
    print("  Small-n MSE is the diagnostic for the mechanism: constant")
    print("  extrapolation helps most where the sample is thinnest, so if the")
    print("  clamp is the cause, MSE(clamp=True) < MSE(clamp=False) at n=500")
    print("  and the two converge as n grows.")
    for target in targets:
        raw_off = res[(target, False)][0]
        raw_on = res[(target, True)][0]
        print(f"    {target:<8}" + "  ".join(
            f"n={n}:{raw_on[n].mean() / raw_off[n].mean():.3f}" for n in ns))
    print()
    print(f"RESULT: theory gate at clamp_inputs=False "
          f"{'RESPECTS' if ok else 'VIOLATES'} the minimax bound")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
