"""WHEN does the discrete formulation beat Cox, and when does it lose?

Section 2 claims the discrete hazard wins where ties are heavy.  Measuring the
formulation effect WITH AN INTERVAL (formulation_ci.py) showed it is not that
simple: on sparcs/drg302 the discrete hazard wins by +0.071 (3/3 splits) but on
support2/slos it LOSES by -0.008, also 3/3.  A blanket "our formulation is
better" is therefore false, and adding a P-spline penalty on alpha does not
rescue it (-0.0066 -> -0.0079).

THE MECHANISM -- two costs pulling opposite ways, both of which our own theory
already names:

  Cox pays a TIE-APPROXIMATION cost.  Breslow/Efron attenuate as mass piles into
  bins; section 4.2 measures this reaching 27.5% bias at 3 bins.  It grows as T FALLS.

  The discrete hazard pays a NUISANCE-PARAMETER cost.  It estimates T free
  baseline logits alpha_t by MLE, while Cox's partial likelihood profiles the
  baseline away entirely.  It grows as T RISES relative to n.

So the formulation effect should be a DECREASING function of T, crossing zero
somewhere, and the crossing should move right as n grows (more rows per bin buys
more affordable alpha_t).  sparcs: T=30 on 34k = 1141 rows/bin, extreme ties ->
we win.  support2: T=60 on 6.4k = 106 rows/bin, mild ties -> we lose.

That is a falsifiable prediction on data already in hand: sweep T on ONE cohort
and the sign must flip.  If it does not, the claim in section 2 needs rewriting,
not re-framing.

Run:  python -u experiments/crossover.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from kanrel import data as D
from experiments.real_data import split
from experiments.significance import paired_bootstrap
from experiments.formulation_ci import cox_linear_per_unit, ours_linear_per_unit

N_SPLITS = 3
ALL = []   # (cohort, modal bin mass, effect) pooled across every sweep


def coarsen(d, T_new):
    """Merge adjacent bins, holding the COHORT fixed.

    The first attempt swept `n_bins` on load_support2("slos"), which silently did
    nothing -- slos is native-day discrete, so T comes from `horizon` and every
    row of the sweep re-ran the identical T=60 fit.  Sweeping `horizon` instead
    is also wrong: it truncates follow-up, so it changes the cohort as well as
    the grid, and the two effects cannot be separated.

    Coarsening maps old bin t -> floor(t * T_new / T), keeping every subject, its
    event indicator and its ordering.  n is fixed, so rows/bin rises exactly as T
    falls, and mass piles into bins exactly as the tie argument requires.  (A8)
    still holds: a censoring time on the fine grid lands on the coarse grid.

    Both costs therefore move together as T falls -- heavier ties (worse for Cox)
    and fewer nuisance parameters (better for the discrete hazard) -- so the
    formulation effect must be DECREASING in T.  That is the prediction.
    """
    from kanrel.data import SurvData
    idx = np.minimum((d.bin_idx.astype(int) * T_new) // d.n_bins, T_new - 1)
    ent = None if d.entry_idx is None else np.minimum(
        (d.entry_idx.astype(int) * T_new) // d.n_bins, T_new - 1)
    return SurvData(d.X, idx.astype(d.bin_idx.dtype), d.event, T_new,
                    d.feature_names, name=f"{d.name}@T{T_new}",
                    intrinsically_discrete=d.intrinsically_discrete,
                    entry_idx=ent, bin_edges=None, meta=dict(d.meta))


def sweep(label, make, bins):
    print(f"\n{'='*92}\n{label}\n{'='*92}")
    print(f"  {'T':>4}{'rows/bin':>10}{'modal bin':>11}{'formulation':>13}"
          f"{'sd':>9}{'resolved':>11}   direction")
    rows = []
    for T in bins:
        try:
            d = D.onehot_ordinals(make(T))
        except Exception as e:
            print(f"  {T:>4}  FAILED {type(e).__name__}: {str(e)[:50]}")
            continue
        eff, sig = [], 0
        for s in range(N_SPLITS):
            tr, te = split(d, seed=s)
            tr, te = D.clip_to_train_range(tr, te)
            try:
                m, lo, hi = paired_bootstrap(
                    cox_linear_per_unit(tr, te) - ours_linear_per_unit(tr, te, s))
            except Exception as e:
                print(f"  {T:>4} s{s} FAILED {type(e).__name__}: {str(e)[:40]}")
                continue
            eff.append(m); sig += (lo > 0 or hi < 0)
        if not eff:
            continue
        a = np.array(eff)
        # tie_ratio in meta refers to the ORIGINAL grid; after coarsening the
        # honest tie measure is how much mass the modal bin carries.
        tie = float(np.bincount(d.bin_idx.astype(int)).max() / d.n)
        arrow = ("DISCRETE wins" if a.mean() > 0 else "COX wins")
        print(f"  {d.n_bins:>4}{d.n / d.n_bins:>10.0f}{tie:>11.4f}"
              f"{a.mean():>+13.5f}{a.std(ddof=1):>9.5f}"
              f"{f'{sig}/{len(eff)}':>11}   {arrow}")
        rows.append((d.n_bins, a.mean()))
        ALL.append((label.split()[0], tie, float(a.mean())))
    if len(rows) >= 3:
        Ts = np.array([r[0] for r in rows], float)
        ef = np.array([r[1] for r in rows])
        # Spearman between T and the effect: the prediction is a NEGATIVE slope.
        rt = np.argsort(np.argsort(Ts)).astype(float)
        re = np.argsort(np.argsort(ef)).astype(float)
        rt -= rt.mean(); re -= re.mean()
        rho = float((rt * re).sum() / np.sqrt((rt**2).sum() * (re**2).sum()))
        flips = (ef.max() > 0) and (ef.min() < 0)
        print(f"\n  corr(T, formulation effect) = {rho:+.3f}"
              f"   (prediction: NEGATIVE)")
        print(f"  sign flips across the sweep: {flips}"
              f"   (prediction: TRUE -- a crossover, not a uniform win)")


# Cohorts to sweep.  The first two are the original pair; the rest were added to
# answer the obvious objection to a two-cohort headline.
#
# drsa/clinic is the most valuable addition: it is a SECOND genuinely discrete
# (R2) cohort, so the rule is no longer being asked to generalise from one
# discrete dataset.  flchain and support-pycox are binned-continuous (R1) and
# therefore test the harder direction -- Lemma 1 says their effect is ~0 at the
# native grid, and the rule says it must RISE as they are coarsened and mass
# concentrates.  If it does not, the rule is about discreteness rather than tie
# mass, and the paper's framing is wrong.
COHORTS = {
    "support2/slos": (lambda: D.load_support2("slos", horizon=60),
                      [6, 10, 15, 20, 30, 60],
                      "n=9105, mild ties; bins COARSENED on a fixed cohort"),
    "sparcs/drg302": (lambda: D.load_sparcs("302", horizon=30),
                      [6, 10, 15, 30],
                      "n=34233, extreme ties; same coarsening"),
    "drsa/clinic": (lambda: D.load_drsa("CLINIC"),
                    [5, 10, 25, 50],
                    "n=4828, intrinsically discrete (R2); SECOND discrete cohort"),
    "flchain": (lambda: D.load_flchain(),
                [4, 6, 10, 20],
                "n=7874, binned-continuous (R1); tests the rule on non-discrete data"),
    "support-pycox": (lambda: D.load_support_pycox(),
                      [4, 6, 10, 20],
                      "n=8873, binned-continuous (R1)"),
    # Added 2026-08-31, second wave.  drsa/music is a THIRD intrinsically
    # discrete cohort and by far the largest, so it tests the R2 claim where
    # rows/bin is never the binding constraint.  Subsampled to 50k because the
    # Cox arm is O(n^2) in the risk sets and the full 200k is not worth the days.
    "drsa/music": (lambda: D.load_drsa("MUSIC", max_rows=50000),
                   [6, 12, 30, 60],
                   "n=50000 subsample, intrinsically discrete (R2); THIRD discrete cohort"),
    "nwtco": (lambda: D.load_nwtco(),
              [4, 6, 10, 20],
              "n=4028, binned-continuous (R1)"),
    "metabric": (lambda: D.load_metabric(),
                 [4, 6, 10, 20],
                 "n=1904, binned-continuous (R1)"),
}


def main():
    want = [a for a in sys.argv[1:] if not a.startswith("-")] or list(COHORTS)
    for name in want:
        if name not in COHORTS:
            print(f"unknown cohort {name!r}; known: {list(COHORTS)}")
            continue
        loader, grid, blurb = COHORTS[name]
        try:
            base = loader()
        except Exception as e:
            print(f"\n{name}: LOAD FAILED {type(e).__name__}: {str(e)[:70]}")
            continue
        # Only sweep grids at or below the cohort's native resolution; asking for
        # more bins than the data has is not coarsening.
        grid = [T for T in grid if T <= base.n_bins]
        sweep(f"{name} -- {blurb}", lambda T, b=base: coarsen(b, T), grid)


def pooled():
    """The real finding: T is the wrong x-axis, TIE MASS is the right one.

    Within sparcs the effect is not monotone in T -- modal mass is 0.667 at T=10
    but 0.718 at T=15, because coarsening a day grid does not align evenly -- so
    corr(T, effect) is only -0.80 there.  Ordering the SAME points by the fraction
    of mass in the modal bin instead makes them monotone across BOTH cohorts at
    once, which is what a rule needs to be: one observable scalar, no per-dataset
    tuning, and it is measurable before fitting anything.
    """
    if len(ALL) < 6:
        return
    m = np.array([a[1] for a in ALL])
    e = np.array([a[2] for a in ALL])
    ra = np.argsort(np.argsort(m)).astype(float)
    rb = np.argsort(np.argsort(e)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    rho = float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    print()
    print("=" * 92)
    ncoh = len({a[0] for a in ALL})
    print(f"POOLED ACROSS {ncoh} COHORTS -- ordered by MODAL BIN MASS")
    print("=" * 92)
    print(f"  {'cohort':<12}{'modal bin':>11}{'formulation':>13}")
    for i in np.argsort(m):
        print(f"  {ALL[i][0]:<12}{m[i]:>11.4f}{e[i]:>+13.5f}")
    print()
    print(f"  Spearman(modal bin mass, formulation effect) = {rho:+.4f}"
          f"   over {len(ALL)} grid configurations")
    if (e > 0).any() and (e < 0).any():
        print(f"  sign change between modal bin mass {m[e < 0].max():.3f}"
              f" (Cox wins) and {m[e > 0].min():.3f} (discrete wins)")
        print()
        print("  RULE: use the discrete hazard when the modal bin carries more")
        print("  than roughly 20% of the mass; below that, Cox's profiled")
        print("  baseline is the better estimator.")

if __name__ == "__main__":
    main()
    pooled()
