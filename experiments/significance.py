"""Are the KAN-vs-linear differences statistically real, or noise?

The large-cohort wins are small in absolute terms -- test NLL +0.0022 on SPARCS
and +0.0086 on KKBox.  Reporting those as "the KAN wins" without a confidence
interval would be indefensible: on a 60,000-row test set a difference of 0.009
may be overwhelming or may be nothing, and the point estimate alone cannot say
which.

Method: both models score the SAME test rows, so the comparison is PAIRED.  We
take the per-unit NLL difference d_i = nll_linear_i - nll_kan_i and bootstrap the
mean of d over units.  Pairing removes the between-unit variance, which is far
larger than the effect and would otherwise swamp it.

Positive d means the KAN assigns higher likelihood, i.e. the KAN is better.

Run:  python experiments/significance.py [dataset ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kanrel import data as D
from kanrel.fit import fit, to_tensors
from kanrel.hazard import DiscreteHazardKAN
from kanrel.likelihood import nll as nll_fn
from kanrel.metrics import antolini_c, paired_c_bootstrap, paired_c_influence
from experiments.real_data import continuous_columns, split

# l1 chosen on VALIDATION in experiments/regularization.py; hard-coded here so
# this script never re-selects, and so the test set is used exactly once.
# Small cohorts (n<2000) now select by 5-FOLD CV, not a single holdout -- pbc's
# holdout was 59 rows and valung's 19, and both picked a worse l1 than the grid
# contained.  CV consistently chooses HEAVIER regularisation, which is the
# expected direction at small n, and materially changed the fits: valung's test
# NLL fell 2.8771 -> 2.5942, flipping its NLL difference from -0.064 to +0.219.
# Those are different models, so they must be re-tested rather than assumed.
SELECTED_L1 = {
    "pbc": 0.1, "gbsg": 0.1, "valung": 0.1, "prostate": 0.3,      # 5-fold CV
    "support2/slos": 0.01, "sparcs/drg302": 0.0, "drsa/music": 0.0,  # holdout
}

# Spline extrapolation.  True (the kan.py default) extends every edge function
# as a constant outside the training range; False reproduces the old unbounded
# behaviour.  Exposed only so extrapolation_ab.py can flip it -- do not change
# it to produce headline numbers.
CLAMP_INPUTS = True

# Winsorising: clip TEST covariates to the training range, for EVERY model.
# Without it the linear baseline is asked to extrapolate to 25 sigma on flchain
# and the KAN is credited for boundedness any sane baseline also has -- the
# section 4.7 error on a different axis.  See kanrel.data.clip_to_train_range.
WINSORIZE = True

# 0.0 clips TEST to the exact training range (train untouched).  A positive
# value additionally winsorises the training tails at that two-sided quantile,
# which is what actually bounds a covariate whose extreme value is IN the
# training set -- flchain's worst row only fell 55.3 -> 25.3 under range-clipping
# because the 25-sigma observation is a training point.
WINSOR_Q = 0.0


def per_unit_nll(m, d):
    X, mask, y = to_tensors(d)
    with torch.no_grad():
        return nll_fn(m(X), mask, y, m.link, reduction=None).numpy(), m.survival(X)


def paired_bootstrap(diff, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def run(name, loader, seed=0):
    d = loader()
    tr, te = split(d, seed=seed)
    if WINSORIZE:
        tr, te = D.clip_to_train_range(tr, te, quantile=WINSOR_Q)
    l1 = SELECTED_L1.get(name, 0.0)

    lin = DiscreteHazardKAN(tr.X.shape[1], tr.n_bins, mode="linear", link="cloglog")
    lin, _ = fit(lin, tr, epochs=500, lr=0.03, val_frac=0.2, patience=60, seed=seed)

    kan = DiscreteHazardKAN(tr.X.shape[1], tr.n_bins, mode="baseline", link="cloglog",
                            hidden=(), grid_size=8, cont_idx=continuous_columns(tr.X),
                            clamp_inputs=CLAMP_INPUTS)
    kan, _ = fit(kan, tr, epochs=400, lr=0.03, val_frac=0.2, patience=60,
                 grid_update_epochs=(30, 100), l1=l1, entropy=1.0, smooth=1e-3, seed=seed)

    nl, Sl = per_unit_nll(lin, te)
    nk, Sk = per_unit_nll(kan, te)
    diff = nl - nk                       # >0 means KAN better
    mean, lo, hi = paired_bootstrap(diff)
    cl = antolini_c(Sl, te.bin_idx, te.event)
    ck = antolini_c(Sk, te.bin_idx, te.event)
    sig = "SIGNIFICANT" if lo > 0 else ("worse" if hi < 0 else "not significant")

    # The C-index is the headline -- mean NLL is not robust to a single blown-up
    # row -- so it needs its own paired interval.  Section 4.5 attacks the
    # literature for reporting exactly this quantity without one.
    cb = paired_c_bootstrap(Sl, Sk, te.bin_idx, te.event, seed=seed)
    csig = ("SIGNIFICANT" if cb["delta_lo"] > 0 else
            "worse" if cb["delta_hi"] < 0 else "not significant")
    # Above ~6000 units the bootstrap SUBSAMPLES (the pair matrices are
    # O(n_events * n)), which is why KKBox's C interval used 6k of 60k rows.
    # The influence-function SE is O(n_events * n) once, so it runs at full
    # resolution; validated against the bootstrap to within 2% on the SE.
    ci_full = None
    if te.n > 6000:
        ci_full = paired_c_influence(Sl, Sk, te.bin_idx, te.event)

    print(f"\n  {name:<16} n_test={te.n:>6}  l1={l1:g}  clamp={CLAMP_INPUTS}  wins={WINSORIZE}/q={WINSOR_Q:g}")
    print(f"    NLL  linear {nl.mean():.4f}   KAN {nk.mean():.4f}")
    print(f"    diff (linear - KAN) = {mean:+.5f}   95% CI [{lo:+.5f}, {hi:+.5f}]"
          f"   -> {sig}")
    print(f"    C-index  linear {cl:.4f}   KAN {ck:.4f}   delta {ck-cl:+.4f}")
    print(f"    C delta  95% CI [{cb['delta_lo']:+.4f}, {cb['delta_hi']:+.4f}]"
          f"   -> {csig}   (bootstrap, {cb['n_units']} of {te.n} units)")
    if ci_full is not None:
        fs = ("SIGNIFICANT" if ci_full["delta_lo"] > 0 else
              "worse" if ci_full["delta_hi"] < 0 else "not significant")
        print(f"    C delta  95% CI [{ci_full['delta_lo']:+.4f}, "
              f"{ci_full['delta_hi']:+.4f}]   -> {fs}"
              f"   (influence, ALL {ci_full['n_units']} units, "
              f"SE {ci_full['se']:.5f})")
        cb, csig = ci_full, fs        # full-resolution interval wins
    return dict(name=name, n_test=te.n, mean=mean, lo=lo, hi=hi, sig=sig,
                c_lin=cl, c_kan=ck, c_lo=cb["delta_lo"], c_hi=cb["delta_hi"],
                c_sig=csig, n_events=cb["n_events"])


def main():
    names = sys.argv[1:] or ["drsa/music", "sparcs/drg302", "support2/slos",
                             "gbsg", "pbc", "valung"]
    avail = dict(D.LOADERS)
    avail["sparcs/drg302"] = lambda: D.load_sparcs("302")
    avail["drsa/music"] = lambda: D.load_drsa("MUSIC", max_rows=200000)

    print("=" * 88)
    print("PAIRED BOOTSTRAP: is the KAN-vs-linear difference real?")
    print("  positive difference = KAN assigns higher likelihood = KAN better")
    print("=" * 88)
    out = []
    for nm in names:
        if nm in avail:
            try:
                out.append(run(nm, avail[nm]))
            except Exception as e:
                print(f"  {nm:<16} FAILED {type(e).__name__}: {str(e)[:60]}")

    print()
    print("=" * 88)
    print(f"  {'dataset':<16}{'n_test':>8}{'NLL diff':>12}{'95% CI':>26}{'verdict':>18}")
    for r in sorted(out, key=lambda r: -r["n_test"]):
        print(f"  {r['name']:<16}{r['n_test']:>8}{r['mean']:>+12.5f}"
              f"   [{r['lo']:+.5f}, {r['hi']:+.5f}]{r['sig']:>18}")


if __name__ == "__main__":
    main()
