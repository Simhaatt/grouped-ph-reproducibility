"""Does the KAN recover a nonlinearity a linear model cannot represent?

This is the gap that most needs closing.  Interpretability is the entire selling
point of KANs -- it is why CoxKAN and SurvKAN exist -- and so far this project
has no positive interpretability result.  The age-40 recovery attempt failed, and
correctly so: SUPPORT is seriously-ill hospitalised adults, not the elective
cataract cohort Nawata et al. studied.

Better target: **physiological variables with U-shaped risk**.  Mean arterial
pressure, temperature and heart rate are all clinically non-monotone -- both
low and high are both bad -- so a linear term in the log-hazard is structurally unable
to represent them, while a spline can.  If the KAN is worth anything
interpretably, this is where it shows.

Design, so that a curve is not just asserted to be real:
  1. fit the additive KAN, extract the partial effect of each continuous covariate
  2. measure NONLINEARITY as the residual after removing the best linear fit,
     as a fraction of total variation -- a scalar per covariate, rankable
  3. confirm it is not noise by refitting on bootstrap resamples and reporting
     a pointwise band
  4. confirm it MATTERS by comparing NLL of the full model against one in which
     that covariate is forced linear

Step 4 is what separates a real finding from a pretty picture: a curve that bends
but does not improve the likelihood is decoration.

Run:  python experiments/interpretability.py [dataset]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from kanrel import data as D
from kanrel.fit import fit, to_tensors
from kanrel.hazard import DiscreteHazardKAN
from kanrel.likelihood import nll as nll_fn
from experiments.real_data import continuous_columns, split


def partial_curve(model, X, j, grid):
    with torch.no_grad():
        return model.partial_effect(torch.as_tensor(X), j, grid).numpy()


def nonlinearity(grid, eff):
    """Fraction of the curve's variation NOT explained by a straight line."""
    A = np.vstack([grid, np.ones_like(grid)]).T
    coef, *_ = np.linalg.lstsq(A, eff, rcond=None)
    resid = eff - A @ coef
    tot = np.var(eff)
    return float(np.var(resid) / tot) if tot > 1e-12 else 0.0


def fit_model(tr, l1, seed, force_linear_col=None):
    """force_linear_col: route that ONE column through the linear term instead of
    a spline, holding everything else fixed.  The NLL difference is then
    attributable to that covariate's nonlinearity alone."""
    ci = continuous_columns(tr.X)
    if force_linear_col is not None:
        ci = [c for c in ci if c != force_linear_col]
    m = DiscreteHazardKAN(tr.X.shape[1], tr.n_bins, mode="baseline", link="cloglog",
                          hidden=(), grid_size=8, cont_idx=ci)
    m, _ = fit(m, tr, epochs=400, lr=0.03, val_frac=0.2, patience=60,
               grid_update_epochs=(30, 100), l1=l1, smooth=1e-3, seed=seed)
    return m


def test_nll(m, te):
    X, mask, y = to_tensors(te)
    with torch.no_grad():
        return float(nll_fn(m(X), mask, y, m.link))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "support2/slos"
    loader = dict(D.LOADERS)[name] if name in D.LOADERS else lambda: D.load_sparcs("302")
    d = loader()
    tr, te = split(d, seed=0)
    l1 = 0.01
    cont = continuous_columns(tr.X)

    print("=" * 92)
    print(f"INTERPRETABILITY -- {d.name}   n={d.n}, {len(cont)} continuous covariates")
    print("=" * 92)

    m = fit_model(tr, l1, seed=0)
    base_nll = test_nll(m, te)
    print(f"  full additive KAN, test NLL {base_nll:.4f}\n")

    rows = []
    for j in cont:
        lo, hi = np.percentile(d.X[:, j], [5, 95])
        if hi - lo < 1e-8:
            continue
        grid = torch.linspace(float(lo), float(hi), 60)
        eff = partial_curve(m, d.X, j, grid)
        rows.append((d.feature_names[j], j, nonlinearity(grid.numpy(), eff),
                     float(eff.max() - eff.min()), grid.numpy(), eff))

    rows.sort(key=lambda r: -r[2] * r[3])          # nonlinearity x amplitude
    print(f"  {'covariate':<12}{'nonlinearity':>14}{'amplitude':>12}"
          f"{'NLL if forced linear':>24}{'delta':>9}")
    for nm, j, nl, amp, grid, eff in rows[:6]:
        m2 = fit_model(tr, l1, seed=0, force_linear_col=j)
        n2 = test_nll(m2, te)
        rows_delta = n2 - base_nll
        print(f"  {nm:<12}{nl:>14.3f}{amp:>12.4f}{n2:>24.4f}{rows_delta:>+9.4f}")

    print()
    best = rows[0]
    nm, j, nl, amp, grid, eff = best
    print(f"  MOST NONLINEAR: {nm}   (nonlinearity {nl:.3f}, amplitude {amp:.4f})")
    print(f"  {'value':>9}{'effect':>10}")
    for a, e in zip(grid[::4], eff[::4]):
        bar = "#" * max(0, int(round(20 + 34 * e)))
        print(f"  {a:>9.2f}{e:>10.4f}  {bar}")

    # bootstrap band -- is the shape stable, or is it noise?
    print(f"\n  bootstrap stability of {nm} (20 resamples):")
    curves = []
    for b in range(20):
        rng = np.random.default_rng(100 + b)
        ix = rng.integers(0, tr.n, tr.n)
        from kanrel.data import SurvData
        sub = SurvData(tr.X[ix], tr.bin_idx[ix], tr.event[ix], tr.n_bins,
                       tr.feature_names, name=tr.name, intrinsically_discrete=True)
        mb = fit_model(sub, l1, seed=b)
        curves.append(partial_curve(mb, d.X, j, grid))
    C = np.array(curves)
    lo_b, hi_b = np.percentile(C, [10, 90], axis=0)
    frac_excl = float(np.mean((lo_b > 0) | (hi_b < 0)))
    print(f"    80% band excludes zero over {100*frac_excl:.0f}% of the range")
    print(f"    mean band width {float(np.mean(hi_b - lo_b)):.4f} vs amplitude {amp:.4f}")
    print(f"    -> {'SHAPE IS STABLE' if np.mean(hi_b-lo_b) < amp else 'shape is within noise'}")


if __name__ == "__main__":
    main()
