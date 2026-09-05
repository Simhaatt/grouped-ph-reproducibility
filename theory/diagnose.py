"""Diagnostics for the two negative results of 2026-08-21.

--------------------------------------------------------------------------
D1.  Why does discrete/cloglog pick up bias at very coarse grids (3-6 bins)?

Lemma 1 says the grouped PH model is EXACT for any bin width: under PH,
    int_{I_t} lambda(s|x) ds = e^{x'beta} int_{I_t} lambda_0,
so eta*_t = alpha_t + x'beta with the SAME beta, whatever the grid.  So a bias at
3 bins cannot come from the hazard model.

HYPOTHESIS: it comes from coarsening the CENSORING times.  `verify_ties.py`
bins obs = min(t, c, horizon), so a subject censored mid-bin is treated as at
risk for the whole bin.  That breaks the "at-risk status is constant within a
bin" premise -- and it breaks it worse as bins get wider.

TEST: rerun with censoring aligned to bin boundaries (administrative censoring,
which is what SPARCS's "120 +" and DRSA's known b actually are).  If the bias
vanishes at 3 bins, the hypothesis holds and the earlier result is a statement
about censoring coarsening, not about the discrete hazard model.

--------------------------------------------------------------------------
D2.  Why does ||ghat - g*||^2 plateau instead of decaying at n^{-8/9}?

The rate needs BOTH G_n ~ n^{1/(2r+1)} AND g inside a Sobolev-r ball with a
modest norm.  g_true contains sin(pi*x2), which over the covariate range
[-3, 3] is THREE full periods -- a large Sobolev norm, so the approximation
error at G = 5..7 knots is big and dominates.

TEST: measure the approximation floor directly by least-squares projecting
g_true onto the spline basis at each G (no noise, no censoring, huge sample).
If the observed MSE plateau matches that floor, the plateau is approximation
bias -- which CONFIRMS the theorem's knot-growth requirement rather than
contradicting it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kanrel.kan import KANLayer
from theory.verify_ties import BETA, coarsen, fit_cox, fit_discrete, HORIZON, SHAPE


# ============================================================ D1
def simulate_censoring(n, seed, mode, n_bins):
    """mode='random'  : c ~ U(0, 4)          -> censoring times get coarsened
       mode='boundary': c on the bin grid    -> administrative censoring
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    g = X @ BETA
    u = rng.uniform(size=n)
    t = (-np.log(u) / np.exp(g)) ** (1.0 / SHAPE)
    edges = np.linspace(0, HORIZON, n_bins + 1)
    if mode == "random":
        c = rng.uniform(0, 4.0, size=n)
    else:
        c = rng.choice(edges[1:], size=n)          # censoring only at bin edges
    obs = np.minimum(t, np.minimum(c, HORIZON))
    event = ((t <= c) & (t <= HORIZON)).astype(float)
    return X, obs, event


def d1(reps=12, n=4000):
    print("=" * 92)
    print("D1  Is the coarse-grid bias caused by coarsening the CENSORING times?")
    print("=" * 92)
    print(f"  {'bins':>5} | {'random censoring':>32} | {'boundary censoring':>32}")
    print(f"  {'':>5} | {'cloglog':>15}{'Cox/Efron':>17} | {'cloglog':>15}{'Cox/Efron':>17}")
    for T in [50, 20, 10, 6, 4, 3]:
        row = f"  {T:>5} |"
        for mode in ("random", "boundary"):
            cl, ef = [], []
            for s in range(reps):
                X, obs, event = simulate_censoring(n, s, mode, T)
                idx, _ = coarsen(obs, T)
                cl.append(fit_discrete(X, idx, event, T, "cloglog"))
                ef.append(fit_cox(X, idx, event, "efron"))
            b1 = np.nanmean(np.array(cl), 0); b2 = np.nanmean(np.array(ef), 0)
            row += f"{100*np.mean(np.abs(b1/BETA-1)):>14.2f}%"
            row += f"{100*np.mean(np.abs(b2/BETA-1)):>16.2f}%  |"
        print(row)
    print("\n  If cloglog is flat under boundary censoring but not random, the bias is")
    print("  censoring coarsening -- a data-collection fact -- not the hazard model.")


# ============================================================ D2
def g_true(X):
    return 0.8 * (X[:, 0] ** 2 - 1.0) + 1.0 * np.sin(np.pi * X[:, 1]) + 0.6 * X[:, 2]


def approximation_floor(grid_size, n_probe=60000, seed=0):
    """Best MSE an additive cubic-spline KAN layer can achieve on g_true.

    Pure least squares on the spline basis: no noise, no censoring, no
    optimiser.  Whatever this floor is, no amount of data can beat it.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_probe, 3)).astype(np.float32)
    y = g_true(X).astype(np.float64)
    y = y - y.mean()
    layer = KANLayer(3, 1, grid_size=grid_size, spline_order=3, grid_range=(-3.5, 3.5))
    with torch.no_grad():
        B = layer.b_splines(torch.as_tensor(X)).double()      # [n, 3, C]
        feats = torch.cat([B.reshape(n_probe, -1),
                           torch.nn.functional.silu(torch.as_tensor(X)).double()], 1)
        feats = torch.cat([feats, torch.ones(n_probe, 1, dtype=torch.float64)], 1)
    # The spline basis, the SiLU columns and the intercept are near-collinear, so
    # the normal equations are rank-deficient.  torch.linalg.lstsq's default CPU
    # driver returned NON-MONOTONE floors (G=6 worse than G=7), which is
    # impossible for a nested basis -- that was numerical breakdown, not
    # approximation error.  numpy's SVD driver handles the deficiency properly.
    A = feats.numpy()
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    return float(np.mean((pred - y) ** 2))


def d2():
    print()
    print("=" * 92)
    print("D2  Is the MSE plateau an APPROXIMATION floor set by the knot count?")
    print("=" * 92)
    print(f"  {'G (knots)':>10}{'approx floor':>16}   <- unbeatable by any n")
    for G in [5, 6, 7, 8, 12, 16, 24, 32]:
        print(f"  {G:>10}{approximation_floor(G):>16.6f}")
    print()
    print("  Observed in verify_theorem4.py: MSE 0.0488 at n=8000, 0.0492 at n=16000,")
    print("  with G_n = 5*(n/500)^(1/9), i.e. G = 5..7 over that range.")
    print("  If the floor at G=7 is close to 0.049, the plateau IS approximation bias:")
    print("  the theorem's knot-growth requirement is being violated in practice,")
    print("  which CONFIRMS the requirement rather than contradicting the rate.")
    print()
    print("  g_true contains sin(pi*x2): over x in [-3,3] that is 3 full periods, so its")
    print("  Sobolev norm is large and the constant in G_n ~ n^{1/(2r+1)} must be large.")


# ============================================================ D3
def d3():
    """Is the large-n MSE limited by OPTIMISATION or by statistics?

    Theorem 4 Part 2 fits slope -0.659 against a predicted -0.889, and the decay
    visibly slows at large n (consecutive doubling factors 1.70, 2.54, 1.51, 1.17,
    1.21 against a theoretical 2^0.889 = 1.85).

    Two candidate causes, distinguished here:
      (a) OPTIMISATION -- `fit` caps at 600 epochs with patience 60 and holds out
          20% for validation.  If the fits are stopping while still improving,
          MSE is training-budget-limited and the slope is an artefact.
      (b) STATISTICS -- the fits are converged and the slope is real.

    Test: refit at fixed n under increasing budgets.  If MSE keeps falling with
    more epochs, it is (a).  `epochs_run` says whether the cap was hit.
    """
    import numpy as np
    from theory.verify_theorem4 import g_true as gt4, simulate
    from kanrel.fit import fit as kfit
    from kanrel.hazard import DiscreteHazardKAN

    rng = np.random.default_rng(999)
    Xte = rng.normal(size=(8000, 3)).astype(np.float32)     # SHARED test set
    gte = gt4(Xte); gte = gte - gte.mean()

    def one(n, epochs, patience, val_frac, seed):
        d = simulate(n, 20, seed, linear=False)
        G = int(max(5, round(5 * (n / 500) ** (1 / 9))))
        m = DiscreteHazardKAN(3, d.n_bins, hidden=(), mode="baseline",
                              link="cloglog", grid_size=G)
        m, h = kfit(m, d, epochs=epochs, lr=0.03, val_frac=val_frac,
                    patience=patience, grid_update_epochs=(30, 80), seed=seed)
        with torch.no_grad():
            gh = (m(torch.as_tensor(Xte))[:, 0] - m.alpha[0]).numpy()
        gh = gh - gh.mean()
        return float(np.mean((gh - gte) ** 2)), h["epochs_run"]

    print()
    print("=" * 92)
    print("D3  Is large-n MSE optimisation-limited or statistics-limited?")
    print("=" * 92)
    print(f"  {'n':>7}{'budget':>26}{'MSE':>12}{'se':>10}{'epochs run':>12}")
    for n in (4000, 16000):
        for label, ep, pat, vf in [("600 ep, pat 60, val .2", 600, 60, 0.2),
                                   ("2500 ep, pat 300, val .2", 2500, 300, 0.2),
                                   ("2500 ep, no early stop", 2500, 10**9, 0.0)]:
            vals = [one(n, ep, pat, vf, s) for s in range(4)]
            mse = np.mean([v[0] for v in vals])
            se = np.std([v[0] for v in vals], ddof=1) / 2
            eps = np.mean([v[1] for v in vals])
            print(f"  {n:>7}{label:>26}{mse:>12.6f}{se:>10.6f}{eps:>12.0f}")
    print()
    print("  If MSE falls materially with a bigger budget, Part 2's slope is")
    print("  training-budget-limited, not the statistical rate.")


if __name__ == "__main__":
    d2()      # cheapest, run first
    d3()
    d1()
