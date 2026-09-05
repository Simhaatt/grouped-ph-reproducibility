"""E9: is the coarsening trend driven by baseline DIMENSION or by tie handling?

Section 5.4 names two costs pulling opposite ways as T falls:

  Cox pays a TIE-APPROXIMATION cost that grows as T falls.
  The grouped model pays a NUISANCE-PARAMETER cost -- T free alpha_t estimated by
  MLE where Cox profiles the baseline away -- that grows as T rises.

The section asserts the second is what makes the grouped model lose on fine
grids.  Nothing in the project measured it, because on every run so far the
number of free alpha_t and the number of bins were the same quantity.

This separates them.  The baseline is constrained to alpha = B(t) gamma with B a
cubic B-spline basis of FIXED dimension q over the bin index, so the grouped
model estimates q + p parameters whatever T is.  Sweeping T then moves tie
density without moving nuisance dimension.

  If the coarsening trend largely vanishes at fixed q, the driver is baseline
  dimension and section 5.4's second cost is the mechanism.
  If it persists, the driver is tie handling and the second cost is a side issue.

Both readings are reportable; the point is that the current text asserts the
first without evidence.  q = T recovers the unconstrained model exactly, and is
run as the control so the constraint's own cost is visible.

Run:  python -u experiments/nuisance_dim.py [cohort ...]
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from kanrel import data as D
from kanrel.likelihood import make_targets, nll
from experiments import cox_arms as CA
from experiments.baselines import standardize
from experiments.crossover import COHORTS, coarsen
from experiments.protocol_decomp import nb_se
from experiments.real_data import split

N_SPLITS = 20
Q_GRID = (3, 5, 8)
OUT = Path(__file__).resolve().parent / "nuisance_dim.txt"


def spline_basis(T, q):
    """Cubic B-spline basis of dimension q over bin index 0..T-1, plus intercept.

    sklearn's SplineTransformer is used rather than a hand-rolled basis so this
    file introduces no second spline implementation; experiments/gam_comparator.py
    already depends on it for the covariate side.
    """
    from sklearn.preprocessing import SplineTransformer
    t = np.arange(T, dtype=float).reshape(-1, 1)
    if q >= T:
        return np.eye(T)                       # exactly the unconstrained model
    n_knots = max(2, q - 2)
    B = SplineTransformer(n_knots=n_knots, degree=3,
                          include_bias=True).fit_transform(t)
    return np.asarray(B, float)


def fit_grouped_constrained(X, tidx, event, T, B, link="cloglog"):
    """Joint MLE of (gamma, beta) with alpha = B gamma."""
    Xt = torch.as_tensor(np.asarray(X, float), dtype=torch.float64)
    Bt = torch.as_tensor(np.asarray(B, float), dtype=torch.float64)
    mask, y = make_targets(tidx, event, T)
    mask, y = mask.double(), y.double()
    q, p = Bt.shape[1], Xt.shape[1]
    th = torch.zeros(q + p, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        th[:q] = -2.5 / max(float(Bt.sum(1).mean()), 1e-6)
    opt = torch.optim.LBFGS([th], max_iter=400, tolerance_grad=1e-10,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        alpha = Bt @ th[:q]
        loss = nll(alpha.unsqueeze(0) + (Xt @ th[q:]).unsqueeze(1), mask, y,
                   link, reduction="sum")
        loss.backward()
        return loss

    opt.step(closure)
    out = th.detach().numpy()
    return out[q:], np.asarray(B, float) @ out[:q]


def run(name, base, grid, log):
    log(f"\n{'='*104}")
    log(f"{name}")
    log(f"{'='*104}")
    log(f"  {'T':>4}{'modal':>8}{'free alpha':>12}"
        + "".join(f"{'q=' + str(q):>20}" for q in Q_GRID)
        + f"{'q=T (control)':>20}")
    log(f"  {'':>4}{'':>8}{'':>12}"
        + "".join(f"{'D_T':>11}{'NB SE':>9}" for q in Q_GRID)
        + f"{'D_T':>11}{'NB SE':>9}")
    rows = []
    for T in [t for t in grid if t <= base.n_bins]:
        d = D.onehot_ordinals(coarsen(base, T))
        modal = float(np.bincount(d.bin_idx.astype(int)).max() / d.n)
        per_q = {q: [] for q in list(Q_GRID) + ["T"]}
        t0 = time.time()
        for s in range(N_SPLITS):
            tr, te = split(d, frac=0.3, seed=s)
            tr, te = D.clip_to_train_range(tr, te)
            trs, tes = standardize(tr, te)
            Xtr, Xte = trs.X.astype(float), tes.X.astype(float)
            try:
                b_ef = CA.fit_cox_ties(Xtr, trs.bin_idx, trs.event, "efron")
                h1 = CA.hazards_breslow(CA.risk(Xtr, b_ef), trs.bin_idx, trs.event,
                                        CA.risk(Xte, b_ef), T)
                n1 = CA.nll_from_hazards(h1, tes.bin_idx, tes.event, T)
            except Exception:
                continue
            for q in list(Q_GRID) + ["T"]:
                qq = T if q == "T" else q
                try:
                    B = spline_basis(T, qq)
                    bb, aa = fit_grouped_constrained(Xtr, trs.bin_idx, trs.event,
                                                     T, B)
                    n3 = CA.nll_from_hazards(CA.hazards_from_alpha(Xte, bb, aa),
                                             tes.bin_idx, tes.event, T)
                    per_q[q].append(float(np.mean(n1 - n3)))
                except Exception:
                    pass
        cells = []
        for q in list(Q_GRID) + ["T"]:
            v = np.array(per_q[q], float)
            if v.size < 2:
                cells.append(f"{'--':>11}{'':>9}")
            else:
                cells.append(f"{v.mean():>+11.5f}{nb_se(v):>9.5f}")
        log(f"  {T:>4}{modal:>8.3f}{T:>12}" + "".join(cells)
            + f"   [{time.time()-t0:.0f}s]")
        rows.append((T, modal, {q: np.array(per_q[q], float) for q in per_q}))

    # The reading: how much of the T-trend survives when q is held fixed?
    if len(rows) >= 3:
        Ts = np.array([r[0] for r in rows], float)
        log("")
        log("  TREND IN D_T ACROSS THE GRID SWEEP (Spearman with T; the paper")
        log("  predicts NEGATIVE -- the grouped model loses as T rises)")
        for q in list(Q_GRID) + ["T"]:
            ef = np.array([r[2][q].mean() if r[2][q].size else np.nan
                           for r in rows])
            ok = np.isfinite(ef)
            if ok.sum() < 3:
                continue
            ra = np.argsort(np.argsort(Ts[ok])).astype(float)
            rb = np.argsort(np.argsort(ef[ok])).astype(float)
            ra -= ra.mean(); rb -= rb.mean()
            rho = float((ra * rb).sum() /
                        np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
            span = float(np.nanmax(ef[ok]) - np.nanmin(ef[ok]))
            lbl = "q=T (free)" if q == "T" else f"q={q} (fixed)"
            log(f"    {lbl:<14} rho = {rho:+.3f}   range of D_T = {span:.5f}")
        log("")
        log("  If the fixed-q rows have a much smaller range than q=T, the")
        log("  coarsening trend is a nuisance-DIMENSION effect.  If the ranges")
        log("  match, it is tie handling and section 5.4's second cost is not")
        log("  the mechanism.")


def main():
    want = [a for a in sys.argv[1:] if not a.startswith("-")] or list(COHORTS)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log("=" * 104)
    log("E9  NUISANCE DIMENSION vs TIE HANDLING")
    log("    alpha = B(t) gamma with dim(gamma) = q held FIXED as T varies.")
    log("    D_T = NLL(Cox/Efron + Breslow) - NLL(grouped, constrained baseline).")
    log(f"    {N_SPLITS} splits; NB SE is Nadeau-Bengio corrected.")
    log("=" * 104)
    for name in want:
        if name not in COHORTS:
            log(f"unknown cohort {name!r}")
            continue
        loader, grid, _ = COHORTS[name]
        try:
            base = loader()
        except Exception as e:
            log(f"\n{name}: LOAD FAILED {type(e).__name__}: {str(e)[:70]}")
            continue
        run(name, base, grid, log)
    log("\nwrote " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
