"""WHY the KAN gains where it gains -- a quantitative complementarity rule.

Section 2 says flexibility pays "where covariates are genuinely continuous".  That is
qualitative, and it does not survive contact with the data: flchain HAS continuous
covariates (free light chains, creatinine, age) and shows dC = +0.0022.  Meanwhile
nwtco shows dC = +0.0001 -- ranking IDENTICAL to linear -- yet a resolved NLL gain
of +0.0085 on 3/3 repeats.  A qualitative rule cannot explain either.

THE HYPOTHESIS.  The two metrics see different things, so they should track
different properties of the recovered edge functions:

  * NLL rewards CALIBRATION.  Any nonlinearity helps, including a MONOTONE one:
    a monotone reshaping of the index changes fitted hazards and so the likelihood.
  * C-index rewards DISCRIMINATION, and is invariant to any monotone transform of
    the risk index.  Only NON-MONOTONE structure -- curvature that REORDERS
    subjects -- can move it.

So: dNLL should track total recovered curvature; dC should track only the
non-monotone part.  nwtco is then not an anomaly but the clean prediction --
monotone nonlinearity, hence NLL gain with exactly zero C gain.

MEASURES, per covariate edge function on a 5-95 percentile grid:
  amplitude   max - min of the curve
  nonlin      fraction of its variance not explained by a straight line
  nonmono     1 - |sum of increments| / sum |increments|;  0 for any monotone
              curve, growing as the curve turns back on itself
Dataset totals weight each by amplitude, since a wiggle in a flat edge is noise.

Also reports Spearman rho between the linear and KAN risk indices: if the KAN is
doing pure monotone recalibration, rho is ~1 and dC must be ~0 by construction.

Run:  python -u experiments/why_gain.py [dataset ...]
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
from kanrel.stats import spearman
from kanrel.fit import fit, to_tensors
from kanrel.hazard import DiscreteHazardKAN
from experiments.real_data import continuous_columns, split
from experiments.interpretability import partial_curve, nonlinearity
from experiments.baselines import SELECTED_L1 as FAIR_L1

# Pooled out-of-fold dC / dNLL from pooled_cv.txt, for the correlation at the end.
POOLED = {"rotgbsg": (0.0173, 0.02135), "support-pycox": (0.0130, 0.00657),
          "metabric": (0.0078, 0.00190), "nwtco": (0.0001, 0.00850),
          "flchain": (0.0022, 0.00161)}


def nonmonotonicity(eff):
    """0 for a monotone curve; grows as the curve reverses direction."""
    dv = np.diff(eff)
    tot = np.abs(dv).sum()
    return float(1.0 - abs(dv.sum()) / tot) if tot > 1e-12 else 0.0


# spearman lives in kanrel.stats; this file had its own ordinal-rank copy.


def risk_index(m, X):
    """Scalar risk per subject: total hazard over the grid (monotone in risk)."""
    with torch.no_grad():
        S = m.survival(torch.as_tensor(X)).numpy()
    return -np.log(np.clip(S[:, -1], 1e-12, 1.0))


def run(nm, loader):
    d = D.onehot_ordinals(loader())
    tr, te = split(d, seed=0)
    tr, te = D.clip_to_train_range(tr, te)
    l1 = FAIR_L1.get(nm, 0.01)

    lin = DiscreteHazardKAN(tr.X.shape[1], tr.n_bins, mode="linear", link="cloglog")
    lin, _ = fit(lin, tr, epochs=500, lr=0.03, val_frac=0.2, patience=60, seed=0)
    cont = continuous_columns(tr.X)
    kan = DiscreteHazardKAN(tr.X.shape[1], tr.n_bins, mode="baseline", link="cloglog",
                            hidden=(), grid_size=8, cont_idx=cont, clamp_inputs=True)
    kan, _ = fit(kan, tr, epochs=400, lr=0.03, val_frac=0.2, patience=60,
                 grid_update_epochs=(30, 100), l1=l1, smooth=1e-3, seed=0)

    rho = spearman(risk_index(lin, te.X), risk_index(kan, te.X))

    tot_nl = tot_nm = 0.0
    detail = []
    for j in cont:
        lo, hi = np.percentile(tr.X[:, j], [5, 95])
        if hi - lo < 1e-8:
            continue
        grid = torch.linspace(float(lo), float(hi), 60)
        eff = partial_curve(kan, tr.X, j, grid)
        amp = float(eff.max() - eff.min())
        nl = nonlinearity(grid.numpy(), eff)
        nm_ = nonmonotonicity(eff)
        tot_nl += amp * nl
        tot_nm += amp * nm_
        detail.append((d.feature_names[j], amp, nl, nm_))

    print(f"\n{'='*92}\n{nm}   n={d.n}  spline covariates={len(detail)}\n{'='*92}")
    print(f"  Spearman rho(linear index, KAN index) = {rho:.5f}"
          + ("   <- essentially IDENTICAL ranking" if rho > 0.999 else ""))
    print(f"  {'covariate':<16}{'amplitude':>11}{'nonlin':>9}{'nonmono':>10}")
    for fn, amp, nl, nm_ in sorted(detail, key=lambda r: -r[1] * r[2])[:6]:
        print(f"  {fn:<16}{amp:>11.4f}{nl:>9.3f}{nm_:>10.3f}")
    print(f"  TOTAL curvature (amp x nonlin)   = {tot_nl:.4f}")
    print(f"  TOTAL reordering (amp x nonmono) = {tot_nm:.4f}")
    return dict(name=nm, rho=rho, nl=tot_nl, nm=tot_nm)


def main():
    names = sys.argv[1:] or ["rotgbsg", "support-pycox", "metabric", "nwtco", "flchain"]
    avail = dict(D.LOADERS)
    out = []
    for nm in names:
        try:
            out.append(run(nm, avail[nm]))
        except Exception as e:
            print(f"  {nm}: FAILED {type(e).__name__}: {str(e)[:70]}")

    print(f"\n{'='*92}\nDOES RECOVERED STRUCTURE PREDICT THE GAIN?\n{'='*92}")
    print(f"  {'dataset':<16}{'rho':>9}{'curvature':>11}{'reordering':>12}"
          f"{'pooled dC':>11}{'pooled dNLL':>13}")
    for r in out:
        dc, dn = POOLED.get(r["name"], (float('nan'),) * 2)
        print(f"  {r['name']:<16}{r['rho']:>9.5f}{r['nl']:>11.4f}{r['nm']:>12.4f}"
              f"{dc:>+11.4f}{dn:>+13.5f}")
    ok = [r for r in out if r["name"] in POOLED]
    if len(ok) >= 3:
        dc = np.array([POOLED[r["name"]][0] for r in ok])
        dn = np.array([POOLED[r["name"]][1] for r in ok])
        nl = np.array([r["nl"] for r in ok]); nmv = np.array([r["nm"] for r in ok])
        print(f"\n  corr(reordering, dC)   = {spearman(nmv, dc):+.3f}"
              f"      <- should be HIGH if only non-monotone structure moves C")
        print(f"  corr(curvature,  dC)   = {spearman(nl, dc):+.3f}")
        print(f"  corr(curvature,  dNLL) = {spearman(nl, dn):+.3f}"
              f"      <- should be HIGH: any curvature helps the likelihood")
        print(f"  corr(reordering, dNLL) = {spearman(nmv, dn):+.3f}")


if __name__ == "__main__":
    main()
