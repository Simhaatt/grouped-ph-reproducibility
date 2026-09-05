"""A confidence interval for the FORMULATION effect -- the paper's headline number.

Section 2 rests on one figure: on sparcs/drg302 the grouped discrete hazard beats
Cox+Breslow by +0.0688 test NLL, holding the linear index fixed.  Every other
headline in this project now carries a paired interval, and section 4.5 attacks
CoxKAN / SurvKAN / KAN-AFT / KAPLAN-HR precisely for quoting differences without
one.  This closes the last gap.

Both models score the SAME test rows through the SAME masked-Bernoulli NLL (the
Cox arm via the Breslow conversion in baselines.py), so the comparison is paired
and the per-unit differences can be bootstrapped directly.

Positive = the discrete hazard assigns higher likelihood = our formulation wins.

Run:  python -u experiments/formulation_ci.py [dataset ...]
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
from experiments.real_data import split
from experiments.significance import paired_bootstrap
from experiments.baselines import (SELECTED_L1 as FAIR_L1, breslow_survival,
                                   nll_from_survival, standardize)

N_SPLITS = 3


def cox_linear_per_unit(tr, te):
    """Cox PH + Breslow, replicating baselines.run_cox_linear EXACTLY.

    An earlier version of this file used lifelines CoxPHFitter, which failed with
    ConvergenceError on 2 of 3 sparcs splits (near-collinear one-hot dummies) and
    produced +0.0648 where the baselines table says +0.0688 -- a different
    estimator annotated with a CI, which is worse than no CI at all.
    baselines.py uses theory.verify_ties.fit_cox; so must this.
    """
    from theory.verify_ties import fit_cox

    trs, tes = standardize(tr, te)
    beta = fit_cox(trs.X.astype(float), trs.bin_idx, trs.event.astype(float), "efron")
    r_tr = np.clip(np.exp(trs.X.astype(float) @ beta), 1e-8, 1e8)
    r_te = np.clip(np.exp(tes.X.astype(float) @ beta), 1e-8, 1e8)
    S = breslow_survival(r_tr, trs.bin_idx, trs.event, r_te, trs.n_bins)
    return nll_from_survival(S, tes.bin_idx, tes.event, trs.n_bins, reduction=None)


def ours_linear_per_unit(tr, te, seed):
    """Replicates baselines.run_ours(mode="linear") EXACTLY -- in particular
    epochs=400 and smooth=1e-3.  Omitting the P-spline penalty on alpha fits a
    different (less regularised) baseline and shifted sparcs by 0.004."""
    m = DiscreteHazardKAN(tr.X.shape[1], tr.n_bins, mode="linear", link="cloglog",
                          hidden=(), grid_size=8, cont_idx=None)
    m, _ = fit(m, tr, epochs=400, lr=0.03, val_frac=0.2, patience=60,
               grid_update_epochs=(), l1=0.0, smooth=1e-3, seed=seed)
    X, mask, y = to_tensors(te)
    with torch.no_grad():
        return nll_fn(m(X), mask, y, m.link, reduction=None).numpy()


def main():
    names = sys.argv[1:] or ["sparcs/drg302", "support2/slos", "rotgbsg"]
    avail = dict(D.LOADERS)
    avail["sparcs/drg302"] = lambda: D.load_sparcs("302")

    print("=" * 92)
    print("FORMULATION EFFECT, WITH AN INTERVAL   (Cox+Breslow minus discrete hazard)")
    print("  index held fixed at LINEAR in both arms; positive favours the discrete hazard")
    print("=" * 92)
    for nm in names:
        if nm not in avail:
            print(f"  unknown dataset {nm!r}"); continue
        d = D.onehot_ordinals(avail[nm]())
        out = []
        for s in range(N_SPLITS):
            tr, te = split(d, seed=s)
            tr, te = D.clip_to_train_range(tr, te)
            try:
                nc = cox_linear_per_unit(tr, te)
                no = ours_linear_per_unit(tr, te, s)
            except Exception as e:
                print(f"  {nm} s{s}: FAILED {type(e).__name__}: {str(e)[:60]}"); continue
            m, lo, hi = paired_bootstrap(nc - no)
            sig = "SIGNIFICANT" if lo > 0 else ("worse" if hi < 0 else "not significant")
            print(f"\n  {nm:<16} split {s}  n_test={te.n}")
            print(f"    Cox+Breslow {nc.mean():.4f}   discrete {no.mean():.4f}")
            print(f"    formulation = {m:+.5f}   95% CI [{lo:+.5f}, {hi:+.5f}]  -> {sig}")
            out.append((m, lo, hi, lo > 0))
        if out:
            a = np.array([o[0] for o in out])
            print(f"  ==> {nm}: {a.mean():+.5f} +/- {a.std(ddof=1):.5f}   "
                  f"resolved on {sum(o[3] for o in out)}/{len(out)} splits")


if __name__ == "__main__":
    main()
