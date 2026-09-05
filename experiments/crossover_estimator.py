"""Is the crossover a property of the LIKELIHOOD or of how it was FITTED?

THE DISCREPANCY.  On support2/slos the paper reports a sign change: the grouped
arm wins at coarse grids and Cox wins at fine ones, resolved 3/3 at T=60 with
D_T = -0.00786 [`crossover.txt`].  The 20-split sweep gives +0.00041 +- 0.00069 on
the same cohort and grid, and D_T is POSITIVE at every T from 6 to 60
[`protocol_decomp.txt`].  Not a smaller effect: the opposite sign.

The two arms are not the same estimator.

    old  experiments/formulation_ci.py::ours_linear_per_unit
         DiscreteHazardKAN in linear mode, fitted by Adam for up to 400 epochs
         with lr 0.03, EARLY STOPPING on a 20% internal validation holdout
         (patience 60), and a P-spline penalty on alpha (smooth = 1e-3)
    new  experiments/cox_arms.py::fit_grouped_joint
         the exact joint MLE of (alpha, beta) by LBFGS on the whole training split

Three differences at once, each of which could cost the grouped arm at fine grids:

    (a) 20% of the training split is spent on the early-stopping holdout, so the
        old arm sees 5,099 rows where the new one sees 6,374
    (b) early stopping returns a point that is not the MLE
    (c) the P-spline penalty biases alpha

Both arms are legitimate and they answer different questions -- for a prediction
comparison early stopping is standard, which is what real_data.py says.  But the
paper presents the crossover as evidence about the grouped LIKELIHOOD and its
nuisance-parameter cost, and if the sign is set by (a), (b) or (c) then it is
evidence about the fitting procedure instead.  This factorial says which.

Run:  python -u experiments/crossover_estimator.py
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
from kanrel.fit import fit, to_tensors
from kanrel.hazard import DiscreteHazardKAN
from kanrel.likelihood import nll as nll_fn
from experiments import cox_arms as CA
from experiments.baselines import standardize
from experiments.crossover import coarsen
from experiments.protocol_decomp import nb_se
from experiments.real_data import split

N_SPLITS = 20
GRIDS = (6, 15, 30, 60)
OUT = Path(__file__).resolve().parent / "crossover_estimator.txt"

# (label, val_frac, patience, smooth).  patience None disables early stopping by
# running the full epoch budget.
ARMS = [
    ("MLE (LBFGS, no penalty)", None, None, None),
    ("SGD val=0.2 stop pen=1e-3", 0.2, 60, 1e-3),   # the paper's old arm
    ("SGD val=0.2 stop pen=0", 0.2, 60, 0.0),
    ("SGD val=0.0 nostop pen=0", 0.0, 10 ** 6, 0.0),
    ("SGD val=0.0 nostop pen=1e-3", 0.0, 10 ** 6, 1e-3),
]


def grouped_sgd(tr, te, seed, val_frac, patience, smooth):
    m = DiscreteHazardKAN(tr.X.shape[1], tr.n_bins, mode="linear", link="cloglog",
                          hidden=(), grid_size=8, cont_idx=None)
    m, _ = fit(m, tr, epochs=400, lr=0.03, val_frac=val_frac, patience=patience,
               grid_update_epochs=(), l1=0.0, smooth=smooth, seed=seed)
    X, mask, y = to_tensors(te)
    with torch.no_grad():
        return nll_fn(m(X), mask, y, m.link, reduction=None).numpy()


def main():
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log("=" * 108)
    log("IS THE CROSSOVER IN THE LIKELIHOOD OR IN THE FIT?  support2/slos,")
    log(f"  {N_SPLITS} splits, Cox/Efron + Breslow held fixed in every column.")
    log("  D_T = NLL(Cox) - NLL(grouped arm).  D_T < 0 is a COX win, i.e. a")
    log("  crossover.  Only the grouped arm's ESTIMATOR changes across columns.")
    log("=" * 108)
    base = D.load_support2("slos", horizon=60)

    header = f"  {'T':>4}{'rows/bin':>10}{'modal':>8}"
    for lbl, *_ in ARMS:
        header += f"{lbl:>29}"
    log(header)
    log(f"  {'':>4}{'':>10}{'':>8}" + "".join(f"{'D_T':>14}{'NB SE':>8}{'':>7}"
                                              for _ in ARMS))

    for T in GRIDS:
        if T > base.n_bins:
            continue
        d = D.onehot_ordinals(coarsen(base, T))
        modal = float(np.bincount(d.bin_idx.astype(int)).max() / d.n)
        t0 = time.time()
        cols = {i: [] for i in range(len(ARMS))}
        for s in range(N_SPLITS):
            tr, te = split(d, frac=0.3, seed=s)
            tr, te = D.clip_to_train_range(tr, te)
            trs, tes = standardize(tr, te)
            Xtr, Xte = trs.X.astype(float), tes.X.astype(float)
            try:
                b = CA.fit_cox_ties(Xtr, trs.bin_idx, trs.event, "efron")
                n1 = CA.nll_from_hazards(
                    CA.hazards_breslow(CA.risk(Xtr, b), trs.bin_idx, trs.event,
                                       CA.risk(Xte, b), T),
                    tes.bin_idx, tes.event, T)
            except Exception:
                continue
            for i, (lbl, vf, pat, sm) in enumerate(ARMS):
                try:
                    if vf is None:
                        bj, aj = CA.fit_grouped_joint(Xtr, trs.bin_idx,
                                                      trs.event, T)
                        n3 = CA.nll_from_hazards(
                            CA.hazards_from_alpha(Xte, bj, aj),
                            tes.bin_idx, tes.event, T)
                    else:
                        n3 = grouped_sgd(trs, tes, s, vf, pat, sm)
                    cols[i].append(float(np.mean(n1 - n3)))
                except Exception:
                    pass
        row = f"  {T:>4}{d.n / T:>10.0f}{modal:>8.4f}"
        for i in range(len(ARMS)):
            v = np.array(cols[i], float)
            if v.size < 2:
                row += f"{'--':>14}{'':>8}{'':>7}"
            else:
                se = nb_se(v)
                tag = ("COX" if v.mean() < -1.959964 * se else
                       "grp" if v.mean() > 1.959964 * se else "ns")
                row += f"{v.mean():>+14.5f}{se:>8.5f}{tag:>7}"
        log(row + f"   [{time.time()-t0:.0f}s]")

    log("")
    log("=" * 108)
    log("READING")
    log("=" * 108)
    log("  A 'COX' tag anywhere is a crossover.  Compare the columns at T=60:")
    log("  if the MLE column never goes negative while a SGD column does, the")
    log("  crossover is a property of the FIT and not of the likelihood, and the")
    log("  paper's section 6.2 mechanism has to be restated in those terms.")
    log("")
    log("  WHAT THIS FACTORIAL DOES AND DOES NOT SEPARATE.  In kanrel.fit, when")
    log("  val_frac is 0 the validation loss is set to the TRAINING loss, which")
    log("  decreases monotonically, so early stopping never triggers.  Withholding")
    log("  data and early stopping therefore cannot be separated in this")
    log("  implementation and the val=0.2 against val=0.0 contrast measures them")
    log("  JOINTLY.  Three things are cleanly isolated:")
    log("")
    log("    optimiser   MLE column against 'SGD val=0.0 nostop pen=0':")
    log("                LBFGS to convergence against 400 Adam epochs, same data,")
    log("                same likelihood, no penalty either side")
    log("    penalty     pen=1e-3 against pen=0 at the same val setting")
    log("    holdout+    val=0.2 against val=0.0, jointly, for the reason above")
    log("      stopping")
    log("")
    log("  Claiming a clean three-way split here would have been wrong, so it is")
    log("  not claimed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
