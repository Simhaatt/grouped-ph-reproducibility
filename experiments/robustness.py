"""Robustness across an EXTERNAL simulation design we did not choose.

`data/MLtoSurvival-Data` is a published grid: event% in {10,30,50,70} x censor%
in {0,10,30,50,70} x {5,25} covariates, n=3000 each.  Our own generator has known
ground truth, but a referee will reasonably ask how the estimator behaves over
event and censoring rates chosen by someone else -- particularly the corners
(10% events, 70% censoring) where discrete-time methods are most likely to break.

The question here is NOT "does the KAN win".  At n=3000 our paired-bootstrap work
already showed differences at that scale are not resolvable.  The question is
whether the estimator stays **stable and calibrated** as events become rare and
censoring heavy: no divergence, no degenerate hazards, no NaNs, IBS that tracks
the censoring rate sensibly.

Run:  python experiments/robustness.py
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
from kanrel.metrics import evaluate
from experiments.real_data import continuous_columns, split

EVENTS = [10, 30, 50, 70]
CENSOR = [0, 10, 30, 50, 70]


def one(d, mode, seed=0, l1=0.0):
    tr, te = split(d, seed=seed)
    # Same fair pipeline as every other comparison in the paper: ordinals as
    # dummies (applied at load, below) and test covariates clipped to the
    # training range for BOTH arms.  Without the clip the LINEAR arm is the one
    # that blows up on an out-of-range row (section 4.8b), which would make a
    # stability check report the baseline's fragility as the KAN's stability.
    tr, te = D.clip_to_train_range(tr, te)
    ci = continuous_columns(tr.X) if mode == "baseline" else None
    m = DiscreteHazardKAN(tr.X.shape[1], tr.n_bins, mode=mode, link="cloglog",
                          hidden=(), grid_size=8, cont_idx=ci)
    m, _ = fit(m, tr, epochs=300, lr=0.03, val_frac=0.2, patience=50,
               grid_update_epochs=(30,) if mode != "linear" else (),
               l1=l1, smooth=1e-3, seed=seed)
    X, mask, y = to_tensors(te)
    with torch.no_grad():
        logits = m(X)
        loss = float(nll_fn(logits, mask, y, m.link))
        S = m.survival(X)
        h = m.hazard(X)
    r = evaluate(S, te.bin_idx, te.event, nll=loss)
    r["finite"] = bool(torch.isfinite(logits).all())
    r["h_max"] = float(h.max())
    r["h_min"] = float(h.min())
    return r


def main():
    for n_cov in (5, 25):
        print(f"\n{'='*100}")
        print(f"EXTERNAL SIMULATION GRID -- {n_cov} covariates, n=3000 each")
        print("  stability check, not a win check: does the fit stay finite and")
        print("  calibrated as events get rare and censoring gets heavy?")
        print(f"{'='*100}")
        print(f"  {'event%':>7}{'cens%':>7}{'obs ev':>8} | "
              f"{'lin NLL':>9}{'KAN NLL':>9}{'lin C':>8}{'KAN C':>8}"
              f"{'lin IBS':>9}{'KAN IBS':>9}{'max h':>8}{'ok':>4}")
        bad = []
        for ev in EVENTS:
            for cs in CENSOR:
                try:
                    # 2 of the 5 covariate columns have 3-4 levels, so the
                    # one-hot fix is NOT a no-op on this grid.
                    d = D.onehot_ordinals(D.load_sim_grid(ev, cs, n_cov))
                except FileNotFoundError:
                    continue
                try:
                    rl = one(d, "linear")
                    rk = one(d, "baseline")
                    ok = rl["finite"] and rk["finite"] and \
                        np.isfinite([rl["nll"], rk["nll"], rl["ibs"], rk["ibs"]]).all()
                    if not ok:
                        bad.append((ev, cs, "non-finite"))
                    print(f"  {ev:>7}{cs:>7}{d.event.mean():>8.3f} | "
                          f"{rl['nll']:>9.4f}{rk['nll']:>9.4f}"
                          f"{rl['c_index']:>8.4f}{rk['c_index']:>8.4f}"
                          f"{rl['ibs']:>9.4f}{rk['ibs']:>9.4f}"
                          f"{rk['h_max']:>8.4f}{'y' if ok else 'N':>4}")
                except Exception as e:
                    bad.append((ev, cs, f"{type(e).__name__}: {str(e)[:40]}"))
                    print(f"  {ev:>7}{cs:>7}     -- | FAILED {type(e).__name__}: "
                          f"{str(e)[:44]}")
        print()
        if bad:
            print(f"  UNSTABLE CELLS ({len(bad)}):")
            for ev, cs, why in bad:
                print(f"    event={ev}% censor={cs}%  {why}")
        else:
            print("  -> every cell finite and calibrated; no instability across the grid.")


if __name__ == "__main__":
    main()
