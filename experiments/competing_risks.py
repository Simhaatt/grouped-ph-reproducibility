"""A7: how wrong is it to treat in-hospital death as censoring?

`support2/slos` has two exits -- discharged alive (70.9%) and died in hospital
(24.8%) -- and every earlier fit in this project treated death as independent
censoring.  That assumption says a patient who dies would eventually have been
discharged.  They would not have been.

Two analyses on IDENTICAL rows:

  naive       discharge hazard with death as censoring, reporting
              1 - prod(1 - h_discharge) as "probability of discharge by day t"
  competing   two-cause multinomial model, reporting the cumulative incidence
              CIF_1(t) = sum_{s<=t} h_1(s) prod_{u<s} P(still in hospital)

The gap between them is the bias, and it is a quantity clinicians would actually
act on: the naive curve answers "what fraction would be discharged if nobody
died", which is not a question anyone asked.

Run:  python experiments/competing_risks.py
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
from kanrel.competing import CompetingHazardKAN
from kanrel.data import SurvData
from kanrel.fit import fit, to_tensors
from kanrel.hazard import DiscreteHazardKAN
from kanrel.likelihood import make_targets_competing, nll_competing
from experiments.real_data import continuous_columns, split


def fit_competing(model, d, cause, epochs=400, lr=0.03, val_frac=0.2,
                  patience=60, l1=0.0, smooth=1e-3, seed=0):
    """Adam + early stopping for the multinomial model."""
    import copy
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(d.X, dtype=torch.float32)
    mask, y = make_targets_competing(d.bin_idx, cause, d.n_bins, model.n_causes)

    perm = rng.permutation(d.n)
    n_val = int(round(val_frac * d.n))
    va = torch.as_tensor(perm[:n_val]); tr = torch.as_tensor(perm[n_val:])
    model.set_standardization(X[tr])
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = nll_competing(model(X[tr]), mask[tr], y[tr])
        (loss + model.penalty(X[tr], l1=l1, smooth=smooth)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            v = float(nll_competing(model(X[va]), mask[va], y[va])) if n_val else float(loss)
        if v < best - 1e-6:
            best, bad, best_state = v, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best


def main():
    dc = D.load_support2_competing()
    cause = dc.meta["cause"]
    tr_ix, te_ix = None, None
    rng = np.random.default_rng(0)
    perm = rng.permutation(dc.n)
    n_te = int(round(0.3 * dc.n))
    te_ix, tr_ix = perm[:n_te], perm[n_te:]

    def sub(d, ix):
        return SurvData(d.X[ix], d.bin_idx[ix], d.event[ix], d.n_bins, d.feature_names,
                        name=d.name, intrinsically_discrete=True, meta=d.meta)

    tr, te = sub(dc, tr_ix), sub(dc, te_ix)
    ci = continuous_columns(tr.X)

    print("=" * 92)
    print("A7  COMPETING RISKS: is treating in-hospital death as censoring harmful?")
    print("=" * 92)
    print(f"  n={dc.n}  train={tr.n} test={te.n}   discharged 70.9% / died 24.8% / censored 4.2%")

    # --- competing-risks model -------------------------------------------------
    cm = CompetingHazardKAN(tr.X.shape[1], tr.n_bins, n_causes=2, hidden=(),
                            grid_size=8, cont_idx=ci)
    cm, vloss = fit_competing(cm, tr, cause[tr_ix], l1=0.01, seed=0)
    Xte = torch.as_tensor(te.X, dtype=torch.float32)
    with torch.no_grad():
        cif = cm.cif(Xte)[..., 0].numpy()          # correct CIF for discharge
        naive_from_cr = cm.naive_cif(Xte, 0).numpy()

    # --- the analysis we have actually been running ---------------------------
    dn = D.load_support2("slos")
    trn, ten = sub(dn, tr_ix), sub(dn, te_ix)
    nm = DiscreteHazardKAN(trn.X.shape[1], trn.n_bins, mode="baseline",
                           link="cloglog", hidden=(), grid_size=8, cont_idx=ci)
    nm, _ = fit(nm, trn, epochs=400, lr=0.03, val_frac=0.2, patience=60,
                grid_update_epochs=(30, 100), l1=0.01, smooth=1e-3, seed=0)
    Xn, _, _ = to_tensors(ten)
    with torch.no_grad():
        naive_curve = (1.0 - nm.survival(Xn)).numpy()   # "P(discharged by t)"

    # --- observed truth on the test split -------------------------------------
    c_te = cause[te_ix]
    obs = np.array([np.mean((te.bin_idx <= t) & (c_te == 1)) for t in range(te.n_bins)])

    print()
    print(f"  {'day':>5}{'observed':>11}{'competing':>12}{'naive(sep fit)':>16}"
          f"{'naive(same fit)':>17}{'bias':>9}")
    print(f"  {'':>5}{'discharged':>11}{'CIF':>12}{'1-prod(1-h)':>16}{'1-prod(1-h)':>17}"
          f"{'pp':>9}")
    for t in [2, 4, 6, 9, 13, 19, 29, 44, 59]:
        b = 100 * (naive_curve[:, t].mean() - cif[:, t].mean())
        print(f"  {t+1:>5}{obs[t]:>11.4f}{cif[:, t].mean():>12.4f}"
              f"{naive_curve[:, t].mean():>16.4f}{naive_from_cr[:, t].mean():>17.4f}"
              f"{b:>+9.1f}")

    final_bias = 100 * (naive_curve[:, -1].mean() - cif[:, -1].mean())
    print()
    print(f"  At day {te.n_bins}: observed discharged {obs[-1]:.3f}, "
          f"competing-risks CIF {cif[:, -1].mean():.3f}, naive {naive_curve[:, -1].mean():.3f}")
    print(f"  -> naive analysis OVERSTATES discharge probability by "
          f"{final_bias:+.1f} percentage points")
    err_cr = abs(cif[:, -1].mean() - obs[-1])
    err_nv = abs(naive_curve[:, -1].mean() - obs[-1])
    print(f"  -> absolute error vs observed:  competing {err_cr:.4f}   naive {err_nv:.4f}")
    print(f"  -> competing-risks model is {'BETTER' if err_cr < err_nv else 'WORSE'} calibrated")


if __name__ == "__main__":
    main()
