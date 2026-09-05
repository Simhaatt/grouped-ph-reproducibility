"""Out-of-fold evaluation, so every row is a test row.

THE PROBLEM.  A 70/30 split throws away 70% of the evaluation data.  On metabric
that leaves n_test = 571, where the paired C-index CI has half-width 0.020 --
against a measured effect of 0.0109.  The result is consistent in sign on 5/5
splits and yet resolvable on 0/5, purely for want of test rows.  That is the same
criticism section 4.5 makes of the literature, applied to us.

THE FIX.  K-fold cross-validation, keeping the OUT-OF-FOLD predictions: each row
is predicted exactly once, by a model that never saw it, so the evaluation set is
the whole cohort.  metabric goes 571 -> 1,904 test rows, a 3.33x gain in pairs
and ~1.8x narrower intervals -- enough to cross the resolution threshold.

Repeats give the split-to-split variation that a single K-fold cannot.

⚠️ CAVEAT, stated because it is easy to oversell.  Out-of-fold predictions come
from K DIFFERENT fitted models, so the pooled bootstrap mixes test-set sampling
with model-to-model variation.  It is the standard cross-validated C-index and it
is the right estimator for "how well does this METHOD rank", but it is not a
statement about one fitted model.  Report it as such.

Run:  python -u experiments/pooled_cv.py [dataset ...]
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
from kanrel.data import SurvData
from kanrel.fit import fit, to_tensors
from kanrel.hazard import DiscreteHazardKAN
from kanrel.likelihood import nll as nll_fn
from kanrel.metrics import paired_c_bootstrap
from experiments.real_data import continuous_columns
from experiments.significance import paired_bootstrap
from experiments.baselines import SELECTED_L1 as FAIR_L1

N_FOLDS, N_REPEATS = 5, 3


def subset(d: SurvData, ix) -> SurvData:
    return SurvData(d.X[ix], d.bin_idx[ix], d.event[ix], d.n_bins, d.feature_names,
                    name=d.name, intrinsically_discrete=d.intrinsically_discrete,
                    entry_idx=None if d.entry_idx is None else d.entry_idx[ix],
                    bin_edges=d.bin_edges, meta=d.meta)


def fit_pair(tr, te, l1, seed):
    lin = DiscreteHazardKAN(tr.X.shape[1], tr.n_bins, mode="linear", link="cloglog")
    lin, _ = fit(lin, tr, epochs=500, lr=0.03, val_frac=0.2, patience=60, seed=seed)
    kan = DiscreteHazardKAN(tr.X.shape[1], tr.n_bins, mode="baseline", link="cloglog",
                            hidden=(), grid_size=8, cont_idx=continuous_columns(tr.X),
                            clamp_inputs=True)
    kan, _ = fit(kan, tr, epochs=400, lr=0.03, val_frac=0.2, patience=60,
                 grid_update_epochs=(30, 100), l1=l1, entropy=1.0, smooth=1e-3, seed=seed)
    X, mask, y = to_tensors(te)
    with torch.no_grad():
        nl = nll_fn(lin(X), mask, y, lin.link, reduction=None).numpy()
        nk = nll_fn(kan(X), mask, y, kan.link, reduction=None).numpy()
        return nl, nk, lin.survival(X).numpy(), kan.survival(X).numpy()


def one_repeat(d, l1, seed):
    """K-fold; return out-of-fold NLLs and survival matrices for ALL n rows."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(d.n)
    folds = np.array_split(order, N_FOLDS)
    nl = np.empty(d.n); nk = np.empty(d.n)
    Sl = np.empty((d.n, d.n_bins)); Sk = np.empty((d.n, d.n_bins))
    for f, te_ix in enumerate(folds):
        tr_ix = np.concatenate([folds[j] for j in range(N_FOLDS) if j != f])
        tr, te = subset(d, tr_ix), subset(d, te_ix)
        tr, te = D.clip_to_train_range(tr, te)      # same fairness fix as elsewhere
        a, b, sa, sb = fit_pair(tr, te, l1, seed * 100 + f)
        nl[te_ix], nk[te_ix] = a, b
        Sl[te_ix], Sk[te_ix] = sa, sb
    return nl, nk, Sl, Sk


def main():
    names = sys.argv[1:] or ["metabric", "rotgbsg", "support-pycox", "nwtco", "flchain"]
    avail = dict(D.LOADERS)
    print("=" * 92)
    print(f"OUT-OF-FOLD EVALUATION  ({N_FOLDS}-fold x {N_REPEATS} repeats, "
          f"winsorised, fair baseline)")
    print("  every row is a test row -> the whole cohort is the evaluation set")
    print("=" * 92)

    summary = []
    for nm in names:
        if nm not in avail:
            print(f"  unknown dataset {nm!r}"); continue
        d = D.onehot_ordinals(avail[nm]())
        l1 = FAIR_L1.get(nm, 0.01)
        print(f"\n{d.summary()}\n  l1={l1:g}   n_eval={d.n} (vs {int(round(0.3*d.n))} "
              f"under a 70/30 split)")
        cd, csig, nd, nsig = [], 0, [], 0
        for r in range(N_REPEATS):
            nl, nk, Sl, Sk = one_repeat(d, l1, seed=r)
            m, lo, hi = paired_bootstrap(nl - nk)
            cb = paired_c_bootstrap(Sl, Sk, d.bin_idx, d.event, seed=r)
            cs = cb["delta_lo"] > 0 or cb["delta_hi"] < 0
            ns = lo > 0 or hi < 0
            csig += cs; nsig += ns
            cd.append(cb["delta"]); nd.append(m)
            print(f"  repeat {r}:  NLL {m:+.5f} [{lo:+.5f}, {hi:+.5f}]"
                  f"{'  SIG' if ns else ''}")
            print(f"             C   {cb['delta']:+.4f} [{cb['delta_lo']:+.4f}, "
                  f"{cb['delta_hi']:+.4f}]{'  SIG' if cs else ''}"
                  f"   (linear {cb['c_a']:.4f} -> KAN {cb['c_b']:.4f})")
        cd, nd = np.array(cd), np.array(nd)
        print(f"  ==> C  {cd.mean():+.4f} +/- {cd.std(ddof=1):.4f}   "
              f"resolved on {csig}/{N_REPEATS} repeats")
        summary.append((nm, d.n, cd.mean(), cd.std(ddof=1), csig, nd.mean(), nsig))

    print("\n" + "=" * 92)
    print(f"  {'dataset':<16}{'n_eval':>8}{'C delta':>12}{'sd':>9}"
          f"{'C resolved':>13}{'NLL delta':>12}{'NLL resolved':>14}")
    for nm, n, c, s, cs, nn, ns in summary:
        print(f"  {nm:<16}{n:>8}{c:>+12.4f}{s:>9.4f}{f'{cs}/{N_REPEATS}':>13}"
              f"{nn:>+12.5f}{f'{ns}/{N_REPEATS}':>14}")


if __name__ == "__main__":
    main()
