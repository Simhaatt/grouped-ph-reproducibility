"""Workstream C: fit the discrete-hazard KAN on real cohorts and compare.

Models compared on identical splits:

  linear/cloglog   Prentice-Gloeckler grouped PH -- THIS IS NAWATA ET AL.'S MODEL
                   and the one our KAN nests.  The reference point.
  linear/logit     the same linear index under a proportional-ODDS link, which is
                   what nnet-survival and most discrete-time deep survival models
                   use.  Included to show the link choice costs something on real
                   data, not only in simulation.
  kan/cloglog      additive KAN [p, 1]: replaces the linear index with a sum of
                   learned univariate functions.  Interpretable edge-by-edge.
  kan-deep/cloglog depth-2 KAN [p, 16, 1]: allows covariate interactions.
  kan-shared/cloglog  time as a KAN input, so effects may vary with t
                   (non-proportional).  This is what SurvKAN / KAPLAN-HR do, but
                   in exact discrete time.

NOTE on early stopping.  The rate experiments in theory/ deliberately disable it,
because Theorem 4 describes the sieve MLE and early stopping targets a different
estimator.  Here the goal is PREDICTION, where early stopping is the appropriate
and standard choice.  The two settings answer different questions.

Run:  python experiments/real_data.py [dataset ...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kanrel import data as D
from kanrel.data import SurvData
from kanrel.fit import fit, to_tensors
from kanrel.hazard import DiscreteHazardKAN
from kanrel.likelihood import nll as nll_fn
from kanrel.metrics import evaluate

MODELS = {
    "linear/cloglog":     dict(mode="linear",   link="cloglog", hidden=(), cont=False),
    "linear/logit":       dict(mode="linear",   link="logit",   hidden=(), cont=False),
    # all-spline: a spline over every column INCLUDING the 0/1 dummies.  This is
    # the naive way to point a KAN at tabular data and is kept as the control.
    "kan-allspline":      dict(mode="baseline", link="cloglog", hidden=(), cont=False),
    # cont-only: splines on continuous columns, a linear term on the dummies.
    "kan/cloglog":        dict(mode="baseline", link="cloglog", hidden=(), cont=True),
    "kan-deep/cloglog":   dict(mode="baseline", link="cloglog", hidden=(16,), cont=True),
    "kan-shared/cloglog": dict(mode="shared",   link="cloglog", hidden=(16,), cont=True),
}


def continuous_columns(X, max_card=2):
    """Columns a univariate spline can meaningfully act on.

    A cubic spline with G knots has G+3 coefficients; over a 0/1 dummy those are
    pinned by two x-values, so the extra capacity is pure variance.  On
    support2/slos 19 of 32 columns are one-hot dummies.
    """
    return [j for j in range(X.shape[1]) if len(np.unique(X[:, j])) > max_card]


def split(d: SurvData, frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(d.n)
    n_te = int(round(frac * d.n))
    te, tr = idx[:n_te], idx[n_te:]

    def sub(ix):
        return SurvData(d.X[ix], d.bin_idx[ix], d.event[ix], d.n_bins,
                        d.feature_names, name=d.name,
                        intrinsically_discrete=d.intrinsically_discrete,
                        entry_idx=None if d.entry_idx is None else d.entry_idx[ix],
                        bin_edges=d.bin_edges, meta=d.meta)
    return sub(tr), sub(te)


def fit_eval(spec, tr, te, epochs, seed):
    ci = continuous_columns(tr.X) if spec.get("cont") else None
    m = DiscreteHazardKAN(tr.X.shape[1], tr.n_bins, mode=spec["mode"],
                          link=spec["link"], hidden=spec["hidden"], grid_size=8,
                          cont_idx=ci)
    t0 = time.time()
    m, hist = fit(m, tr, epochs=epochs, lr=0.03, val_frac=0.2, patience=60,
                  grid_update_epochs=(30, 100) if spec["mode"] != "linear" else (),
                  smooth=1e-3, seed=seed)
    secs = time.time() - t0
    Xte, mask, y = to_tensors(te)
    Xtr, mtr, ytr = to_tensors(tr)
    with torch.no_grad():
        loss = float(nll_fn(m(Xte), mask, y, m.link))
        # TRAIN NLL is the diagnostic that separates the two ways a flexible model
        # can fail to beat a linear one:
        #   train NLL ~ linear's  -> there is NO nonlinear signal to find, and the
        #                            KAN is correctly reporting that
        #   train NLL << linear's -> signal was fitted but did not generalise, i.e.
        #                            overfitting, and regularisation is the answer
        # Without it, a null test result is uninterpretable.
        train_loss = float(nll_fn(m(Xtr), mtr, ytr, m.link))
        S = m.survival(Xte)
    res = evaluate(S, te.bin_idx, te.event, nll=loss)
    res["train_nll"] = train_loss
    res["secs"] = secs
    res["epochs"] = hist["epochs_run"]
    return m, res


def run(name, loader, epochs=500, seed=0):
    d = loader()
    tr, te = split(d, seed=seed)
    print(f"\n{'='*94}")
    print(f"{d.summary()}")
    print(f"  train {tr.n} / test {te.n}   tie_ratio={d.meta.get('tie_ratio', float('nan')):.4f}")
    print(f"{'='*94}")
    print(f"  {'model':<20}{'train NLL':>11}{'test NLL':>11}{'C-index':>10}"
          f"{'IBS':>10}{'epochs':>8}{'secs':>8}")
    out = {}
    for label, spec in MODELS.items():
        try:
            m, r = fit_eval(spec, tr, te, epochs, seed)
            out[label] = (m, r)
            print(f"  {label:<20}{r['train_nll']:>11.4f}{r['nll']:>11.4f}"
                  f"{r['c_index']:>10.4f}{r['ibs']:>10.4f}{r['epochs']:>8}"
                  f"{r['secs']:>8.1f}")
        except Exception as e:
            print(f"  {label:<20}  FAILED {type(e).__name__}: {str(e)[:50]}")
    if "linear/cloglog" in out and "kan/cloglog" in out:
        lin, kan = out["linear/cloglog"][1], out["kan/cloglog"][1]
        dtr = lin["train_nll"] - kan["train_nll"]
        dte = lin["nll"] - kan["nll"]
        print()
        print(f"  DIAGNOSIS  KAN minus linear:  train {dtr:+.4f}   test {dte:+.4f}")
        if dtr < 0.005:
            print("    -> train fit barely improves: NO detectable nonlinear signal.")
            print("       The KAN is correctly finding nothing; this is a property of")
            print("       the data, not a tuning failure.")
        elif dte < 0:
            print("    -> train improves but test does not: OVERFITTING; needs")
            print("       stronger regularisation (l1/entropy or fewer knots).")
        else:
            print("    -> genuine nonlinear signal recovered.")
    return d, tr, te, out


def age_curve(d, model, n_points=60):
    """Partial effect of age -- the Nawata age-40 break, if it is there."""
    if "age" not in d.feature_names:
        return None
    j = d.feature_names.index("age")
    # 5-95 percentile: the earlier 1-99 range put the reported "steepest
    # change" at age 81 purely from the sparse upper tail.
    lo, hi = np.percentile(d.X[:, j], [5, 95])
    grid = torch.linspace(float(lo), float(hi), n_points)
    eff = model.partial_effect(torch.as_tensor(d.X), j, grid).numpy()
    return grid.numpy(), eff


def main():
    names = sys.argv[1:] or ["support2/slos"]
    avail = dict(D.LOADERS)
    avail["sparcs/drg302"] = lambda: D.load_sparcs("302")
    avail["drsa/music"] = lambda: D.load_drsa("MUSIC", max_rows=200000)

    for nm in names:
        if nm not in avail:
            print(f"unknown dataset {nm!r}; available: {sorted(avail)}")
            continue
        d, tr, te, out = run(nm, avail[nm])

        if nm == "support2/slos" and "kan/cloglog" in out:
            m, _ = out["kan/cloglog"]
            got = age_curve(d, m)
            if got is not None:
                grid, eff = got
                print(f"\n  AGE partial effect on the log-hazard of discharge")
                print(f"  (Nawata et al. report the age effect CHANGING AT 40)")
                print(f"    {'age':>7}{'effect':>11}")
                for a, e in zip(grid[::5], eff[::5]):
                    bar = "#" * max(0, int(round(18 + 18 * e)))
                    print(f"    {a:>7.1f}{e:>11.4f}  {bar}")
                slopes = np.gradient(eff, grid)
                kink = grid[1:-1][np.argmax(np.abs(np.diff(slopes[:-1])))]
                print(f"    steepest change in slope at age {kink:.1f}")


if __name__ == "__main__":
    main()
