"""Symbolic formula extraction, and what substituting the formula COSTS.

Symbolic extraction is CoxKAN's signature feature and the longest-standing gap
in this project (status.md section 9.3).  A referee will ask why it is missing.

What the incumbent literature does: fit a KAN, replace each edge function by the
closed form that best matches it, print the formula.  What it does NOT do is
report what that substitution costs in predictive terms -- and a formula that
looks like the curve but predicts worse is a picture, not a model.

So this script has two halves.

  HALF 1 -- EXTRACTION.  For each genuinely continuous covariate, take the
  identifiable partial effect (hazard.partial_effect: one covariate swept with
  the rest at their median -- NOT a raw edge plot, which is not identifiable in
  a depth->=2 net) and fit a library of closed forms  a*f(b*x + c) + d  by
  least squares.  Rank by BIC so that a 4-parameter fit has to earn its extra
  parameters, and report the top candidates with R^2.

  HALF 2 -- THE COST, which is the part that makes this a result.  Substitute
  every winning formula into the design matrix and refit a LINEAR model on the
  transformed covariates.  That surrogate is fully symbolic: it is a closed-form
  log-hazard, no splines.  Score it on the same test rows as the KAN and the
  untransformed linear baseline.  The number that matters is the retention

      retention = (C_symbolic - C_linear) / (C_KAN - C_linear)

  i.e. what fraction of the KAN's discrimination gain survives symbolisation.

This inherits every correction the pipeline has already absorbed: one-hot
ordinals for every model (section 4.7), winsorising to the training range
(section 4.8b), and the l1 selected in regularization.py.  Without those the
comparison would repeat the error section 4.7 documents.

Run:  python -u experiments/symbolic.py [dataset ...]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from kanrel import data as D
from kanrel.fit import fit, to_tensors
from kanrel.hazard import DiscreteHazardKAN
from kanrel.likelihood import nll as nll_fn
from kanrel.metrics import evaluate
from experiments.baselines import SELECTED_L1
from experiments.real_data import continuous_columns, split

N_GRID = 200          # points at which the partial effect is sampled
SEEDS = (0, 1, 2, 3, 4)   # 5 splits: retention is a ratio, so it needs the spread


# --------------------------------------------------------------------- library
# Each entry: name, callable, n_free_shape_params, pretty-printer.
# The affine wrapper a*f(b*x + c) + d is common to all of them, so the only
# difference in complexity between candidates is the inner function itself --
# which is why BIC is computed on a common parameter count plus a small
# complexity charge that prefers a polynomial to a transcendental at equal fit.
def _safe(f):
    def g(z):
        with np.errstate(all="ignore"):
            out = f(z)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return g


LIBRARY = [
    ("x",           _safe(lambda z: z),                       0),
    ("x^2",         _safe(lambda z: z ** 2),                  1),
    ("x^3",         _safe(lambda z: z ** 3),                  1),
    ("sqrt|x|",     _safe(lambda z: np.sqrt(np.abs(z))),      1),
    ("|x|",         _safe(lambda z: np.abs(z)),               1),
    ("log(1+|x|)",  _safe(lambda z: np.log1p(np.abs(z))),     1),
    ("exp",         _safe(np.exp),                            2),
    ("tanh",        _safe(np.tanh),                           2),
    ("sin",         _safe(np.sin),                            2),
    ("1/(1+x^2)",   _safe(lambda z: 1.0 / (1.0 + z ** 2)),    2),
    ("sigmoid",     _safe(lambda z: 1.0 / (1.0 + np.exp(-z))), 2),
]


def _fit_one(f, x, y):
    """Least squares for a*f(b*x + c) + d, with restarts.

    curve_fit from a single start is unreliable on the transcendentals (exp and
    sin both have flat regions where the Jacobian vanishes), so several starts
    are tried and the best SSE kept.  This is the same failure mode as
    methodological lesson 10: a bad optimum masquerades as a bad basis.
    """
    def model(z, a, b, c, d):
        return a * f(b * z + c) + d

    best = None
    sx = np.std(x) or 1.0
    for b0 in (1.0 / sx, 2.0 / sx, 0.5 / sx, -1.0 / sx):
        for c0 in (0.0, -np.mean(x) * b0):
            try:
                p, _ = curve_fit(model, x, y, p0=[np.std(y) or 1.0, b0, c0, np.mean(y)],
                                 maxfev=6000)
            except Exception:
                continue
            sse = float(np.sum((y - model(x, *p)) ** 2))
            if np.isfinite(sse) and (best is None or sse < best[0]):
                best = (sse, p)
    return best


def suggest(x, y, top=3):
    """Rank the library against one recovered curve. -> [(name, r2, bic, params)]"""
    n = len(x)
    tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    out = []
    for name, f, extra in LIBRARY:
        got = _fit_one(f, x, y)
        if got is None:
            continue
        sse, p = got
        r2 = 1.0 - sse / tot
        # 4 affine parameters for every candidate, plus a complexity charge so
        # that a transcendental must beat a polynomial by more than noise.
        k = 4 + extra
        bic = n * np.log(max(sse, 1e-300) / n) + k * np.log(n)
        out.append((name, r2, bic, p))
    out.sort(key=lambda r: r[2])
    return out[:top]


def apply_formula(name, p, z):
    f = dict((nm, fn) for nm, fn, _ in LIBRARY)[name]
    a, b, c, d = p
    return a * f(b * z + c) + d


# --------------------------------------------------------------------- models
def fit_ours(tr, mode, l1=0.0, seed=0, cont_idx=None):
    m = DiscreteHazardKAN(tr.X.shape[1], tr.n_bins, mode=mode, link="cloglog",
                          hidden=(), grid_size=8,
                          cont_idx=cont_idx if mode == "baseline" else None)
    m, _ = fit(m, tr, epochs=400, lr=0.03, val_frac=0.2, patience=60,
               grid_update_epochs=(30, 100) if mode != "linear" else (),
               l1=l1, smooth=1e-3, seed=seed)
    return m


def score(m, te):
    X, mask, y = to_tensors(te)
    with torch.no_grad():
        loss = float(nll_fn(m(X), mask, y, m.link))
        S = m.survival(X).numpy()
    return evaluate(S, te.bin_idx, te.event, nll=loss)


def transformed_copy(d, cont, formulas):
    """A SurvData whose continuous columns carry the fitted closed forms.

    Only the continuous columns move; dummies pass through untouched, so the
    surrogate and the linear baseline differ in exactly one respect -- whether
    the continuous covariates enter through their symbolic transform.
    """
    X = d.X.copy()
    for j, (name, p) in zip(cont, formulas):
        X[:, j] = apply_formula(name, p, d.X[:, j]).astype(np.float32)
    return D.SurvData(X, d.bin_idx, d.event, d.n_bins, d.feature_names,
                      name=d.name, intrinsically_discrete=d.intrinsically_discrete,
                      entry_idx=d.entry_idx, bin_edges=d.bin_edges, meta=d.meta)


# ----------------------------------------------------------------------- main
def run(name, loader):
    d = D.onehot_ordinals(loader())          # fair baseline, section 4.7
    l1 = SELECTED_L1.get(name, 0.01)
    print("=" * 96)
    print(f"SYMBOLIC EXTRACTION -- {name}   n={d.n}  T={d.n_bins}  l1={l1:g}")
    print("=" * 96)

    rows = {k: {"nll": [], "c": []} for k in ("linear", "KAN", "symbolic")}
    picked = {}

    for seed in SEEDS:
        tr, te = split(d, seed=seed)
        tr, te = D.clip_to_train_range(tr, te)      # winsorise, section 4.8b
        cont = continuous_columns(tr.X)

        m_lin = fit_ours(tr, "linear", seed=seed)
        m_kan = fit_ours(tr, "baseline", l1=l1, seed=seed, cont_idx=cont)

        # --- HALF 1: read a closed form off each recovered partial effect -----
        formulas = []
        for j in cont:
            lo, hi = np.percentile(tr.X[:, j], [1, 99])
            if not np.isfinite(lo) or hi - lo < 1e-9:
                formulas.append(("x", np.array([0.0, 1.0, 0.0, 0.0])))
                continue
            grid = np.linspace(lo, hi, N_GRID).astype(np.float32)
            with torch.no_grad():
                curve = m_kan.partial_effect(torch.as_tensor(tr.X), j,
                                             torch.as_tensor(grid)).numpy()
            cands = suggest(grid, curve)
            if not cands:
                formulas.append(("x", np.array([0.0, 1.0, 0.0, 0.0])))
                continue
            best = cands[0]
            formulas.append((best[0], best[3]))
            if seed == SEEDS[0]:
                picked[j] = (d.feature_names[j], cands)

        # --- HALF 2: what does substituting them cost? -----------------------
        tr_s = transformed_copy(tr, cont, formulas)
        te_s = transformed_copy(te, cont, formulas)
        m_sym = fit_ours(tr_s, "linear", seed=seed)

        for key, m, tset in (("linear", m_lin, te), ("KAN", m_kan, te),
                             ("symbolic", m_sym, te_s)):
            r = score(m, tset)
            rows[key]["nll"].append(r["nll"])
            rows[key]["c"].append(r["c_index"])

    # ------------------------------------------------------------- reporting
    print("\n  RECOVERED FORMULAE (seed 0; ranked by BIC, top 3 shown)")
    print(f"  {'covariate':<18}{'best form':>14}{'R^2':>8}   runners-up")
    for j, (fname, cands) in picked.items():
        alt = ", ".join(f"{c[0]} ({c[1]:.2f})" for c in cands[1:])
        print(f"  {fname:<18}{cands[0][0]:>14}{cands[0][1]:>8.3f}   {alt}")

    print(f"\n  PREDICTIVE COST OF SYMBOLISATION  ({len(SEEDS)} seeds)")
    print(f"  {'model':<24}{'test NLL':>18}{'C-index':>18}")
    for key in ("linear", "KAN", "symbolic"):
        nl = np.array(rows[key]["nll"])
        cc = np.array(rows[key]["c"])
        print(f"  {key:<24}{nl.mean():>10.4f}+/-{nl.std(ddof=1):<7.4f}"
              f"{cc.mean():>10.4f}+/-{cc.std(ddof=1):<7.4f}")

    lin = np.array(rows["linear"]["c"])
    kan = np.array(rows["KAN"]["c"])
    sym = np.array(rows["symbolic"]["c"])
    gain, sgain = kan - lin, sym - lin
    print(f"\n  KAN gain over linear      dC = {gain.mean():+.4f}"
          f" +/- {gain.std(ddof=1):.4f}")
    print(f"  symbolic gain over linear dC = {sgain.mean():+.4f}"
          f" +/- {sgain.std(ddof=1):.4f}")

    # THE COST OF SYMBOLISATION IS A DIFFERENCE, NOT A RATIO.
    #
    # The natural-sounding statistic, retention = (C_sym - C_lin)/(C_KAN - C_lin),
    # is unusable here and the first version of this script used it anyway.  Its
    # denominator is an effect of ~0.02 with a per-split sd of ~0.015, so on
    # rotgbsg the five splits returned -58%, 125%, 95%, 462%, 92% -- a mean of
    # 143% +/- 192% that says nothing.  That is exactly the `tail_share` failure
    # section 4.3d already documented (-864% on metabric) and section 4.3i replaced
    # with a denominator-free diagnostic.  Same error, second occasion; see
    # methodological lesson 16 -- fix the CLASS of statistic, not one script.
    #
    # C_KAN - C_symbolic is paired, well conditioned, and answers the question
    # directly: how much discrimination does writing the model in closed form
    # cost?  Zero means symbolisation is free.
    cost = kan - sym
    m, s = cost.mean(), cost.std(ddof=1)
    se = s / np.sqrt(len(cost))
    print(f"\n  COST OF SYMBOLISATION   C_KAN - C_symbolic = {m:+.4f} +/- {s:.4f}"
          f"   (SE {se:.4f})")
    print(f"    per split: {', '.join(f'{c:+.4f}' for c in cost)}")
    # 1.96, not 2.802.  2.802 = z_{0.975} + z_{0.80} is the multiplier for a
    # MINIMUM DETECTABLE EFFECT at 80% power, which answers "how large would a
    # difference have to be for this dataset to find it reliably".  It is not a
    # critical value, and using it to decide whether an observed difference is
    # non-zero makes the test conservative at roughly the 0.5% level rather than
    # the 5% level.  Two scripts here did exactly that.
    if abs(m) < 1.959964 * se:
        print("    -> NOT distinguishable from zero at this split count:")
        print("       the closed-form model matches the KAN's discrimination.")
    elif m > 0:
        print("    -> symbolisation costs discrimination: the formulae describe")
        print("       the fitted curves without carrying their predictive content.")
    else:
        print("    -> the symbolic surrogate RANKS BETTER than the KAN it was read")
        print("       off, which means the spline flexibility was not buying the gain.")
    print(f"    for scale, the KAN's own gain over linear is {gain.mean():+.4f}"
          f" +/- {gain.std(ddof=1):.4f}")
    return rows


def main():
    names = sys.argv[1:] or ["rotgbsg", "support-pycox", "metabric"]
    avail = dict(D.LOADERS)
    for nm in names:
        if nm not in avail:
            print(f"unknown dataset {nm!r}; known: {sorted(avail)}")
            continue
        run(nm, avail[nm])
        print()


if __name__ == "__main__":
    main()
