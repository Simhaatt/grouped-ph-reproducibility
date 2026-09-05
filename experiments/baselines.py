"""Head-to-head against the published CoxKAN, on identical splits and metrics.

Every earlier real-data comparison was against a LINEAR discrete hazard.  That is
the wrong reference point for a paper whose framing is that the KAN survival
literature models time incorrectly: the obvious referee question is "did you
actually run CoxKAN?".  This does.

Fairness requires care, because the two model families emit different objects:

  ours     an [N, T] logit matrix -> discrete hazards directly
  CoxKAN   a scalar partial hazard per subject, with NO baseline (that is the
           point of the partial likelihood)

So CoxKAN's risk scores are turned into survival curves ON OUR BIN GRID with a
Breslow baseline estimated from the TRAINING split:

    dH0(t) = d_t / sum_{i in R(t)} r_i ,   S(t|x) = exp(-H0(t) * r_x)

and the resulting curves are scored with the SAME Antolini C, IPCW Brier and
masked-Bernoulli NLL as our own models.  Without this, "NLL" would mean partial
log-likelihood for one family and full likelihood for the other, and the numbers
would not be comparable at all.

Also fixed here: every previous real-data number used ONE split at seed 0, so the
bootstrap CIs captured test-set sampling but not training variability.  This runs
REPEATED SPLITS and reports mean +/- sd across them.

Run:  python experiments/baselines.py [dataset ...]
"""
from __future__ import annotations

import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from kanrel import data as D
from kanrel.fit import fit, to_tensors
from kanrel.hazard import DiscreteHazardKAN
from kanrel.likelihood import make_targets
from kanrel.likelihood import nll as nll_fn
from kanrel.metrics import evaluate
from experiments.real_data import continuous_columns, split

N_SPLITS = 5


# ------------------------------------------------------------------ Breslow
def breslow_survival(risk_tr, bin_tr, ev_tr, risk_te, n_bins):
    """Breslow baseline from the training split -> [N_te, T] survival curves.

    dH0(t) = d_t / sum_{i at risk in t} r_i ,  S(t|x) = exp(-H0(t) r_x)
    """
    risk_tr = np.asarray(risk_tr, dtype=float).ravel()
    risk_te = np.asarray(risk_te, dtype=float).ravel()
    bin_tr = np.asarray(bin_tr).astype(int)
    ev_tr = np.asarray(ev_tr).astype(float)

    dH0 = np.zeros(n_bins)
    for t in range(n_bins):
        at_risk = bin_tr >= t
        denom = risk_tr[at_risk].sum()
        d_t = float(((bin_tr == t) & (ev_tr == 1)).sum())
        dH0[t] = d_t / denom if denom > 1e-12 else 0.0
    H0 = np.cumsum(dH0)
    S = np.exp(-np.outer(risk_te, H0))
    return np.clip(S, 1e-12, 1.0)


def nll_from_survival(S, bin_idx, event, n_bins, reduction="mean"):
    """Masked-Bernoulli NLL from a survival matrix, so Cox-family models are
    scored on exactly the same quantity as ours.

    reduction=None returns the PER-UNIT vector (whose mean is the scalar), which
    is what a paired bootstrap of the formulation effect needs -- see
    experiments/formulation_ci.py."""
    S = np.clip(np.asarray(S), 1e-12, 1 - 1e-12)
    prev = np.concatenate([np.ones((S.shape[0], 1)), S[:, :-1]], axis=1)
    h = np.clip(1.0 - S / prev, 1e-12, 1 - 1e-12)
    mask, y = make_targets(bin_idx, event, n_bins)
    mask, y = mask.numpy(), y.numpy()
    ll = y * np.log(h) + (1 - y) * np.log(1 - h)
    per_unit = -(ll * mask).sum(axis=1)
    return per_unit if reduction is None else float(per_unit.mean())


def standardize(tr, te):
    """Standardise covariates using TRAINING statistics only.

    `run_ours` standardises inside the model, but the Cox-family baselines were
    receiving RAW features.  On flchain that is fatal: `sample.yr` is ~1997, so
    exp(X @ beta) overflows, the risk scores clip to a constant, and BOTH Cox
    models reported C-index exactly 0.5000 +/- 0.0000 -- a collapsed fit reported
    as if it were a result.  Comparing a standardised model against unstandardised
    baselines is not a fair comparison, so this is applied to every model.
    """
    mu = tr.X.mean(axis=0)
    sd = tr.X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    from kanrel.data import SurvData

    def z(d):
        return SurvData(((d.X - mu) / sd).astype(np.float32), d.bin_idx, d.event,
                        d.n_bins, d.feature_names, name=d.name,
                        intrinsically_discrete=d.intrinsically_discrete,
                        entry_idx=d.entry_idx, bin_edges=d.bin_edges, meta=d.meta)
    return z(tr), z(te)


# ------------------------------------------------------------------- models
def run_ours(tr, te, mode, l1=0.0, seed=0):
    ci = continuous_columns(tr.X) if mode == "baseline" else None
    m = DiscreteHazardKAN(tr.X.shape[1], tr.n_bins, mode=mode, link="cloglog",
                          hidden=(), grid_size=8, cont_idx=ci)
    m, _ = fit(m, tr, epochs=400, lr=0.03, val_frac=0.2, patience=60,
               grid_update_epochs=(30, 100) if mode != "linear" else (),
               l1=l1, smooth=1e-3, seed=seed)
    X, mask, y = to_tensors(te)
    with torch.no_grad():
        loss = float(nll_fn(m(X), mask, y, m.link))
        S = m.survival(X).numpy()
    return evaluate(S, te.bin_idx, te.event, nll=loss)


def run_coxkan(tr, te, seed=0, steps=400):
    """The published CoxKAN, scored on our grid via a Breslow baseline."""
    from coxkan import CoxKAN

    tr, te = standardize(tr, te)
    cols = [c.replace(".", "_").replace(" ", "_") for c in tr.feature_names]

    def frame(d):
        f = pd.DataFrame(d.X, columns=cols)
        f["duration"] = d.bin_idx.astype(float) + 1.0
        f["event"] = d.event.astype(float)
        return f

    ftr, fte = frame(tr), frame(te)
    n_val = max(20, int(0.2 * len(ftr)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ftr))
    sub_tr = ftr.iloc[perm[n_val:]].reset_index(drop=True)
    sub_va = ftr.iloc[perm[:n_val]].reset_index(drop=True)

    ck = CoxKAN(width=[len(cols), 1], grid=8, k=3, seed=seed)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ck.train(sub_tr, sub_va, duration_col="duration", event_col="event",
                 steps=steps, lr=0.01, log=10**9, lamb=0.0)
        r_tr = np.asarray(ck.predict_partial_hazard(ftr)).ravel()
        r_te = np.asarray(ck.predict_partial_hazard(fte)).ravel()

    r_tr = np.clip(r_tr, 1e-8, 1e8)
    r_te = np.clip(r_te, 1e-8, 1e8)
    S = breslow_survival(r_tr, tr.bin_idx, tr.event, r_te, tr.n_bins)
    loss = nll_from_survival(S, te.bin_idx, te.event, tr.n_bins)
    return evaluate(S, te.bin_idx, te.event, nll=loss)


def run_cox_linear(tr, te, seed=0, method="efron"):
    """CONTROL: plain linear Cox (partial likelihood + Breslow), same pipeline.

    CoxKAN's scores reach our metrics through a Breslow baseline WE implemented.
    If that conversion were lossy it would penalise CoxKAN for our code rather
    than for the method.  A linear Cox pushed through the identical path is the
    check: it should land close to our own linear cloglog model, because
    Prentice-Gloeckler grouping of a continuous PH model is exact (Lemma 1).
    A large gap here would indict the conversion, not CoxKAN.
    """
    from theory.verify_ties import fit_cox

    tr, te = standardize(tr, te)
    beta = fit_cox(tr.X.astype(float), tr.bin_idx, tr.event.astype(float), method)
    r_tr = np.clip(np.exp(tr.X.astype(float) @ beta), 1e-8, 1e8)
    r_te = np.clip(np.exp(te.X.astype(float) @ beta), 1e-8, 1e8)
    S = breslow_survival(r_tr, tr.bin_idx, tr.event, r_te, tr.n_bins)
    loss = nll_from_survival(S, te.bin_idx, te.event, tr.n_bins)
    return evaluate(S, te.bin_idx, te.event, nll=loss)


# Per-dataset l1 from the CV/holdout selection in experiments/regularization.py.
# A previous version hard-coded l1=0.01 everywhere, which is simply the wrong
# model on the small cohorts -- valung's CV choice is 0.1, and at 0.01 the fit is
# unstable across splits (NLL 5.69 +/- 5.28).  Repeated splits exposed that; a
# single split had hidden it.
SELECTED_L1 = {
    "pbc": 0.1, "gbsg": 0.1, "valung": 0.1, "prostate": 0.3,        # 5-fold CV
    "support2/slos": 0.01, "sparcs/drg302": 0.0, "drsa/music": 0.0,  # holdout
    # KAPLAN-HR's benchmark set.  All n < 10^4, i.e. inside the range our own
    # bootstrap showed is not resolvable, so l1 follows the small-n default
    # rather than being tuned per dataset on results we cannot distinguish.
    "metabric": 0.01, "support-pycox": 0.01, "rotgbsg": 0.01,
    "flchain": 0.01, "nwtco": 0.01,
}


def models_for(dataset_name):
    l1 = SELECTED_L1.get(dataset_name, 0.01)
    return {
        "ours/linear-cloglog": lambda tr, te, s: run_ours(tr, te, "linear", seed=s),
        "ours/KAN-cloglog": lambda tr, te, s: run_ours(tr, te, "baseline", l1=l1, seed=s),
        "CoxKAN (published)": lambda tr, te, s: run_coxkan(tr, te, seed=s),
        "Cox-linear+Breslow*": lambda tr, te, s: run_cox_linear(tr, te, seed=s),
    }


def run(name, loader, n_splits=N_SPLITS):
    d = loader()
    # FAIR BASELINE: expand low-cardinality ordinals to dummies for EVERY model.
    # Without this the linear comparator codes e.g. cancer as 0/1/2 (equal
    # spacing) and the all-spline KAN is credited for fixing the encoding --
    # inflating its measured advantage ~3x on support-pycox.
    d = D.onehot_ordinals(d)
    MODELS = models_for(name)
    print(f"\n{'='*92}")
    print(f"{d.summary()}")
    print(f"  {n_splits} repeated splits -- mean +/- sd ACROSS splits (not within one)")
    print(f"  l1 = {SELECTED_L1.get(name, 0.01):g} (selected in regularization.py)")
    print(f"  * Cox-linear+Breslow is the CONTROL: it shows what our Breslow")
    print(f"    conversion itself costs, so CoxKAN is not charged for our pipeline.")
    print(f"{'='*92}")
    acc = {k: {"nll": [], "c_index": [], "ibs": []} for k in MODELS}
    for s in range(n_splits):
        tr, te = split(d, seed=s)
        for label, fn in MODELS.items():
            try:
                r = fn(tr, te, s)
                for k in ("nll", "c_index", "ibs"):
                    acc[label][k].append(r[k])
            except Exception as e:
                print(f"    split {s} {label}: FAILED {type(e).__name__}: {str(e)[:44]}")

    print(f"  {'model':<22}{'test NLL':>18}{'C-index':>18}{'IBS':>18}")
    for label in MODELS:
        a = acc[label]
        if not a["nll"]:
            print(f"  {label:<22}{'--':>18}")
            continue
        def ms(k):
            v = np.array(a[k])
            return f"{v.mean():.4f}+/-{v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():.4f}"
        print(f"  {label:<22}{ms('nll'):>18}{ms('c_index'):>18}{ms('ibs'):>18}")
    return acc


def main():
    names = sys.argv[1:] or ["gbsg", "pbc", "support2/slos"]
    avail = dict(D.LOADERS)
    avail["sparcs/drg302"] = lambda: D.load_sparcs("302")
    avail["drsa/music"] = lambda: D.load_drsa("MUSIC", max_rows=200000)
    for nm in names:
        if nm in avail:
            run(nm, avail[nm])
        else:
            print(f"unknown dataset {nm!r}")


if __name__ == "__main__":
    main()
