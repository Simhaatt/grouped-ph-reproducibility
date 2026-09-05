"""Cox-family estimators and prediction protocols, factored out for E1-E3.

WHY THIS FILE EXISTS.  Until now the project had exactly one Cox arm: Efron
coefficients pushed through a Breslow baseline (experiments/formulation_ci.py ->
baselines.breslow_survival -> baselines.nll_from_survival).  Reading that chain
end to end shows something the manuscript never said out loud:

    S(t|x)   = exp(-H0(t) e^{b'x})
    h(t|x)   = 1 - S(t)/S(t-1) = 1 - exp(-dH0(t) e^{b'x})
             = 1 - exp(-exp( log dH0(t) + b'x ))

which is the grouped cloglog hazard with alpha_t = log dH0(t).  The Cox arm and
the grouped arm therefore have IDENTICAL functional form under that protocol, so
D_T measures ESTIMATION (partial-likelihood beta with a profiled Breslow alpha,
against joint MLE of (alpha, beta)) and NOT specification.  Nothing in the paper
told the reader which of the two was being measured.  This module makes both
axes explicit and sweepable:

  COEFFICIENTS   Breslow ties, Efron ties, exact-discrete conditional PL
  BASELINE       Breslow, Nelson-Aalen (globally centred), Kalbfleisch-Prentice

AND THE ABLATION SETTLES IT THE OTHER WAY.  The first draft of this file claimed
Kalbfleisch-Prentice would break the shared functional form, because it produces a
discrete baseline survival a_t with h(t|x) = 1 - a_t^{e^{b'x}}.  It does not:

    1 - a_t^{r}  =  1 - exp(r log a_t)  =  1 - exp(-exp( log(-log a_t) + b'x ))

so KP is ALSO a cloglog hazard, with alpha_t = log(-log a_t).  Worse (for that
claim) the KP defining equation

    sum_{i in D_t} r_i / (1 - a_t^{r_i})  =  sum_{i in R_t} r_i

is algebraically the profile score for alpha_t in the cloglog model with beta
held fixed -- differentiate the grouped log-likelihood in c_t = e^{alpha_t} and
the two equations coincide.  The KP protocol and the E3 arm-2 profile are the
same estimator, which is why they agree to five decimals in protocol_decomp.txt.

The conclusion is stronger than the ablation was designed to find: ALL THREE
Cox prediction protocols yield a grouped cloglog hazard, differing only in how
alpha_t is estimated (Breslow: moment; Nelson-Aalen: moment, covariate-blind;
KP: profile MLE).  D_T therefore measures ESTIMATION under every one of them,
never specification.  The only arms in this project that change the
specification are the logit-link ones -- the unconditional logit MLE and the
exact-discrete conditional PL, both of which target a log ODDS ratio.

Everything returns a per-test-row hazard matrix [N_te, T], so every arm is scored
through the one masked-Bernoulli NLL the rest of the project uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EPS = 1e-12


# ------------------------------------------------------------------ helpers
def risk(X, beta):
    return np.clip(np.exp(np.asarray(X, float) @ np.asarray(beta, float)), 1e-8, 1e8)


def _counts(bin_tr, ev_tr, n_bins):
    bin_tr = np.asarray(bin_tr).astype(int)
    ev_tr = np.asarray(ev_tr).astype(float)
    d = np.array([float(((bin_tr == t) & (ev_tr == 1)).sum()) for t in range(n_bins)])
    m = np.array([float((bin_tr >= t).sum()) for t in range(n_bins)])
    return bin_tr, ev_tr, d, m


# ------------------------------------------------------- coefficient arms
def fit_cox_ties(X, tidx, event, method):
    """Breslow / Efron partial likelihood.  Delegates to the project's own
    implementation so this file introduces no second Cox estimator."""
    from theory.verify_ties import fit_cox
    return fit_cox(np.asarray(X, float), np.asarray(tidx).astype(int),
                   np.asarray(event, float), method)


NEG = -1.0e30      # a finite stand-in for log 0; see _exact_log_pl


def _exact_log_pl(eta, order, block_ends, Kmax):
    """Log exact-discrete partial likelihood in ONE pass over the sample.

    The naive implementation runs the elementary-symmetric recursion separately
    for every risk set, which costs sum_t |R_t| sequential steps.  But the risk
    sets here are NESTED -- R_t = {i : k_i >= t} is a suffix once the sample is
    sorted by descending exit bin -- so a single accumulation visits each subject
    once and every R_t is a prefix of that accumulation.  Snapshotting log e_{d_t}
    at each block boundary therefore costs n steps in total rather than sum_t|R_t|,
    which on a 30-bin cohort is a ~15x reduction and is the difference between
    minutes and hours.

    The pad value is a finite NEG rather than -inf.  With -inf the very first
    steps evaluate logaddexp(-inf, -inf), whose derivative is exp(-inf + inf) =
    NaN, and the whole gradient is poisoned -- the first version of this function
    returned beta = [nan nan nan] for exactly that reason.  A finite NEG has
    derivative 1/2 there, and those cells are multiplied by exp(NEG - finite) = 0
    as soon as they meet a real value, so they never reach the answer.
    """
    L = torch.cat([eta.new_zeros(1), eta.new_full((Kmax,), NEG)])
    es = eta[order]
    out = eta.new_zeros(())
    ptr = 0
    for (end, d, died_ix) in zip(block_ends[0], block_ends[1], block_ends[2]):
        while ptr < end:
            L = torch.cat([L[:1], torch.logaddexp(L[1:], L[:-1] + es[ptr])])
            ptr += 1
        out = out + eta[died_ix].sum() - L[d]
    return out


def exact_discrete_cost(tidx, event, n_bins):
    """n * (max_t d_t + 1): the work the single-pass recursion must do.

    Reported so that "infeasible" is a number in the output rather than a silent
    omission.  The binding term is max_t d_t -- the modal event count -- which is
    precisely what the R2 cohorts make large, so the exact method is hardest to
    run exactly where the paper most wants it."""
    tidx = np.asarray(tidx).astype(int)
    event = np.asarray(event).astype(float)
    dmax = 0
    for t in range(n_bins):
        d = int(((tidx == t) & (event == 1)).sum())
        dmax = max(dmax, d)
    return float(len(tidx)) * (dmax + 1), dmax


def fit_cox_exact_discrete(X, tidx, event, max_iter=40, budget=4.0e6):
    """Cox's EXACT discrete partial likelihood (conditional logistic).

        L_t = exp(b' s_{D_t}) / e_{d_t}( { exp(b' x_i) : i in R_t } )

    This is the likelihood designed for intrinsically discrete time, and its
    absence was the single most exposed point in the paper.  It eliminates the
    baseline by CONDITIONING rather than by profiling, so it is the honest
    discrete counterpart of the partial likelihood -- not the same object as the
    unconditional logit MLE, which estimates T free alpha_t.

    Returns (beta, info).  info["feasible"] is False, and beta is None, when the
    recursion cost exceeds `budget`; the caller must then say so rather than drop
    the row.
    """
    X = np.asarray(X, float)
    tidx = np.asarray(tidx).astype(int)
    event = np.asarray(event, float)
    n_bins = int(tidx.max()) + 1
    cost, dmax = exact_discrete_cost(tidx, event, n_bins)
    info = {"cost": cost, "max_order": dmax, "feasible": cost <= budget,
            "n": int(len(tidx))}
    if not info["feasible"]:
        return None, info

    # Sort by DESCENDING exit bin so that every risk set is a prefix.
    order = np.argsort(-tidx, kind="stable")
    t_sorted = tidx[order]
    ends, ds, died_idx = [], [], []
    for t in sorted(set(tidx.tolist()), reverse=True):
        d = int(((tidx == t) & (event == 1)).sum())
        end = int(np.searchsorted(-t_sorted, -t, side="right"))
        if d == 0:
            continue
        ends.append(end)
        ds.append(d)
        died_idx.append(torch.as_tensor(np.flatnonzero((tidx == t) & (event == 1))))
    blocks = (ends, ds, died_idx)

    Xt = torch.as_tensor(X, dtype=torch.float64)
    order_t = torch.as_tensor(order)
    # Warm start from Efron.  The two likelihoods have the same argmin to first
    # order, so this cuts LBFGS from ~40 iterations to ~10 and is the difference
    # between this arm being runnable on 34 configurations and not.
    try:
        b0 = fit_cox_ties(X, tidx, event, "efron")
    except Exception:
        b0 = np.zeros(X.shape[1])
    b = torch.as_tensor(np.asarray(b0, float), dtype=torch.float64).clone()
    b.requires_grad_(True)
    opt = torch.optim.LBFGS([b], max_iter=max_iter, tolerance_grad=1e-8,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        eta = Xt @ b
        loss = -_exact_log_pl(eta, order_t, blocks, dmax)
        loss.backward()
        return loss

    opt.step(closure)
    out = b.detach().numpy()
    if not np.all(np.isfinite(out)):
        info["feasible"] = False
        info["failure"] = "non-finite"
        return None, info
    return out, info


def fit_grouped_joint(X, tidx, event, n_bins, link="cloglog", entry=None):
    """Joint MLE of (alpha, beta) -- the grouped model, linear index."""
    from kanrel.likelihood import make_targets, nll
    Xt = torch.as_tensor(np.asarray(X, float), dtype=torch.float64)
    mask, y = make_targets(tidx, event, n_bins, entry)
    mask, y = mask.double(), y.double()
    T, p = n_bins, Xt.shape[1]
    th = torch.zeros(T + p, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        th[:T] = -2.5
    opt = torch.optim.LBFGS([th], max_iter=300, tolerance_grad=1e-10,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = th[:T].unsqueeze(0) + (Xt @ th[T:]).unsqueeze(1)
        loss = nll(logits, mask, y, link, reduction="sum")
        loss.backward()
        return loss

    opt.step(closure)
    out = th.detach().numpy()
    return out[T:], out[:T]


def profile_alpha(X, tidx, event, n_bins, beta, link="cloglog", entry=None):
    """alpha_t by MLE with beta HELD FIXED -- arm 2 of the E3 decomposition.

    Arm 1 (Cox beta, Breslow alpha) -> arm 2 isolates the baseline
    representation, because only the way alpha is obtained changes.  Arm 2 ->
    arm 3 (joint MLE) then isolates coefficient estimation, because only beta
    changes.  Section 5.4 asserted this split; this measures it.
    """
    from kanrel.likelihood import make_targets, nll
    Xt = torch.as_tensor(np.asarray(X, float), dtype=torch.float64)
    off = Xt @ torch.as_tensor(np.asarray(beta, float), dtype=torch.float64)
    mask, y = make_targets(tidx, event, n_bins, entry)
    mask, y = mask.double(), y.double()
    a = torch.full((n_bins,), -2.5, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([a], max_iter=300, tolerance_grad=1e-10,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nll(a.unsqueeze(0) + off.unsqueeze(1), mask, y, link, reduction="sum")
        loss.backward()
        return loss

    opt.step(closure)
    return a.detach().numpy()


# ----------------------------------------------------------- baseline arms
def hazards_breslow(r_tr, bin_tr, ev_tr, r_te, n_bins):
    """h_t(x) = 1 - exp(-dH0_t r_x),  dH0_t = d_t / sum_{R_t} r_i."""
    bin_tr, ev_tr, d, _ = _counts(bin_tr, ev_tr, n_bins)
    dH0 = np.zeros(n_bins)
    for t in range(n_bins):
        den = r_tr[bin_tr >= t].sum()
        dH0[t] = d[t] / den if den > EPS else 0.0
    return np.clip(1.0 - np.exp(-np.outer(r_te, dH0)), EPS, 1 - EPS)


def hazards_nelson_aalen(r_tr, bin_tr, ev_tr, r_te, n_bins):
    """Marginal Nelson-Aalen increments, rescaled by a GLOBAL mean risk.

    dLambda_t = d_t / |R_t| ignores the covariates entirely; the baseline is then
    recovered as dH0_t = dLambda_t / rbar with rbar the training mean risk.
    Using the per-risk-set mean instead would reproduce Breslow EXACTLY
    (dH0 = d_t/(|R_t| rbar_t)), which is why the global constant is the version
    that actually differs -- it ignores the drift of mean risk through time that
    Breslow tracks."""
    bin_tr, ev_tr, d, m = _counts(bin_tr, ev_tr, n_bins)
    rbar = float(np.mean(r_tr))
    dH0 = np.where(m > 0, d / np.maximum(m, 1.0), 0.0) / max(rbar, EPS)
    return np.clip(1.0 - np.exp(-np.outer(r_te, dH0)), EPS, 1 - EPS)


def hazards_kalbfleisch_prentice(r_tr, bin_tr, ev_tr, r_te, n_bins):
    """Kalbfleisch-Prentice discrete baseline: h_t(x) = 1 - a_t^{r_x}.

    a_t in (0,1] solves  sum_{i in D_t} r_i / (1 - a_t^{r_i}) = sum_{i in R_t} r_i.

    That equation IS the profile score for alpha_t in the grouped cloglog model
    with beta held fixed, and 1 - a_t^r is a cloglog hazard with
    alpha_t = log(-log a_t).  So this protocol does not give the Cox arm a
    different model -- it gives it the best alpha_t the shared model admits, and
    it coincides with profile_alpha() below to machine precision.  That identity
    is asserted in smoke_cox_arms.py, because two routes to one quantity is the
    cheapest validation available for either of them.  Solved by bisection on
    log a_t.
    """
    bin_tr = np.asarray(bin_tr).astype(int)
    ev_tr = np.asarray(ev_tr).astype(float)
    a = np.ones(n_bins)
    for t in range(n_bins):
        died = (bin_tr == t) & (ev_tr == 1)
        if not died.any():
            continue
        rd = r_tr[died]
        R = float(r_tr[bin_tr >= t].sum())
        if R <= EPS:
            continue
        lo, hi = -50.0, -1e-12                    # log a in (-inf, 0)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            am = np.exp(mid)
            g = float((rd / np.maximum(1.0 - am ** rd, EPS)).sum()) - R
            # g is INCREASING in a: as a -> 0 it tends to sum_D r - R < 0, and as
            # a -> 1 the denominators vanish and it tends to +inf.  So g > 0 means
            # a_t is too LARGE and the root lies below mid.  The first version had
            # this branch inverted -- the fourth inverted verdict predicate in this
            # project -- and drove every a_t to the clip, which showed up as a test
            # NLL of 88 against 1.7 for the other two baselines.
            if g > 0:
                hi = mid
            else:
                lo = mid
        a[t] = float(np.exp(0.5 * (lo + hi)))
    a = np.clip(a, EPS, 1.0)
    h = 1.0 - np.power(a[None, :], r_te[:, None])
    return np.clip(h, EPS, 1 - EPS)


def hazards_from_alpha(X_te, beta, alpha, link="cloglog"):
    eta = np.asarray(X_te, float) @ np.asarray(beta, float)
    logits = alpha[None, :] + eta[:, None]
    if link == "logit":
        return np.clip(1.0 / (1.0 + np.exp(-logits)), EPS, 1 - EPS)
    return np.clip(-np.expm1(-np.exp(np.clip(logits, -30, 10))), EPS, 1 - EPS)


# ------------------------------------------------------------------ scoring
def nll_from_hazards(h, bin_idx, event, n_bins, reduction=None):
    """The one scoring function.  Per-unit so every comparison stays paired."""
    from kanrel.likelihood import make_targets
    h = np.clip(np.asarray(h, float), EPS, 1 - EPS)
    mask, y = make_targets(bin_idx, event, n_bins)
    mask, y = mask.numpy(), y.numpy()
    ll = y * np.log(h) + (1 - y) * np.log(1 - h)
    per = -(ll * mask).sum(axis=1)
    return per if reduction is None else float(per.mean())


BASELINES = {
    "breslow": hazards_breslow,
    "nelson-aalen": hazards_nelson_aalen,
    "kalbfleisch-prentice": hazards_kalbfleisch_prentice,
}
