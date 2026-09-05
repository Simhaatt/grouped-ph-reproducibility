"""Smoke test for cox_arms: does the refactor reproduce the existing pipeline?

Three things must hold before any of E1-E3 is worth running.

  1. hazards_breslow + nll_from_hazards must equal breslow_survival +
     nll_from_survival to machine precision.  If it does not, every D_T in the
     new tables is a different quantity from every D_T in the old ones and the
     two cannot be compared.
  2. The exact-discrete conditional PL must recover a known beta on a small
     simulation where it is the correct likelihood.
  3. The three baselines must actually differ from each other.  A "protocol
     ablation" whose arms coincide measures nothing.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from experiments import cox_arms as CA
from experiments.baselines import breslow_survival, nll_from_survival


def sim_logistic_discrete(n, T, beta, seed=0, a0=-2.0):
    """Cox's 1972 DISCRETE model: h_t(x)/(1-h_t(x)) = (h0_t/(1-h0_t)) e^{b'x}.

    WHICH LAW MAKES THE EXACT METHOD CORRECT.  The exact-discrete conditional
    likelihood -- the one built from elementary symmetric polynomials, and what
    R's ties="exact" and SAS's TIES=DISCRETE maximise -- is the conditional law
    of the failure SET under the LOGISTIC discrete model above.  It estimates a
    log ODDS ratio.

    It is NOT the correct likelihood for (1 - h_t(x)) = (1 - h0_t)^{exp(b'x)}.
    That relation rearranges to cloglog(h_t) = alpha_t + b'x -- it IS the grouped
    Prentice-Gloeckler model this paper fits, not a rival to it.  Any falsification
    design must therefore use the logistic law here, and the first version of this
    smoke test simulated the cloglog law by mistake and reported 16.6% "bias" in
    an estimator that was simply targeting a different parameter.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(beta)))
    eta = X @ beta
    idx = np.full(n, T - 1, dtype=int)
    ev = np.zeros(n)
    alive = np.ones(n, bool)
    for t in range(T):
        h = 1.0 / (1.0 + np.exp(-(a0 + eta)))
        die = alive & (rng.uniform(size=n) < h)
        idx[die] = t
        ev[die] = 1.0
        alive &= ~die
    idx[alive] = T - 1                      # grid-aligned administrative censoring
    return X, idx, ev


def main():
    rng = np.random.default_rng(0)
    n, T, p = 4000, 8, 3
    beta_true = np.array([0.7, -0.5, 0.3])
    X, idx, ev = sim_logistic_discrete(n, T, beta_true, seed=1)
    n_tr = 2800
    Xtr, itr, etr = X[:n_tr], idx[:n_tr], ev[:n_tr]
    Xte, ite, ete = X[n_tr:], idx[n_tr:], ev[n_tr:]

    print("=" * 88)
    print("SMOKE TEST 1: refactored Breslow path == existing baselines.py path")
    print("=" * 88)
    b = CA.fit_cox_ties(Xtr, itr, etr, "efron")
    r_tr, r_te = CA.risk(Xtr, b), CA.risk(Xte, b)
    old = nll_from_survival(breslow_survival(r_tr, itr, etr, r_te, T), ite, ete, T,
                            reduction=None)
    new = CA.nll_from_hazards(CA.hazards_breslow(r_tr, itr, etr, r_te, T), ite, ete, T)
    gap = float(np.max(np.abs(old - new)))
    print(f"  max |old - new| per unit = {gap:.3e}   -> {'MATCH' if gap < 1e-9 else 'DIFFER'}")

    print()
    print("=" * 88)
    print("SMOKE TEST 2: exact discrete PL recovers beta where it is correct")
    print("=" * 88)
    cost, dmax = CA.exact_discrete_cost(itr, etr, T)
    print(f"  recursion cost = {cost:,.0f} vector-steps, max order = {dmax}")
    t0 = time.time()
    be, info = CA.fit_cox_exact_discrete(Xtr, itr, etr, budget=1e9)
    el = time.time() - t0
    print(f"  fit took {el:.1f}s   feasible={info['feasible']}")
    if be is not None:
        for nm, est in (("exact-discrete", be),
                        ("Cox/Efron", CA.fit_cox_ties(Xtr, itr, etr, "efron")),
                        ("Cox/Breslow", CA.fit_cox_ties(Xtr, itr, etr, "breslow"))):
            rel = float(np.mean(np.abs(est / beta_true - 1)))
            print(f"  {nm:<16} {np.array2string(est, precision=3)}"
                  f"   mean |rel bias| = {100*rel:5.2f}%")

    print()
    print("=" * 88)
    print("SMOKE TEST 3: the three baselines are genuinely different")
    print("=" * 88)
    hs = {k: f(r_tr, itr, etr, r_te, T) for k, f in CA.BASELINES.items()}
    keys = list(hs)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = float(np.max(np.abs(hs[keys[i]] - hs[keys[j]])))
            print(f"  max |h_{keys[i]} - h_{keys[j]}| = {d:.5f}")
    for k, h in hs.items():
        print(f"  test NLL under {k:<22} = "
              f"{CA.nll_from_hazards(h, ite, ete, T, reduction='mean'):.5f}")

    # The KP solver has a closed form at d_t = 1: a_t = (1 - r_i/R)^{1/r_i}.
    # Checking the bisection against it is what catches an inverted branch, and
    # the first version of the solver failed exactly here.
    rng2 = np.random.default_rng(7)
    rr = np.exp(rng2.normal(scale=0.4, size=400))
    bb = np.arange(400) % 5
    ee = np.zeros(400)
    ee[np.flatnonzero(bb == 2)[0]] = 1.0          # exactly one death in bin 2
    hk = CA.hazards_kalbfleisch_prentice(rr, bb, ee, np.array([1.0]), 5)
    i0 = np.flatnonzero((bb == 2) & (ee == 1))[0]
    Rt = rr[bb >= 2].sum()
    a_closed = (1.0 - rr[i0] / Rt) ** (1.0 / rr[i0])
    a_solved = 1.0 - hk[0, 2]                     # r_te = 1 so h = 1 - a
    print(f"  KP at d_t=1: closed form {a_closed:.9f}  bisection {a_solved:.9f}"
          f"   -> {'MATCH' if abs(a_closed-a_solved) < 1e-6 else 'DIFFER'}")

    print()
    print("=" * 88)
    print("SMOKE TEST 5: Kalbfleisch-Prentice IS the cloglog profile for alpha")
    print("=" * 88)
    # The KP defining equation is the profile score for alpha_t in the grouped
    # cloglog model with beta fixed, so h_KP must equal hazards_from_alpha with
    # alpha from profile_alpha.  Two independent routes -- a bisection on a_t and
    # an LBFGS on alpha_t -- landing on one answer validates both.  It also
    # retires the claim that KP gives the Cox arm a different functional form.
    a_prof = CA.profile_alpha(Xtr, itr, etr, T, b, "cloglog")
    h_prof = CA.hazards_from_alpha(Xte, b, a_prof)
    gap_kp = float(np.max(np.abs(hs["kalbfleisch-prentice"] - h_prof)))
    print(f"  max |h_KP - h_profile| = {gap_kp:.3e}"
          f"   -> {'SAME ESTIMATOR' if gap_kp < 1e-5 else 'DIFFER'}")

    print()
    print("=" * 88)
    print("SMOKE TEST 4: E3 three-arm decomposition runs")
    print("=" * 88)
    a2 = CA.profile_alpha(Xtr, itr, etr, T, b)
    bj, aj = CA.fit_grouped_joint(Xtr, itr, etr, T)
    arms = {
        "arm1 Cox beta + Breslow alpha":
            CA.nll_from_hazards(hs["breslow"], ite, ete, T, reduction="mean"),
        "arm2 Cox beta + profiled alpha":
            CA.nll_from_hazards(CA.hazards_from_alpha(Xte, b, a2), ite, ete, T,
                                reduction="mean"),
        "arm3 joint MLE (alpha, beta)":
            CA.nll_from_hazards(CA.hazards_from_alpha(Xte, bj, aj), ite, ete, T,
                                reduction="mean"),
    }
    for k, v in arms.items():
        print(f"  {k:<34} {v:.5f}")
    v = list(arms.values())
    print(f"  baseline representation (arm1 -> arm2) = {v[0]-v[1]:+.5f}")
    print(f"  coefficient estimation  (arm2 -> arm3) = {v[1]-v[2]:+.5f}")
    print(f"  total D_T               (arm1 -> arm3) = {v[0]-v[2]:+.5f}")
    return 0 if gap < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
