"""1a: the exact-discrete coefficient comparator on the E5 Weibull design.

The decomposition the paper currently reports has three arms.  The review asks
for a fourth, so that the Efron approximation and the baseline representation
can be separated from each other:

    A   beta_Efron  + Breslow baseline
    B   beta_Efron  + cloglog profile baseline   (= Kalbfleisch-Prentice)
    C   beta_exact  + cloglog profile baseline
    D   grouped joint MLE of (alpha, beta)

    A - B   baseline representation
    B - C   Efron versus exact tie handling, coefficients only
    C - D   what is left: conditional PL versus joint likelihood

Arm C profiles alpha under CLOGLOG, not logit.  E7 in simulations.py profiles
the exact arm under logit because there the logistic law is the true one; here
the question is whether exact tie handling recovers the coefficient the grouped
model gets, so B and C must differ in beta ALONE.

The generating design, grid and seeds are byte-identical to e5() in
simulations.py -- same stable_seed(shape, censor, n, T, s) -- so every cell here
is the same data the frozen E5 log was computed on and the two are directly
comparable.

FEASIBILITY.  The exact recursion costs n x (max_t d_t + 1) and max_t d_t grows
with n, so the cost is quadratic.  At n=10000 the coarse grids exceed the budget
and arm C does not exist there.  That is reported per cell, never silently
dropped: it is the practical finding that exact tie handling is least available
exactly where ties are heaviest.

Run:  python -u experiments/exact_arm_sim.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from experiments import cox_arms as CA
from experiments.protocol_decomp import nb_se
from experiments.simulations import sim_weibull_grouped
from kanrel.stats import stable_seed

OUT = Path(__file__).resolve().parent / "exact_arm_sim.txt"
REPS = int(os.environ.get("SIM_REPS", "20"))
TEST_FRAC = 0.3
BUDGET = float(os.environ.get("EXACT_BUDGET", "8e6"))
BETA = np.array([0.7, -0.5, 0.3, 0.4, -0.6])

SHAPES = (0.8, 1.0, 1.5)
CENSORS = (0.0, 0.2, 0.4)
NS = (500, 2000, 10000)
TS = (2, 4, 8, 20, 40)


def four_arms(X, idx, ev, T, rng):
    """Arms A-D on one replication, all scored on the same held-out rows."""
    n = len(idx)
    perm = rng.permutation(n)
    n_te = int(round(TEST_FRAC * n))
    te_i, tr_i = perm[:n_te], perm[n_te:]
    Xtr, itr, etr = X[tr_i], idx[tr_i], ev[tr_i]
    Xte, ite, ete = X[te_i], idx[te_i], ev[te_i]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd

    r = {}
    cost, dmax = CA.exact_discrete_cost(itr, etr, T)
    r["cost"], r["dmax"] = float(cost), int(dmax)

    # arm A: Efron coefficients pushed through a Breslow baseline
    b_ef = CA.fit_cox_ties(Ztr, itr, etr, "efron")
    r["beta_efron"] = b_ef / sd            # b/sd, not b*sd: see one_rep()
    hA = CA.hazards_breslow(CA.risk(Ztr, b_ef), itr, etr, CA.risk(Zte, b_ef), T)
    nA = CA.nll_from_hazards(hA, ite, ete, T)

    # arm B: same coefficients, baseline profiled by cloglog MLE
    aB = CA.profile_alpha(Ztr, itr, etr, T, b_ef, "cloglog")
    nB = CA.nll_from_hazards(CA.hazards_from_alpha(Zte, b_ef, aB), ite, ete, T)

    # arm D: joint MLE
    bD, aD = CA.fit_grouped_joint(Ztr, itr, etr, T, "cloglog")
    r["beta_grouped"] = bD / sd
    nD = CA.nll_from_hazards(CA.hazards_from_alpha(Zte, bD, aD), ite, ete, T)

    # The exact-discrete coefficient is a log ODDS ratio, so scoring it through
    # a cloglog baseline pairs a coefficient from one model with a hazard from
    # another.  protocol_decomp.py already found that on metabric the mismatch
    # alone cost the exact arm +0.0022 against Efron.  So every predictive
    # comparison is reported TWICE: once in the cloglog family against the
    # cloglog joint MLE, and once in the logit family against the logit joint
    # MLE, which is the exact method judged in its own metric.  Without the
    # second, "arm C predicts worse" is a statement about link choice.
    bL, aL = CA.fit_grouped_joint(Ztr, itr, etr, T, "logit")
    nDl = CA.nll_from_hazards(CA.hazards_from_alpha(Zte, bL, aL, "logit"),
                              ite, ete, T)
    aBl = CA.profile_alpha(Ztr, itr, etr, T, b_ef, "logit")
    nBl = CA.nll_from_hazards(CA.hazards_from_alpha(Zte, b_ef, aBl, "logit"),
                              ite, ete, T)

    r["D_A"] = float(np.mean(nA - nD))
    r["D_B"] = float(np.mean(nB - nD))
    r["D_Bl"] = float(np.mean(nBl - nDl))
    r["feasible"] = bool(cost <= BUDGET)

    # arm C: exact-discrete coefficients, same cloglog profile baseline as B
    if r["feasible"]:
        bC, info = CA.fit_cox_exact_discrete(Ztr, itr, etr, budget=BUDGET)
        if bC is not None:
            aC = CA.profile_alpha(Ztr, itr, etr, T, bC, "cloglog")
            nC = CA.nll_from_hazards(CA.hazards_from_alpha(Zte, bC, aC),
                                     ite, ete, T)
            r["beta_exact"] = bC / sd
            r["D_C"] = float(np.mean(nC - nD))
            # arm C': the same coefficient in its own link
            aCl = CA.profile_alpha(Ztr, itr, etr, T, bC, "logit")
            nCl = CA.nll_from_hazards(
                CA.hazards_from_alpha(Zte, bC, aCl, "logit"), ite, ete, T)
            r["D_Cl"] = float(np.mean(nCl - nDl))
        else:
            r["feasible"] = False
            r["failure"] = info.get("failure", "infeasible")
    return r


def agg(rows, key):
    v = np.array([r[key] for r in rows if key in r], float)
    if v.size < 2:
        return float("nan"), float("nan")
    return float(v.mean()), nb_se(v)


def relbias(rows, key, beta=BETA):
    """Mean over components of |mean_reps(beta_hat_j)/beta_j - 1|."""
    v = np.array([r[key] for r in rows if key in r], float)
    if v.size == 0:
        return float("nan")
    return float(np.mean(np.abs(v.mean(0) / beta - 1.0)))


def shrink(rows, key, beta=BETA):
    """Attenuation factor: the slope of mean(beta_hat) on beta through zero.

    Coarsening attenuates the Cox coefficient towards zero, so a single number
    for 'how much of beta survives' is more readable than five component biases.
    1.0 is unbiased; 0.7 means 30 percent of the signal is gone.
    """
    v = np.array([r[key] for r in rows if key in r], float)
    if v.size == 0:
        return float("nan")
    m = v.mean(0)
    return float((m @ beta) / (beta @ beta))


def main():
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    t_start = time.time()
    log("=" * 122)
    log("1a  EXACT-DISCRETE COEFFICIENT COMPARATOR -- E5 WEIBULL DESIGN")
    log("    A = Efron + Breslow | B = Efron + cloglog profile | "
        "C = exact + cloglog profile | D = grouped joint MLE")
    log("    D_x = NLL(arm x) - NLL(arm D).  A-B is the baseline, B-C is Efron "
        "vs exact, C-D is what remains.")
    log("    shrink = slope of mean(beta_hat) on the true beta; 1.000 unbiased, "
        "< 1 attenuated.  " + str(REPS) + " reps per cell.")
    log("    true beta = " + str(BETA) + "   exact budget = "
        + format(BUDGET, ",.0f"))
    log("=" * 122)
    log("    cloglog family: D_A, D_B, D_C vs the cloglog joint MLE.  "
        "logit family: D_Bl, D_Cl vs the LOGIT joint MLE --")
    log("    the exact method judged in its own metric, so that B vs C is tie "
        "handling and not link choice.")
    log("=" * 122)
    log(f"    {'shape':>6}{'cens':>6}{'n':>7}{'T':>4}{'modal':>7}{'dmax':>6}"
        f"{'D_A':>10}{'D_B':>10}{'D_C':>10}{'D_Bl':>10}{'D_Cl':>10}"
        f"{'shrEfron':>10}{'shrExact':>10}{'shrGrp':>9}"
        f"{'|b|Efr':>8}{'|b|Exa':>8}{'|b|Grp':>8}  exactC")

    blocks = []
    for shape in SHAPES:
        for censor in CENSORS:
            for n in NS:
                for T in TS:
                    rows, modal = [], []
                    for s in range(REPS):
                        rng = np.random.default_rng(
                            stable_seed(shape, censor, n, T, s))
                        X, idx, ev = sim_weibull_grouped(n, T, BETA, shape,
                                                         rng, censor)
                        modal.append(np.bincount(idx, minlength=T).max() / n)
                        try:
                            rows.append(four_arms(X, idx, ev, T, rng))
                        except Exception as e:
                            print("      rep " + str(s) + " FAILED "
                                  + type(e).__name__ + ": " + str(e)[:60],
                                  flush=True)
                    if len(rows) < 2:
                        continue
                    nfeas = sum(1 for r in rows if r.get("feasible"))
                    dA, sA = agg(rows, "D_A")
                    dB, sB = agg(rows, "D_B")
                    dC, sC = agg(rows, "D_C")
                    dBl, sBl = agg(rows, "D_Bl")
                    dCl, sCl = agg(rows, "D_Cl")
                    cost = float(np.mean([r["cost"] for r in rows]))
                    dmax = int(np.mean([r["dmax"] for r in rows]))
                    b = {
                        "shape": shape, "censor": censor, "n": n, "T": T,
                        "modal": float(np.mean(modal)), "dmax": dmax,
                        "cost": cost, "reps": len(rows), "n_feasible": nfeas,
                        "D_A": dA, "se_A": sA, "D_B": dB, "se_B": sB,
                        "D_C": dC, "se_C": sC,
                        "D_Bl": dBl, "se_Bl": sBl, "D_Cl": dCl, "se_Cl": sCl,
                        "shrink_efron": shrink(rows, "beta_efron"),
                        "shrink_exact": shrink(rows, "beta_exact"),
                        "shrink_grouped": shrink(rows, "beta_grouped"),
                        "relbias_efron": relbias(rows, "beta_efron"),
                        "relbias_exact": relbias(rows, "beta_exact"),
                        "relbias_grouped": relbias(rows, "beta_grouped"),
                        "beta_efron": np.array(
                            [r["beta_efron"] for r in rows]).mean(0).tolist(),
                        "beta_grouped": np.array(
                            [r["beta_grouped"] for r in rows]).mean(0).tolist(),
                        "beta_exact": (np.array(
                            [r["beta_exact"] for r in rows
                             if "beta_exact" in r]).mean(0).tolist()
                            if nfeas else None),
                    }
                    blocks.append(b)
                    dc = f"{dC:>+10.5f}" if nfeas else f"{'--':>10}"
                    dcl = f"{dCl:>+10.5f}" if nfeas else f"{'--':>10}"
                    sx = f"{b['shrink_exact']:>10.3f}" if nfeas else f"{'--':>10}"
                    bx = (f"{100 * b['relbias_exact']:>7.1f}%" if nfeas
                          else f"{'--':>8}")
                    log(f"    {shape:>6.1f}{censor:>6.0%}{n:>7}{T:>4}"
                        f"{np.mean(modal):>7.3f}{dmax:>6}"
                        f"{dA:>+10.5f}{dB:>+10.5f}{dc}"
                        f"{dBl:>+10.5f}{dcl}"
                        f"{b['shrink_efron']:>10.3f}{sx}"
                        f"{b['shrink_grouped']:>9.3f}"
                        f"{100 * b['relbias_efron']:>7.1f}%{bx}"
                        f"{100 * b['relbias_grouped']:>7.1f}%"
                        f"  {nfeas}/{len(rows)}")
                    OUT.with_suffix(".json").write_text(
                        json.dumps(blocks, indent=1), encoding="utf-8")

    feas = [b for b in blocks if b["n_feasible"] > 0]
    log("")
    log("=" * 122)
    log("READING")
    log("=" * 122)
    log("  cells: " + str(len(blocks)) + "   arm C available in "
        + str(len(feas)) + "   infeasible in " + str(len(blocks) - len(feas)))
    if feas:
        se = np.array([b["shrink_efron"] for b in feas])
        sx = np.array([b["shrink_exact"] for b in feas])
        sg = np.array([b["shrink_grouped"] for b in feas])
        log(f"  mean attenuation where arm C exists:  Efron {se.mean():.3f}   "
            f"exact {sx.mean():.3f}   grouped {sg.mean():.3f}")
        gap = sg - se
        rec = np.where(np.abs(gap) > 1e-9, (sx - se) / np.where(
            np.abs(gap) > 1e-9, gap, 1.0), np.nan)
        rec = rec[np.isfinite(rec)]
        if rec.size:
            log(f"  fraction of the Efron-to-grouped coefficient gap that exact "
                f"tie handling closes: median {np.median(rec):.2f} "
                f"over {rec.size} cells")
        coarse = [b for b in feas if b["T"] <= 4]
        if coarse:
            ce = np.mean([b["shrink_efron"] for b in coarse])
            cx = np.mean([b["shrink_exact"] for b in coarse])
            cg = np.mean([b["shrink_grouped"] for b in coarse])
            log(f"  coarse grids only (T <= 4, {len(coarse)} cells):  "
                f"Efron {ce:.3f}   exact {cx:.3f}   grouped {cg:.3f}")
        # The predictive comparison, stated in both families so that the link
        # mismatch cannot be mistaken for a tie-handling effect.
        cg_worse = sum(1 for b in feas
                       if b["D_C"] is not None and b["D_B"] is not None
                       and np.isfinite(b["D_C"]) and b["D_C"] > b["D_B"])
        lg_worse = sum(1 for b in feas
                       if b.get("D_Cl") is not None and b.get("D_Bl") is not None
                       and np.isfinite(b["D_Cl"]) and b["D_Cl"] > b["D_Bl"])
        log(f"  arm C predicts worse than arm B in {cg_worse}/{len(feas)} cells "
            f"scored in the CLOGLOG family (confounded with link),")
        log(f"                                and {lg_worse}/{len(feas)} cells "
            f"scored in the LOGIT family (exact's own metric).")
    log("")
    log("  wall time " + format(time.time() - t_start, ".0f") + "s")
    log("  wrote " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
