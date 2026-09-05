"""E5-E8: what the regimes look like when the truth is known.

The real-data sweep can show that D_T moves with modal bin mass.  It cannot show
WHY, because on real data the model is never correct and the estimand is never
known.  These four designs fix the law and vary one thing at a time.

  E5  Simulation A -- R1 verification.  Continuous Weibull PH, then grouped.
      Lemma 1 says the grouped likelihood is the exact law of the coarsened data,
      so D_T should sit at zero and beta should be unbiased at EVERY grid.  This
      is the controlled version of the nwtco observation.

  E6  Simulation B -- R2, cloglog-correct.  Discrete hazards generated to satisfy
      cloglog(h_t) = alpha_t + b'x exactly.  The grouped model is correct and Cox
      is not, so this is the CEILING on D_T available from specification alone.

  E7  Simulation C -- the falsification test.  Discrete hazards generated from the
      LOGISTIC law, under which the exact-discrete conditional partial likelihood
      is correct and cloglog is wrong.  If D_T goes negative here, the direction
      of the effect tracks which model is correctly specified rather than how many
      ties there are, and the regime framing has real force.  If it stays positive,
      the paper's interpretation needs revising.

      A CORRECTION TO THE DESIGN AS PROPOSED.  The review asked for two
      sub-designs: (i) logit-link discrete hazards, and (ii) "Cox's discrete PH,
      (1 - h_t) = (1 - h_0t)^{exp(b'x)}, under which the exact-discrete Cox
      likelihood is correct and cloglog is wrong".  Design (ii) cannot do that job:

          (1 - h_t) = (1 - h_0t)^{exp(b'x)}
            <=>  log(-log(1 - h_t)) = log(-log(1 - h_0t)) + b'x
            <=>  cloglog(h_t) = alpha_t + b'x

      which is the Prentice-Gloeckler grouped model this paper fits.  Design (ii)
      is design E6 -- our own model, correctly specified -- not a misspecification
      of it.  The exact-discrete likelihood is the conditional law of the failure
      SET under the LOGISTIC model, and estimates a log odds ratio; design (i) is
      therefore the whole of the falsification test.  It is run here at two
      effect sizes, because logit and cloglog agree to first order when hazards
      are small and only separate when they are not, so a weak-effect design would
      have produced a null for a reason that has nothing to do with the claim.

  E8  Factorial separation of modal bin mass from n/T.  Section 7.4 contrasts
      drsa/music (~1,667 rows per interval) with support2 (~304) and reads the
      difference as an effect of rows per interval.  Modal mass and n/T move
      together on real cohorts, so that reading is not identified there.  Here
      they are crossed on an orthogonal grid.

Run:  python -u experiments/simulations.py [e5 e6 e7 e8]
"""
from __future__ import annotations

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

OUT = Path(__file__).resolve().parent
REPS = int(os.environ.get('SIM_REPS', '20'))
TEST_FRAC = 0.3


# ------------------------------------------------------------- generators
def covariates(n, p, rng):
    """Mixed continuous / binary, as the review asked: p-2 normal, 2 Bernoulli."""
    X = rng.normal(size=(n, p))
    if p >= 2:
        X[:, -1] = (rng.uniform(size=n) < 0.4).astype(float)
        X[:, -2] = (rng.uniform(size=n) < 0.5).astype(float)
    return X


def sim_weibull_grouped(n, T, beta, shape, rng, rand_censor=0.0, horizon=3.0):
    """E5: continuous Weibull PH, then grouped onto T equal-width bins.

    Random censoring is applied on the CONTINUOUS scale and then coarsened, which
    violates (A8) -- a subject censored mid-interval is treated as at risk for the
    whole interval.  That is deliberate: the review asked for 20% and 40% random
    censoring, and the resulting bias is a property of the censoring scheme rather
    than of either estimator, so it must be visible in both arms rather than
    assumed away.
    """
    X = covariates(n, len(beta), rng)
    u = rng.uniform(size=n)
    t = (-np.log(u) / np.exp(X @ beta)) ** (1.0 / shape)
    obs = np.minimum(t, horizon)
    ev = (t <= horizon).astype(float)
    if rand_censor > 0:
        # Exponential censoring with rate chosen to censor about `rand_censor`.
        c = rng.exponential(scale=horizon / max(rand_censor, 1e-6) * 0.5, size=n)
        cens = c < obs
        obs = np.where(cens, c, obs)
        ev = np.where(cens, 0.0, ev)
    edges = np.linspace(0, horizon, T + 1)
    idx = np.clip(np.searchsorted(edges, obs, side="left") - 1, 0, T - 1)
    return X, idx.astype(int), ev


def sim_discrete(n, T, beta, rng, link, a0=None, scale=1.0):
    """E6 / E7: discrete hazards from an exactly specified law.

    link='cloglog'  h_t = 1 - exp(-exp(alpha_t + b'x))   -- grouped model correct
    link='logit'    h_t = 1 / (1 + exp(-(alpha_t + b'x)))-- exact-discrete correct
    """
    p = len(beta)
    X = covariates(n, p, rng)
    eta = (X @ beta) * scale
    if a0 is None:
        a0 = np.linspace(-2.2, -1.4, T)      # mildly increasing baseline
    idx = np.full(n, T - 1, dtype=int)
    ev = np.zeros(n)
    alive = np.ones(n, bool)
    for t in range(T):
        z = a0[t] + eta
        h = (1.0 / (1.0 + np.exp(-z)) if link == "logit"
             else -np.expm1(-np.exp(np.clip(z, -30, 10))))
        die = alive & (rng.uniform(size=n) < h)
        idx[die] = t
        ev[die] = 1.0
        alive &= ~die
    idx[alive] = T - 1                        # grid-aligned administrative censoring
    return X, idx, ev


# ------------------------------------------------------------------- core
def one_rep(X, idx, ev, T, rng, want_exact=False):
    """D_T and beta for one replication.  Same arms as protocol_decomp.py."""
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
    b_ef = CA.fit_cox_ties(Ztr, itr, etr, "efron")
    # Z = (X - mu)/sd, so eta = sum b_j Z_j = sum (b_j/sd_j) X_j + const and the
    # original-scale coefficient is b/sd, NOT b*sd.  The first version multiplied,
    # which left the two normal covariates near-right (sd ~ 1) and the two Bernoulli
    # ones wrong by a factor of four, showing up as a flat ~30% "bias" in models
    # that were in fact unbiased.
    r["beta_efron"] = b_ef / sd
    h1 = CA.hazards_breslow(CA.risk(Ztr, b_ef), itr, etr, CA.risk(Zte, b_ef), T)
    n1 = CA.nll_from_hazards(h1, ite, ete, T)
    a2 = CA.profile_alpha(Ztr, itr, etr, T, b_ef, "cloglog")
    n2 = CA.nll_from_hazards(CA.hazards_from_alpha(Zte, b_ef, a2), ite, ete, T)
    bj, aj = CA.fit_grouped_joint(Ztr, itr, etr, T, "cloglog")
    r["beta_grouped"] = bj / sd
    n3 = CA.nll_from_hazards(CA.hazards_from_alpha(Zte, bj, aj), ite, ete, T)
    bl, al = CA.fit_grouped_joint(Ztr, itr, etr, T, "logit")
    n3l = CA.nll_from_hazards(CA.hazards_from_alpha(Zte, bl, al, "logit"),
                              ite, ete, T)

    r["D_T"] = float(np.mean(n1 - n3))            # the paper's headline quantity
    r["D_baseline"] = float(np.mean(n1 - n2))     # arm1 -> arm2
    r["D_coef"] = float(np.mean(n2 - n3))         # arm2 -> arm3
    r["D_link"] = float(np.mean(n3l - n3))        # logit MLE minus cloglog MLE
    if want_exact:
        be, info = CA.fit_cox_exact_discrete(Ztr, itr, etr, budget=8e6)
        if be is not None:
            ale = CA.profile_alpha(Ztr, itr, etr, T, be, "logit")
            ne = CA.nll_from_hazards(CA.hazards_from_alpha(Zte, be, ale, "logit"),
                                     ite, ete, T)
            r["beta_exact"] = be / sd
            r["D_exact_vs_logit"] = float(np.mean(ne - n3l))
            r["D_exact_vs_cloglog"] = float(np.mean(ne - n3))
    return r


def agg(rows, key):
    v = np.array([r[key] for r in rows if key in r], float)
    if v.size < 2:
        return float("nan"), float("nan")
    return float(v.mean()), nb_se(v)


def relbias(rows, key, beta):
    v = np.array([r[key] for r in rows if key in r], float)
    if v.size == 0:
        return float("nan")
    return float(np.mean(np.abs(v.mean(0) / beta - 1.0)))


# ------------------------------------------------------------------- E5
def e5(log):
    log("=" * 108)
    log("E5  SIMULATION A -- R1 VERIFICATION.  Continuous Weibull PH, then grouped.")
    log("    Lemma 1 predicts D_T = 0 and unbiased beta at every grid.")
    log("=" * 108)
    beta = np.array([0.7, -0.5, 0.3, 0.4, -0.6])
    log(f"    true beta = {beta}   ({REPS} reps per cell)")
    log(f"    {'shape':>6}{'censor':>8}{'n':>7}{'T':>5}{'modal':>8}"
        f"{'D_T':>10}{'NB SE':>9}{'|bias| Cox':>12}{'|bias| grouped':>16}")
    for shape in (0.8, 1.0, 1.5):
        for censor in (0.0, 0.2, 0.4):
            for n in (500, 2000, 10000):
                for T in (2, 4, 8, 20, 40):
                    rows, modal = [], []
                    for s in range(REPS):
                        rng = np.random.default_rng(hash((shape, censor, n, T, s))
                                                    % (2**32))
                        X, idx, ev = sim_weibull_grouped(n, T, beta, shape, rng,
                                                         censor)
                        modal.append(np.bincount(idx, minlength=T).max() / n)
                        try:
                            rows.append(one_rep(X, idx, ev, T, rng))
                        except Exception:
                            pass
                    if len(rows) < 2:
                        continue
                    m, se = agg(rows, "D_T")
                    log(f"    {shape:>6.1f}{censor:>8.0%}{n:>7}{T:>5}"
                        f"{np.mean(modal):>8.3f}{m:>+10.5f}{se:>9.5f}"
                        f"{100*relbias(rows,'beta_efron',beta):>11.1f}%"
                        f"{100*relbias(rows,'beta_grouped',beta):>15.1f}%")


# ------------------------------------------------------------------- E6
def e6(log):
    log("")
    log("=" * 108)
    log("E6  SIMULATION B -- R2, CLOGLOG-CORRECT.  The ceiling on D_T from")
    log("    specification alone: the grouped model is right, Cox is not.")
    log("=" * 108)
    beta = np.array([0.7, -0.5, 0.3, 0.4, -0.6])
    log(f"    {'n':>7}{'T':>5}{'scale':>7}{'modal':>8}{'D_T':>10}{'NB SE':>9}"
        f"{'D_base':>10}{'D_coef':>10}{'D_link':>10}{'|bias| grouped':>16}")
    for n in (2000, 10000):
        for T in (2, 4, 8, 20, 40):
            for scale in (1.0, 2.0):
                rows, modal = [], []
                for s in range(REPS):
                    rng = np.random.default_rng(hash((n, T, scale, s)) % (2**32))
                    X, idx, ev = sim_discrete(n, T, beta, rng, "cloglog",
                                              scale=scale)
                    modal.append(np.bincount(idx, minlength=T).max() / n)
                    try:
                        rows.append(one_rep(X, idx, ev, T, rng))
                    except Exception:
                        pass
                if len(rows) < 2:
                    continue
                m, se = agg(rows, "D_T")
                log(f"    {n:>7}{T:>5}{scale:>7.1f}{np.mean(modal):>8.3f}"
                    f"{m:>+10.5f}{se:>9.5f}{agg(rows,'D_baseline')[0]:>+10.5f}"
                    f"{agg(rows,'D_coef')[0]:>+10.5f}{agg(rows,'D_link')[0]:>+10.5f}"
                    f"{100*relbias(rows,'beta_grouped',beta*scale):>15.1f}%")


# ------------------------------------------------------------------- E7
def e7(log):
    log("")
    log("=" * 108)
    log("E7  SIMULATION C -- FALSIFICATION.  Logistic discrete hazards: the")
    log("    exact-discrete conditional PL is correct, cloglog is misspecified.")
    log("    The review's second sub-design, (1-h) = (1-h0)^{exp(b'x)}, is")
    log("    algebraically cloglog(h) = alpha + b'x -- it IS the grouped model,")
    log("    so it is E6 above and cannot serve as a misspecification.")
    log("=" * 108)
    beta = np.array([0.7, -0.5, 0.3, 0.4, -0.6])
    log(f"    {'n':>7}{'T':>5}{'scale':>7}{'modal':>8}{'D_T':>10}{'NB SE':>9}"
        f"{'D_link':>10}{'exact-logit':>13}{'exact-clog':>12}   direction")
    for n in (2000, 10000):
        for T in (2, 4, 8, 20):
            for scale in (1.0, 2.0):
                rows, modal = [], []
                for s in range(REPS):
                    rng = np.random.default_rng(hash((n, T, scale, s, "L"))
                                                % (2**32))
                    X, idx, ev = sim_discrete(n, T, beta, rng, "logit",
                                              scale=scale)
                    modal.append(np.bincount(idx, minlength=T).max() / n)
                    try:
                        rows.append(one_rep(X, idx, ev, T, rng, want_exact=True))
                    except Exception:
                        pass
                if len(rows) < 2:
                    continue
                m, se = agg(rows, "D_T")
                arrow = ("grouped still wins" if m > 1.96 * se else
                         "COX WINS -- falsified" if m < -1.96 * se else
                         "not resolved")
                ex1 = agg(rows, "D_exact_vs_logit")[0]
                ex2 = agg(rows, "D_exact_vs_cloglog")[0]
                log(f"    {n:>7}{T:>5}{scale:>7.1f}{np.mean(modal):>8.3f}"
                    f"{m:>+10.5f}{se:>9.5f}{agg(rows,'D_link')[0]:>+10.5f}"
                    f"{ex1:>+13.5f}{ex2:>+12.5f}   {arrow}")
    log("")
    log("    D_link < 0 means the logit MLE beats the cloglog MLE, i.e. the")
    log("    misspecification is doing what the design intends.  If D_T stays")
    log("    positive while D_link is negative, the formulation effect is not")
    log("    about which link is correct.")


# ------------------------------------------------------------------- E8
def e8(log):
    log("")
    log("=" * 108)
    log("E8  MODAL BIN MASS vs ROWS PER INTERVAL, CROSSED.  On real cohorts the")
    log("    two move together and section 7.4 cannot separate them.")
    log("=" * 108)
    beta = np.array([0.7, -0.5, 0.3, 0.4, -0.6])
    log("    modal mass is set by the baseline level, independently of n and T.")
    log("    'modal' counts EVERY subject in the busiest bin; 'ev modal' counts")
    log("    only EVENTS there, as a fraction of all events.  On real cohorts the")
    log("    two move together, so section 6.3 could not tell which it was using.")
    log(f"    {'n':>7}{'T':>5}{'n/T':>8}{'target':>8}{'modal':>8}{'ev modal':>10}"
        f"{'ev/bin':>9}{'D_T':>10}{'NB SE':>9}{'D_base':>10}{'D_coef':>10}")
    surface = []
    for n in (1000, 4000, 16000):
        for T in (5, 10, 20):
            for lvl, target in ((-3.2, "low"), (-1.6, "mid"), (-0.4, "high")):
                a0 = np.full(T, lvl)
                rows, modal, evmodal, evbin = [], [], [], []
                for s in range(REPS):
                    rng = np.random.default_rng(hash((n, T, lvl, s, "F"))
                                                % (2**32))
                    X, idx, ev = sim_discrete(n, T, beta, rng, "cloglog", a0=a0)
                    modal.append(np.bincount(idx, minlength=T).max() / n)
                    ec = np.bincount(idx[ev == 1], minlength=T)
                    tot = max(int(ec.sum()), 1)
                    evmodal.append(ec.max() / tot)
                    evbin.append(tot / T)
                    try:
                        rows.append(one_rep(X, idx, ev, T, rng))
                    except Exception:
                        pass
                if len(rows) < 2:
                    continue
                m, se = agg(rows, "D_T")
                surface.append((float(np.mean(modal)), float(np.mean(evmodal)),
                                float(np.mean(evbin)), m))
                log(f"    {n:>7}{T:>5}{n/T:>8.0f}{target:>8}{np.mean(modal):>8.3f}"
                    f"{np.mean(evmodal):>10.3f}{np.mean(evbin):>9.0f}"
                    f"{m:>+10.5f}{se:>9.5f}{agg(rows,'D_baseline')[0]:>+10.5f}"
                    f"{agg(rows,'D_coef')[0]:>+10.5f}")

    # WHICH SCALAR ORDERS D_T?  Section 6.3 uses modal bin mass.  On real cohorts
    # modal mass, modal EVENT mass and events per bin are near-collinear, so that
    # choice was never tested.  Here they are crossed, and these correlations say
    # which one the effect follows.
    #
    # This block is emitted from inside e8() deliberately.  A previous attempt
    # appended it by str.replace on a pattern that matched nothing, and because
    # that replace was not asserted the run completed -- 1454 seconds -- with the
    # table printed and the conclusion missing.  Every insertion patch in this
    # project is asserted now.
    if len(surface) >= 6:
        a = np.array(surface)

        def _rho(u, v):
            ru = np.argsort(np.argsort(u)).astype(float)
            rv = np.argsort(np.argsort(v)).astype(float)
            ru -= ru.mean()
            rv -= rv.mean()
            return float((ru * rv).sum() /
                         np.sqrt((ru ** 2).sum() * (rv ** 2).sum()))

        log("")
        log(f"    ORDERING over {len(a)} cells, Spearman against D_T:")
        for j, nm in enumerate(("modal bin mass (section 6.3)",
                                "modal EVENT mass", "events per bin")):
            log(f"      {nm:<30} rho = {_rho(a[:, j], a[:, 3]):+.4f}")
        log("      The largest |rho| is the variable section 6.3 should use.")



# ------------------------------------------------------------------ E6c
def e6c(log):
    """E6 by COARSENING a fixed cohort -- the design the real sweep actually uses.

    E6 above varies T by generating a different number of discrete hazards, so
    each row is a DIFFERENT data-generating process and the table is a ceiling at
    each T rather than a crossover.  The real-data sweep does something else: it
    fixes the cohort and merges adjacent bins, so n is held while nuisance
    dimension falls and tie mass rises together.  Nothing in E5-E8 reproduced that
    under a known truth, which is the one setting where the two costs of section
    6.2 can be separated from each other.

    Coarsening preserves the model exactly.  With cloglog hazards,

        S(a_t | x) = prod_{s<=t} (1 - h_s) = exp(-e^{b'x} sum_{s<=t} e^{alpha_s})

    so the coarsened process is again cloglog with
    alpha'_t = log( Lam0(a_t) - Lam0(a_{t-1}) ).  The grouped model is therefore
    correctly specified at EVERY coarsening -- this is Lemma 1 again -- and any
    trend in D_T is estimation, not creeping misspecification.  (Under the logistic
    law coarsening does NOT preserve the link, which is why this design uses
    cloglog.)
    """
    log("")
    log("=" * 108)
    log("E6c  R2 BY COARSENING A FIXED COHORT -- the real sweep's design, under a")
    log("     known truth.  Generated at T0=40 cloglog, then adjacent bins merged.")
    log("     The grouped model stays EXACTLY correct at every grid (Lemma 1), so")
    log("     any trend in D_T is estimation and not creeping misspecification.")
    log("=" * 108)
    beta = np.array([0.7, -0.5, 0.3, 0.4, -0.6])
    T0 = 40
    for n in (2000, 10000):
        for scale in (1.0, 2.0):
            log(f"    n={n}  scale={scale:g}")
            log(f"      {'T':>5}{'rows/bin':>10}{'modal':>8}{'D_T':>11}{'NB SE':>9}"
                f"{'D_base':>10}{'D_coef':>10}{'base %':>9}")
            rows = []
            for T in (2, 4, 8, 20, 40):
                acc, modal = [], []
                for s in range(REPS):
                    rng = np.random.default_rng(hash((n, scale, s, "C")) % (2**32))
                    X, idx0, ev = sim_discrete(n, T0, beta, rng, "cloglog",
                                               scale=scale)
                    idx = np.minimum((idx0 * T) // T0, T - 1)   # merge adjacent
                    modal.append(np.bincount(idx, minlength=T).max() / n)
                    try:
                        acc.append(one_rep(X, idx, ev, T, rng))
                    except Exception:
                        pass
                if len(acc) < 2:
                    continue
                m, se = agg(acc, "D_T")
                db = agg(acc, "D_baseline")[0]
                dc = agg(acc, "D_coef")[0]
                pct = 100 * db / m if abs(m) > 1e-9 else float("nan")
                log(f"      {T:>5}{n/T:>10.0f}{np.mean(modal):>8.3f}{m:>+11.5f}"
                    f"{se:>9.5f}{db:>+10.5f}{dc:>+10.5f}{pct:>8.1f}%")
                rows.append((T, m))
            if len(rows) >= 3:
                Ts = np.array([r[0] for r in rows], float)
                ef = np.array([r[1] for r in rows])
                ra = np.argsort(np.argsort(Ts)).astype(float)
                rb = np.argsort(np.argsort(ef)).astype(float)
                ra -= ra.mean(); rb -= rb.mean()
                rho = float((ra * rb).sum() /
                            np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
                flips = bool((ef > 0).any() and (ef < 0).any())
                log(f"      corr(T, D_T) = {rho:+.3f}   sign flips: {flips}"
                    f"   (section 6.2 predicts negative, and a flip)")
            log("")


# ------------------------------------------------------------------ E6d
def e6d(log):
    """Does the CROSSOVER reproduce under a known truth, or only the slope?

    E6c confirms the first half of section 6.2: coarsening a fixed cohort makes
    D_T monotone decreasing in T, with corr(T, D_T) = -1.000 in every cell.  It
    does NOT reproduce the second half.  D_T stays positive all the way down to
    50 rows per bin, so the SIGN FLIP -- the crossover the paper's rule is built
    on -- is absent.

    Section 6.2 says the flip comes from the nuisance-parameter cost: the grouped
    model estimates T free alpha_t where Cox profiles the baseline away, and that
    cost grows as T rises relative to n.  E6c never made T/n extreme enough to
    see it: at T=40 on n=2000 there are still 50 rows per bin.  This pushes the
    ratio hard -- T up to 160 on n as small as 600 -- which is the regime where
    support2 at T=60 on 6,374 training rows lives, and where E9 located the
    effect on real data.

    If the flip appears here, the mechanism is confirmed end to end under a known
    truth.  If it does not, the crossover on real cohorts is driven by something
    this design lacks, and section 6.2 must say what.
    """
    log("")
    log("=" * 108)
    log("E6d  DOES THE SIGN FLIP REPRODUCE?  T pushed hard against n.")
    log("     Generated at T0=160 cloglog, coarsened; the grouped model is")
    log("     EXACTLY correct at every grid, so a flip can only be estimation.")
    log("=" * 108)
    beta = np.array([0.7, -0.5, 0.3, 0.4, -0.6])
    T0 = 160
    # E6D_N lets a focused confirmation run re-visit one n at high replication
    # without re-running the whole sweep; the crossing is directional at 20 reps
    # and needs more to resolve.
    for n in [int(x) for x in os.environ.get("E6D_N", "600,1200,2400").split(",")]:
        log(f"    n={n}  (train n is 0.7n, so rows/bin at T=160 is {0.7*n/160:.1f})")
        log(f"      {'T':>5}{'rows/bin':>10}{'tr rows/bin':>13}{'modal':>8}"
            f"{'D_T':>11}{'NB SE':>9}{'D_base':>10}{'D_coef':>10}   verdict")
        rows = []
        for T in (5, 10, 20, 40, 80, 160):
            acc, modal = [], []
            for s in range(REPS):
                rng = np.random.default_rng(hash((n, s, T, "D")) % (2**32))
                X, idx0, ev = sim_discrete(n, T0, beta, rng, "cloglog")
                idx = np.minimum((idx0 * T) // T0, T - 1)
                modal.append(np.bincount(idx, minlength=T).max() / n)
                try:
                    acc.append(one_rep(X, idx, ev, T, rng))
                except Exception:
                    pass
            if len(acc) < 2:
                continue
            m, se = agg(acc, "D_T")
            v = ("grouped wins" if m > 1.96 * se else
                 "COX WINS -- flip" if m < -1.96 * se else "not resolved")
            log(f"      {T:>5}{n/T:>10.1f}{0.7*n/T:>13.1f}{np.mean(modal):>8.3f}"
                f"{m:>+11.5f}{se:>9.5f}{agg(acc,'D_baseline')[0]:>+10.5f}"
                f"{agg(acc,'D_coef')[0]:>+10.5f}   {v}")
            rows.append((T, m, se))
        if rows:
            ef = np.array([r[1] for r in rows])
            flip = bool((ef < 0).any())
            res_flip = any(r[1] < -1.96 * r[2] for r in rows)
            log(f"      any negative D_T: {flip}    RESOLVED negative: {res_flip}")
        log("")
    log("    READING.  A resolved negative row is the crossover reproduced under a")
    log("    known truth, which would complete section 6.2's mechanism.  Only")
    log("    unresolved negatives, or none at all, means the real-data flip is")
    log("    driven by something absent here -- and the honest options are a")
    log("    misspecification the simulation does not have, or a covariate")
    log("    dimension effect, since p is 5 here and 32 on support2/slos.")


def main():
    which = [a.lower() for a in sys.argv[1:]] or ["e5", "e6", "e6c", "e7", "e8"]
    lines = []
    # SIM_TAG keeps a focused rerun from sharing an output path with a sweep that
    # is still running.  Launching a high-replication e6d confirmation while the
    # original e6d was still writing put two processes on one file and nearly
    # destroyed the 20-rep results -- the same failure mode as the status.md
    # truncation recorded in lesson 26.  A rerun with different settings gets a
    # different file, always.
    tag = os.environ.get("SIM_TAG", "")
    path = OUT / ("simulations_" + "".join(which) + (f"_{tag}" if tag else "") + ".txt")

    def log(s=""):
        print(s, flush=True)
        lines.append(s)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    t0 = time.time()
    for k in which:
        {"e5": e5, "e6": e6, "e6c": e6c, "e6d": e6d, "e7": e7, "e8": e8}[k](log)
    log("")
    log(f"total {time.time()-t0:.0f}s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
