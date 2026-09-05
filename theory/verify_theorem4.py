"""Monte-Carlo gate for Theorem 4 (fixed-grid semiparametric rate, regime R2).

Two claims, tested separately.

PART 1 -- sqrt(n)-normality and efficiency of the Euclidean part.
  With T fixed and alpha in R^T unknown, beta should be sqrt(n)-consistent and
  asymptotically normal, with the observed-information sandwich giving valid
  standard errors.  Tests: z = (betahat - beta)/se is standard normal, and the
  nominal 95% CI covers at 95%.  If the information is EFFICIENT, se should also
  track the empirical sd -- ratio ~1.  A ratio far from 1 means the plug-in
  information is wrong, which would falsify the efficiency half of Theorem 4.

PART 2 -- does the risk of ghat respect the minimax BOUND?
  Theorem 4 gives an UPPER BOUND on risk, ||ghat - g*||^2 = O(n^{-2r/(2r+1)}).
  Converging faster than that does not violate it; only converging slower would.
  The gate therefore checks the bound holds, NOT that it is tight.

  Two earlier versions of this gate had the direction wrong -- one demanded the
  slope equal -2r/(2r+1), the other demanded two targets of different smoothness
  separate.  Both asked the bound to be tight, which the theorem never claims.

  Observed on both targets: the PARAMETRIC rate n^-1.
      smooth  0.8(x0^2-1) + sin(x1) + 0.6x2   r >> 1, bound -0.889, got -1.005
      kinked  1.2|x0| + 0.6x2                 r = 1,  bound -0.667, got -1.043

  Why the bound is loose here is measurable, not mysterious (diagnose.py D2):
  the spline approximation floor is 1.6e-5 for the smooth target and ~0.002 for
  1.2|x0|, against observed MSE of 0.0064 and 0.0038 at n=16000.  With bias that
  far below variance the sieve behaves like a correctly-specified ~30-parameter
  parametric model and earns n^-1.  A step function (floor 0.26-0.41) would be
  needed before the nonparametric regime showed at these sample sizes.

  Each slope carries a bootstrap CI over replicates, since with affordable
  replicate counts a point estimate of the slope is not interpretable alone.

Both parts use grouped Weibull PH data, where Lemma 1 makes the discrete model
EXACTLY correct -- so any failure is estimation error, not misspecification.

Run:  python theory/verify_theorem4.py [--quick]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kanrel.data import SurvData
from kanrel.fit import fit, to_tensors
from kanrel.hazard import DiscreteHazardKAN
from kanrel.likelihood import nll

SHAPE, HORIZON = 1.5, 2.0
BETA = np.array([0.70, -0.50, 0.30], dtype=np.float32)


def g_true(X):
    """SMOOTH target (r large).  See the note on sin(pi*x1) in the git history:
    three periods over the covariate range gave a huge Sobolev norm and resonated
    with the uniform knot grid.  sin(x1) is one period; its spline approximation
    floor is 1.6e-5 at G=5, i.e. negligible against the statistical error."""
    return 0.8 * (X[:, 0] ** 2 - 1.0) + 1.0 * np.sin(X[:, 1]) + 0.6 * X[:, 2]


def g_kinked(X):
    """NON-SMOOTH target: |x| has a bounded first derivative but an unbounded
    second, so it sits in a Sobolev ball of smoothness r = 1 rather than r >> 1.

    This is the contrast that makes Theorem 4's rate testable.  A smooth target is
    approximated so well by a modest spline basis that the sieve behaves like a
    correctly-specified parametric model and converges at n^-1 -- FASTER than the
    minimax rate, which is legitimate because minimax is a worst case over the
    ball, not a prediction for every member of it.  Only a target that is genuinely
    hard for the basis exposes the nonparametric rate.

    Predicted slope at r = 1:  -2r/(2r+1) = -0.667.
    """
    return 1.2 * (np.abs(X[:, 0]) - np.sqrt(2.0 / np.pi)) + 0.6 * X[:, 2]


# Shared evaluation set, fixed once for every replicate and every n.  Drawing a
# fresh test set per replicate injects sampling noise that is common across fits
# and so contributes pure variance to the fitted slope; freezing it removes that
# term without touching the estimator.
_TEST_X = np.random.default_rng(20_260_821).normal(size=(8000, 3)).astype(np.float32)
_TEST_SETS = {"smooth": (_TEST_X, g_true(_TEST_X) - g_true(_TEST_X).mean()),
              "kinked": (_TEST_X, g_kinked(_TEST_X) - g_kinked(_TEST_X).mean())}
TARGETS = {"smooth": g_true, "kinked": g_kinked}
# What the theorem PREDICTS as an upper bound on risk: the minimax rate for the
# smoothness class the target belongs to.  Cubic splines cap the usable
# smoothness at r = 4, so a very smooth target is bounded by -2*4/(2*4+1);
# |x| sits at r = 1 and is bounded by -2/3.
MINIMAX_BOUND = {"smooth": -2 * 4 / (2 * 4 + 1), "kinked": -2 * 1 / (2 * 1 + 1)}
# What we EXPECT to observe, which is faster, because the approximation error is
# far below the statistical error at these knot counts (see diagnose.py D2).
PRED_SLOPE = {"smooth": -1.0, "kinked": -1.0}


def rate_holds(ci, bound):
    """THE verdict predicate for Part 2.  Import this; never restate it.

    Theorem 4's rate is an UPPER BOUND ON RISK: the risk decays at least as fast
    as n^bound.  On a log-log plot "at least as fast" means a slope no SHALLOWER
    than the bound, i.e. no greater than it.  So the bound holds when even the
    shallowest plausible slope -- ci[1], the upper end -- is still <= bound.
    Converging FASTER (a more negative slope) can never violate an upper bound;
    only converging slower can.

    This lives in one place because the direction has been inverted three times
    in this project: twice in earlier versions of this gate, and once in
    theory/theorem4_clamp_ab.py, which restated the test instead of importing it
    and duly reported "VIOLATES BOUND" for a slope of -1.006 that reproduces a
    known-good run to three decimals.  Any script that judges this proposition
    must call this function.
    """
    return ci[1] <= bound


def simulate(n, n_bins, seed, linear=True, p=3, censor_scale=3.0, target="smooth"):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    g = (X @ BETA) if linear else TARGETS[target](X)
    u = rng.uniform(size=n)
    t = (-np.log(u) / np.exp(g)) ** (1.0 / SHAPE)
    c = rng.uniform(0, censor_scale, size=n)
    obs = np.minimum(t, np.minimum(c, HORIZON))
    event = ((t <= c) & (t <= HORIZON)).astype(np.float32)
    edges = np.linspace(0, HORIZON, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, obs, side="left") - 1, 0, n_bins - 1)
    return SurvData(X, idx.astype(np.int64), event, n_bins,
                    [f"x{i}" for i in range(p)], name="thm4",
                    intrinsically_discrete=False, bin_edges=edges,
                    meta={"g_true": g})


# ------------------------------------------------------------------- PART 1
def beta_with_se(d, seed):
    """Exact MLE by LBFGS, plus observed-information SEs on the raw covariate scale.

    NOTE the earlier version initialised with Adam and read the Hessian there.
    That is wrong: the observed-information sandwich is only valid AT the MLE, and
    Adam at a fixed step size sits near it, not on it.  The grouped-PH cloglog
    log-likelihood is concave in (alpha, beta), so LBFGS with strong-Wolfe finds
    the exact optimum and the SEs become valid.
    """
    X, mask, y = to_tensors(d)
    T, p = d.n_bins, d.X.shape[1]
    Xd, maskd, yd = X.double(), mask.double(), y.double()
    mu, sd = Xd.mean(0), Xd.std(0).clamp_min(1e-9)
    z = (Xd - mu) / sd

    def total_nll(theta):
        a, b = theta[:T], theta[T:]
        logits = a.unsqueeze(0) + (z @ b).unsqueeze(1)
        return nll(logits, maskd, yd, "cloglog", reduction="sum")

    theta = torch.zeros(T + p, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        theta[:T] = -2.5
    opt = torch.optim.LBFGS([theta], max_iter=500, tolerance_grad=1e-12,
                            tolerance_change=1e-14, history_size=50,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = total_nll(theta)
        loss.backward()
        return loss

    opt.step(closure)
    theta_hat = theta.detach()
    H = torch.autograd.functional.hessian(total_nll, theta_hat)
    cov = torch.linalg.inv(H)
    se_std = torch.sqrt(torch.diag(cov)[T:].clamp_min(1e-18))
    return (theta_hat[T:] / sd).numpy(), (se_std / sd).numpy()


def part1(n=6000, n_bins=20, reps=60):
    est, ses = [], []
    for s in range(reps):
        b, se = beta_with_se(simulate(n, n_bins, s, linear=True), s)
        est.append(b); ses.append(se)
    est, ses = np.array(est), np.array(ses)
    z = (est - BETA) / ses
    emp_sd = est.std(0, ddof=1)
    mean_se = ses.mean(0)
    cover = np.abs(z) < 1.959964
    return dict(est=est.mean(0), emp_sd=emp_sd, mean_se=mean_se,
                ratio=mean_se / emp_sd, z_mean=z.mean(0), z_sd=z.std(0, ddof=1),
                coverage=cover.mean(0), reps=reps, n=n)


# ------------------------------------------------------------------- PART 2
def g_mse(n, n_bins, seed, hidden=(), grid_scale=True, epochs=3000, target="smooth"):
    """One replicate of ||ghat - g*||^2.

    hidden=()  -> a PURE ADDITIVE KAN [p, 1].  This is the function class the
                  cited rate (Liu, Chatzi & Lai) actually covers, and g_true is
                  additive, so this is the honest test of Theorem 4's rate.
    hidden=(8,) -> a depth-2 composition: a strictly larger class whose rate the
                  cited theorem does NOT describe.  Reported for contrast.

    grid_scale grows the knot count as G_n ~ n^{1/(2r+1)} with r=4, which the
    theorem REQUIRES.  Holding G fixed leaves an approximation-bias floor that
    flattens the log-log slope at large n.
    """
    d = simulate(n, n_bins, seed, linear=False, target=target)
    grid_size = int(max(5, round(5 * (n / 500) ** (1 / 9)))) if grid_scale else 8
    # clamp_inputs=False IS REQUIRED HERE, and is not a stylistic choice.
    #
    # Theorem 4 Part 2 is a statement about the SIEVE MLE -- the unconstrained
    # maximiser over the spline sieve.  Clamping each edge input to the knot range
    # constrains the function class, so a clamped fit is a DIFFERENT ESTIMATOR and
    # is not the object the theorem describes.
    #
    # This is not hypothetical.  When clamp_inputs became the DEFAULT in
    # kanrel/kan.py, this gate's smooth-target slope moved -1.005 -> -0.873 and
    # crossed the -0.889 minimax bound, and it looked like a theory failure.  The
    # controlled A/B (theory/theorem4_clamp_ab.py) shows the estimator never
    # changed: with the clamp off it reproduces the original run to five decimals
    # at every n (0.197813/0.197798, 0.090065/0.090048, 0.040393/0.040338,
    # 0.021446/0.021440).  The clamp lowers MSE most at SMALL n -- constant
    # extrapolation helps where the sample is thinnest -- and that is precisely
    # what flattens the log-log slope.
    #
    # So: clamp OFF here, where the estimand is the sieve MLE; clamp ON in
    # experiments/, where unbounded predictions off the training support are
    # indefensible (status.md 4.3f).  Methodological lesson 4 -- deep-learning
    # defaults change the estimand -- with the same resolution as the first time.
    m = DiscreteHazardKAN(d.X.shape[1], d.n_bins, hidden=hidden, mode="baseline",
                          link="cloglog", grid_size=grid_size, clamp_inputs=False)
    # NO early stopping and NO validation split, deliberately.
    #
    # Theorem 4 is a statement about the sieve MLE -- the actual maximiser of the
    # discrete likelihood.  Early stopping is an implicit regulariser that targets
    # a DIFFERENT estimator, and a validation split discards 20% of the sample.
    # Diagnostic D3 showed the earlier settings (600 epochs, patience 60, val 0.2)
    # hit the epoch cap at 568/596 epochs and were still improving: MSE fell 27% at
    # n=4000 and 61% at n=16000 under a larger budget.  Because the penalty grew
    # with n, it flattened the log-log slope -- over n=4000..16000 the fitted slope
    # was -0.248 budget-limited against -0.747 converged.
    m, _ = fit(m, d, epochs=epochs, lr=0.03, val_frac=0.0, patience=10**9,
               grid_update_epochs=(30, 80), seed=seed)
    # Shared test set across every replicate and every n.  Drawing a fresh test
    # set per replicate injects test-sampling noise that is common to all fits and
    # therefore pure variance in the slope estimate; holding it fixed removes it.
    Xte, gt = _TEST_SETS[target]
    with torch.no_grad():
        eta = m(torch.as_tensor(Xte))
        gh = (eta[:, 0] - m.alpha[0]).numpy()
    gh = gh - gh.mean()          # identifiability: E[g] = 0
    return float(np.mean((gh - gt) ** 2))


def part2(ns, reps=6, n_bins=20, hidden=(), grid_scale=True, n_boot=2000, seed=0,
          target="smooth"):
    """Returns per-n replicate values, the fitted slope, and a bootstrap CI.

    A point estimate of the slope is not interpretable on its own here: with the
    replicate counts that are affordable, the per-n MSE carries ~50% Monte-Carlo
    error, so the slope does too.  The question that matters is whether the fitted
    slope is CONSISTENT with the predicted -2r/(2r+1), and only a confidence
    interval can answer that.  Resampling replicates within each n propagates the
    MC error into the slope.
    """
    raw = {}
    for n in ns:
        raw[n] = np.array([g_mse(n, n_bins, s, hidden=hidden, grid_scale=grid_scale,
                                 target=target) for s in range(reps)])
    logn = np.log(np.array(ns, dtype=float))
    slope = float(np.polyfit(logn, np.log([raw[n].mean() for n in ns]), 1)[0])

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        means = [raw[n][rng.integers(0, len(raw[n]), len(raw[n]))].mean() for n in ns]
        if min(means) <= 0:
            continue
        boots.append(np.polyfit(logn, np.log(means), 1)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out = {n: (float(raw[n].mean()), float(raw[n].std(ddof=1) / np.sqrt(reps)))
           for n in ns}
    return out, slope, (float(lo), float(hi))


def main():
    quick = "--quick" in sys.argv
    torch.set_num_threads(max(1, torch.get_num_threads()))
    ok = True

    print("=" * 82)
    print("PART 1  sqrt(n)-normality, valid SEs, and efficiency of betahat")
    print("=" * 82)
    r = part1(n=2000 if quick else 6000, reps=12 if quick else 60)
    print(f"  n={r['n']}  reps={r['reps']}  T=20   true beta = {BETA}")
    hdr = "".join(f"{f'x{i}':>11}" for i in range(len(BETA)))
    print(f"    {'':<14}{hdr}")
    for k, lbl in [("est", "estimate"), ("emp_sd", "empirical sd"), ("mean_se", "mean SE"),
                   ("ratio", "SE/sd"), ("z_mean", "mean z"), ("z_sd", "sd of z"),
                   ("coverage", "95% coverage")]:
        print(f"    {lbl:<14}" + "".join(f"{v:>11.4f}" for v in np.atleast_1d(r[k])))
    # Tolerance must reflect Monte-Carlo error, not a round number.  The empirical
    # sd from R replicates has relative SE ~ 1/sqrt(2(R-1)), so the SE/sd ratio
    # inherits it; anything inside 3 of those is consistent with 1.  The earlier
    # fixed 0.12 was tighter than the noise floor at R=60 (0.092) and so reported
    # a FAIL for ratios that were within 1.6 standard errors of 1.
    se_tol = 3.0 / np.sqrt(2 * (r["reps"] - 1))
    good_se = np.all(np.abs(r["ratio"] - 1) < se_tol)
    print(f"    (SE/sd tolerance {se_tol:.3f} = 3 x MC error at {r['reps']} reps)")
    good_z = np.all(np.abs(r["z_sd"] - 1) < 0.25) and np.all(np.abs(r["z_mean"]) < 0.5)
    good_cov = np.all(np.abs(r["coverage"] - 0.95) < 0.09)
    ok &= good_se and good_z and good_cov
    print(f"    -> SE/sd ~ 1 : {'PASS' if good_se else 'FAIL'}"
          f" | z ~ N(0,1) : {'PASS' if good_z else 'FAIL'}"
          f" | coverage : {'PASS' if good_cov else 'FAIL'}")

    print()
    print("=" * 82)
    print("PART 2  nonparametric rate for ghat   ||ghat-g*||^2 ~ n^(-2r/(2r+1))")
    print("=" * 82)
    ns = [500, 1000, 2000] if quick else [500, 1000, 2000, 4000, 8000, 16000]
    # What is actually testable here is that the rate TRACKS SMOOTHNESS.
    #
    # An earlier version of this gate demanded the slope equal -2r/(2r+1) with
    # r=4.  That was the wrong proposition.  -2r/(2r+1) is a MINIMAX rate: a worst
    # case over the whole Sobolev ball, not a prediction for any particular member
    # of it.  On the smooth target the spline approximation floor is 1.6e-5 against
    # an MSE of 0.006 at n=16000 -- 400x smaller -- so the sieve behaves like a
    # correctly-specified ~30-parameter parametric model and converges at n^-1,
    # BEATING the minimax bound.  That is not a failure; it is what minimax means.
    #
    # The falsifiable claim is the contrast: a target that is genuinely hard for
    # the basis must converge measurably more slowly.
    reps = 3 if quick else 12
    res = {}
    for tgt in ("smooth", "kinked"):
        out, slope, ci = part2(ns, reps=reps, target=tgt)
        res[tgt] = (slope, ci)
        print(f"\n  target = {tgt}   (predicted {PRED_SLOPE[tgt]:.3f})")
        print(f"    {'n':>8}{'MSE':>14}{'se':>12}")
        for n, (mm, ss) in out.items():
            print(f"    {n:>8}{mm:>14.6f}{ss:>12.6f}")
        print(f"    slope = {slope:.3f}   95% bootstrap CI [{ci[0]:.3f}, {ci[1]:.3f}]")

    # Theorem 4 is an UPPER BOUND on risk.  Converging faster than the minimax
    # rate does not violate it -- only converging SLOWER would.  Two earlier
    # versions of this gate got the direction wrong: the first demanded equality
    # with -2r/(2r+1), the second demanded the two targets separate.  Both were
    # asking the bound to be TIGHT, which the theorem never claims.
    #
    # Why it is not tight here is measurable, not mysterious (theory/diagnose.py
    # D2): the spline approximation floor is 1.6e-5 for the smooth target and
    # ~0.002 for 1.2|x0|, against observed MSE of 0.0064 and 0.0038 at n=16000.
    # With bias that far below variance the sieve behaves like a correctly
    # specified ~30-parameter parametric model and earns n^-1.  A target would
    # have to be much harder for the basis (a step function's floor is 0.26-0.41)
    # before the nonparametric regime became visible at these sample sizes.
    print()
    print(f"    {'target':<10}{'slope':>9}{'95% CI':>20}{'minimax bound':>16}{'verdict':>12}")
    all_ok = True
    for tgt in ("smooth", "kinked"):
        slope, ci = res[tgt]
        bound = MINIMAX_BOUND[tgt]
        good = rate_holds(ci, bound)
        all_ok &= good
        print(f"    {tgt:<10}{slope:>9.3f}   [{ci[0]:>6.3f}, {ci[1]:>6.3f}]{bound:>16.3f}"
              f"{'OK' if good else 'VIOLATED':>12}")
    good_slope = all_ok
    ok &= good_slope
    print()
    print(f"    -> {'PASS' if good_slope else 'FAIL'}: risk decays AT LEAST as fast as "
          f"the minimax bound n^(-2r/(2r+1))")
    print(f"       (both targets in fact reach the parametric rate; see the")
    print(f"        approximation-floor analysis in theory/diagnose.py D2)")

    print()
    print("=" * 82)
    print("RESULT:", "THEOREM 4 GATE PASSES" if ok else "GATE FAILED - see above")
    print("=" * 82)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
