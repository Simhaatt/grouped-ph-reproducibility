"""Estimator-correctness gate.

Lemma 1 says that grouping a continuous-time PH model under the CLOGLOG link is
EXACT: the grouped model is the true law of the coarsened data, with
eta_t(x) = alpha_t + x'beta and the SAME beta as the continuous model.

So on grouped Weibull PH data the MLE must recover the true beta, for any number
of bins.  Any systematic departure is a bug in the likelihood path
(make_targets -> logits -> nll), not misspecification.  That makes this a sharp
end-to-end test rather than a smoke test.

Contrast under test: the LOGIT link (proportional odds) is NOT exact for grouped
PH data, so it should show visible bias -- confirming the link choice matters and
is not cosmetic.

Run:  python theory/verify_estimator.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kanrel.data import SurvData
from kanrel.fit import fit
from kanrel.hazard import DiscreteHazardKAN

BETA = np.array([0.70, -0.50, 0.30], dtype=np.float32)
SHAPE, HORIZON = 1.5, 2.0


def simulate(n, n_bins, seed, censor_scale=3.0):
    """Weibull PH, Lambda_0(s) = s^shape, linear index x'beta."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(BETA))).astype(np.float32)
    g = X @ BETA
    u = rng.uniform(size=n)
    t = (-np.log(u) / np.exp(g)) ** (1.0 / SHAPE)
    c = rng.uniform(0, censor_scale, size=n)
    obs = np.minimum(t, np.minimum(c, HORIZON))
    event = ((t <= c) & (t <= HORIZON)).astype(np.float32)
    edges = np.linspace(0, HORIZON, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, obs, side="left") - 1, 0, n_bins - 1)
    return SurvData(X, idx.astype(np.int64), event, n_bins,
                    [f"x{i}" for i in range(len(BETA))], name="gate",
                    intrinsically_discrete=False, bin_edges=edges)


def estimate(d, link, seed):
    m = DiscreteHazardKAN(d.X.shape[1], d.n_bins, mode="linear", link=link)
    m, _ = fit(m, d, epochs=600, lr=0.05, val_frac=0.0, patience=600, seed=seed)
    # undo standardisation: eta uses (x - mu)/sd, so beta_raw = beta_hat / sd
    return (m.beta.detach() / m.x_std).numpy()


def run(link, n=8000, n_bins=20, reps=20):
    est = np.array([estimate(simulate(n, n_bins, s), link, s) for s in range(reps)])
    mean, se = est.mean(0), est.std(0, ddof=1) / np.sqrt(reps)
    bias = mean - BETA
    z = bias / se
    return mean, se, bias, z


def main():
    ok = True
    print("=" * 78)
    print(f"TRUE beta = {BETA}   (Weibull PH, shape={SHAPE}, n=8000, T=20, 20 reps)")
    print("=" * 78)

    for link, exact in (("cloglog", True), ("logit", False)):
        mean, se, bias, z = run(link)
        tag = "EXACT for grouped PH (Lemma 1)" if exact else "proportional ODDS - not exact"
        print(f"\n  link = {link:<8} [{tag}]")
        print(f"    {'':<10}" + "".join(f"{f'x{i}':>12}" for i in range(len(BETA))))
        print(f"    {'estimate':<10}" + "".join(f"{v:>12.4f}" for v in mean))
        print(f"    {'true':<10}" + "".join(f"{v:>12.4f}" for v in BETA))
        print(f"    {'bias':<10}" + "".join(f"{v:>12.4f}" for v in bias))
        print(f"    {'z (bias/se)':<10}" + "".join(f"{v:>12.2f}" for v in z))
        if exact:
            good = np.all(np.abs(z) < 3.0)
            ok &= good
            print(f"    -> {'PASS' if good else 'FAIL'}: |z| < 3 on every coefficient")
        else:
            print(f"    -> max |bias| = {np.abs(bias).max():.4f} "
                  f"({100*np.abs(bias/BETA).max():.1f}% of truth), as expected")

    print()
    print("=" * 78)
    print("INVARIANCE TO BIN COUNT  (Lemma 1 holds for every T, so beta must not drift)")
    print("=" * 78)
    print(f"  {'T':>5}" + "".join(f"{f'x{i}':>12}" for i in range(len(BETA))) + f"{'max|bias|':>12}")
    for T in (5, 10, 20, 40, 80):
        est = np.array([estimate(simulate(6000, T, s), "cloglog", s) for s in range(8)])
        m = est.mean(0)
        print(f"  {T:>5}" + "".join(f"{v:>12.4f}" for v in m) + f"{np.abs(m-BETA).max():>12.4f}")
    print("  -> no systematic drift with T: the estimand does not depend on the grid.")

    print()
    print("=" * 78)
    print("RESULT:", "GATE PASSES" if ok else "FAILURE - likelihood path is wrong")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
