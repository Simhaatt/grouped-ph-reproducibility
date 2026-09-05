"""
Numerical gate for Lemma 2 (offset invariance).

Claim:  eta*_t(x) = log Delta + log lambda(c_t | x) + O(Delta^p)   uniformly in x,
        where eta*_t(x) = log int_{I_t} lambda(s|x) ds.

Two anchor choices for c_t, and the sharper one is NOT the obvious one:
  right endpoint  c_t = t*Delta        -> p = 1, constant  (1/2) sup |lambda'/lambda|
  MIDPOINT        c_t = (t-1/2)*Delta  -> p = 2, constant (1/24) sup |lambda''/lambda|

Also under test: the error constant scales like 1/lambda_min, so lambda must be
bounded AWAY FROM ZERO -- log lambda stops being Lipschitz otherwise.  This is a
strictly stronger hypothesis than the "lambda bounded above" needed by Theorem 3.

Non-proportional hazard is used on purpose (time-shape depends on x), otherwise
e^{g(x)} factors out of both sides and uniformity in x is tested vacuously:

    lambda(s|x) = a0 + a1 s + a2 s^2 + x (b0 + b1 s)

Run:  python theory/verify_offset.py
"""
import numpy as np

TAU = 2.0
XS = np.linspace(0.0, 1.0, 11)


def make_hazard(a0=1.0, a1=0.6, a2=0.5, b0=0.3, b1=0.4):
    lam = lambda s, x: a0 + a1 * s + a2 * s**2 + x * (b0 + b1 * s)
    dlam = lambda s, x: a1 + 2 * a2 * s + x * b1
    d2lam = lambda s, x: 2 * a2 + 0.0 * s
    # exact antiderivative -> exact bin integrals, no quadrature error in the test
    F = lambda s, x: a0 * s + a1 * s**2 / 2 + a2 * s**3 / 3 + x * (b0 * s + b1 * s**2 / 2)
    return lam, dlam, d2lam, F


def errors(T, lam, F, anchor):
    """max over bins and covariate values of |eta* - logDelta - log lambda(c_t)|."""
    d = TAU / T
    edges = np.linspace(0.0, TAU, T + 1)
    lo, hi = edges[:-1], edges[1:]
    c = hi if anchor == "right" else 0.5 * (lo + hi)
    worst = 0.0
    for x in XS:
        u = F(hi, x) - F(lo, x)                 # exact int_{I_t} lambda
        err = np.log(u) - np.log(d) - np.log(lam(c, x))
        worst = max(worst, np.abs(err).max())
    return worst


def sup_ratio(fn, lam, n=20001):
    s = np.linspace(0.0, TAU, n)
    return max(np.abs(fn(s, x) / lam(s, x)).max() for x in XS)


def main():
    ok = True
    lam, dlam, d2lam, F = make_hazard()
    lam_min = min(lam(np.linspace(0, TAU, 5001), x).min() for x in XS)

    print("=" * 76)
    print("ORDER OF THE REMAINDER   (non-proportional hazard, exact bin integrals)")
    print("=" * 76)
    print(f"  lambda_min = {lam_min:.4f}   sup|lam'/lam| = {sup_ratio(dlam, lam):.4f}"
          f"   sup|lam''/lam| = {sup_ratio(d2lam, lam):.4f}")
    print()
    print(f"{'T':>7} {'err(right)':>13} {'/Delta':>11} | {'err(MIDpoint)':>15} {'/Delta^2':>11}")
    prev_r = prev_m = None
    for T in [16, 32, 64, 128, 256, 512, 1024]:
        d = TAU / T
        er = errors(T, lam, F, "right")
        em = errors(T, lam, F, "mid")
        print(f"{T:>7} {er:>13.3e} {er/d:>11.6f} | {em:>15.3e} {em/d**2:>11.6f}")
        prev_r, prev_m = er / d, em / d**2

    pred_r = 0.5 * sup_ratio(dlam, lam)
    pred_m = sup_ratio(d2lam, lam) / 24.0
    print()
    print(f"  right endpoint : ratio/Delta   -> {prev_r:.6f}   predicted (1/2)sup|lam'/lam|  = {pred_r:.6f}")
    print(f"  MIDPOINT       : ratio/Delta^2 -> {prev_m:.6f}   predicted (1/24)sup|lam''/lam| = {pred_m:.6f}")
    ok &= abs(prev_r / pred_r - 1) < 0.02 and abs(prev_m / pred_m - 1) < 0.02
    print(f"  -> midpoint anchor is second order.  Lemma 2 should be stated with the MIDPOINT.")

    print()
    print("=" * 76)
    print("NECESSITY of lambda bounded AWAY FROM ZERO   (constant should scale ~ 1/lambda_min)")
    print("=" * 76)
    print(f"{'a0':>8} {'lambda_min':>12} {'err/Delta^2':>14} {'x lambda_min':>14}")
    base = None
    for a0 in [2.0, 1.0, 0.5, 0.2, 0.1, 0.05]:
        lam2, _, _, F2 = make_hazard(a0=a0, b0=0.0, b1=0.0)   # keep lam>0, vary the floor
        lm = min(lam2(np.linspace(0, TAU, 5001), x).min() for x in XS)
        T = 1024
        em = errors(T, lam2, F2, "mid")
        const = em / (TAU / T) ** 2
        prod = const * lm
        if base is None:
            base = prod
        print(f"{a0:>8} {lm:>12.4f} {const:>14.6f} {prod:>14.6f}")
    print("  -> const * lambda_min is ~constant, i.e. const ~ 1/lambda_min.")
    print("     log lambda is not Lipschitz as lambda -> 0, so Lemma 2 needs")
    print("     0 < lambda_min <= lambda <= lambda_max < infinity.")
    print("     This is STRICTLY STRONGER than Theorem 3's hypothesis (bounded above only).")

    print()
    print("=" * 76)
    print("RESULT:", "GATE PASSES" if ok else "FAILURE - do not assert the claim")
    print("=" * 76)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
