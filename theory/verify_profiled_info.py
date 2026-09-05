"""
Numerical gate for A4(d): does the Delta^2 law survive PROFILING OUT the baseline?

Theorem 3 computes information about beta with alpha treated as KNOWN.  In regime
(R2) alpha is T unknown parameters.  Because d eta_t / d alpha_s is nonzero only
when s == t, the alpha block of the information matrix is DIAGONAL, so profiling
has a closed form:

    W_t(x)      = pi_t(x) * g(u_t(x))              at-risk weight x per-bin info
    I*_T        = sum_t [ E(v^2 W_t) - E(v W_t)^2 / E(W_t) ]
                = sum_t E[W_t] * Var_{W_t}(v)

i.e. a sum over bins of (total at-risk information) x (weighted covariate variance)
-- the discrete analogue of Cox's risk-set variance.

Continuous limit, with m_k(s) = E[ v^k S(s|X) lambda(s|X) ]:

    I*_inf      = int_0^tau [ m_2(s) - m_1(s)^2 / m_0(s) ] ds

QUESTION UNDER TEST (genuinely open before running this):
Theorem 3's first-order cancellation came from an EXACT identity,
int_{I_t} S lambda = S(a_{t-1})(1 - e^{-u_t}).  The profiled information is
NONLINEAR in the weights (ratio term), so that exactness may not survive.
Is  I*_inf - I*_T  of order Delta or Delta^2 ?

Run:  python theory/verify_profiled_info.py
"""
import numpy as np

# covariate: 5 atoms, v(x) = x  (needs nondegenerate spread or Var == 0)
XS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
PX = np.full(5, 0.2)


def g_info(u):
    return u**2 * np.exp(-u) / (-np.expm1(-u))


def make_model(k, beta, tau):
    Lam = lambda s, x: s**k * np.exp(beta * x)
    lam = lambda s, x: k * s ** (k - 1) * np.exp(beta * x)
    return Lam, lam


def I_profiled_T(T, k, beta, tau):
    """Profiled (efficient) information for beta with alpha_1..alpha_T unknown."""
    Lam, _ = make_model(k, beta, tau)
    e = np.linspace(0.0, tau, T + 1)
    M0 = np.zeros(T); M1 = np.zeros(T); M2 = np.zeros(T)
    for x, p in zip(XS, PX):
        L = Lam(e, x)
        W = np.exp(-L[:-1]) * g_info(np.diff(L))       # pi_t(x) * g(u_t(x))
        M0 += p * W
        M1 += p * x * W
        M2 += p * x**2 * W
    return np.sum(M2 - M1**2 / M0)


def I_profiled_inf(k, beta, tau, n=4_000_001):
    """int_0^tau [ m2 - m1^2/m0 ] ds  with m_j(s) = E[ v^j S lambda ]."""
    Lam, lam = make_model(k, beta, tau)
    s = np.linspace(1e-12, tau, n)
    m0 = np.zeros_like(s); m1 = np.zeros_like(s); m2 = np.zeros_like(s)
    for x, p in zip(XS, PX):
        f = np.exp(-Lam(s, x)) * lam(s, x)
        m0 += p * f
        m1 += p * x * f
        m2 += p * x**2 * f
    return np.trapezoid(m2 - m1**2 / m0, s)


def unprofiled_T(T, k, beta, tau):
    """Theorem 3's quantity, for side-by-side comparison."""
    Lam, _ = make_model(k, beta, tau)
    e = np.linspace(0.0, tau, T + 1)
    tot = 0.0
    for x, p in zip(XS, PX):
        L = Lam(e, x)
        tot += p * x**2 * np.sum(np.exp(-L[:-1]) * g_info(np.diff(L)))
    return tot


def unprofiled_inf(k, beta, tau):
    Lam, _ = make_model(k, beta, tau)
    return sum(p * x**2 * (1 - np.exp(-Lam(tau, x))) for x, p in zip(XS, PX))


def const_profiled_closed_form(k, beta, tau, n=2_000_001):
    """Candidate closed form for the profiled Delta^2 constant.

    Decompose I*_inf - I*_T into two pieces.

    (B) Theorem-3 term.  Since (1-e^-u) - g(u) = psi(u), the first-order
        perturbation of F(a,b,c) = c - b^2/a is
            dF = dN_2 - 2 vbar dN_1 + vbar^2 dN_0 = E[(v - vbar)^2 S psi(u)]
        and psi(u) = u^3/12 + ... contributes
            (1/12) int E[(v - vbar)^2 S lambda^3] ds.

    (A) Nonlinearity term.  F is homogeneous of degree 1, so F(N) = Delta F(mbar)
        exactly, and int_{I_t} F(m) - Delta F(mbar) = (1/2) int mtil' Hess(F) mtil.
        The Hessian quadratic form of c - b^2/a is -(2/a)(btil - (b/a) atil)^2, and
        m1' - vbar m0' = m0 vbar', so this is -(Delta^3/12) m0 vbar'^2 per bin:
            -(1/12) int m0 vbar'^2 ds.

    The drift term enters with a MINUS sign: estimating the baseline REDUCES the
    information lost to grouping.
    """
    s = np.linspace(1e-9, tau, n)
    S = lambda x: np.exp(-s**k * np.exp(beta * x))
    lam = lambda x: k * s ** (k - 1) * np.exp(beta * x)
    m0 = sum(p * S(x) * lam(x) for x, p in zip(XS, PX))
    m1 = sum(p * x * S(x) * lam(x) for x, p in zip(XS, PX))
    vbar = m1 / m0
    dvbar = np.gradient(vbar, s)
    thm3 = np.trapezoid(
        sum(p * (x - vbar) ** 2 * S(x) * lam(x) ** 3 for x, p in zip(XS, PX)), s
    ) / 12.0
    drift = np.trapezoid(m0 * dvbar**2, s) / 12.0
    return thm3 - drift, thm3, drift


def main():
    print("=" * 84)
    print("A4(d)  PROFILED information: does the Delta^2 law survive an unknown baseline?")
    print("=" * 84)

    ok = True
    emp = {}
    for k, beta, tau in [(1.5, 0.7, 2.0), (2.5, 0.4, 1.5), (1.0, 1.0, 2.0)]:
        Iinf = I_profiled_inf(k, beta, tau)
        print(f"\n  k={k}  beta={beta}  tau={tau}    I*_inf = {Iinf:.12f}")
        print(f"  {'T':>7} {'I*_inf - I*_T':>16} {'/Delta':>13} {'/Delta^2':>13}"
              f" {'ratio prev/cur':>15}")
        prev = None
        for T in [16, 32, 64, 128, 256, 512, 1024, 2048]:
            d = tau / T
            gap = Iinf - I_profiled_T(T, k, beta, tau)
            ratio = (prev / gap) if prev else float("nan")
            print(f"  {T:>7} {gap:>16.6e} {gap/d:>13.6f} {gap/d**2:>13.6f} {ratio:>15.3f}")
            if T == 256:
                emp[(k, beta, tau)] = gap / d**2
            prev = gap
        # halving Delta should quarter an O(Delta^2) gap
        ok &= abs(ratio - 4.0) < 0.25

    print()
    print("=" * 84)
    print("CLOSED FORM for the profiled constant")
    print("  I*_inf - I*_T = (Delta^2/12)[ int E[(v-vbar)^2 S lam^3] - int m0 vbar'^2 ] + O(Delta^3)")
    print("=" * 84)
    print(f"  {'k':>5}{'beta':>6}{'tau':>5} | {'predicted':>12} {'empirical':>12} {'rel err':>10}"
          f" | {'Thm3 term':>11} {'drift credit':>13}")
    for (k, beta, tau), e in emp.items():
        c, t1, t2 = const_profiled_closed_form(k, beta, tau)
        print(f"  {k:>5}{beta:>6}{tau:>5} | {c:>12.6f} {e:>12.6f} {abs(c/e-1):>10.2e}"
              f" | {t1:>11.6f} {t2:>13.6f}")
        ok &= abs(c / e - 1) < 1e-3
    print("  drift credit is subtracted: estimating the baseline REDUCES the grouping loss.")

    print()
    print("=" * 84)
    print("SIDE BY SIDE at T=512  (profiled vs Theorem 3's known-baseline quantity)")
    print("=" * 84)
    k, beta, tau = 1.5, 0.7, 2.0
    T, d = 512, 2.0 / 512
    gp = I_profiled_inf(k, beta, tau) - I_profiled_T(T, k, beta, tau)
    gu = unprofiled_inf(k, beta, tau) - unprofiled_T(T, k, beta, tau)
    print(f"  profiled   (alpha unknown): gap={gp:.6e}   gap/Delta^2={gp/d**2:.6f}")
    print(f"  unprofiled (alpha known)  : gap={gu:.6e}   gap/Delta^2={gu/d**2:.6f}")

    print()
    print("=" * 84)
    print("VERDICT:", "Delta^2 SURVIVES profiling" if ok else
          "ORDER CHANGES under profiling - Theorem 3 does NOT extend as stated")
    # Standard machine-readable verdict line, emitted by EVERY gate so that
    # run_gates.py can classify the run without a per-gate special case.  Without
    # it this gate's self-report was unclassifiable and run_gates scored it "?",
    # which its own summary then counted as a failure -- 5/6 reported for six
    # passing gates.  A verdict a tool cannot read is a verdict nobody reads.
    print(f"RESULT: GATE {'PASSES' if ok else 'FAILED'}")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
