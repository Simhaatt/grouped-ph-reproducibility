"""Evaluation metrics for discrete-time survival predictions.

Everything here consumes an [N, T] survival matrix S[i, t] = P(T_i > bin t | x_i)
and the observed (bin_idx, event) pair, so it works identically for our estimator
and for any baseline that can produce a survival curve on the same grid.

Two choices worth stating because they are easy to get subtly wrong:

* **Antolini's C-index**, not Harrell's.  Harrell's C compares a single scalar
  risk score, which presumes proportional hazards.  Antolini's compares the
  survival probabilities AT each unit's own event time, so it stays meaningful
  when hazards are non-proportional -- exactly the case our "shared" mode allows.

* **IPCW Brier with a discrete Kaplan-Meier censoring estimate.**  Censored units
  are reweighted by the probability of remaining uncensored, otherwise the score
  is biased by the censoring pattern rather than the model.
"""

from __future__ import annotations

import numpy as np


def _as_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def km_censoring(bin_idx, event, n_bins: int) -> np.ndarray:
    """Kaplan-Meier estimate of the CENSORING survival G(t), on the bin grid.

    Same recursion as ordinary KM with the event indicator flipped: here a
    "death" is a censoring event.  Returns G evaluated at bins 0..T-1.
    """
    bin_idx = _as_np(bin_idx).astype(int)
    event = _as_np(event).astype(float)
    n = len(bin_idx)
    G = np.ones(n_bins)
    surv = 1.0
    for t in range(n_bins):
        at_risk = np.sum(bin_idx >= t)
        cens = np.sum((bin_idx == t) & (event == 0))
        if at_risk > 0:
            surv *= (1.0 - cens / at_risk)
        G[t] = surv
    return np.clip(G, 1e-8, 1.0)


def antolini_c(S, bin_idx, event) -> float:
    """Time-dependent concordance (Antolini et al., 2005).

    Concordant when the unit that fails first has the lower survival probability
    at that unit's own event time.  Ties in the predicted probability score 0.5.
    """
    S = _as_np(S)
    k = _as_np(bin_idx).astype(int)
    d = _as_np(event).astype(bool)
    n = len(k)

    num = den = 0.0
    for i in np.flatnonzero(d):
        comparable = k > k[i]              # j outlives i's event bin
        m = int(comparable.sum())
        if m == 0:
            continue
        si = S[i, k[i]]
        sj = S[comparable, k[i]]
        num += float(np.sum(sj > si) + 0.5 * np.sum(sj == si))
        den += m
    return float(num / den) if den else float("nan")


def brier(S, bin_idx, event, n_bins=None, G=None):
    """IPCW Brier score at each bin.  Returns an array of length T."""
    S = _as_np(S)
    k = _as_np(bin_idx).astype(int)
    d = _as_np(event).astype(float)
    T = n_bins or S.shape[1]
    if G is None:
        G = km_censoring(k, d, T)

    out = np.full(T, np.nan)
    for t in range(T):
        st = S[:, t]
        # died at or before t, uncensored -> true survival is 0
        died = (k <= t) & (d == 1)
        # still alive after t -> true survival is 1
        alive = k > t
        w_died = np.zeros_like(st)
        w_died[died] = 1.0 / G[np.clip(k[died] - 1, 0, T - 1)]
        w_alive = np.zeros_like(st)
        w_alive[alive] = 1.0 / G[t]
        out[t] = float(np.mean(w_died * (0.0 - st) ** 2 + w_alive * (1.0 - st) ** 2))
    return out


def integrated_brier(S, bin_idx, event, n_bins=None) -> float:
    b = brier(S, bin_idx, event, n_bins)
    return float(np.nanmean(b))


def evaluate(S, bin_idx, event, nll=None) -> dict:
    """Bundle the three headline numbers."""
    return {
        "c_index": antolini_c(S, bin_idx, event),
        "ibs": integrated_brier(S, bin_idx, event),
        "nll": float(nll) if nll is not None else float("nan"),
    }


# --------------------------------------------------------------- C-index CIs
def paired_c_bootstrap(S_a, S_b, bin_idx, event, n_boot=1000, seed=0,
                       max_units=6000, chunk=250):
    """Paired bootstrap CI for the Antolini C-index DIFFERENCE (b minus a).

    WHY THIS EXISTS.  Every other bootstrap in this repo is on per-unit NLL,
    because NLL is additive over units and so trivially resampled.  But mean NLL
    turned out not to be robust at these sample sizes -- on flchain a single test
    row with NLL 59.8 carried 99.7% of a "significant" difference over 2,362 rows
    -- so the headline moved to the C-index, which is rank-based and cannot be
    moved by one blown-up prediction.  That left the headline with no interval at
    all, which is precisely what section 4.5 criticises CoxKAN, SurvKAN, KAN-AFT
    and KAPLAN-HR for.  This closes it.

    METHOD.  Antolini's C is a ratio of sums over comparable ORDERED PAIRS, not a
    mean over units, so it cannot be resampled unit-by-unit like NLL.  Resample
    units with multiplicities w (multinomial), and every pair (i, j) then carries
    weight w_i * w_j:

        C(w) = (w_ev . (P w)) / (w_ev . (M w))

    with M[j, a] = 1{k_j > k_a} the comparability indicator for event a, and
    P[j, a] the concordance contribution (1, or 0.5 on a predicted tie).  Both
    models are scored on the SAME resample and the SAME denominator, which is
    what makes the comparison paired -- the between-unit variation that swamps a
    0.02 effect cancels.

    Self-pairs need no special handling: M[i, i] = 1{k_i > k_i} = 0 already.

    Returns c_a, c_b, delta and the 95% percentile interval for each.
    """
    S_a, S_b = _as_np(S_a), _as_np(S_b)
    k = _as_np(bin_idx).astype(int)
    d = _as_np(event).astype(bool)
    rng = np.random.default_rng(seed)

    n = len(k)
    if n > max_units:
        # The pair matrices are O(n_events * n); above a few thousand units they
        # stop fitting comfortably in memory.  Subsample UNITS once, identically
        # for both models, so the comparison stays paired.
        keep = np.sort(rng.choice(n, max_units, replace=False))
        S_a, S_b, k, d = S_a[keep], S_b[keep], k[keep], d[keep]
        n = max_units

    ev = np.flatnonzero(d)
    if len(ev) == 0:
        return {}
    kev = k[ev]

    M = (k[:, None] > kev[None, :]).astype(np.float32)        # [n, n_ev]

    def conc(S):
        at = S[:, kev]                                        # [n, n_ev]
        si = S[ev, kev][None, :]                              # [1, n_ev]
        return (((at > si).astype(np.float32)
                 + 0.5 * (at == si).astype(np.float32)) * M)
    P_a, P_b = conc(S_a), conc(S_b)

    # float64 for the POINT estimates: these are sums over millions of float32
    # entries, and the accumulation error is ~1e-7 -- two orders of magnitude
    # above the effects being reported.  The bootstrap replicates stay float32,
    # where resampling noise dominates by far.
    den0 = M.sum(dtype=np.float64)
    c_a0 = float(P_a.sum(dtype=np.float64) / den0) if den0 else float("nan")
    c_b0 = float(P_b.sum(dtype=np.float64) / den0) if den0 else float("nan")

    da, db, dd = [], [], []
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=m).astype(np.float32).T  # [n, m]
        Wev = W[ev]                                           # [n_ev, m]
        den = (M.T @ W * Wev).sum(axis=0)
        na = (P_a.T @ W * Wev).sum(axis=0)
        nb = (P_b.T @ W * Wev).sum(axis=0)
        ok = den > 0
        ca, cb = na[ok] / den[ok], nb[ok] / den[ok]
        da.append(ca); db.append(cb); dd.append(cb - ca)

    da, db, dd = np.concatenate(da), np.concatenate(db), np.concatenate(dd)
    q = lambda v: tuple(float(x) for x in np.percentile(v, [2.5, 97.5]))
    lo, hi = q(dd)
    return dict(c_a=c_a0, c_b=c_b0, delta=c_b0 - c_a0,
                delta_lo=lo, delta_hi=hi,
                c_a_ci=q(da), c_b_ci=q(db),
                sig=(lo > 0) or (hi < 0), n_units=n, n_events=len(ev))


def paired_c_influence(S_a, S_b, bin_idx, event, chunk=256):
    """Analytic SE for the Antolini C-index difference, via influence functions.

    WHY.  `paired_c_bootstrap` materialises the [n, n_events] pair matrices, which
    caps it at a few thousand units -- on KKBox (n_test=60,000) it had to subsample
    to 6,000, so the C interval there used 10% of the data and came out marginal
    while the NLL result on all 60,000 rows was decisive.  Chunking the bootstrap
    does not rescue it: n_events * n * n_boot is ~1.8e12 flops for KKBox.

    THE FIX.  C = N/D is a ratio of sums over ordered pairs, so it is a smooth
    functional of the empirical measure and has an influence function.  Under the
    multinomial bootstrap weights w,

        N(w) = sum_{a in E} sum_j w_a w_j P[j,a],   D(w) likewise with M,

    so unit i perturbs N through BOTH roles it plays -- as an event (a row of
    comparisons) and as a comparator (a column):

        dN/dw_i = 1{i in E} * sum_j P[j,i]  +  sum_a P[i,a]

    and IF_i = (dN/dw_i - C * dD/dw_i) / D by the delta method.  Var(C) is then
    the centred sum of squares of the IFs.  For the PAIRED difference the
    influences subtract, IF_i = IF_i^b - IF_i^a, which is what makes the
    between-unit variance cancel exactly as pairing intends.

    Cost is O(n_events * n) ONCE, computed in chunks over events with O(n) memory
    -- no resampling, so KKBox becomes tractable at full resolution.

    Validated against paired_c_bootstrap wherever both are computable; they agree
    to ~1e-3 on the SE.  Returns the same keys as paired_c_bootstrap.
    """
    S_a, S_b = _as_np(S_a), _as_np(S_b)
    k = _as_np(bin_idx).astype(int)
    d = _as_np(event).astype(bool)
    n = len(k)
    ev = np.flatnonzero(d)
    if len(ev) == 0:
        return {}
    kev = k[ev]

    # Accumulators for dN/dw and dD/dw, split by the two roles a unit plays.
    colN_a = np.zeros(n); colN_b = np.zeros(n); colD = np.zeros(n)
    rowN_a = np.zeros(len(ev)); rowN_b = np.zeros(len(ev)); rowD = np.zeros(len(ev))
    tot_a = tot_b = tot_d = 0.0

    for s in range(0, len(ev), chunk):
        sl = slice(s, min(s + chunk, len(ev)))
        kc = kev[sl]
        M = (k[:, None] > kc[None, :]).astype(np.float32)          # [n, c]

        def conc(S):
            at = S[:, kc]
            si = S[ev[sl], kc][None, :]
            return (((at > si).astype(np.float32)
                     + 0.5 * (at == si).astype(np.float32)) * M)
        Pa, Pb = conc(S_a), conc(S_b)

        colN_a += Pa.sum(axis=1, dtype=np.float64)
        colN_b += Pb.sum(axis=1, dtype=np.float64)
        colD += M.sum(axis=1, dtype=np.float64)
        rowN_a[sl] = Pa.sum(axis=0, dtype=np.float64)
        rowN_b[sl] = Pb.sum(axis=0, dtype=np.float64)
        rowD[sl] = M.sum(axis=0, dtype=np.float64)
        tot_a += float(Pa.sum(dtype=np.float64))
        tot_b += float(Pb.sum(dtype=np.float64))
        tot_d += float(M.sum(dtype=np.float64))

    if tot_d <= 0:
        return {}
    c_a, c_b = tot_a / tot_d, tot_b / tot_d

    def influence(col, row, c):
        g = col.copy()                       # role: comparator (column)
        g[ev] += row                         # role: event (row)
        gd = colD.copy()
        gd[ev] += rowD
        return (g - c * gd) / tot_d
    ia, ib = influence(colN_a, rowN_a, c_a), influence(colN_b, rowN_b, c_b)

    dif = ib - ia                            # paired: the shared part cancels
    se = float(np.sqrt(((dif - dif.mean()) ** 2).sum()))
    z = 1.959964
    delta = c_b - c_a
    lo, hi = delta - z * se, delta + z * se
    return dict(c_a=c_a, c_b=c_b, delta=delta, delta_lo=lo, delta_hi=hi,
                se=se, sig=(lo > 0) or (hi < 0), n_units=n, n_events=len(ev),
                method="influence")
