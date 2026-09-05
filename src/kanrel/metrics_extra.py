"""E16: the metrics Antolini concordance cannot carry on its own.

Antolini's C is rank-only.  It cannot distinguish a model whose predicted
survival curves are correctly ordered but badly calibrated from one that is
right, and section 8's claim is about predictive quality rather than ordering.
Three more are added here, in a separate module so that nothing already running
against kanrel.metrics is disturbed:

  Uno's IPCW concordance   censoring-weighted, so it estimates a population
                           concordance rather than one that drifts with the
                           censoring distribution of the particular test split.
  integrated Brier score   already in kanrel.metrics; re-exported for one call.
  ICI                      integrated calibration index: mean |observed -
                           predicted| survival, with the observed side from a
                           smoothed calibration curve.  This is what actually
                           catches a model whose ordering is fine and whose
                           probabilities are not.
"""
from __future__ import annotations

import numpy as np

from kanrel.metrics import antolini_c, integrated_brier, km_censoring


def uno_c(S, bin_idx, event, n_bins=None, tau=None):
    """Uno's IPCW concordance on the discrete grid.

    Pairs are ordered by predicted survival at the earlier subject's exit bin, and
    weighted by 1/G(k_i)^2 with G the Kaplan-Meier estimate of the censoring
    distribution.  Restricting to k_i < tau keeps the weights bounded, which is
    the usual and necessary truncation: without it the last few uncensored
    subjects carry unbounded weight.
    """
    S = np.asarray(S, float)
    bin_idx = np.asarray(bin_idx).astype(int)
    event = np.asarray(event).astype(float)
    T = int(S.shape[1] if n_bins is None else n_bins)
    G = km_censoring(bin_idx, event, T)
    G = np.clip(np.asarray(G, float), 1e-6, None)
    if tau is None:
        ev_bins = bin_idx[event == 1]
        tau = int(np.quantile(ev_bins, 0.90)) if ev_bins.size else T - 1
    num = den = 0.0
    for i in np.flatnonzero((event == 1) & (bin_idx <= tau)):
        ki = bin_idx[i]
        comp = bin_idx > ki                       # still at risk after i's exit
        if not comp.any():
            continue
        w = 1.0 / (G[ki] ** 2)
        si = S[i, ki]
        sj = S[comp, ki]
        num += w * float((sj > si).sum() + 0.5 * (sj == si).sum())
        den += w * float(comp.sum())
    return float(num / den) if den > 0 else float("nan")


def ici(S, bin_idx, event, n_bins=None, horizon=None, n_groups=10):
    """Integrated calibration index at a fixed horizon.

    Subjects are grouped into deciles of predicted survival at `horizon`; the
    observed side is the Kaplan-Meier estimate within each group, which handles
    censoring correctly where a raw event proportion would not.  Reported as the
    size-weighted mean |observed - predicted|, so 0 is perfect and the units are
    probability.
    """
    S = np.asarray(S, float)
    bin_idx = np.asarray(bin_idx).astype(int)
    event = np.asarray(event).astype(float)
    T = int(S.shape[1] if n_bins is None else n_bins)
    if horizon is None:
        horizon = int(np.median(bin_idx[event == 1])) if (event == 1).any() else T // 2
    horizon = int(np.clip(horizon, 0, T - 1))
    p = S[:, horizon]
    if np.allclose(p, p[0]):
        return float("nan")
    q = np.quantile(p, np.linspace(0, 1, n_groups + 1))
    q[0] -= 1e-9
    grp = np.clip(np.searchsorted(q, p, side="left") - 1, 0, n_groups - 1)
    tot, wsum = 0.0, 0.0
    for g in range(n_groups):
        m = grp == g
        if m.sum() < 5:
            continue
        # Kaplan-Meier within the group, evaluated at `horizon`.
        surv, at_risk = 1.0, int(m.sum())
        bi, ei = bin_idx[m], event[m]
        for t in range(horizon + 1):
            d = int(((bi == t) & (ei == 1)).sum())
            if at_risk > 0 and d > 0:
                surv *= 1.0 - d / at_risk
            at_risk -= int((bi == t).sum())
            if at_risk <= 0:
                break
        tot += m.sum() * abs(surv - float(p[m].mean()))
        wsum += m.sum()
    return float(tot / wsum) if wsum > 0 else float("nan")


def evaluate_full(S, bin_idx, event, nll=None, n_bins=None):
    """Every metric in one dict, so no arm is scored on a different set."""
    return {
        "nll": float(nll) if nll is not None else float("nan"),
        "c_antolini": antolini_c(S, bin_idx, event),
        "c_uno": uno_c(S, bin_idx, event, n_bins),
        "ibs": integrated_brier(S, bin_idx, event, n_bins),
        "ici": ici(S, bin_idx, event, n_bins),
    }


METRIC_ORDER = ("nll", "c_antolini", "c_uno", "ibs", "ici")
HIGHER_IS_BETTER = {"nll": False, "c_antolini": True, "c_uno": True,
                    "ibs": False, "ici": False}
