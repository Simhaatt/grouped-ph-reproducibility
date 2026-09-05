"""Discrete-time survival likelihood.

The right-censored discrete likelihood factorizes into Bernoulli terms over the
at-risk set, so the loss is a masked BCE on an [N, T] logit matrix:

    l = sum_i sum_{t=e_i}^{k_i} [ y_it log h_it + (1 - y_it) log(1 - h_it) ]

with y_it = delta_i * 1{t == k_i}.  This is *exactly* the person-period
expansion, without materialising N*T rows of covariates.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

LINKS = ("cloglog", "logit")

# exp(ETA_MAX) must not overflow; exp(ETA_MIN) must not underflow to 0.
ETA_MAX, ETA_MIN = 10.0, -30.0


def log_hazard(logits: torch.Tensor, link: str = "cloglog"):
    """-> (log h, log(1 - h)), numerically stable."""
    if link == "logit":
        return F.logsigmoid(logits), F.logsigmoid(-logits)
    if link != "cloglog":
        raise ValueError(f"link must be one of {LINKS}, got {link!r}")

    eta = logits.clamp(ETA_MIN, ETA_MAX)
    u = eta.exp()                       # cumulative hazard increment
    log_1mh = -u                        # log(1 - h) = -exp(eta)
    # log h = log(1 - exp(-u)) via the stable log1mexp split at log 2
    big = u > 0.693147180559945
    log_h = torch.where(
        big,
        torch.log1p(-torch.exp(-torch.where(big, u, torch.full_like(u, 1.0)))),
        torch.log(-torch.expm1(-torch.where(big, torch.full_like(u, 1.0), u))),
    )
    return log_h.clamp_min(-60.0), log_1mh


def hazard(logits: torch.Tensor, link: str = "cloglog") -> torch.Tensor:
    if link == "logit":
        return torch.sigmoid(logits)
    return -torch.expm1(-logits.clamp(ETA_MIN, ETA_MAX).exp())


def survival(logits: torch.Tensor, link: str = "cloglog") -> torch.Tensor:
    """S(t) = prod_{s<=t} (1 - h(s)).  -> [N, T]"""
    _, log_1mh = log_hazard(logits, link)
    return log_1mh.cumsum(dim=1).exp()


def make_targets(bin_idx, event, n_intervals: int, entry_idx=None):
    """Build the at-risk mask and event target.

    bin_idx  : [N] 0-based index of the bin holding the event / last follow-up
    event    : [N] 1 = failure, 0 = right-censored
    entry_idx: [N] 0-based first at-risk bin (left truncation / staggered entry)
    """
    bin_idx = torch.as_tensor(bin_idx, dtype=torch.long)
    event = torch.as_tensor(event, dtype=torch.float32)
    t = torch.arange(n_intervals, device=bin_idx.device).unsqueeze(0)  # [1, T]

    at_risk = t <= bin_idx.unsqueeze(1)
    if entry_idx is not None:
        entry_idx = torch.as_tensor(entry_idx, dtype=torch.long)
        at_risk &= t >= entry_idx.unsqueeze(1)

    mask = at_risk.to(torch.float32)
    y = (t == bin_idx.unsqueeze(1)).to(torch.float32) * event.unsqueeze(1)
    return mask, y


def nll(logits, mask, y, link: str = "cloglog", reduction: str = "mean"):
    """Negative log-likelihood, summed over bins then averaged over units."""
    log_h, log_1mh = log_hazard(logits, link)
    ll = (y * log_h + (1.0 - y) * log_1mh) * mask
    per_unit = -ll.sum(dim=1)
    if reduction == "mean":
        return per_unit.mean()
    if reduction == "sum":
        return per_unit.sum()
    return per_unit


# ---------------------------------------------------------- competing risks
def make_targets_competing(bin_idx, cause, n_intervals: int, n_causes: int,
                           entry_idx=None):
    """At-risk mask and one-hot cause target for competing risks.

    cause: [N] integer, 0 = censored, 1..K = exited via that cause.
    Returns mask [N, T] and y [N, T, K].
    """
    bin_idx = torch.as_tensor(bin_idx, dtype=torch.long)
    cause = torch.as_tensor(cause, dtype=torch.long)
    t = torch.arange(n_intervals, device=bin_idx.device).unsqueeze(0)

    at_risk = t <= bin_idx.unsqueeze(1)
    if entry_idx is not None:
        at_risk &= t >= torch.as_tensor(entry_idx, dtype=torch.long).unsqueeze(1)
    mask = at_risk.to(torch.float32)

    exit_here = (t == bin_idx.unsqueeze(1))                       # [N, T]
    y = torch.zeros(len(bin_idx), n_intervals, n_causes)
    for k in range(1, n_causes + 1):
        y[:, :, k - 1] = (exit_here & (cause == k).unsqueeze(1)).float()
    return mask, y


def nll_competing(logits, mask, y, reduction: str = "mean"):
    """Multinomial discrete-time competing-risks NLL.

    logits: [N, T, K] -- one per cause.  The reference category is "survives the
    bin", whose logit is fixed at 0, so

        P(exit via k) = e^{eta_k} / (1 + sum_j e^{eta_j})
        P(survive)    =        1  / (1 + sum_j e^{eta_j})

    This is the K-cause generalisation of the Bernoulli likelihood: at K=1 it
    reduces exactly to the logit-link discrete hazard.

    Why a competing-risks model is needed at all: treating a competing event as
    independent censoring assumes those subjects would eventually have
    experienced the event of interest.  For in-hospital death vs discharge that
    is false -- a patient who dies is never discharged -- and the naive analysis
    therefore OVERSTATES the discharge hazard.
    """
    zeros = torch.zeros_like(logits[..., :1])
    full = torch.cat([zeros, logits], dim=-1)                     # [N, T, K+1]
    logp = torch.log_softmax(full, dim=-1)
    survived = 1.0 - y.sum(dim=-1)                                # [N, T]
    ll = logp[..., 0] * survived + (logp[..., 1:] * y).sum(dim=-1)
    per_unit = -(ll * mask).sum(dim=1)
    if reduction == "mean":
        return per_unit.mean()
    if reduction == "sum":
        return per_unit.sum()
    return per_unit


def cause_hazards(logits):
    """[N, T, K] logits -> cause-specific hazards h_k(t), plus P(survive bin)."""
    zeros = torch.zeros_like(logits[..., :1])
    p = torch.softmax(torch.cat([zeros, logits], dim=-1), dim=-1)
    return p[..., 1:], p[..., 0]


def cumulative_incidence(logits):
    """CIF_k(t) = sum_{s<=t} h_k(s) * prod_{u<s} P(survive bin u).

    This is the quantity the naive analysis gets wrong.  Treating a competing
    event as censoring and reporting 1 - prod(1 - h_k) OVERSTATES the incidence
    of cause k, because it credits subjects who actually left via another cause.
    """
    h, p_surv = cause_hazards(logits)                             # [N,T,K], [N,T]
    shifted = torch.cat([torch.ones_like(p_surv[:, :1]), p_surv[:, :-1]], dim=1)
    overall = torch.cumprod(shifted, dim=1)                       # P(still at risk)
    return torch.cumsum(h * overall.unsqueeze(-1), dim=1)
