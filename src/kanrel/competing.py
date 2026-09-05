"""Discrete-time competing-risks hazard KAN (A7).

Kept separate from `DiscreteHazardKAN` on purpose: that class is validated by
seven gates and there is no reason to risk it for an extension.

Model, for causes k = 1..K:

    eta_t^k(x) = alpha_t^k + g_k(x)          "survive the bin" is the reference
    P(exit via k in bin t | at risk) = softmax over {survive, 1..K}

Motivation.  `support2/slos` records length of stay with TWO exits: discharged
alive (70.9%) and died in hospital (25.9%).  Every earlier fit treated death as
independent censoring, which assumes those patients would eventually have been
discharged.  They would not have been.  The naive analysis therefore overstates
the discharge hazard, and the cumulative incidence of discharge is overstated
further still, because 1 - prod(1 - h_discharge) credits subjects who actually
died to the discharge curve.

The quantity that makes the bias visible is the cumulative incidence function,
not the hazard: CIF weights each bin's cause-specific hazard by the probability
of still being at risk, which correctly accounts for the competing exit.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .kan import KAN
from .likelihood import cause_hazards, cumulative_incidence, nll_competing


class CompetingHazardKAN(nn.Module):
    def __init__(self, n_features: int, n_bins: int, n_causes: int = 2,
                 hidden: tuple[int, ...] = (), grid_size: int = 8,
                 spline_order: int = 3, alpha_init: float = -2.5,
                 cont_idx=None, linear: bool = False, **kan_kwargs):
        super().__init__()
        self.n_bins, self.n_causes, self.linear = n_bins, n_causes, linear
        self.register_buffer("x_mean", torch.zeros(n_features))
        self.register_buffer("x_std", torch.ones(n_features))

        cont = list(range(n_features)) if cont_idx is None else sorted(int(i) for i in cont_idx)
        binf = [i for i in range(n_features) if i not in set(cont)]
        self.register_buffer("cont_idx", torch.tensor(cont, dtype=torch.long))
        self.register_buffer("bin_idx_feat", torch.tensor(binf, dtype=torch.long))
        self.n_cont, self.n_bin_feat = len(cont), len(binf)

        self.alpha = nn.Parameter(torch.full((n_bins, n_causes), float(alpha_init)))
        if linear:
            self.beta = nn.Parameter(torch.zeros(n_features, n_causes))
            self.kan = None
        else:
            self.kan = KAN([self.n_cont, *hidden, n_causes],
                           grid_size=grid_size, spline_order=spline_order, **kan_kwargs)
            if self.n_bin_feat:
                self.beta_bin = nn.Parameter(torch.zeros(self.n_bin_feat, n_causes))

    @torch.no_grad()
    def set_standardization(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        self.x_mean.copy_(X.mean(0))
        self.x_std.copy_(X.std(0).clamp_min(1e-6))

    def _z(self, X):
        return (torch.as_tensor(X, dtype=torch.float32) - self.x_mean) / self.x_std

    def forward(self, X) -> torch.Tensor:
        """-> [N, T, K] logits."""
        z = self._z(X)
        if self.linear:
            g = z @ self.beta                                   # [N, K]
        else:
            g = self.kan(z.index_select(1, self.cont_idx))      # [N, K]
            if self.n_bin_feat:
                g = g + z.index_select(1, self.bin_idx_feat) @ self.beta_bin
        return self.alpha.unsqueeze(0) + g.unsqueeze(1)         # [N, T, K]

    def loss(self, X, mask, y):
        return nll_competing(self.forward(X), mask, y)

    def penalty(self, X, l1: float = 0.0, entropy: float = 1.0, smooth: float = 0.0):
        pen = torch.zeros((), device=self.x_mean.device)
        if l1 > 0 and self.kan is not None:
            pen = pen + l1 * self.kan.regularization(
                self._z(X).index_select(1, self.cont_idx), entropy)
        if smooth > 0 and self.n_bins > 2:
            d2 = self.alpha[2:] - 2 * self.alpha[1:-1] + self.alpha[:-2]
            pen = pen + smooth * d2.pow(2).sum()
        return pen

    # ------------------------------------------------------------ quantities
    def hazards(self, X):
        return cause_hazards(self.forward(X))[0]                # [N, T, K]

    def cif(self, X):
        """Cumulative incidence, correctly accounting for the competing exit."""
        return cumulative_incidence(self.forward(X))            # [N, T, K]

    def naive_cif(self, X, cause: int = 0):
        """What the censoring-based analysis reports: 1 - prod(1 - h_k).

        Wrong whenever a competing exit exists, because it credits subjects who
        left via the OTHER cause to this cause's curve.  Provided here so the
        bias can be measured rather than asserted.
        """
        h = self.hazards(X)[..., cause]
        return 1.0 - torch.cumprod(1.0 - h, dim=1)
