"""Discrete-hazard KAN models.

All modes emit an [N, T] matrix of logits consumed directly by
``kanrel.likelihood.nll``.  Under the cloglog link, Lemma 1 makes that objective
the EXACT likelihood of the coarsened data -- no quadrature anywhere -- which is
the property the continuous-time KAN survival models do not have.

Modes
-----
linear    eta_t(x) = alpha_t + x'beta
          Prentice-Gloeckler grouped PH.  This is Nawata et al.'s model and the
          comparator our KAN nests; also the estimator-correctness gate, since
          data from ``make_grouped_weibull`` is exactly this model.

baseline  eta_t(x) = alpha_t + g_KAN(x)                       <- Theorem 4, regime (R2)
          Free baseline plus one nonparametric covariate function.  Proportional
          in the discrete sense; g is interpretable edge-by-edge.

shared    eta_t(x) = KAN([x, tau_t])
          Time as a KAN input, so effects may vary with t.  This is what SurvKAN
          and KAPLAN-HR do, but in exact discrete time.

multi     eta_t(x) = alpha_t + g_KAN(x)_t
          KAN emits all T logits; maximally flexible, least interpretable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .kan import KAN
from .likelihood import hazard as _hazard
from .likelihood import nll as _nll
from .likelihood import survival as _survival

MODES = ("linear", "baseline", "shared", "multi")


class DiscreteHazardKAN(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_bins: int,
        hidden: tuple[int, ...] = (16,),
        mode: str = "baseline",
        link: str = "cloglog",
        grid_size: int = 8,
        spline_order: int = 3,
        alpha_init: float = -2.5,
        time_chunk: int = 8,
        cont_idx=None,
        **kan_kwargs,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode, self.link, self.n_bins, self.n_features = mode, link, n_bins, n_features
        self.time_chunk = time_chunk        # bins per forward pass in "shared" mode

        # Which columns deserve a spline.
        #
        # A cubic spline with G knots is G+3 coefficients.  Over a 0/1 dummy those
        # are constrained by exactly TWO distinct x-values, so the extra capacity
        # is pure variance and the fitted edge function is arbitrary between 0 and
        # 1.  On support2/slos 19 of 32 columns are one-hot dummies -- 59% of the
        # KAN's capacity spent on features that cannot be nonlinear.  Passing
        # cont_idx routes those columns through a plain linear term instead, so
        # the splines are spent only where nonlinearity is possible.
        if cont_idx is None:
            cont = list(range(n_features))
        else:
            cont = sorted(int(i) for i in cont_idx)
        binf = [i for i in range(n_features) if i not in set(cont)]
        self.register_buffer("cont_idx", torch.tensor(cont, dtype=torch.long))
        self.register_buffer("bin_idx_feat", torch.tensor(binf, dtype=torch.long))
        self.n_cont, self.n_bin_feat = len(cont), len(binf)
        if self.n_bin_feat and mode != "linear":
            self.beta_bin = nn.Parameter(torch.zeros(self.n_bin_feat))

        # standardisation, filled by set_standardization
        self.register_buffer("x_mean", torch.zeros(n_features))
        self.register_buffer("x_std", torch.ones(n_features))

        # free baseline log-hazard; not used by "shared", which learns it on the
        # time axis of the KAN itself (Lemma 2: that axis IS the baseline).
        if mode != "shared":
            self.alpha = nn.Parameter(torch.full((n_bins,), float(alpha_init)))

        kw = dict(grid_size=grid_size, spline_order=spline_order, **kan_kwargs)
        if mode == "linear":
            self.beta = nn.Parameter(torch.zeros(n_features))
            self.kan = None
        elif mode == "baseline":
            self.kan = KAN([self.n_cont, *hidden, 1], **kw)
        elif mode == "shared":
            self.kan = KAN([self.n_cont + 1, *hidden, 1], **kw)
        else:  # multi
            self.kan = KAN([self.n_cont, *hidden, n_bins], **kw)

    # ------------------------------------------------------------------ setup
    @torch.no_grad()
    def set_standardization(self, X: torch.Tensor):
        X = torch.as_tensor(X, dtype=torch.float32)
        self.x_mean.copy_(X.mean(0))
        self.x_std.copy_(X.std(0).clamp_min(1e-6))

    def _z(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.x_mean) / self.x_std

    # ---------------------------------------------------------------- forward
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """-> [N, T] logits."""
        z = self._z(torch.as_tensor(X, dtype=torch.float32))
        n = z.shape[0]

        if self.mode == "linear":
            return self.alpha.unsqueeze(0) + (z @ self.beta).unsqueeze(1)

        # splines on the continuous columns, a linear term on the binary ones
        zc = z.index_select(1, self.cont_idx)
        lin = (z.index_select(1, self.bin_idx_feat) @ self.beta_bin).unsqueeze(1) \
            if self.n_bin_feat else 0.0

        if self.mode == "baseline":
            return self.alpha.unsqueeze(0) + self.kan(zc) + lin   # [N,1] broadcasts

        if self.mode == "multi":
            return self.alpha.unsqueeze(0) + self.kan(zc) + lin   # [N,T]

        z = zc                                                     # shared: KAN on cont+time

        # shared: evaluate time slices in chunks.
        #
        # The obvious implementation expands to [N*T, p+1] in one go.  That costs
        # O(N*T*p*(G+k)) floats once the B-spline basis is built -- ~555 MB at
        # N=6.4k, T=60, p=33, G=8, and utterly infeasible at KKBox scale
        # (200k x 240 bins).  Chunking over t does the same FLOPs with T/chunk
        # times less peak memory.
        tau = torch.linspace(-1.0, 1.0, self.n_bins, device=z.device, dtype=z.dtype)
        chunk = max(1, min(self.time_chunk, self.n_bins))
        cols = []
        for lo in range(0, self.n_bins, chunk):
            hi = min(lo + chunk, self.n_bins)
            w = hi - lo
            zt = torch.cat(
                [z.repeat_interleave(w, 0), tau[lo:hi].repeat(n).unsqueeze(1)], dim=1
            )
            cols.append(self.kan(zt).view(n, w))
        return torch.cat(cols, dim=1) + lin

    # ------------------------------------------------------------- quantities
    def hazard(self, X) -> torch.Tensor:
        return _hazard(self.forward(X), self.link)

    def survival(self, X) -> torch.Tensor:
        return _survival(self.forward(X), self.link)

    def loss(self, X, mask, y) -> torch.Tensor:
        return _nll(self.forward(X), mask, y, self.link)

    # -------------------------------------------------------- grid refitting
    @torch.no_grad()
    def refit_grids(self, X):
        """Refit each KAN layer's knots to the empirical input distribution.

        Owned by the model rather than the training loop: only the model knows
        which columns reach the KAN (``cont_idx``) and whether a time axis is
        appended (``shared``).  A previous version had ``fit`` build this input
        itself, which broke the moment binary columns stopped being passed to the
        KAN -- it fed 32 columns to a network expecting 13.
        """
        if self.kan is None:
            return
        z = self._kan_input(torch.as_tensor(X, dtype=torch.float32))
        self.kan.update_grids(z)

    def _kan_input(self, X_or_z, standardized: bool = False):
        """Columns the KAN actually consumes, with a zero time axis if needed."""
        z = X_or_z if standardized else self._z(X_or_z)
        z = z.index_select(1, self.cont_idx)
        if self.mode == "shared":
            tau = torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
            z = torch.cat([z, tau], dim=1)
        return z

    # ------------------------------------------------------- regularisation
    def penalty(self, X, l1: float = 0.0, entropy: float = 1.0, smooth: float = 0.0):
        """KAN sparsification + second-difference (P-spline) penalty on alpha."""
        pen = torch.zeros((), device=self.x_mean.device)
        if l1 > 0 and self.kan is not None:
            z = self._kan_input(torch.as_tensor(X, dtype=torch.float32))
            pen = pen + l1 * self.kan.regularization(z, entropy)
        if smooth > 0 and hasattr(self, "alpha") and self.n_bins > 2:
            d2 = self.alpha[2:] - 2 * self.alpha[1:-1] + self.alpha[:-2]
            pen = pen + smooth * d2.pow(2).sum()
        return pen

    # ------------------------------------------------------ interpretability
    @torch.no_grad()
    def partial_effect(self, X, feature: int, grid: torch.Tensor):
        """Sweep one covariate with the others held at their median.

        Model-agnostic and identifiable as stated: it reports g at a specified
        reference point, which is what makes it comparable across fits.  It is
        NOT the same object as a raw KAN edge plot, which is not identifiable in
        a multi-layer network (see the identifiability opening in the survey).
        """
        X = torch.as_tensor(X, dtype=torch.float32)
        grid = torch.as_tensor(grid, dtype=torch.float32)   # accept numpy too
        ref = X.median(dim=0).values.repeat(len(grid), 1)
        ref[:, feature] = grid
        eta = self.forward(ref)
        return eta[:, 0] - eta[:, 0].mean()

    @torch.no_grad()
    def feature_importance(self, X) -> torch.Tensor:
        if self.kan is None:
            return self.beta.abs().detach()
        z = self._kan_input(torch.as_tensor(X, dtype=torch.float32))
        return self.kan.input_importance(z)
